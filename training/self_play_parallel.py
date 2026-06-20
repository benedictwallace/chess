import os
import multiprocessing as mp
import queue
import threading
import time
import torch
import numpy as np

from training.self_play import play_game


class RemoteEvaluator:
    """
    A lightweight proxy object passed to MCTS workers instead of a real neural network.
    Routes evaluation requests to the central GPU batcher and blocks until response arrives.
    """
    def __init__(self, worker_id, request_queue, response_pipe):
        self.is_proxy = True
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_pipe = response_pipe

    def evaluate_remote(self, planes):
        self.request_queue.put((self.worker_id, planes))
        policy_logits, value = self.response_pipe.recv()
        return policy_logits, value


def gpu_batcher_loop(net, request_queue, response_pipes, stop_event, batch_size=64, timeout=0.002):
    """
    Runs on the main thread. Collects cross-worker positions, batches them, 
    and performs a high-efficiency GPU forward pass under a healthy TLS state.
    """
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")
    net.eval()
    
    while not stop_event.is_set() or not request_queue.empty():
        requests = []
        start_time = time.time()
        
        while len(requests) < batch_size:
            try:
                rem = timeout - (time.time() - start_time)
                if rem <= 0 and len(requests) > 0:
                    break
                req = request_queue.get(timeout=max(0.0001, rem) if len(requests) > 0 else 0.05)
                requests.append(req)
            except queue.Empty:
                if len(requests) > 0:
                    break
                if stop_event.is_set():
                    return

        if not requests:
            continue
            
        worker_ids = [r[0] for r in requests]
        planes_list = [r[1] for r in requests]
        
        batch_x = torch.from_numpy(np.stack(planes_list)).to(device)
        
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    out = net(batch_x)
            else:
                out = net(batch_x)
        # out[2:] (ease/aux heads) is ignored: search only needs policy+value,
        # and ease targets are derived from the search tree, not the net.
        policy_logits = out[0].cpu().numpy()
        value_tensor = out[1].cpu().numpy()
        
        for idx, worker_id in enumerate(worker_ids):
            response_pipes[worker_id].send((policy_logits[idx], float(value_tensor[idx, 0])))


def _worker_process_loop(worker_id, task_queue, result_queue, request_queue, response_pipe, 
                          iterations, max_plies, temp_moves, c, ease_signal=None):
    """
    Worker process loop: pin to one thread and run asynchronous searches.
    """
    torch.set_num_threads(1)
    evaluator = RemoteEvaluator(worker_id, request_queue, response_pipe)
    
    while True:
        try:
            _task_idx = task_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break
            
        try:
            game_examples = play_game(evaluator, iterations, max_plies, temp_moves, c,
                                      ease_signal=ease_signal)
        except Exception:
            # A crashed worker must STILL report something, or the collector
            # blocks forever on result_queue.get(), stop_event is never set,
            # and the main-thread batcher spins indefinitely. Emit an empty
            # game as a sentinel so the run can complete.
            import traceback
            traceback.print_exc()
            game_examples = []
        result_queue.put(game_examples)


def generate_games_parallel(net, num_games, iterations=100, max_plies=200,
                            temp_moves=30, c=1.5, workers=None, verbose=True,
                            ease_signal=None):
    """
    Generates parallel games while executing the GPU model inference exclusively
    on the main execution thread to preserve CUDA Graph context.
    """
    if num_games <= 0:
        return []
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, num_games))

    ctx = mp.get_context("spawn")
    
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    request_queue = ctx.Queue()
    
    for i in range(num_games):
        task_queue.put(i)
        
    response_pipes_parent = []
    response_pipes_child = []
    for _ in range(workers):
        p_conn, c_conn = ctx.Pipe(duplex=True)
        response_pipes_parent.append(p_conn)
        response_pipes_child.append(c_conn)
        
    processes = []
    for worker_id in range(workers):
        p = ctx.Process(
            target=_worker_process_loop,
            args=(worker_id, task_queue, result_queue, request_queue, response_pipes_child[worker_id],
                  iterations, max_plies, temp_moves, c, ease_signal)
        )
        p.start()
        processes.append(p)
        
    all_examples = []
    stop_event = threading.Event()

    # Define a lightweight background thread to collect finished game histories
    def result_collector_loop():
        done = 0
        while done < num_games:
            examples = result_queue.get()
            all_examples.extend(examples)
            done += 1
            if verbose:
                print(f"  game {done}/{num_games}: {len(examples)} positions (total {len(all_examples)})", flush=True)
        # Once all games have successfully landed, signal the main thread batcher to wrap up
        stop_event.set()

    collector_thread = threading.Thread(target=result_collector_loop)
    collector_thread.start()
    
    # Run the main GPU batcher loop on the MAIN THREAD.
    # batch_size is capped by how many requests can be in flight at once: each
    # worker blocks on its pipe after one request, so at most `workers` are
    # outstanding. A larger batch_size just made the loop wait out the full
    # timeout every batch for requests that can never arrive.
    gpu_batcher_loop(net, request_queue, response_pipes_parent, stop_event,
                     batch_size=max(1, workers), timeout=0.002)
    
    # Clean up and join all subprocesses and collection threads
    for p in processes:
        p.join()
        
    collector_thread.join()
    
    return all_examples
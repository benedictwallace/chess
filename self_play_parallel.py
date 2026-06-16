"""
Parallel self-play.

Self-play games are independent, so we run many at once across CPU processes.
Each worker holds its own CPU copy of the (small) network and plays whole games;
the GPU is left free for training and for other cluster users. This is the right
split for this engine because self-play is bottlenecked on pure-Python board
logic, not on network inference -- so N cores buys ~N x throughput.

Drop-in for self_play.generate_games: same arguments (plus `workers`), same
return value (a flat list of (planes, policy_target, value_target,
ease_target, ease_mask) tuples).

Two pitfalls handled here:
  * Thread oversubscription -- each worker is pinned to ONE torch thread, so
    `workers` processes don't each spawn `ncores` BLAS threads and thrash.
  * PyTorch + multiprocessing -- we use the 'spawn' start method (fork after a
    CUDA context is initialised is unsafe; workers also stay on CPU).
"""

import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from network import ChessNet
from self_play import play_game


# Set in each worker process by the initializer; never touched in the parent.
_WORKER_NET = None


def _init_worker(state_dict, channels, num_blocks):
    """Runs once per worker process: pin to one thread, rebuild the net on CPU."""
    global _WORKER_NET
    torch.set_num_threads(1)          # one core per worker -- avoid oversubscription
    net = ChessNet(channels=channels, num_blocks=num_blocks)
    net.load_state_dict(state_dict)
    net.eval()
    _WORKER_NET = net


def _play_one(args):
    """Play a single game with this worker's net. Top-level so it is picklable."""
    iterations, max_plies, temp_moves, c = args
    return play_game(_WORKER_NET, iterations, max_plies, temp_moves, c)


def generate_games_parallel(net, num_games, iterations=100, max_plies=200,
                            temp_moves=30, c=1.5, workers=None, verbose=True):
    """
    Generate self-play examples using `workers` CPU processes.

    Args mirror self_play.generate_games. `workers=None` uses all cores.
    The net may live on any device; its weights are snapshotted to CPU and
    each worker rebuilds its own copy. Returns a flat list of example tuples.
    """
    if num_games <= 0:
        return []
    if workers is None:
        workers = os.cpu_count() or 1
    workers = max(1, min(workers, num_games))

    # snapshot weights + architecture for the workers (read straight off the net)
    channels = net.stem[0].out_channels
    num_blocks = len(net.blocks)
    state_dict = {k: v.detach().cpu() for k, v in net.state_dict().items()}

    ctx = mp.get_context("spawn")     # safe with torch; workers re-import cleanly
    task = (iterations, max_plies, temp_moves, c)

    all_examples = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx,
        initializer=_init_worker, initargs=(state_dict, channels, num_blocks),
    ) as ex:
        futures = [ex.submit(_play_one, task) for _ in range(num_games)]
        for fut in as_completed(futures):
            examples = fut.result()
            all_examples.extend(examples)
            done += 1
            if verbose:
                print(f"  game {done}/{num_games}: {len(examples)} positions "
                      f"(total {len(all_examples)})", flush=True)

    return all_examples
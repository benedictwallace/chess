"""
GPU-batched parallel Elo scoring.

Architecture (mirrors training/self_play_parallel.py, extended to many nets):

    CPU worker processes run the matches' MCTS (one torch thread each, 'spawn').
    They hold NO network -- every position evaluation is a RemoteEvaluator proxy
    call that ships (worker_id, net_id, planes) to a central batcher.

    The MAIN process runs gpu_batcher_loop: it collects requests from ALL workers,
    groups them by net_id, runs one batched GPU forward per net, and routes the
    results back over per-worker pipes.

Why this and not "device=cuda in each worker": a 128x8 net evaluated at batch
size 1 across N independent CUDA contexts is launch-overhead-bound and contends
for one GPU -- typically SLOWER than multi-core CPU, and prone to OOM on context
memory. The win on GPU comes from BATCHING across concurrent games, which only the
central batcher can do. (For a net this small the speedup is modest; the bottleneck
is the Python MCTS on the CPU workers, not the forward pass -- so keep `workers`
high to fill batches, and don't expect miracles over the CPU version.)

Bonus fix over the old CPU script: nets are loaded from disk ONCE on the main
process, not re-loaded inside every match.

Scheduling for GPU throughput: the unit of parallel work is a single GAME, and
games are emitted grouped by pairing. So at any instant the workers are clustered
on the same one or two pairings -> only ~2 distinct nets are ever in flight -> the
batcher builds per-net batches of ~workers/2, the same batching profile self-play
gets. (Parallelising whole matches instead spread the workers across ~20 different
checkpoints at once, collapsing every GPU batch to size 1 -- the cause of the
"hours with no progress" slowdown.)

Same schedule, same Elo fit, same resumable cache as score_elo.py.

Usage:
    python -m evaluation.score_elo_parallel --ckpt-dir checkpoints --workers 18 --games 30 --iterations 400
    python -m evaluation.score_elo_parallel --ckpt-dir checkpoints --stride 5 --round-robin
"""

import argparse
import os
import queue
import random
import sys
import threading
import time
import multiprocessing as mp

import numpy as np
import torch

from evaluation.arena import make_remote_agent, play_game, load_net
from evaluation.score_elo import discover_checkpoints, fit_elo, load_cache, append_cache
from model.network import ChessNet


# --------------------------------------------------------------------------- #
# progress bar
# --------------------------------------------------------------------------- #
def _fmt(secs):
    secs = int(secs)
    h, secs = divmod(secs, 3600)
    m, s = divmod(secs, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _progress(done, total, t0, width=28):
    frac = done / total if total else 1.0
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else 0.0
    sys.stdout.write(
        f"\r  matches |{bar}| {done}/{total} ({frac*100:3.0f}%)  "
        f"elapsed {_fmt(elapsed)}  eta {_fmt(eta)}  ({rate:.2f}/s)   "
    )
    sys.stdout.flush()


# --------------------------------------------------------------------------- #
# central GPU batcher
# --------------------------------------------------------------------------- #
def _infer_torch(net, planes_np, device, use_amp):
    """One batched forward. Returns (policy_logits_np (B,A), values_np (B,1))."""
    batch = torch.from_numpy(np.ascontiguousarray(planes_np)).to(device)
    with torch.no_grad():
        if use_amp:
            with torch.autocast("cuda"):
                logits, values = net(batch)
        else:
            logits, values = net(batch)
    return logits.float().cpu().numpy(), values.float().cpu().numpy()


def _serve_requests(requests, nets, response_pipes, infer):
    """
    Group a batch of (worker_id, net_id, planes) by net_id, run one forward per
    net, and send (logits, value) back to each requesting worker. Pure routing
    logic -- `infer(net, planes_np)->(logits_np, values_np)` is injected so this
    is unit-testable without torch or a GPU.
    """
    buckets = {}
    for (worker_id, net_id, planes) in requests:
        buckets.setdefault(net_id, []).append((worker_id, planes))

    for net_id, items in buckets.items():
        planes_np = np.stack([p for _, p in items])
        logits_np, values_np = infer(nets[net_id], planes_np)
        for idx, (worker_id, _) in enumerate(items):
            response_pipes[worker_id].send((logits_np[idx], float(values_np[idx, 0])))


def gpu_batcher_loop(nets, request_queue, response_pipes, stop_event,
                     batch_size=32, timeout=0.002):
    """Runs on the MAIN process/thread. Drains requests until all matches are done."""
    any_net = next(iter(nets.values()))
    device = next(any_net.parameters()).device
    use_amp = (device.type == "cuda")
    for n in nets.values():
        n.eval()

    def infer(net, planes_np):
        return _infer_torch(net, planes_np, device, use_amp)

    while not stop_event.is_set() or not request_queue.empty():
        requests = []
        start = time.time()
        while len(requests) < batch_size:
            try:
                rem = timeout - (time.time() - start)
                if rem <= 0 and requests:
                    break
                req = request_queue.get(timeout=max(0.0001, rem) if requests else 0.05)
                requests.append(req)
            except queue.Empty:
                if requests:
                    break
                if stop_event.is_set():
                    return
        if requests:
            _serve_requests(requests, nets, response_pipes, infer)


# --------------------------------------------------------------------------- #
# worker process
# --------------------------------------------------------------------------- #
def _worker_loop(worker_id, task_queue, result_queue, request_queue, response_pipe,
                 max_plies, iterations, c, opening_plies):
    import traceback
    torch.set_num_threads(1)            # one core per worker; MCTS is CPU-bound here

    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            break
        except Exception:
            break

        pidx, a_name, b_name, a_spec, b_spec, a_white, seed = task
        try:
            rng = random.Random(seed)
            A = make_remote_agent(a_spec, a_spec, worker_id, request_queue,
                                  response_pipe, rng, iterations, c, opening_plies)
            B = make_remote_agent(b_spec, b_spec, worker_id, request_queue,
                                  response_pipe, rng, iterations, c, opening_plies)
            # ONE game per task (not a whole match). Tasks are emitted grouped by
            # pairing, so at any instant the workers share only ~2 nets and the
            # batcher builds big per-net batches instead of many batch-1 forwards.
            if a_white:
                s_a = play_game(A, B, max_plies)
            else:
                s_a = 1.0 - play_game(B, A, max_plies)
            result_queue.put((pidx, a_name, b_name, s_a, True))
        except Exception:
            # A crashed game must still report, or the collector blocks forever,
            # stop_event never fires, and the batcher spins. Flag ok=False.
            traceback.print_exc()
            result_queue.put((pidx, a_name, b_name, None, False))


# --------------------------------------------------------------------------- #
# net loading (main process, on the GPU, once each)
# --------------------------------------------------------------------------- #
def _build_nets(tasks, device):
    specs = set()
    for (_pidx, _a, _b, a_spec, b_spec, _white, _seed) in tasks:
        for spec in (a_spec, b_spec):
            if spec not in ("random", "material"):
                specs.add(spec)
    nets = {}
    for spec in specs:
        if spec == "untrained":
            nets[spec] = ChessNet().to(device).eval()
        else:
            nets[spec] = load_net(spec, device)
    return nets


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Parallel GPU-batched checkpoint Elo scoring")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--anchors", default="random,material")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=400,
                    help="PUCT sims/move (match training, default 400)")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1,
                    help="test every Nth checkpoint (1 = all); final always kept")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="processes (None = all cores)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="max requests per GPU batch (default = workers)")
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpts = discover_checkpoints(args.ckpt_dir)
    if not ckpts:
        print(f"no net_iter*.pt found in {args.ckpt_dir}")
        return
    if args.stride > 1:
        kept = ckpts[::args.stride]
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept

    anchors = [a for a in args.anchors.split(",") if a]
    ckpt_names = [f"iter{it}" for it, _ in ckpts]
    spec = {f"iter{it}": os.path.join(args.ckpt_dir, f"net_iter{it}.pt") for it, _ in ckpts}
    for a in anchors:
        spec[a] = a
    players = anchors + ckpt_names
    name_idx = {n: i for i, n in enumerate(players)}
    workers = args.workers or os.cpu_count() or 1

    # schedule: each checkpoint vs each anchor; chain (or round-robin)
    schedule = []
    for cn in ckpt_names:
        for a in anchors:
            schedule.append((cn, a))
    if args.round_robin:
        for x in range(len(ckpt_names)):
            for y in range(x + 1, len(ckpt_names)):
                schedule.append((ckpt_names[x], ckpt_names[y]))
    else:
        for x in range(len(ckpt_names) - 1):
            schedule.append((ckpt_names[x], ckpt_names[x + 1]))

    cache_path = os.path.join(args.ckpt_dir, "elo_matches.csv")
    cache = load_cache(cache_path)

    # split cached vs pending
    cached, pending = [], []
    for (a, b) in schedule:
        if (a, b) in cache:
            aw, dr, bw, g = cache[(a, b)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        elif (b, a) in cache:
            bw, dr, aw, g = cache[(b, a)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        else:
            pending.append((a, b))

    # ONE task per GAME, emitted grouped by pairing. Per-game granularity keeps
    # net diversity low at any instant -- workers cluster on the current pairing,
    # so ~2 nets are in flight and the batcher builds large per-net batches
    # (~workers/2) instead of many batch-1 forwards. It also makes the progress
    # bar advance per game rather than only per completed match.
    tasks = []
    for pidx, (a, b) in enumerate(pending):
        for g in range(args.games):
            a_white = (g % 2 == 0)
            seed = args.seed + pidx * args.games + g
            tasks.append((pidx, a, b, spec[a], spec[b], a_white, seed))

    print(f"{len(ckpts)} checkpoints, anchors={anchors}, {args.games} games/match, "
          f"{args.iterations} sims/move, {workers} workers, device={device}")
    print(f"{len(cached)} pairings cached, {len(pending)} to play "
          f"({len(pending) * args.games} games)")

    results = list(cached)

    if tasks:
        nets = _build_nets(tasks, device)
        print(f"loaded {len(nets)} distinct nets onto {device}\n")

        ctx = mp.get_context("spawn")
        task_queue = ctx.Queue()
        result_queue = ctx.Queue()
        request_queue = ctx.Queue()
        for t in tasks:
            task_queue.put(t)

        parent_pipes, child_pipes = [], []
        for _ in range(workers):
            p, c = ctx.Pipe(duplex=True)
            parent_pipes.append(p)
            child_pipes.append(c)

        procs = []
        for wid in range(workers):
            pr = ctx.Process(
                target=_worker_loop,
                args=(wid, task_queue, result_queue, request_queue, child_pipes[wid],
                      args.max_plies, args.iterations, args.c, args.opening_plies),
            )
            pr.start()
            procs.append(pr)

        stop_event = threading.Event()
        t0 = time.time()

        # aggregate per-game results back into per-pairing W/D/L, then cache +
        # add to the Elo fit once a pairing's games are all accounted for.
        agg = {pidx: {"a": a, "b": b, "w": 0, "d": 0, "l": 0, "n": 0}
               for pidx, (a, b) in enumerate(pending)}
        received = {pidx: 0 for pidx in range(len(pending))}

        def collector():
            done = 0
            total = len(tasks)
            _progress(0, total, t0)
            while done < total:
                pidx, a_name, b_name, s_a, ok = result_queue.get()
                done += 1
                received[pidx] += 1
                rec = agg[pidx]
                if ok:
                    if s_a == 1.0:
                        rec["w"] += 1
                    elif s_a == 0.0:
                        rec["l"] += 1
                    else:
                        rec["d"] += 1
                    rec["n"] += 1
                if received[pidx] == args.games:           # pairing fully accounted for
                    if rec["n"] > 0:
                        append_cache(cache_path, rec["a"], rec["b"],
                                     rec["w"], rec["d"], rec["l"], rec["n"])
                        results.append((name_idx[rec["a"]], name_idx[rec["b"]],
                                        rec["w"] + 0.5 * rec["d"], rec["n"]))
                    else:
                        sys.stdout.write(f"\n[warn] pairing {rec['a']} vs {rec['b']} "
                                         f"all games failed; skipped\n")
                _progress(done, total, t0)
            stop_event.set()

        ct = threading.Thread(target=collector)
        ct.start()

        gpu_batcher_loop(nets, request_queue, parent_pipes, stop_event,
                         batch_size=args.batch_size or max(1, workers), timeout=0.002)

        for pr in procs:
            pr.join()
        ct.join()
        print()

    # fit + output
    pin = "random" if "random" in players else players[0]
    elo = fit_elo(players, results, pin=pin)

    print(f"\n{'='*46}\n  Elo ratings ({pin} = 0)\n{'='*46}")
    for a in anchors:
        print(f"  {a:14} {elo[a]:+7.0f}")
    print("  " + "-" * 30)
    rows = []
    for it, _ in ckpts:
        name = f"iter{it}"
        print(f"  {name:14} {elo[name]:+7.0f}")
        rows.append((it, elo[name]))

    out_csv = os.path.join(args.ckpt_dir, "elo_ratings.csv")
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "elo"])
        for it, e in rows:
            w.writerow([it, f"{e:.1f}"])
    print(f"\n  wrote {out_csv}")

    if rows:
        es = [e for _, e in rows]
        lo, hi = min(es), max(es)
        span = (hi - lo) or 1.0
        print(f"\n  Elo vs iteration  ({lo:+.0f} .. {hi:+.0f})")
        for it, e in rows:
            bar = "#" * int(1 + 40 * (e - lo) / span)
            print(f"  {it:>4} {bar}")


if __name__ == "__main__":
    main()
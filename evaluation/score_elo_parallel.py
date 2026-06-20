"""
Parallel version of elo.py.

Matches are independent, so they run across CPU worker processes (one torch
thread each, 'spawn' start method) instead of one at a time. Same schedule,
same Elo fit, same resumable cache as elo.py -- just N x faster.

Usage:
    python elo_parallel.py --ckpt-dir checkpoints --workers 24 --games 16 --iterations 40
    python elo_parallel.py --ckpt-dir checkpoints --stride 5 --workers 24
"""

import argparse
import csv
import math
import os
import random
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from evaluation.arena import make_agent, match
from evaluation.score_elo import discover_checkpoints, fit_elo, load_cache, append_cache


def _fmt(secs):
    secs = int(secs)
    h, secs = divmod(secs, 3600)
    m, s = divmod(secs, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _progress(done, total, t0, width=28):
    """In-place progress bar with elapsed / ETA / rate."""
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


def _init_worker():
    torch.set_num_threads(1)          # one core per worker -- no oversubscription


def _run_match(task):
    """Play one match on CPU. Top-level + picklable args for 'spawn'."""
    (a_name, b_name, a_spec, b_spec,
     games, max_plies, iters, c, opening_plies, seed) = task
    dev = torch.device("cpu")
    rng = random.Random(seed)
    A = make_agent(a_spec, dev, rng, iters, c, opening_plies)
    B = make_agent(b_spec, dev, rng, iters, c, opening_plies)
    st = match(A, B, games=games, max_plies=max_plies, alternate=True, verbose=False)
    return (a_name, b_name, st["wins"], st["draws"], st["losses"], st["games"])


def main():
    ap = argparse.ArgumentParser(description="Parallel checkpoint Elo scoring")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--anchors", default="random,material")
    ap.add_argument("--games", type=int, default=16)
    ap.add_argument("--iterations", type=int, default=40)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--stride", type=int, default=1,
                    help="test every Nth checkpoint (1 = all); final always kept")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--workers", type=int, default=None, help="processes (None = all cores)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

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
    spec = {f"iter{it}": path for it, path in ckpts}
    for a in anchors:
        spec[a] = a
    players = anchors + ckpt_names
    name_idx = {n: i for i, n in enumerate(players)}
    workers = args.workers or os.cpu_count() or 1
    print(f"{len(ckpts)} checkpoints, anchors={anchors}, {args.games} games/match, "
          f"{args.iterations} sims/move, {workers} workers\n")

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
    results, pending = [], []
    for (a, b) in schedule:
        if (a, b) in cache:
            aw, dr, bw, g = cache[(a, b)]
            results.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        elif (b, a) in cache:
            bw, dr, aw, g = cache[(b, a)]
            results.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        else:
            pending.append((a, b))

    print(f"{len(results)} cached, {len(pending)} to play")

    tasks = [
        (a, b, spec[a], spec[b], args.games, args.max_plies,
         args.iterations, args.c, args.opening_plies, args.seed + i)
        for i, (a, b) in enumerate(pending)
    ]

    ctx = mp.get_context("spawn")
    if tasks:
        t0 = time.time()
        done = 0
        _progress(0, len(tasks), t0)
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_init_worker) as ex:
            futures = [ex.submit(_run_match, t) for t in tasks]
            for fut in as_completed(futures):
                a, b, aw, dr, bw, g = fut.result()
                append_cache(cache_path, a, b, aw, dr, bw, g)
                results.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
                done += 1
                _progress(done, len(tasks), t0)
        print()   # newline after the bar finishes

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
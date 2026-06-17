"""
Score checkpoints on a single Elo scale.

Discovers net_iter*.pt in a directory, plays each against fixed anchors
(random / material) and against its neighbours, then fits one Elo rating per
checkpoint by maximum likelihood (Bradley-Terry / Elo) with `random` pinned at
0. Prints an Elo-vs-iteration table and writes elo_ratings.csv.

Why anchors AND neighbours: anchors give an absolute floor, but strong
checkpoints sweep them and the number saturates. Neighbour games (ckpt i vs
i+1) compare strong-vs-strong, so the scale keeps rising past the anchors.

Match results are cached to elo_matches.csv as they complete, so a long
evaluation can be interrupted and resumed without replaying games.

Usage:
    python elo.py --ckpt-dir checkpoints --games 20 --iterations 40
    python elo.py --ckpt-dir checkpoints --round-robin          # fuller, O(n^2)
"""

import argparse
import csv
import glob
import math
import os
import re
import random

import torch

from arena import make_agent, match


# --------------------------------------------------------------------------- #
# checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_checkpoints(ckpt_dir):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "net_iter*.pt")):
        m = re.search(r"net_iter(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)            # [(iteration, path), ...] ascending


# --------------------------------------------------------------------------- #
# Elo fit  (Bradley-Terry MM, ties as half-wins, light prior, random pinned 0)
# --------------------------------------------------------------------------- #
def fit_elo(names, results, pin="random", prior_games=2.0, steps=400):
    """
    names    : list of player names (index = player id)
    results  : list of (i, j, score_i, n)  -- score_i = i's points over n games
    Returns  : dict name -> Elo, with `pin` at 0.
    """
    P = len(names)
    gamma = [1.0] * P                      # BT strengths; Elo = 400*log10(gamma)
    wins = [0.0] * P                       # total points (wins + 0.5*draws)
    pairs = {}                             # i -> list of (j, n)
    for i in range(P):
        pairs[i] = []
    for (i, j, s_i, n) in results:
        wins[i] += s_i
        wins[j] += (n - s_i)
        pairs[i].append((j, n))
        pairs[j].append((i, n))

    # MM iterations: gamma_i = (W_i + prior/2) / ( sum_j n_ij/(gamma_i+gamma_j) + prior/(gamma_i+1) )
    for _ in range(steps):
        new = list(gamma)
        for i in range(P):
            denom = prior_games / (gamma[i] + 1.0)        # virtual draws vs rating 0
            for (j, n) in pairs[i]:
                denom += n / (gamma[i] + gamma[j])
            if denom > 0:
                new[i] = (wins[i] + 0.5 * prior_games) / denom
        gamma = new

    pin_idx = names.index(pin) if pin in names else 0
    ref = gamma[pin_idx]
    elo = {}
    for i, name in enumerate(names):
        elo[name] = 400.0 * math.log10(gamma[i] / ref) if gamma[i] > 0 else float("-inf")
    return elo


# --------------------------------------------------------------------------- #
# match cache (resumable)
# --------------------------------------------------------------------------- #
def load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["a"], row["b"])
            cache[key] = (int(row["a_wins"]), int(row["draws"]),
                          int(row["b_wins"]), int(row["games"]))
    return cache


def append_cache(path, a, b, wins, draws, losses, games):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["a", "b", "a_wins", "draws", "b_wins", "games"])
        w.writerow([a, b, wins, draws, losses, games])


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Score checkpoints on one Elo scale")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--anchors", default="random,material",
                    help="fixed reference players (comma-separated)")
    ap.add_argument("--games", type=int, default=20, help="games per match")
    ap.add_argument("--iterations", type=int, default=40, help="PUCT sims/move for eval")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--round-robin", action="store_true",
                    help="play all checkpoint pairs (O(n^2)), not just neighbours")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    ckpts = discover_checkpoints(args.ckpt_dir)
    if not ckpts:
        print(f"no net_iter*.pt found in {args.ckpt_dir}")
        return
    anchors = [a for a in args.anchors.split(",") if a]
    ckpt_names = [f"iter{it}" for it, _ in ckpts]
    spec = {f"iter{it}": path for it, path in ckpts}
    for a in anchors:
        spec[a] = a
    players = anchors + ckpt_names
    print(f"{len(ckpts)} checkpoints, anchors={anchors}, "
          f"{args.games} games/match, {args.iterations} sims/move\n")

    # build each agent once and reuse across matches
    agents = {name: make_agent(spec[name], device, rng,
                               args.iterations, args.c, args.opening_plies)
              for name in players}

    # schedule: every checkpoint vs every anchor; checkpoint chain (or round-robin)
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

    name_idx = {n: i for i, n in enumerate(players)}
    results = []
    for k, (a, b) in enumerate(schedule, 1):
        if (a, b) in cache:
            aw, dr, bw, g = cache[(a, b)]
        elif (b, a) in cache:          # symmetric; flip
            bw, dr, aw, g = cache[(b, a)]
        else:
            print(f"[{k}/{len(schedule)}] {a} vs {b}")
            st = match(agents[a], agents[b], games=args.games,
                       max_plies=args.max_plies, alternate=True, verbose=False)
            aw, dr, bw, g = st["wins"], st["draws"], st["losses"], st["games"]
            append_cache(cache_path, a, b, aw, dr, bw, g)
            print(f"      {a}: +{aw} ={dr} -{bw}")
        s_a = aw + 0.5 * dr
        results.append((name_idx[a], name_idx[b], s_a, g))

    pin = "random" if "random" in players else players[0]
    elo = fit_elo(players, results, pin=pin)

    # output
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

    # tiny ascii curve
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
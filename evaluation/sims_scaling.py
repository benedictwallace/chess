"""
Search-scaling diagnostic: is the net SEARCH-limited or CAPACITY-limited?

THE QUESTION
------------
When Elo stops moving, there are two very different causes and they need
opposite remedies:

  SEARCH-LIMITED -- the net's priors and value head are good enough that the
      search keeps finding improvements, but you are only running 400 sims, so
      the policy targets are soft. Remedy: MORE SIMS per move (better targets),
      accepting less data.

  CAPACITY-LIMITED -- the net has saturated what the search can extract; extra
      simulations converge to the same move it already preferred. Remedy: a
      BIGGER TRUNK. More sims would cost data rate and buy nothing.

Guessing wrong is expensive: raising sims when capacity-limited halves your
data rate for no gain, and growing the trunk when search-limited makes a
data-starved run worse.

THE TEST
--------
Play the net against ITSELF at N and 2N simulations. The score difference is
the value of doubling search, measured directly. No external opponent, so no
anchor-calibration issues, no Stockfish, no assumption that the opponent's
strength is known.

Interpretation (the reference point is AlphaZero-style engines, where doubling
search is worth roughly 100 Elo in this strength range):

    2N scores >= ~0.64  (>100 Elo)  -> SEARCH-LIMITED.
        The search is still extracting real improvements the net does not
        already know. Raise search_iterations; your targets are the weak link.

    2N scores ~0.57-0.64 (50-100 Elo) -> mixed. Search still pays, but less
        than a healthy engine. Modest sims increase, or leave as is.

    2N scores <= ~0.57  (<50 Elo)   -> CAPACITY-LIMITED.
        Extra search converges to the move the net already preferred. This is
        the trigger for a bigger trunk (channels 128->160, blocks 8->10).

A SECOND READING
----------------
Run with --budgets "100,200,400,800,1600" to get the whole curve rather than
one ratio. A curve that is steep at the low end and flat at the high end tells
you where YOUR net's search saturates -- and that saturation point is a
sensible choice for search_iterations, since simulations past it are wasted.

WHY SELF-PLAY MEASUREMENT IS FAIR HERE
--------------------------------------
Both sides are the same network with the same weights, so the only difference
is the number of simulations. Opening diversity comes from opening_plies
sampling, applied symmetrically. There is no colour or book asymmetry to
correct for because colours are alternated ticket by ticket.

USAGE
    python -m evaluation.sims_scaling --ckpt checkpoints/net_iter20768.pt \\
        --budgets 400,800 --games 200
"""

import argparse
import itertools
import math
import os

import torch

from evaluation.arena import load_net
from evaluation.score_elo_batched import (
    run_elo_matches_batched, _make_eval_fn,
)


def wilson(s, n, z=1.96):
    """Wilson interval on a score in [0,1] -- correct near 0 and 1, unlike the
    normal approximation."""
    if n == 0:
        return (0.0, 1.0)
    d = 1 + z * z / n
    c = (s + z * z / (2 * n)) / d
    h = z * math.sqrt(max(s * (1 - s) / n + z * z / (4 * n * n), 0.0)) / d
    return (c - h, c + h)


def elo(s):
    if s <= 0.0:
        return float("-inf")
    if s >= 1.0:
        return float("inf")
    return -400.0 * math.log10(1.0 / s - 1.0)


def main():
    ap = argparse.ArgumentParser(
        description="Measure the Elo value of doubling search for one net.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--budgets", default="400,800",
                    help="comma-separated sim counts, ascending")
    ap.add_argument("--games", type=int, default=200,
                    help="games per budget PAIR (even; colours alternate)")
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--fpu-reduction", type=float, default=0.25)
    ap.add_argument("--opening-plies", type=int, default=8,
                    help="temperature-sampled opening plies. Applied to BOTH "
                         "sides, so it costs neither -- it is only here to "
                         "stop all games being identical.")
    ap.add_argument("--opening-temp", type=float, default=1.0)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--concurrency", type=int, default=128)
    ap.add_argument("--device", default="")
    ap.add_argument("--all-pairs", action="store_true",
                    help="play every pair of budgets, not just consecutive "
                         "ones (n^2 games; gives a redundant consistency check)")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    budgets.sort()
    if len(budgets) < 2:
        raise SystemExit("need at least two budgets, e.g. --budgets 400,800")
    if args.games % 2:
        print(f"[warn] --games {args.games} is odd; colours will not balance.")

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_net(args.ckpt, device)
    fn = _make_eval_fn(net)

    # ONE net, registered under a distinct id per budget. eval_fns all point at
    # the same weights -- the id exists only so the runner can look up how many
    # simulations that side is allowed.
    ids = [f"n{b}" for b in budgets]
    eval_fns = {i: fn for i in ids}
    iters = {i: b for i, b in zip(ids, budgets)}

    pairs = (list(itertools.combinations(range(len(budgets)), 2))
             if args.all_pairs
             else [(i, i + 1) for i in range(len(budgets) - 1)])

    print(f"net    : {os.path.basename(args.ckpt)}")
    print(f"device : {device}")
    print(f"budgets: {budgets}")
    print(f"{len(pairs)} pair(s) x {args.games} games "
          f"= {len(pairs) * args.games} games\n")

    results = []
    for lo_i, hi_i in pairs:
        lo, hi = budgets[lo_i], budgets[hi_i]
        a_id, b_id = ids[hi_i], ids[lo_i]        # 'a' is the HIGHER budget
        tickets = []
        for g in range(args.games):
            a_white = (g % 2 == 0)
            tickets.append((0, a_white,
                            a_id if a_white else b_id,
                            b_id if a_white else a_id))
        tot = {"s": 0.0, "n": 0}

        def on_done(pidx, a_score, _t=tot):
            _t["s"] += a_score
            _t["n"] += 1
            if _t["n"] % 40 == 0:
                print(f"    {_t['n']}/{len(tickets)} games", flush=True)

        print(f"  {hi} sims vs {lo} sims ...", flush=True)
        run_elo_matches_batched(
            tickets, eval_fns, iterations=iters, c=args.c,
            fpu_reduction=args.fpu_reduction,
            opening_plies=args.opening_plies, opening_temp=args.opening_temp,
            max_plies=args.max_plies, concurrency=args.concurrency,
            on_game_done=on_done)

        s = tot["s"] / max(tot["n"], 1)
        lo_ci, hi_ci = wilson(s, tot["n"])
        results.append((lo, hi, s, lo_ci, hi_ci, tot["n"]))
        print(f"    {hi} sims scored {s:.3f} "
              f"[{lo_ci:.3f}, {hi_ci:.3f}]  -> {elo(s):+.0f} Elo "
              f"[{elo(lo_ci):+.0f}, {elo(hi_ci):+.0f}]\n")

    print("=" * 68)
    print(f"{'comparison':<22}{'score':>8}{'Elo':>10}{'95% CI':>20}")
    for lo, hi, s, l, h, n in results:
        print(f"{f'{hi} vs {lo} sims':<22}{s:8.3f}{elo(s):+10.0f}"
              f"{f'[{elo(l):+.0f}, {elo(h):+.0f}]':>20}")

    # verdict on the widest doubling measured
    lo, hi, s, l, h, n = results[-1]
    print("\n" + "-" * 68)
    if hi >= 2 * lo:
        if l > 0.64:
            v = ("SEARCH-LIMITED. Doubling search is worth >100 Elo even at the "
                 "bottom of\nthe interval, so the search keeps finding moves the "
                 "net does not already\nprefer. Raise search_iterations -- your "
                 "policy TARGETS are the weak link,\nnot the network.")
        elif h < 0.57:
            v = ("CAPACITY-LIMITED. Doubling search is worth <50 Elo even at the "
                 "top of the\ninterval: extra simulations converge on the move "
                 "the net already preferred.\nMore sims would cost data rate for "
                 "nothing. This is the trigger to grow the\ntrunk (channels "
                 "128->160, blocks 8->10) -- which needs a fresh run.")
        else:
            v = ("MIXED / INCONCLUSIVE. The interval spans the 50-100 Elo band, "
                 "so this run\ncannot separate the two cases. Either raise "
                 "--games (the interval narrows as\n1/sqrt(n)) or widen the "
                 "budget ratio to 4x, where the effect is larger.")
    else:
        v = (f"{hi}/{lo} is not a doubling, so the 100-Elo reference point does "
             f"not apply.\nRe-run with a 2x ratio for a verdict.")
    print(v)
    print("-" * 68)


if __name__ == "__main__":
    main()

    
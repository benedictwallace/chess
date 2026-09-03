"""
Choose the delta for delta-constrained forgiving selection -- from the search's
own resolution rather than by eye.

THE QUESTION
------------
The forgiving selector plays the most forgiving member of

    S = { a : Q1 - Q(a) <= delta }

so delta is the whole experiment's knob: it is the value you agree to give up
per move in exchange for slack. Two failure modes bracket it, and they are not
symmetric.

  delta TOO SMALL -> |S| = 1 almost everywhere. The selector never fires, the
      treatment arm is the control arm, and the result is a null you cannot
      interpret: you have not learned that forgiveness does not help, only
      that you never tried it.

  delta TOO LARGE -> |S| is the whole candidate set and membership is decided
      by estimation noise rather than by the position. The agent then trades
      real value for a statistic computed on noise, and the arm loses strength
      for no reason a reviewer will accept.

The second bound is the one nobody checks. A root Q at 99 visits is not exact;
if delta sits below the error on Q1 - Q2, then S is a coin flip. This script
measures that error and reports the window between the two bounds.

HOW THE NOISE FLOOR IS MEASURED
-------------------------------
Not by reseeding. MCTS here is deterministic given the network -- no Dirichlet
at a probe root, and Gumbel off -- so re-running the same position produces
the same tree and an apparent standard error of exactly zero. That number
would be a lie.

Instead the same positions are searched at TWO budgets, and the disagreement
between them is the error scale:

    gap_lo = Q1 - Q2 at the deployment budget
    gap_hi = Q1 - Q2 at a large reference budget

    rms |gap_lo - gap_hi|      how far the deployment gap sits from a
                               better-resolved estimate of the same quantity
    P(argmax flips)            how often the deployment budget disagrees with
                               the reference about which move is even best

This is a lower bound on the true error (the reference budget is not the
truth, only a better estimate of it), which is the safe direction: a delta
above this floor is above the real one too.

SEARCH CONFIGURATION -- THE SAME ROOT THE ARENA USES
----------------------------------------------------
The searches here run Gumbel top-m + SEQUENTIAL HALVING and take the candidate
set from SHState.stat_children(), exactly as the arena does. That is not
pedantry about matching visit counts; the candidate SET itself differs, and it
differs in the direction that would bias delta.

A forced-visit root takes its top-m BY PRIOR. A halving root takes the top-m by
prior and then eliminates on VALUE across phases, so the four survivors are the
four the search judged best. Those four sit closer together in Q than four
picked by prior would -- selection on value compresses the gaps. Calibrating on
a prior-selected set therefore reports a WIDER gap distribution than the arena
will ever see, and a delta chosen from those percentiles is too large: it would
look reasonable here and saturate |S| in the arena.

The visit profile matches too (at 800 sims, m=8: two finalists at 234 and two
semi-finalists at 99), and Gumbel is off by default so the run is deterministic
and reproducible, again as in the arena.

    arena  --sims 800 --sh-m 8 --sh-stat-width 4
    match  --sims 800 --sh-m 8 --sh-stat-width 4

WHAT IT PRINTS
--------------
  * the gap distribution at the deployment budget (percentiles)
  * the noise floor (rms disagreement, argmax flip rate)
  * a delta sweep: for each candidate delta, the mean |S|, the share of
    positions where the selector can fire at all (|S| >= 2), the share where
    it is unconstrained (|S| = all candidates), and the worst-case value
    given up
  * a recommendation, with the reasoning shown so you can overrule it

Usage
    python calibrate_delta.py --checkpoint checkpoints/net_iter7500.pt \\
        --positions 400 --sims 800 --sims-hi 5000 \\
        --force-m 4 --force-n 99 --out delta_calibration.csv

    # skip the noise floor (fast, but then you only get the |S| half)
    python calibrate_delta.py --checkpoint ... --sims-hi 0
"""

# MUST be first: patches Board.legalMoves() with the Cython generator
# before anything imports engine.board. The search is move-generation
# bound, so this is worth roughly 3x end to end.
try:
    from evaluation.fast_movegen_boot import ensure_fast_movegen
except ImportError:
    from fast_movegen_boot import ensure_fast_movegen
ensure_fast_movegen()

import argparse
import csv
import json
import math
import os

import numpy as np

try:                                        # repo-layout tolerant
    from evaluation.probe_forgiveness import (_probe_eval_fn, _uci,
                                              harvest_positions)
except ImportError:
    from probe_forgiveness import _probe_eval_fn, _uci, harvest_positions
try:
    from evaluation.score_elo_batched import _ZeroGumbel
except ImportError:
    from score_elo_batched import _ZeroGumbel
try:
    from search.sequential_halving import SHState, plan_phases
except ImportError:
    from sequential_halving import SHState, plan_phases
try:
    from search.puct import node_fpu_q
except ImportError:
    from puct import node_fpu_q
try:
    from training.self_play_batched import Node, _expand, _backprop, _softmax
except ImportError:
    from self_play_batched import Node, _expand, _backprop, _softmax
try:
    from model.encoding import encode_env
    from model.move_encoding import encodeMovePOV
except ImportError:
    from encoding import encode_env
    from move_encoding import encodeMovePOV
try:
    from evaluation.arena import load_net
except ImportError:
    from arena import load_net
try:
    from engine.fen import env_from_fen, board_to_fen
except ImportError:
    from fen import env_from_fen, board_to_fen


# --------------------------------------------------------------------------- #
# batched sequential-halving search over a set of positions
# --------------------------------------------------------------------------- #
class _Item:
    """One position under search. Needs its own class rather than probe's
    _ProbeItem because that one is slotted without an `sh` field."""
    __slots__ = ("env", "ply", "fen", "root", "sims", "sh")

    def __init__(self, env, ply):
        self.env = env
        self.ply = ply
        self.fen = board_to_fen(env.board)
        self.root = None
        self.sims = 0
        self.sh = None


def search_positions_sh(items, eval_fn, *, sims=800, m=8, stat_width=4,
                        c=1.5, fpu_reduction=0.25, c_visit=50.0, c_scale=0.02,
                        gumbel=False, seed=0, tag=""):
    """One Gumbel + sequential-halving root search per item, batched across
    items: one simulation per item per round, a single network forward per
    round. Interior nodes stay on PUCT, as in training and in the arena.

    Mutates each item's .root and .sh in place; read the candidate set back
    with it.sh.stat_children()."""
    rng = np.random.default_rng(seed) if gumbel else _ZeroGumbel()
    for it in items:
        it.root = Node()
        it.root.moverSign = 0
        it.sims = 0
        it.sh = None

    live = list(items)
    while live:
        batch_planes, batch_meta = [], []
        for it in live:
            node = it.root
            env = it.env.clone()
            path = [node]
            at_root = True
            while node.expanded and not node.terminal and node.children:
                best = None
                if at_root:
                    if it.sh is None:
                        it.sh = SHState(node, budget=sims - node.visits, m=m,
                                        rng=rng, c_visit=c_visit,
                                        c_scale=c_scale, stat_width=stat_width)
                    best = it.sh.next_child()
                if best is None:
                    sqrt_pv = math.sqrt(node.visits)
                    fpu_q = node_fpu_q(node, fpu_reduction)
                    best_score = -1e30
                    for ch in node.children:
                        v = ch.visits
                        q = ch.value / v if v else fpu_q
                        s = q + c * ch.prior * sqrt_pv / (1 + v)
                        if s > best_score:
                            best_score = s
                            best = ch
                node = best
                env.step(node.move)
                path.append(node)
                at_root = False

            if node.terminal:
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                it.sims += 1
                continue
            legal = env.legalMoves()
            if not legal:
                node.terminal = True
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                it.sims += 1
                continue
            if env.isRepetition() or env.isFiftyMove():
                node.terminal = True
                _backprop(path, 0.0)
                it.sims += 1
                continue

            batch_planes.append(encode_env(env))
            batch_meta.append((it, node, env, legal, path,
                               env.board.sideToMove))

        if batch_planes:
            logits_b, values_b = eval_fn(batch_planes)
            for (it, node, env, legal, path, mover), logits, value in zip(
                    batch_meta, logits_b, values_b):
                idxs = [encodeMovePOV(m_, mover) for m_ in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {mv: float(p) for mv, p in zip(legal, probs)}
                # net_value MUST be passed: the Gumbel v_mix completion reads
                # it, and a root's own .value accumulator is pinned to 0.
                _expand(node, priors, mover, False, False, 0.0, 0.0,
                        net_value=float(value))
                _backprop(path,
                          float(value) if mover == "white" else -float(value))
                it.sims += 1

        live = [it for it in live if it.sims < sims]


def stat_set_row(it):
    """Chooser-POV Qs of the halving statistics set, descending, plus the
    visit floor backing them and the Q-argmax move (the point the delta
    window is measured from)."""
    if it.sh is None:
        return None
    stat = [ch for ch in it.sh.stat_children() if ch.visits > 0]
    if len(stat) < 2:
        return None
    pairs = sorted(((ch.value / ch.visits, ch) for ch in stat),
                   key=lambda t: -t[0])
    return {"qs": [q for q, _ in pairs],
            "moves": [_uci(ch.move) for _, ch in pairs],
            # Every visited child, not just the statistics set: pricing a flip
            # means looking up the OTHER budget's value for the move this
            # budget preferred, and that move need not have survived halving
            # at the other budget. Without this the flip cost is missing
            # precisely in the cases where the two budgets disagree most.
            "all_q": {_uci(ch.move): ch.value / ch.visits
                      for ch in it.root.children if ch.visits > 0},
            "n_qual": len(pairs),
            "stat_floor": min(ch.visits for ch in stat),
            "n_legal": len(it.root.children),
            "best_move": _uci(pairs[0][1].move)}


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def delta_set_size(qs, delta):
    """|S| = how many candidates sit within delta of the best."""
    q1 = qs[0]
    return sum(1 for q in qs if q1 - q <= delta)


def worst_sacrifice(qs, delta):
    """The most value the selector could give up at this delta: the distance
    from Q1 to the worst member of S. This is the WORST case, not the realised
    one -- the selector picks by forgiveness, not by minimum Q -- but it is
    the bound delta actually guarantees, and it is what delta means."""
    q1 = qs[0]
    inside = [q for q in qs if q1 - q <= delta]
    return q1 - min(inside)


def summarise_gaps(gaps):
    g = np.asarray(gaps, dtype=np.float64)
    ps = (5, 10, 25, 50, 75, 90, 95)
    return {p: float(np.percentile(g, p)) for p in ps}


def per_position_rows(items, lo_rows, hi_rows=None):
    """One row per probed position: the deployment-budget statistics, and the
    reference-budget ones where available.

    These are what make the noise floor re-conditionable after the fact. An
    aggregate rms cannot be sliced; a table can."""
    out = []
    for i, (it, lo) in enumerate(zip(items, lo_rows)):
        if lo is None or len(lo["qs"]) < 2:
            continue
        row = dict(idx=i, fen=it.fen, ply=it.ply,
                   n_legal=lo["n_legal"], n_qual=lo["n_qual"],
                   stat_floor=lo["stat_floor"],
                   best_move=lo["best_move"],
                   gap=round(lo["qs"][0] - lo["qs"][1], 6))
        for j, q in enumerate(lo["qs"][:4]):
            row[f"q{j + 1}"] = round(q, 6)
        if hi_rows is not None:
            hi = hi_rows[i]
            if hi is not None and len(hi["qs"]) >= 2:
                row["gap_hi"] = round(hi["qs"][0] - hi["qs"][1], 6)
                row["best_move_hi"] = hi["best_move"]
                row["gap_err"] = round(row["gap"] - row["gap_hi"], 6)
                row["flipped"] = int(lo["best_move"] != hi["best_move"])
                # How much the flip COST, by the reference budget's own
                # reckoning: the reference Q of the move the deployment
                # budget preferred, against the reference best. A flip
                # between two genuinely tied moves costs ~0 and is not an
                # error worth pricing into delta.
                hi_q = hi.get("all_q", {})
                if lo["best_move"] in hi_q:
                    row["flip_cost"] = round(hi["qs"][0]
                                             - hi_q[lo["best_move"]], 6)
        out.append(row)
    return out


def noise_floor(lo_rows, hi_rows, gap_max=None):
    """Disagreement between the two budgets on the SAME positions.

    gap_max: restrict to positions whose deployment-budget gap is at most
    this. THIS IS THE POINT. An unconditional rms over a skewed gap
    distribution is dominated by brittle positions where the budgets disagree
    by a lot in absolute terms -- and those positions can never have |S| >= 2,
    so their error is irrelevant to delta. The error that bounds delta is the
    error among positions that are actually near-tied.

    Returns a dict with the rms, the median absolute error (robust to the
    tail the rms chases), the flip rate, and the count."""
    diffs, flips, costs = [], 0, []
    for lo, hi in zip(lo_rows, hi_rows):
        if lo is None or hi is None:
            continue
        if len(lo["qs"]) < 2 or len(hi["qs"]) < 2:
            continue
        gap_lo = lo["qs"][0] - lo["qs"][1]
        if gap_max is not None and gap_lo > gap_max:
            continue
        diffs.append(gap_lo - (hi["qs"][0] - hi["qs"][1]))
        if lo["best_move"] != hi["best_move"]:
            flips += 1
            hi_q = hi.get("all_q", {})
            if lo["best_move"] in hi_q:
                costs.append(hi["qs"][0] - hi_q[lo["best_move"]])
    if not diffs:
        return None
    d = np.asarray(diffs, dtype=np.float64)
    return dict(rms=float(np.sqrt((d ** 2).mean())),
                mad=float(np.median(np.abs(d))),
                p90=float(np.percentile(np.abs(d), 90)),
                flip_rate=flips / len(diffs),
                mean_flip_cost=(float(np.mean(costs)) if costs else None),
                n=len(diffs))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Calibrate the delta window for forgiving selection.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--positions", type=int, default=400,
                   help="positions to harvest (ignored with --fens)")
    p.add_argument("--fens", default=None,
                   help="probe your own positions instead, one FEN per line")
    p.add_argument("--sims", type=int, default=800,
                   help="DEPLOYMENT budget -- match the arena's --sims")
    p.add_argument("--sims-hi", type=int, default=5000,
                   help="reference budget for the noise floor (0 = skip)")
    p.add_argument("--sh-m", type=int, default=8,
                   help="root actions considered, top-m -- match the arena")
    p.add_argument("--sh-stat-width", type=int, default=4,
                   help="candidates in the statistics set -- match the arena")
    p.add_argument("--sh-c-visit", type=float, default=50.0)
    p.add_argument("--sh-c-scale", type=float, default=0.02)
    p.add_argument("--sh-gumbel", action="store_true",
                   help="keep the Gumbel draws (default off, as in the arena)")
    p.add_argument("--deltas", default="0.01,0.02,0.03,0.05,0.075,0.1,0.15",
                   help="comma list of deltas to evaluate")
    p.add_argument("--fire-target", type=float, default=0.25,
                   help="minimum share of positions where the selector should "
                        "be able to fire (|S| >= 2) for the experiment to "
                        "have any power")
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--min-ply", type=int, default=6)
    p.add_argument("--max-ply", type=int, default=140)
    p.add_argument("--harvest-temp", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--cond-gaps", default="0.02,0.05,0.1,0.2,1.0",
                   help="gap thresholds at which to report the conditional "
                        "noise floor")
    p.add_argument("--floor-gap", type=float, default=0.1,
                   help="which conditional slice supplies the floor used for "
                        "the NOISE/ok verdicts and the recommendation. The "
                        "default 0.1 covers every position a sensible delta "
                        "could act in, without being so tight that the slice "
                        "is only the exact ties.")
    p.add_argument("--positions-out", default=None,
                   help="one row per probed position (FEN, stat-set Qs, gap "
                        "at both budgets, flip and its cost). Defaults to "
                        "<out>_positions.csv -- keep it, it is what lets the "
                        "floor be re-conditioned without re-searching.")
    p.add_argument("--out", default="delta_calibration.csv")
    p.add_argument("--fens-out", default=None,
                   help="write the probed FENs here (feed to --fens to "
                        "re-probe the SAME positions later)")
    return p.parse_args()


def main():
    import torch
    args = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_net(args.checkpoint, device)
    eval_fn = _probe_eval_fn(net, amp=args.amp)
    print(f"device: {device}   checkpoint: {args.checkpoint}")

    # ---- positions ----
    if args.fens:
        with open(args.fens) as f:
            fens = [ln.strip() for ln in f if ln.strip()]
        items = [_Item(env_from_fen(fen), -1) for fen in fens]
        print(f"loaded {len(items)} positions from {args.fens}")
    else:
        print(f"harvesting {args.positions} positions "
              f"(policy playouts, plies {args.min_ply}-{args.max_ply})")
        snaps = harvest_positions(
            eval_fn, args.positions, min_ply=args.min_ply,
            max_ply=args.max_ply, concurrency=args.concurrency,
            temp=args.harvest_temp, seed=args.seed)
        items = [_Item(env, ply) for env, ply in snaps]

    if args.fens_out:
        with open(args.fens_out, "w") as f:
            for it in items:
                f.write(it.fen + "\n")
        print(f"positions -> {args.fens_out}")

    def run_all(budget, tag):
        out = []
        for i in range(0, len(items), args.concurrency):
            chunk = items[i:i + args.concurrency]
            search_positions_sh(
                chunk, eval_fn, sims=budget, m=args.sh_m,
                stat_width=args.sh_stat_width, c_visit=args.sh_c_visit,
                c_scale=args.sh_c_scale, gumbel=args.sh_gumbel,
                seed=args.seed, tag=tag)
            out.extend(stat_set_row(it) for it in chunk)
            print(f"  {tag}{min(i + args.concurrency, len(items))}"
                  f"/{len(items)} at {budget} sims", flush=True)
        return out

    plan = plan_phases(args.sims, args.sh_m)
    cum, tot = [], 0
    for n_c, per in plan:
        tot += per
        cum.append((n_c, tot))
    print(f"\nsearching at the deployment budget ({args.sims} sims, "
          f"halving m={args.sh_m}, gumbel="
          f"{'on' if args.sh_gumbel else 'off'})")
    print(f"  schedule {cum}")
    lo_rows = run_all(args.sims, "")

    usable = [r for r in lo_rows if r is not None and len(r["qs"]) >= 2]
    if not usable:
        raise SystemExit("no position produced a usable statistics set -- "
                         "raise --sims or lower --sh-m")
    floors = [r["stat_floor"] for r in usable]
    print(f"statistics set: {np.mean([r['n_qual'] for r in usable]):.1f} "
          f"actions, backed by {int(np.median(floors))} visits (median)")
    gaps = [r["qs"][0] - r["qs"][1] for r in usable]
    pct = summarise_gaps(gaps)
    print(f"\naction gap at {args.sims} sims over {len(usable)} positions")
    for p in sorted(pct):
        print(f"  p{p:<3d} {pct[p]:.4f}")

    # ---- noise floor ----
    rms = flip = None
    nf_all = nf_cond = None
    hi_rows = None
    if args.sims_hi > args.sims:
        print(f"\nre-searching the same positions at {args.sims_hi} sims "
              f"for the noise floor")
        hi_rows = run_all(args.sims_hi, "hi-")
        nf_all = noise_floor(lo_rows, hi_rows)
        if nf_all:
            print(f"\nnoise floor, ALL {nf_all['n']} positions")
            print(f"  rms |gap({args.sims}) - gap({args.sims_hi})| "
                  f"= {nf_all['rms']:.4f}")
            print(f"  median |error| = {nf_all['mad']:.4f}   "
                  f"p90 = {nf_all['p90']:.4f}")
            print(f"  argmax disagrees: {nf_all['flip_rate']:.1%}"
                  + (f" (mean cost {nf_all['mean_flip_cost']:.4f})"
                     if nf_all["mean_flip_cost"] is not None else ""))
            print("  NOTE: this is unconditional. The gap distribution is "
                  "skewed, so the rms is\n        dominated by brittle "
                  "positions -- which can never have |S| >= 2 and so\n"
                  "        cannot constrain delta. The conditional figure "
                  "below is the one to use.")

        # ---- the delta-relevant floor: error among NEAR-TIED positions ---- #
        print(f"\nnoise floor CONDITIONED on the deployment gap "
              f"(the positions delta can actually act in)")
        print(f"  {'gap <=':>8} {'n':>5} {'rms':>8} {'median':>8} {'p90':>8} "
              f"{'flips':>7} {'flip cost':>10}")
        for gm in [float(x) for x in args.cond_gaps.split(",")]:
            nf = noise_floor(lo_rows, hi_rows, gap_max=gm)
            if not nf:
                continue
            fc = ("" if nf["mean_flip_cost"] is None
                  else f"{nf['mean_flip_cost']:.4f}")
            print(f"  {gm:>8.3f} {nf['n']:>5} {nf['rms']:>8.4f} "
                  f"{nf['mad']:>8.4f} {nf['p90']:>8.4f} "
                  f"{nf['flip_rate']:>6.0%} {fc:>10}")
        # The floor used downstream is computed directly, so it does not
        # depend on --floor-gap appearing in --cond-gaps.
        nf_cond = noise_floor(lo_rows, hi_rows, gap_max=args.floor_gap)
        if nf_cond:
            rms = nf_cond["rms"]
            flip = nf_cond["flip_rate"]
            print(f"\n  -> floor taken at gap <= {args.floor_gap}: "
                  f"rms {rms:.4f} over {nf_cond['n']} positions")
            print(f"     a delta below ~{rms:.3f} selects mostly on "
                  f"estimation error")
    else:
        print("\nnoise floor SKIPPED (--sims-hi 0): the lower bound on delta "
              "is unmeasured, so the sweep below only tells you about power, "
              "not about validity.")

    # ---- delta sweep ----
    deltas = [float(x) for x in args.deltas.split(",")]
    print(f"\ndelta sweep ({len(usable)} positions, candidate set = "
          f"{np.mean([r['n_qual'] for r in usable]):.1f} actions on average)")
    print(f"  {'delta':>7} {'mean|S|':>8} {'fires':>7} {'saturated':>10} "
          f"{'worst sac':>10} {'vs floor':>9}")
    rows = []
    for d in deltas:
        sizes = [delta_set_size(r["qs"], d) for r in usable]
        sacs = [worst_sacrifice(r["qs"], d) for r in usable]
        fires = float(np.mean([s >= 2 for s in sizes]))
        sat = float(np.mean([s == r["n_qual"]
                             for s, r in zip(sizes, usable)]))
        verdict = "-" if rms is None else (
            "NOISE" if d < rms else ("ok" if d < 4 * rms else "wide"))
        print(f"  {d:>7.3f} {np.mean(sizes):>8.2f} {fires:>6.0%} "
              f"{sat:>9.0%} {np.mean(sacs):>10.4f} {verdict:>9}")
        rows.append(dict(delta=d, mean_delta_set=round(float(np.mean(sizes)), 3),
                         fire_rate=round(fires, 4),
                         saturated_rate=round(sat, 4),
                         mean_worst_sacrifice=round(float(np.mean(sacs)), 5),
                         noise_floor=("" if rms is None else round(rms, 5)),
                         verdict=verdict))

    # ---- recommendation ----
    print("\nrecommendation")
    ok = [r for r in rows if r["fire_rate"] >= args.fire_target
          and r["saturated_rate"] <= 0.5
          and (rms is None or r["delta"] >= rms)]
    if not ok:
        print(f"  NONE of the swept deltas both fires in >= "
              f"{args.fire_target:.0%} of positions and sits above the noise "
              f"floor.")
        if rms is not None:
            need = [r for r in rows if r["delta"] >= rms]
            if need:
                print(f"  The smallest valid delta is {need[0]['delta']:.3f}, "
                      f"firing in {need[0]['fire_rate']:.0%} of positions.")
            print("  If that fire rate is too low to give the experiment "
                  "power, the honest conclusion is that this search budget "
                  "cannot resolve near-ties at this scale -- raise --sims "
                  "rather than lowering delta below the floor.")
    else:
        best = min(ok, key=lambda r: r["delta"])
        print(f"  delta = {best['delta']:.3f}")
        print(f"    fires in {best['fire_rate']:.0%} of positions "
              f"(|S| >= 2), mean |S| = {best['mean_delta_set']:.2f}")
        print(f"    guarantees giving up at most {best['delta']:.3f} per move; "
              f"worst case realised {best['mean_worst_sacrifice']:.4f} on "
              f"average")
        if rms is not None:
            print(f"    sits {best['delta'] / rms:.1f}x above the "
                  f"{rms:.4f} noise floor")
        print("  Smallest delta meeting all three conditions -- prefer the "
              "smallest, since delta is value surrendered and the point is to "
              "buy slack cheaply. Run the sweep at a second delta either side "
              "and report the sensitivity.")

    meta = dict(checkpoint=os.path.abspath(args.checkpoint),
                sims=args.sims, sims_hi=args.sims_hi,
                sh_m=args.sh_m, sh_stat_width=args.sh_stat_width,
                sh_gumbel=args.sh_gumbel,
                median_stat_floor=int(np.median(floors)),
                positions=len(usable),
                gap_percentiles=pct,
                floor_gap=args.floor_gap,
                noise_all=nf_all, noise_conditional=nf_cond,
                noise_rms=rms, argmax_flip_rate=flip)

    pos_path = args.positions_out or (os.path.splitext(args.out)[0]
                                      + "_positions.csv")
    prows = per_position_rows(items, lo_rows, hi_rows)
    if prows:
        keys = []
        for r in prows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(pos_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, restval="")
            w.writeheader()
            w.writerows(prows)
        print(f"per-position rows -> {pos_path} ({len(prows)} rows)")
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.splitext(args.out)[0] + "_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {args.out} and "
          f"{os.path.splitext(args.out)[0]}_meta.json")


if __name__ == "__main__":
    main()

    
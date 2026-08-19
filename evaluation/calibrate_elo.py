"""
Place a checkpoint on an EXTERNAL, citable rating scale -- with honest error bars.

WHY THIS EXISTS
---------------
score_elo_external.py answers "how strong is this net relative to
sf:skill=0+nodes=1?", and pins that anchor at 0. That number is reproducible
and internally consistent, but it is not a chess rating: nobody has published
what `Stockfish 18 at Skill Level 0 with a 1-node cap` is worth in Elo, and the
published Skill-Level tables do not transfer because they assume a normal time
control with no node cap.

This script fixes the scale to a reference player whose rating on a named human
scale IS documented, then propagates that reference's own uncertainty into the
final answer. The output is deliberately of the form

    net_iter9375 = 1043 Elo (Lichess blitz)   [95% CI 961 - 1120]
      of which +-62 is match sampling noise and +-55 is the reference's own
      rating uncertainty

rather than a bare number, because a bare number here would be false precision.

THE THREE THINGS THAT MAKE A PLACEMENT VALID
--------------------------------------------
1. BRACKETING. The net must both beat some anchors and lose to others. If it
   loses to everything, Bradley-Terry extrapolates off the end of the ladder
   and the rating is determined by the prior, not the data. The script checks
   this and refuses to report an absolute number if the net is not bracketed.

2. CONNECTIVITY. Every player must be joined to the reference by a path of
   played matches. Two disconnected clusters have no defined rating difference.
   The script checks this with a union-find over the match graph.

3. LOCAL RESOLUTION. Pairs with a 100-0 sweep carry almost no information about
   the size of the gap. The script reports each pair's score with a Wilson
   interval and flags saturated pairs.

REFERENCE PLAYERS
-----------------
The default reference is Maia, a Leela-derived network trained to imitate human
play at a specific Lichess rating band. It is the closest thing the engine
community has to a calibrated weak opponent, and unlike Skill Level it errs the
way humans err rather than by injecting random blunders. Requires the `lc0`
binary and the maia weights:

    # weights: https://github.com/CSSLab/maia-chess  (maia-1100.pb.gz etc.)
    python -m evaluation.calibrate_elo --lc0 /path/to/lc0 \\
        --maia-weights-dir /path/to/maia_weights ...

Without lc0 the script falls back to Stockfish's own UCI_Elo anchors, which are
a WEAKER reference: Stockfish converts UCI_Elo internally to a Skill Level, the
mapping has been repeatedly reported as poorly calibrated, and the floor
(UCI_Elo 1320 in recent builds) may sit above a developing net entirely. The
script says so in its output rather than hiding it.

CAVEAT YOU MUST CARRY INTO THE WRITE-UP
---------------------------------------
Lichess blitz ratings are NOT FIDE ratings; at the low end Lichess runs several
hundred points higher, and the offset is not a constant. Report the scale name
alongside the number, always. This script never converts between scales.
"""

import argparse
import glob
import math
import os
import random
import sys

from evaluation.openings import BOOK
from evaluation.score_elo_external import (
    UCIAgent, match, load_matches, append_match, canon,
    make_ckpt_agent, _RandomAgent, wilson_elo_ci,
)


# --------------------------------------------------------------------------- #
# Reference players with a documented rating on a named external scale.
#
# `sigma` is the uncertainty in the REFERENCE'S OWN rating -- not measurement
# noise from our matches. It is propagated into the final interval, because a
# placement can never be tighter than the peg it hangs from.
# --------------------------------------------------------------------------- #
REFERENCES = {
    "maia:1100": dict(
        rating=1100, sigma=75, scale="Lichess blitz",
        note="Maia-1100: trained on Lichess games in the 1100-1199 band. "
             "Its own playing strength is close to, but not identical to, the "
             "band it was trained on -- hence the sigma."),
    "maia:1500": dict(
        rating=1500, sigma=75, scale="Lichess blitz",
        note="Maia-1500: trained on the 1500-1599 band."),
    "maia:1900": dict(
        rating=1900, sigma=75, scale="Lichess blitz",
        note="Maia-1900: trained on the 1900-1999 band."),
    "sf:elo=1320": dict(
        rating=1320, sigma=200, scale="Stockfish UCI_Elo (self-reported)",
        note="Stockfish's own UCI_Elo floor. LARGE sigma on purpose: this is a "
             "self-reported figure, implemented by converting to a Skill Level, "
             "and multiple reports find it does not match the scale it claims. "
             "Use only if lc0/Maia is unavailable, and say so in the write-up."),
}


# A ladder of node-limited Stockfish rungs. Node caps ONLY -- no Skill Level.
# Skill Level weakens the engine mainly by randomly blundering, which produces
# an opponent that is weak in a way no human is; node limits weaken it by
# making it shallow, which is closer to how weaker humans actually fail. These
# rungs have no known absolute rating -- they exist purely to CHAIN the net to
# the reference, and the fit gives them coordinates on the reference's scale.
DEFAULT_LADDER = [
    "sf:nodes=1",
    "sf:nodes=8",
    "sf:nodes=32",
    "sf:nodes=128",
    "sf:nodes=512",
    "sf:nodes=2048",
]


# --------------------------------------------------------------------------- #
# agents
# --------------------------------------------------------------------------- #
class _Lc0Agent(UCIAgent):
    """Maia / any lc0 network. lc0 speaks UCI, so UCIAgent does the work; we
    only need to point it at a weights file and force a 1-node search.

    Maia is designed to be read at ONE node: the raw policy head is the
    human-move predictor, and letting lc0 search on top of it produces a
    stronger, less human player -- which would silently break the calibration.
    """

    def __init__(self, name, lc0_path, weights_path):
        super().__init__("nodes=1", lc0_path, name=name)
        self._opts = {"WeightsFile": weights_path,
                      "Backend": "eigen",     # CPU; override with --lc0-backend
                      "MinibatchSize": "1"}


def build_agent(spec, args, ckpt_paths):
    """spec -> (agent, needs_close). Understands:
         net:<iter>          a checkpoint from --ckpt-dir
         maia:<band>         lc0 + maia-<band>.pb.gz
         sf:<opts>           Stockfish, e.g. sf:nodes=128 or sf:elo=1320
         random              uniform legal mover (the floor of the ladder)
    """
    if spec == "random":
        return _RandomAgent(seed=1234), False

    if spec.startswith("net:"):
        it = int(spec.split(":", 1)[1])
        if it not in ckpt_paths:
            raise SystemExit(f"no checkpoint net_iter{it}.pt in {args.ckpt_dir!r}")
        return make_ckpt_agent(ckpt_paths[it], args.sims, args.c,
                               args.device, opening_plies=0), False

    if spec.startswith("maia:"):
        band = spec.split(":", 1)[1]
        if not args.lc0:
            raise SystemExit("maia anchors need --lc0 /path/to/lc0")
        wp = os.path.join(args.maia_weights_dir, f"maia-{band}.pb.gz")
        if not os.path.exists(wp):
            raise SystemExit(f"maia weights not found: {wp}\n"
                             f"  get them from https://github.com/CSSLab/maia-chess")
        a = _Lc0Agent(spec, args.lc0, wp)
        if args.lc0_backend:
            a._opts["Backend"] = args.lc0_backend
        return a, True

    if spec.startswith("sf:"):
        return UCIAgent(spec, args.engine, name=canon(spec)), True

    raise SystemExit(f"unrecognised player spec: {spec!r}")


# --------------------------------------------------------------------------- #
# schedule: a connected graph, not a round robin
# --------------------------------------------------------------------------- #
def build_schedule(players, net_name, neighbours=2):
    """Players are given in ASCENDING expected strength.

    Each player meets its next `neighbours` rungs. That is enough for a
    connected graph with redundant paths, at O(n) matches instead of the
    O(n^2) a full round robin would cost -- and adjacent pairs are where the
    information is: a 100-0 sweep between distant rungs constrains the fit far
    less than a 60-40 between neighbours.

    The net additionally meets every rung within +-3 of its expected slot, so
    its own placement is over-determined rather than resting on two pairings.
    """
    pairs = []
    n = len(players)
    for i in range(n):
        for k in range(1, neighbours + 1):
            if i + k < n:
                pairs.append((players[i], players[i + k]))
    if net_name in players:
        i = players.index(net_name)
        for j in range(max(0, i - 3), min(n, i + 4)):
            if players[j] != net_name:
                p = (net_name, players[j])
                if p not in pairs and (p[1], p[0]) not in pairs:
                    pairs.append(p)
    return pairs


# --------------------------------------------------------------------------- #
# fit + bootstrap
# --------------------------------------------------------------------------- #
def fit_bt(matches, pin, prior_games=2.0, steps=800):
    """Bradley-Terry MM, ties as half wins, light prior, `pin` fixed at 0.
    Same estimator as score_elo_external.fit_elo -- reimplemented here only so
    the bootstrap can call it on resampled counts."""
    names = sorted({n for (a, b) in matches for n in (a, b)})
    idx = {n: i for i, n in enumerate(names)}
    P = len(names)
    wins = [0.0] * P
    pair_n = {}
    for (a, b), (w, d, l, n) in matches.items():
        i, j = idx[a], idx[b]
        wins[i] += w + 0.5 * d
        wins[j] += l + 0.5 * d
        key = (min(i, j), max(i, j))
        pair_n[key] = pair_n.get(key, 0) + n
    gamma = [1.0] * P
    for _ in range(steps):
        new = []
        for i in range(P):
            num = wins[i] + 0.5 * prior_games
            den = prior_games / (gamma[i] + 1.0)
            for (a, b), nn in pair_n.items():
                if a == i:
                    den += nn / (gamma[i] + gamma[b])
                elif b == i:
                    den += nn / (gamma[i] + gamma[a])
            new.append(num / max(den, 1e-12))
        s = sum(new) / P
        gamma = [g / s for g in new]
    base = gamma[idx[pin]] if pin in idx else 1.0
    return {n: 400.0 * math.log10(gamma[idx[n]] / base) for n in names}


def bootstrap(matches, pin, reps=400, seed=0):
    """Percentile CIs from resampling each pair's outcome.

    Each pair's (w,d,l) is redrawn from a multinomial with the observed
    proportions and the same game count, then the whole ladder is refit. This
    captures how much of the final rating is an artefact of which games
    happened to be won -- including the compounding of that noise along the
    chain from the net to the reference, which a per-pair Wilson interval
    cannot see.
    """
    rng = random.Random(seed)
    draws = {n: [] for n in {x for k in matches for x in k}}
    for _ in range(reps):
        res = {}
        for (a, b), (w, d, l, n) in matches.items():
            p = [w / n, d / n, l / n]
            ww = dd = ll = 0
            for _ in range(n):
                u = rng.random()
                if u < p[0]:
                    ww += 1
                elif u < p[0] + p[1]:
                    dd += 1
                else:
                    ll += 1
            res[(a, b)] = (ww, dd, ll, n)
        for k, v in fit_bt(res, pin).items():
            draws[k].append(v)
    out = {}
    for k, v in draws.items():
        v.sort()
        lo = v[int(0.025 * len(v))]
        hi = v[int(0.975 * len(v)) - 1]
        out[k] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# validity checks
# --------------------------------------------------------------------------- #
def connected(matches, a, b):
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (p, q) in matches:
        ra, rb = find(p), find(q)
        if ra != rb:
            parent[ra] = rb
    return a in parent and b in parent and find(a) == find(b)


def bracketing(matches, who):
    """Return (n_beaten, n_lost_to, saturated_pairs) for `who`."""
    above = below = 0
    saturated = []
    for (a, b), (w, d, l, n) in matches.items():
        if who not in (a, b):
            continue
        s = (w + 0.5 * d) / n if a == who else (l + 0.5 * d) / n
        other = b if a == who else a
        if s > 0.5:
            below += 1
        elif s < 0.5:
            above += 1
        if s <= 0.02 or s >= 0.98:
            saturated.append((other, s, n))
    return below, above, saturated


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Place a checkpoint on an external rating scale.")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--iter", type=int, required=True,
                    help="checkpoint iteration to place")
    ap.add_argument("--sims", type=int, default=400,
                    help="net search budget. MUST match whatever you quote "
                         "elsewhere -- search budget is worth real Elo.")
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--device", default="")
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", ""),
                    help="Stockfish binary (for the chaining ladder)")
    ap.add_argument("--lc0", default="", help="lc0 binary (for Maia references)")
    ap.add_argument("--maia-weights-dir", default=".",
                    help="directory holding maia-<band>.pb.gz")
    ap.add_argument("--lc0-backend", default="",
                    help="lc0 Backend option, e.g. cuda-auto; default eigen (CPU)")
    ap.add_argument("--reference", default="maia:1100",
                    help="which reference to pin the scale to; "
                         f"known: {', '.join(REFERENCES)}")
    ap.add_argument("--ladder", default=",".join(DEFAULT_LADDER),
                    help="comma-separated chaining rungs, ascending strength")
    ap.add_argument("--extra-references", default="",
                    help="comma-separated additional references to include, "
                         "e.g. 'maia:1500'. Including a second one lets you "
                         "CHECK the calibration: fit pinned to each in turn and "
                         "see whether they agree.")
    ap.add_argument("--games", type=int, default=100,
                    help="games per pair (even, so book colours balance)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--no-book", action="store_true")
    ap.add_argument("--matches-file", default="checkpoints/calib_matches.csv")
    ap.add_argument("--bootstrap", type=int, default=400)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the schedule and exit without playing")
    args = ap.parse_args()

    if args.games % 2:
        print(f"[warn] --games {args.games} is odd; book colours will not "
              f"balance exactly.")

    ref = args.reference
    if ref not in REFERENCES:
        print(f"[warn] {ref!r} has no documented rating in REFERENCES; the fit "
              f"will be RELATIVE to it and no absolute placement is reported.")

    ckpts = {}
    for p in glob.glob(os.path.join(args.ckpt_dir, "net_iter*.pt")):
        m = os.path.basename(p)[len("net_iter"):-len(".pt")]
        if m.isdigit():
            ckpts[int(m)] = p

    net = f"net:{args.iter}"
    refs = [ref] + [s for s in args.extra_references.split(",") if s.strip()]
    ladder = [s.strip() for s in args.ladder.split(",") if s.strip()]

    # ascending expected strength: random floor, then node rungs, then the net
    # slotted in near the bottom (it is weak), then the references on top.
    players = ["random"] + ladder[:2] + [net] + ladder[2:] + refs
    seen, ordered = set(), []
    for p in players:
        if p not in seen:
            seen.add(p)
            ordered.append(p)
    players = ordered

    pairs = build_schedule(players, net)
    print(f"players ({len(players)}), ascending expected strength:")
    for p in players:
        tag = "  <- placing" if p == net else ("  <- reference" if p in refs else "")
        print(f"   {p}{tag}")
    print(f"\n{len(pairs)} pairs x {args.games} games = "
          f"{len(pairs) * args.games} games total")
    if args.dry_run:
        for a, b in pairs:
            print(f"   {a:22s} vs {b}")
        return

    book = None if args.no_book else BOOK
    matches = load_matches(args.matches_file)
    agents, closers = {}, []

    def get(spec):
        if spec not in agents:
            a, needs_close = build_agent(spec, args, ckpts)
            agents[spec] = a
            if needs_close:
                closers.append(a)
        return agents[spec]

    try:
        for a, b in pairs:
            if (a, b) in matches or (b, a) in matches:
                print(f"  cached: {a} vs {b}")
                continue
            print(f"  playing {a} vs {b} ...", flush=True)
            w, d, l = match(get(a), get(b), args.games,
                            max_plies=args.max_plies, verbose=False,
                            label=f"{a} vs {b}", book=book)
            append_match(args.matches_file, a, b, w, d, l, args.games)
            matches[(a, b)] = (w, d, l, args.games)
            print(f"    +{w} ={d} -{l}")
    finally:
        for c in closers:
            try:
                c.close()
            except Exception:
                pass

    # ---------------- report ----------------
    print("\n" + "=" * 72)
    ok = True
    if not connected(matches, net, ref):
        print(f"INVALID: {net} and {ref} are not connected by played matches.")
        ok = False
    below, above, sat = bracketing(matches, net)
    print(f"bracketing: {net} scored >50% vs {below} player(s), "
          f"<50% vs {above} player(s)")
    if below == 0 or above == 0:
        print("INVALID: the net is not bracketed -- its rating would be "
              "EXTRAPOLATED off the end of the ladder, not measured. Add "
              "weaker rungs (--ladder 'sf:nodes=1,...') if it beat nothing, "
              "or stronger ones if it beat everything.")
        ok = False
    for other, s, n in sat:
        print(f"  [note] saturated pair vs {other}: score {s:.3f} over {n} "
              f"games carries little information about the size of the gap")

    fit = fit_bt(matches, pin=ref)
    print(f"\nratings relative to {ref} (= 0):")
    for p in sorted(fit, key=lambda k: fit[k]):
        print(f"   {p:24s} {fit[p]:+8.1f}")

    print(f"\nbootstrapping ({args.bootstrap} resamples) ...", flush=True)
    ci = bootstrap(matches, pin=ref, reps=args.bootstrap)

    rel = fit[net]
    lo, hi = ci[net]
    print(f"\n{net} vs {ref}: {rel:+.1f} Elo  [95% CI {lo:+.1f}, {hi:+.1f}]")

    if ok and ref in REFERENCES:
        R = REFERENCES[ref]
        abs_r = R["rating"] + rel
        stat = (hi - lo) / 2.0
        tot = math.sqrt(stat ** 2 + R["sigma"] ** 2)
        print("\n" + "-" * 72)
        print(f"PLACEMENT: {net} ~ {abs_r:.0f} Elo on the {R['scale']} scale")
        print(f"  95% interval {abs_r - tot:.0f} - {abs_r + tot:.0f}")
        print(f"  of which +-{stat:.0f} is match sampling noise")
        print(f"        and +-{R['sigma']:.0f} is {ref}'s own rating uncertainty")
        print(f"  reference: {R['note']}")
        print("-" * 72)
        print("Report the SCALE NAME with the number. Lichess blitz is not "
              "FIDE;\nat this level Lichess runs several hundred points higher "
              "and the\noffset is not constant. Do not convert between scales.")
        if len(refs) > 1:
            print("\nCross-check: refit pinned to each reference in turn --")
            for r2 in refs:
                if r2 in REFERENCES and connected(matches, net, r2):
                    f2 = fit_bt(matches, pin=r2)
                    print(f"   pinned to {r2:12s} -> "
                          f"{REFERENCES[r2]['rating'] + f2[net]:.0f} "
                          f"({REFERENCES[r2]['scale']})")
            print("   Large disagreement means the ladder is mis-specified "
                  "(non-transitive\n   matchups, or a rung whose reference "
                  "rating is wrong).")
    elif not ok:
        print("\nNo absolute placement reported -- see the INVALID lines above.")

    print(f"\nmatches cached in {args.matches_file}")
    print(f"metadata to record: net sims={args.sims}, games/pair={args.games}, "
          f"book={'off' if args.no_book else f'{len(BOOK)} lines'}, "
          f"engine={args.engine or '(none)'}, lc0={args.lc0 or '(none)'}")


if __name__ == "__main__":
    main()

    
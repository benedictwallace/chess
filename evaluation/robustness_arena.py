"""
Robustness arena: measure how two models' strength DEGRADES under injected
mistakes -- the evaluation the forgiveness project is aiming at ("policies
tolerant to noise, approximation error, or small distribution shifts").

Two match designs
-----------------
HEAD-TO-HEAD (default):    A vs B, both perturbed IDENTICALLY at each noise
    level. Isolates *relative* robustness: if A's play collapses faster under
    noise, its score drops as epsilon rises even if A wins the clean match.
BENCHMARK (--benchmark C.pt):  A vs C and B vs C at each level, where the
    benchmark C always plays CLEAN (no perturbation, argmax after the shared
    opening). A common, fixed yardstick: you get two independent degradation
    curves score_A(eps), score_B(eps) against the same opponent. This is the
    cleaner design for the report -- head-to-head confounds "A got worse"
    with "B got better at punishing".

Mistake mechanisms (combinable; all activate only AFTER the shared
opening-randomisation plies so they don't confound opening diversity)
---------------------------------------------------------------------
eps + --mode random    with prob eps play a UNIFORM RANDOM legal move
                       (worst-case noise -- an actuator fault, not a chess
                       mistake).
eps + --mode blunder   with prob eps play a searched-but-inferior move:
                       sampled from the root's NON-best children by visits.
                       A chess-shaped mistake -- plausible but suboptimal.
--vnoise sigma         Gaussian noise added to the value head during search.
                       NOTE: the runner caches evals per position, so this is
                       a FIXED PER-POSITION evaluation error, not iid noise
                       per visit -- i.e. a consistently-wrong evaluator,
                       which is precisely "approximation error / distribution
                       shift" rather than jitter that averages out over
                       visits.
--temp-noise t         sample every post-opening move from visits^(1/t)
                       instead of argmax (diffuse imprecision). NOTE:
                       bypasses --select-* (it replaces the base selection).

Selection policies (--select-a / --select-b; the benchmark always plays
greedy) -- THE forgiveness experiment: same checkpoint, different move rule
---------------------------------------------------------------------------
greedy            argmax visits after the opening (legacy behaviour).
forgiving_tree    delta-constrained forgiveness selection from SUBTREE
                  statistics: among { a : Q1 - Q(a) <= --delta } play the
                  candidate with the highest child_forgiveness (--sel-stat /
                  --sel-agg / --sel-parity; parity=1 = my future slack).
forgiving_head    same delta-set, but scored by the checkpoint's TRAINED
                  FORGIVENESS HEAD evaluated on each candidate's SUCCESSOR
                  position (one batched forward per decision).

PERSPECTIVE FLIP for forgiving_head -- read this before picking a head
checkpoint: the head sees the successor s_a, where the side to move is the
OPPONENT. A head trained on a *_me target therefore predicts the OPPONENT's
slack at s_a, and a *_opp head predicts YOUR post-move slack. So to select
for your own robustness, load the *_opp* head checkpoint (e.g.
net_iter7500_forgiveness_flat_entropy_opp.pt); unfiltered heads
(flat_entropy, tree_entropy) predict the parity-blended slack of the
continuation and need no flip.

Typical selection experiment (same trunk, three rules, clean benchmark):
    python robustness_arena.py \
        forgiveness_heads_iter7500/net_iter7500_forgiveness_flat_entropy_opp.pt \
        forgiveness_heads_iter7500/net_iter7500_forgiveness_flat_entropy_opp.pt \
        --benchmark checkpoints/net_iter2400.pt \
        --select-a greedy --select-b forgiving_head --delta 0.05 \
        --games 200 --sims 300 --eps 0,0.05,0.1,0.2 --mode blunder \
        --out sel_head.csv
(A and B are the SAME checkpoint -- only the selection rule differs, so the
two degradation curves isolate the selector. Run again with
--select-b forgiving_tree for the subtree-statistic agent.)

Sweeps: --eps and --vnoise each take a comma list; levels are their cartesian
product. Both sides of a perturbed pairing get the SAME level.

Output: a printed table and a CSV with one row per (level, pairing):
games, W/D/L, score, Elo diff with a 95% CI, and injected-mistakes-per-game
(the realised mistake rate -- e.g. eps=0.1 over ~40 post-opening moves is ~4
forced mistakes a game; report this, reviewers will ask).

Examples
--------
# sanity run (small, fast):
python robustness_arena.py checkpoints/net_iter2400.pt checkpoints/net_iter1698.pt \
    --games 40 --sims 200 --eps 0,0.1

# the real sweep, vs a clean benchmark:
python robustness_arena.py A.pt B.pt --benchmark checkpoints/net_iter2400.pt \
    --games 200 --sims 300 --eps 0,0.02,0.05,0.1,0.2 --mode blunder \
    --out robustness.csv

# evaluator-error axis instead of move-error axis:
python robustness_arena.py A.pt B.pt --games 200 --sims 300 \
    --eps 0 --vnoise 0,0.05,0.1,0.2
"""

import argparse
import csv
import math
import os
import time
import zlib

import numpy as np
import torch

try:                                    # repo-layout tolerant imports
    from evaluation.score_elo_batched import (run_elo_matches_batched,
                                              _make_eval_fn, select_move)
except ImportError:
    from score_elo_batched import (run_elo_matches_batched,
                                   _make_eval_fn, select_move)
try:
    from evaluation.arena import load_net
except ImportError:
    from arena import load_net
try:
    from search.forgiveness import (select_move_forgiving,
                                    flat_forgiveness_hybrid)
except ImportError:
    from forgiveness import select_move_forgiving, flat_forgiveness_hybrid
try:
    from model.encoding import encode_env
except ImportError:
    from encoding import encode_env


# --------------------------------------------------------------------------- #
# perturbations
# --------------------------------------------------------------------------- #
def wrap_value_noise(eval_fn, sigma, seed):
    """Add N(0, sigma) to the value head, clipped to [-1, 1]. Combined with
    the runner's per-position eval cache this realises a fixed per-position
    evaluation error (see module docstring)."""
    if sigma <= 0:
        return eval_fn
    rng = np.random.default_rng(seed)

    def noisy(planes_list):
        logits, values = eval_fn(planes_list)
        v = np.asarray(values, dtype=np.float64)
        return logits, np.clip(v + rng.normal(0.0, sigma, size=v.shape),
                               -1.0, 1.0)
    return noisy


def make_head_F_fn(net):
    """planes list -> forgiveness-head outputs in [0, 1] (one batched
    forward, return_forgiveness=True)."""
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def head_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    _logits, _value, f = net(x, return_forgiveness=True)
            else:
                _logits, _value, f = net(x, return_forgiveness=True)
        return f.float().cpu().numpy().reshape(-1)
    return head_fn


class Perturber:
    """decide_move callback for run_elo_matches_batched. Applies the shared
    opening temperature to EVERYONE, then per-player BASE SELECTION
    (greedy / forgiving_tree / forgiving_head, see `selectors`) with
    per-player mistakes layered on top for perturbed ids. Counts decisions
    and injected mistakes per player for the realised-rate report, plus
    forgiveness-selection stats (delta-set size, switch rate, Q sacrifice)."""

    def __init__(self, perturbed_ids, eps, mode, temp_noise,
                 opening_plies, opening_temp, seed, selectors=None,
                 diag_delta=None):
        self.perturbed = set(perturbed_ids)
        self.eps = float(eps)
        self.mode = mode
        self.temp_noise = float(temp_noise)
        self.opening_plies = opening_plies
        self.opening_temp = opening_temp
        self.rng = np.random.default_rng(seed)
        self.selectors = selectors or {}    # net_id -> selection spec dict
        self.diag_delta = diag_delta        # log greedy |S| at this delta
        self.decisions = {}             # net_id -> post-opening decisions
        self.mistakes = {}              # net_id -> injected mistakes
        self.sel_stats = {}             # net_id -> dict(n, switched, sacrifice,
                                        #               ndelta_sum)

    def _head_score_children(self, g, head_fn):
        """score_children hook: forgiveness-head value of each candidate's
        SUCCESSOR position, one batched forward. See the module docstring's
        PERSPECTIVE FLIP note for which head that value belongs to."""
        def score(children):
            planes = []
            for c in children:
                env2 = g.env.clone()
                env2.step(c.move)
                planes.append(encode_env(env2))
            return list(head_fn(planes))
        return score

    def _hybrid_score_children(self, g, spec):
        """score_children hook: flat subtree statistic with the head imputing
        the undefined frontier (flat_forgiveness_hybrid). Load a *_me flat
        head for parity=1 -- imputation nodes are positions where the
        aggregate's owner is to move, so no perspective flip (module
        docstring)."""
        head_fn = spec["head_fn"]

        def score(children):
            out = []
            for c in children:
                base = g.env.clone()
                base.step(c.move)

                def impute(paths, _base=base):
                    planes = []
                    for path in paths:
                        env2 = _base.clone()
                        for mv in path:
                            env2.step(mv)
                        planes.append(encode_env(env2))
                    return list(head_fn(planes))

                f, info = flat_forgiveness_hybrid(
                    c, spec["tau"], impute, stat=spec["stat"],
                    parity=spec["parity"],
                    impute_min_visits=spec["impute_min_visits"],
                    max_impute=spec["max_impute"], return_info=True)
                st = self.sel_stats.setdefault(
                    g.search_net, dict(n=0, switched=0, sacrifice=0.0,
                                       ndelta_sum=0))
                st["imputed_mass"] = (st.get("imputed_mass", 0.0)
                                      + info["imputed_mass"])
                st["imputed_calls"] = st.get("imputed_calls", 0) + 1
                out.append(f)
            return out
        return score

    def _base_move(self, g, visit_counts, pid, best):
        """The player's UNPERTURBED move under its selection policy."""
        spec = self.selectors.get(pid)
        if not spec or spec["kind"] == "greedy":
            if self.diag_delta is not None:      # |S| DIAGNOSTIC only: how
                kids = [c for c in g.root.children if c.visits > 0]
                if len(kids) >= 2:               # flat are the positions this
                    qs = [c.value / c.visits for c in kids]   # player reaches
                    q1 = max(qs)                 # (steering control) -- the
                    nd = sum(q1 - q <= self.diag_delta for q in qs)
                    st = self.sel_stats.setdefault(
                        pid, dict(n=0, switched=0, sacrifice=0.0,
                                  ndelta_sum=0))
                    st["n"] += 1
                    st["ndelta_sum"] += nd       # move played stays `best`
            return best
        score_children = None
        if spec["kind"] == "forgiving_head":
            score_children = self._head_score_children(g, spec["head_fn"])
        elif spec["kind"] == "forgiving_hybrid":
            score_children = self._hybrid_score_children(g, spec)
        move, info = select_move_forgiving(
            g.root, spec["delta"], spec["tau"], floor=0,
            gamma=spec["gamma"], stat=spec["stat"], agg=spec["agg"],
            parity=spec["parity"], score_children=score_children,
            return_info=True)
        st = self.sel_stats.setdefault(
            pid, dict(n=0, switched=0, sacrifice=0.0, ndelta_sum=0))
        st["n"] += 1
        st["switched"] += bool(info.get("switched"))
        st["sacrifice"] += info.get("q1", 0.0) - info.get("q_played", 0.0)
        st["ndelta_sum"] += info.get("n_delta", 1)
        return move if move is not None else best

    def _blunder(self, visit_counts, best):
        """A searched-but-inferior move: sample non-best children by visits.
        Falls back to the best move if it is the only child."""
        alts = [(m, v) for m, v in visit_counts.items() if m != best and v > 0]
        if not alts:
            alts = [(m, 1) for m in visit_counts if m != best]
        if not alts:
            return best
        moves, w = zip(*alts)
        w = np.asarray(w, dtype=np.float64)
        return moves[self.rng.choice(len(moves), p=w / w.sum())]

    def __call__(self, g, visit_counts):
        if g.ply < self.opening_plies:                    # shared opening
            return select_move(visit_counts, self.opening_temp, self.rng)

        pid = g.search_net
        best = max(visit_counts, key=visit_counts.get)
        base = self._base_move(g, visit_counts, pid, best)
        if pid not in self.perturbed:
            return base                                    # clean benchmark

        self.decisions[pid] = self.decisions.get(pid, 0) + 1
        if self.eps > 0 and self.rng.random() < self.eps:
            self.mistakes[pid] = self.mistakes.get(pid, 0) + 1
            if self.mode == "random":
                legal = g.env.legalMoves()
                return legal[self.rng.integers(len(legal))]
            return self._blunder(visit_counts, best)
        if self.temp_noise > 0:
            return select_move(visit_counts, self.temp_noise, self.rng)
        return base


# --------------------------------------------------------------------------- #
# match plumbing
# --------------------------------------------------------------------------- #
def make_tickets(pairing_name, tracked_id, opp_id, games):
    """`games` tickets with alternating colours; a_score in the results is
    always the TRACKED player's score."""
    out = []
    for i in range(games):
        a_white = (i % 2 == 0)
        w, b = (tracked_id, opp_id) if a_white else (opp_id, tracked_id)
        out.append(((pairing_name, i), a_white, w, b))
    return out


def elo_ci(scores):
    """Mean score -> Elo difference with a 95% CI from the per-game sample
    std (scores in {0, 0.5, 1}). Probabilities are clamped away from 0/1 so
    a whitewash maps to a finite bound rather than +-inf."""
    n = len(scores)
    s = np.asarray(scores, dtype=np.float64)
    m = s.mean()
    se = s.std(ddof=1) / math.sqrt(n) if n > 1 else 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    eps = 1.0 / (4.0 * max(n, 1))

    def to_elo(p):
        p = min(max(p, eps), 1.0 - eps)
        return -400.0 * math.log10(1.0 / p - 1.0)
    return m, to_elo(m), to_elo(lo), to_elo(hi)


def run_level(nets, pairings, level, args, seed):
    """One noise level: fresh (possibly value-noised) evaluators, one batched
    run over all pairings' tickets, per-pairing score lists back."""
    eps, vnoise = level
    eval_fns = {}
    for pid, net in nets.items():
        fn = _make_eval_fn(net)
        if pid in pairings["perturbed"]:
            # crc32, NOT hash(): str hashes are salted per process, so hash(pid)
            # made value-noise streams irreproducible across runs -- the one
            # place (paper-facing degradation curves) reproducibility matters.
            fn = wrap_value_noise(fn, vnoise,
                                  seed + zlib.crc32(pid.encode()) % 10_000)
        eval_fns[pid] = fn

    selectors = {}
    for pid, spec in pairings.get("selectors", {}).items():
        spec = dict(spec)
        if spec["kind"] in ("forgiving_head", "forgiving_hybrid"):
            spec["head_fn"] = make_head_F_fn(nets[pid])
        selectors[pid] = spec

    perturb = Perturber(pairings["perturbed"], eps, args.mode,
                        args.temp_noise, args.opening_plies,
                        args.opening_temp, seed, selectors=selectors,
                        diag_delta=(args.delta if pairings.get("selectors")
                                    else None))

    tickets = []
    for name, tracked, opp in pairings["pairs"]:
        tickets.extend(make_tickets(name, tracked, opp, args.games))

    done = [0]
    t0 = time.time()

    def progress(pidx, a_score):
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"    {done[0]}/{len(tickets)} games "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    results = run_elo_matches_batched(
        tickets, eval_fns, iterations=args.sims, c=args.c,
        fpu_reduction=args.fpu_reduction,
        opening_plies=args.opening_plies, opening_temp=args.opening_temp,
        max_plies=args.max_plies, concurrency=args.concurrency,
        use_cache=True, cache_cap=args.cache_cap,
        decide_move=perturb, on_game_done=progress,
        rng=np.random.default_rng(seed))

    by_pairing = {name: [] for name, _, _ in pairings["pairs"]}
    for (name, _i), score in results:
        by_pairing[name].append(score)
    return by_pairing, perturb


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness arena: two models under injected mistakes.")
    p.add_argument("model_a", help="checkpoint for model A")
    p.add_argument("model_b", help="checkpoint for model B")
    p.add_argument("--benchmark", default=None,
                   help="optional third checkpoint; A and B each play it "
                        "CLEAN instead of playing each other")
    p.add_argument("--games", type=int, default=200,
                   help="games per pairing per level (default 200; CIs at "
                        "40 games are ~+-100 Elo -- fine for a sanity run, "
                        "useless for a curve)")
    p.add_argument("--sims", type=int, default=300)
    p.add_argument("--eps", default="0,0.05,0.1,0.2",
                   help="comma list of mistake probabilities")
    p.add_argument("--vnoise", default="0",
                   help="comma list of value-noise sigmas (levels = cartesian "
                        "product with --eps)")
    p.add_argument("--mode", choices=["random", "blunder"], default="blunder")
    p.add_argument("--temp-noise", type=float, default=0.0,
                   help="post-opening sampling temperature (0 = argmax; "
                        "bypasses --select-*)")
    sel = ["greedy", "forgiving_tree", "forgiving_head", "forgiving_hybrid"]
    p.add_argument("--select-a", choices=sel, default="greedy",
                   help="model A's post-opening selection policy")
    p.add_argument("--select-b", choices=sel, default="greedy",
                   help="model B's post-opening selection policy")
    p.add_argument("--delta", type=float, default=0.05,
                   help="near-optimal Q window for forgiving selection")
    p.add_argument("--sel-tau", type=float, default=None,
                   help="forgiveness temperature for subtree scoring; "
                        "default: checkpoint config forgiveness_tau, else 0.044")
    p.add_argument("--sel-gamma", type=float, default=0.85)
    p.add_argument("--sel-stat", choices=["gap", "entropy"], default="entropy")
    p.add_argument("--sel-agg", choices=["tree", "flat"], default="tree")
    p.add_argument("--hybrid-impute-min-visits", type=int, default=2,
                   help="forgiving_hybrid: only impute undefined nodes with "
                        "at least this many visits")
    p.add_argument("--hybrid-max-impute", type=int, default=32,
                   help="forgiving_hybrid: head-impute at most this many "
                        "nodes per candidate (heaviest by visits)")
    p.add_argument("--sel-parity", type=int, default=1, choices=[0, 1],
                   help="forgiving_tree only: 1 = my future decision nodes "
                        "(default), 0 = opponent's. forgiving_head "
                        "perspective is set by WHICH head checkpoint you "
                        "load -- see the module docstring")
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--opening-temp", type=float, default=1.0)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--cache-cap", type=int, default=250_000)
    p.add_argument("--c", type=float, default=1.5)
    p.add_argument("--fpu-reduction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="robustness.csv")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    nets = {"A": load_net(args.model_a, device),
            "B": load_net(args.model_b, device)}

    # selection policies for the tracked players (benchmark stays greedy)
    if args.sel_tau is None and (args.select_a != "greedy"
                                 or args.select_b != "greedy"):
        cfg = torch.load(args.model_a, map_location="cpu",
                         weights_only=False).get("config", {})
        args.sel_tau = cfg.get("forgiveness_tau", 0.044)
        print(f"--sel-tau from checkpoint config: {args.sel_tau}")
    selectors = {}
    for pid, kind in (("A", args.select_a), ("B", args.select_b)):
        if kind != "greedy":
            selectors[pid] = dict(kind=kind, delta=args.delta,
                                  tau=args.sel_tau, gamma=args.sel_gamma,
                                  stat=args.sel_stat, agg=args.sel_agg,
                                  parity=args.sel_parity,
                                  impute_min_visits=args.hybrid_impute_min_visits,
                                  max_impute=args.hybrid_max_impute)
    print(f"selection: A={args.select_a}  B={args.select_b}"
          + (f"  delta={args.delta}" if selectors else ""))
    if args.benchmark:
        nets["BM"] = load_net(args.benchmark, device)
        pairings = {"pairs": [("A_vs_BM", "A", "BM"),
                              ("B_vs_BM", "B", "BM")],
                    "perturbed": {"A", "B"},
                    "selectors": selectors}
        print(f"benchmark mode: A and B each play a clean {args.benchmark}")
    else:
        pairings = {"pairs": [("A_vs_B", "A", "B")],
                    "perturbed": {"A", "B"},
                    "selectors": selectors}
        print("head-to-head mode: A vs B, both perturbed identically")

    eps_levels = [float(x) for x in args.eps.split(",")]
    vn_levels = [float(x) for x in args.vnoise.split(",")]
    levels = [(e, v) for v in vn_levels for e in eps_levels]
    print(f"{len(levels)} level(s) x {args.games} games/pairing x "
          f"{len(pairings['pairs'])} pairing(s), {args.sims} sims/move, "
          f"mode={args.mode}")

    rows = []
    for li, level in enumerate(levels):
        eps, vn = level
        print(f"\n== level {li+1}/{len(levels)}: eps={eps} vnoise={vn} ==")
        by_pairing, perturb = run_level(nets, pairings, level, args,
                                        seed=args.seed + 1000 * li)
        # HEAD-TO-HEAD (no --benchmark) has ONE pair whose two players each
        # have their own selector and diagnostics; emit a row per player so
        # the opponent's switch/sacrifice/imputed stats are not lost. The
        # opponent's row mirrors the result (score/Elo negated) and carries
        # ITS own selection + mistake counters. Benchmark mode is unchanged.
        report = []
        for name, tracked, opp in pairings["pairs"]:
            report.append((name, tracked, False))
            if not args.benchmark:
                report.append((f"{name}(B)", opp, True))
        for name, tracked, mirrored in report:
            base = name[:-3] if mirrored else name
            scores = by_pairing[base]
            if mirrored:                      # this player's own perspective
                scores = [1.0 - s for s in scores]
            n = len(scores)
            w = sum(s == 1.0 for s in scores)
            d = sum(s == 0.5 for s in scores)
            l = n - w - d
            m, elo, elo_lo, elo_hi = elo_ci(scores)
            dec = perturb.decisions.get(tracked, 0)
            mis = perturb.mistakes.get(tracked, 0)
            mpg = mis / max(n, 1)
            st = perturb.sel_stats.get(tracked)
            sw = st["switched"] / st["n"] if st and st["n"] else 0.0
            sac = st["sacrifice"] / st["n"] if st and st["n"] else 0.0
            nds = st["ndelta_sum"] / st["n"] if st and st["n"] else 1.0
            imass = (st["imputed_mass"] / st["imputed_calls"]
                     if st and st.get("imputed_calls") else None)
            sel_note = (f"  switch {sw:.0%}, sac {sac:.4f}, "
                        f"|S| {nds:.2f}" if st else "")
            if imass is not None:
                sel_note += f", imputed {imass:.0%}"
            print(f"  {name:9s} [{tracked:2s}] {w:3d}W {d:3d}D {l:3d}L  "
                  f"score {m:.3f}  Elo {elo:+7.1f} "
                  f"[{elo_lo:+7.1f}, {elo_hi:+7.1f}]  "
                  f"injected {mpg:.2f} mistakes/game{sel_note}")
            sel_kind = (pairings.get("selectors", {})
                        .get(tracked, {}).get("kind", "greedy"))
            rows.append(dict(eps=eps, vnoise=vn, pairing=name,
                             player=tracked,
                             selector=sel_kind, delta=args.delta, games=n,
                             wins=w, draws=d, losses=l,
                             score=round(m, 4), elo=round(elo, 1),
                             elo_lo=round(elo_lo, 1), elo_hi=round(elo_hi, 1),
                             mistakes_per_game=round(mpg, 3),
                             post_opening_decisions=dec,
                             switch_rate=round(sw, 4),
                             mean_q_sacrifice=round(sac, 5),
                             mean_delta_set=round(nds, 3),
                             imputed_mass=(round(imass, 4)
                                           if imass is not None else "")))
        # write incrementally so a long sweep is never lost
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    print(f"\nresults -> {os.path.abspath(args.out)}")

    if len(eps_levels) > 1:
        print("\ndegradation summary (score vs opponent):")
        for name in (("A_vs_BM", "B_vs_BM") if args.benchmark
                     else ("A_vs_B", "A_vs_B(B)")):
            pts = [(r["eps"], r["score"]) for r in rows
                   if r["pairing"] == name and r["vnoise"] == vn_levels[0]]
            if len(pts) > 1:
                slope = np.polyfit(*zip(*pts), 1)[0]
                print(f"  {name}: d(score)/d(eps) = {slope:+.3f} "
                      f"(less negative = more robust)")


if __name__ == "__main__":
    main()


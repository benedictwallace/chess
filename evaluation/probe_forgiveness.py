"""
Forgiveness probe: measure per-position forgiveness statistics with a pretrained
checkpoint, for calibration and visualization -- BEFORE committing an forgiveness head
or a search modification to any particular definition.

What it does:
  1. Harvests diverse positions by playing quick policy-only games with the
     checkpoint (sampling moves from the network priors -- no search), snapshotting
     each game once at a random ply. (--fens FILE skips this and probes your own
     positions instead, one FEN per line.)
  2. Runs a full root search on every position, batched across positions exactly
     like batched self-play, with FORCED ROOT VISITS: the top --force-m children
     by prior each receive at least a floor of visits, so their Q values have
     matched standard errors and the action gap is measuring the position, not
     PUCT's visit allocation. No Dirichlet noise (we want the net's honest
     assessment).
  3. From each root's qualified children (visits >= floor) computes, chooser-POV,
     the TWO local forgiveness statistics (plus their recursive aggregates):
       q1, q2, gap = q1 - q2          the action gap
       F_gap = exp(-gap / tau)        gap forgiveness (tau auto-calibrated: median
                                      gap -> F = 0.5, unless --tau given)
       eff_actions = exp(H)           H = entropy of softmax(Q / tau) --
                                      "effective number of good moves"
       forgiveness_entropy                   H normalized to [0,1] by log(#qualified)
       F_tree_gap, F_flat_gap         the recursive / flat-subtree aggregates
       F_tree_entropy, F_flat_entropy of EACH local statistic ("tree" uses
                                      the --gamma decay; "flat" is implicit)
  4. Writes everything (with FENs) to a CSV for visualization, prints summary
     percentiles, an ASCII histogram of F_gap, and the most brittle / most
     forgiving positions found.
  5. Optionally (--sims-hi N) re-searches every position at a larger budget and
     reports the Spearman rank correlation of the gaps across budgets -- the
     "is my cheap-search forgiveness a good proxy for expensive-search forgiveness?" check
     that decides whether forgiveness-head targets can be harvested from ordinary
     self-play searches or need a dedicated high-budget labelling pass.

Usage:
    python probe_forgiveness.py --checkpoint checkpoints/net_iter200.pt
    python probe_forgiveness.py --checkpoint ... --positions 500 --sims 700 --sims-hi 5000
    python probe_forgiveness.py --checkpoint ... --fens my_positions.txt --out probe.csv

Visualization: the CSV has one row per position including its FEN. Sort by
F_gap (or forgiveness_entropy) and paste FENs into any board GUI / lichess analysis
to eyeball what the metric calls brittle vs forgiving; or plot the columns
directly (gap vs ply, F_gap histogram, eff_actions vs v_root, ...).
"""

import argparse
import csv
import math

import numpy as np
import torch

from engine.gameEnv import Chess
from model.encoding import encode_env
from search.puct import node_fpu_q
from model.move_encoding import encodeMovePOV
from evaluation.arena import load_net
from training.self_play_batched import (
    Node, _expand, _backprop, _softmax,
)
from search.forgiveness import forgiveness_from_qs, tree_forgiveness, flat_forgiveness

try:
    from engine.fen import board_to_fen, env_from_fen, square_to_alg
except ImportError:                       # fen.py at repo root
    from fen import board_to_fen, env_from_fen, square_to_alg


def _uci(move):
    p = move.promotion.lower() if move.promotion else ""
    return f"{square_to_alg(move.fromSq)}{square_to_alg(move.toSq)}{p}"


def _probe_eval_fn(net, amp=False):
    """Evaluator for ANALYSIS tools: full fp32 by default, sanitized outputs.

    The training-side evaluator (_make_torch_eval_fn) runs under fp16 autocast
    for throughput. Heavily trained (especially overtrained) checkpoints can
    emit policy logits large enough to overflow float16 (max ~65,504) on some
    positions; the overflow becomes inf, the softmax becomes NaN, and anything
    sampling from those probabilities dies. Probing and interactive play are
    not throughput-critical, so full precision is the right default here
    (--amp opts back in). Outputs are sanitized either way: NaN logits become
    a huge negative (prior ~ 0 for that move; uniform if a whole row is bad),
    inf is capped, and values are clamped to the tanh range [-1, 1]."""
    import torch
    device = next(net.parameters()).device

    def eval_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if amp and device.type == "cuda":
                with torch.autocast("cuda"):
                    logits, value = net(x)
            else:
                logits, value = net(x)
        lg = logits.float().cpu().numpy()
        vl = value.float().cpu().numpy().reshape(-1)
        lg = np.nan_to_num(lg, nan=-1e9, posinf=60.0, neginf=-1e9)
        vl = np.clip(np.nan_to_num(vl, nan=0.0, posinf=1.0, neginf=-1.0),
                     -1.0, 1.0)
        return lg, vl

    return eval_fn


# --------------------------------------------------------------------------- #
# position harvesting: policy-only playouts, one snapshot per game
# --------------------------------------------------------------------------- #
class _HarvestGame:
    __slots__ = ("env", "ply", "target")

    def __init__(self, rng, min_ply, max_ply):
        self.env = Chess()
        self.env.reset()
        self.ply = 0
        self.target = int(rng.integers(min_ply, max_ply + 1))


def harvest_positions(eval_fn, n_positions, *, min_ply=6, max_ply=140,
                      concurrency=64, temp=1.0, seed=0, verbose=True):
    """Play policy-sampled games (no search) and snapshot each once at a random
    target ply in [min_ply, max_ply]. Games that end before their target restart
    without a snapshot. Returns a list of (env_clone, ply) -- clones, not FENs,
    so repetition / fifty-move counters survive into the probe searches."""
    rng = np.random.default_rng(seed)
    games = [_HarvestGame(rng, min_ply, max_ply) for _ in range(concurrency)]
    out = []
    while len(out) < n_positions:
        planes, metas = [], []
        for g in games:
            if g.env.isTerminal():
                g.__init__(rng, min_ply, max_ply)
            planes.append(encode_env(g.env))
            metas.append(g)
        logits_b, _ = eval_fn(planes)
        for g, logits in zip(metas, logits_b):
            legal = g.env.legalMoves()
            if not legal:
                g.__init__(rng, min_ply, max_ply)
                continue
            mover = g.env.board.sideToMove
            idxs = [encodeMovePOV(m, mover) for m in legal]
            probs = _softmax(np.asarray(logits, dtype=np.float64)[idxs] / temp)
            move = legal[int(rng.choice(len(legal), p=probs))]
            g.env.step(move)
            g.ply += 1
            if g.ply >= g.target:
                if not g.env.isTerminal() and g.env.legalMoves():
                    out.append((g.env.clone(), g.ply))
                    if verbose and len(out) % 50 == 0:
                        print(f"  harvested {len(out)}/{n_positions} positions",
                              flush=True)
                g.__init__(rng, min_ply, max_ply)
            elif g.env.isTerminal():
                g.__init__(rng, min_ply, max_ply)
    return out[:n_positions]


# --------------------------------------------------------------------------- #
# batched root search over a set of positions (forced visits, no noise)
# --------------------------------------------------------------------------- #
class _ProbeItem:
    __slots__ = ("env", "ply", "fen", "root", "sims", "forced_set")

    def __init__(self, env, ply):
        self.env = env
        self.ply = ply
        self.fen = board_to_fen(env.board)
        self.root = None
        self.sims = 0
        self.forced_set = None


def search_positions(items, eval_fn, *, sims=700, c=1.5, fpu_reduction=0.25,
                     force_m=8, force_n=40, verbose=True, tag=""):
    """One forced-visit PUCT search per item, batched across items (one
    simulation per item per round, single network forward per round). Returns
    the effective per-child visit floor actually applied."""
    force_eff = 0
    if force_m > 0 and force_n > 0:
        force_eff = min(int(force_n), max(1, sims // (2 * force_m)))
    for it in items:
        it.root = Node()
        it.root.moverSign = 0
        it.sims = 0
        it.forced_set = None

    live = list(items)
    rounds = 0
    while live:
        batch_planes, batch_meta = [], []
        for it in live:
            node = it.root
            env = it.env.clone()
            path = [node]
            at_root = True
            while node.expanded and not node.terminal and node.children:
                best = None
                if at_root and force_eff > 0:
                    if it.forced_set is None:
                        it.forced_set = sorted(node.children,
                                               key=lambda ch: ch.prior,
                                               reverse=True)[:force_m]
                    for ch in it.forced_set:
                        if ch.visits < force_eff:
                            best = ch
                            break
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
                idxs = [encodeMovePOV(m, mover) for m in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {m: float(p) for m, p in zip(legal, probs)}
                _expand(node, priors, mover, False, False, 0.0, 0.0)
                _backprop(path, float(value) if mover == "white" else -float(value))
                it.sims += 1

        live = [it for it in live if it.sims < sims]
        rounds += 1
        if verbose and rounds % 200 == 0:
            print(f"  {tag}search round {rounds}: {len(live)} positions still "
                  f"running", flush=True)
    return force_eff


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def root_q_vector(root, force_eff):
    """Chooser-POV Q values of the root's QUALIFIED children (visits >= floor),
    sorted descending, plus bookkeeping. Sibling Qs share moverSign, so
    comparing them is legitimate; the floor is what makes them comparable in
    variance."""
    visited = [ch for ch in root.children if ch.visits > 0]
    if not visited:
        return None
    tot = sum(ch.visits for ch in visited)
    v_root = sum(ch.value for ch in visited) / tot
    if force_eff > 0:
        qual = [ch for ch in root.children if ch.visits >= force_eff]
    else:
        qual = visited
    if len(qual) < 2:
        qual = sorted(visited, key=lambda ch: -ch.visits)[:2]
    qs = sorted((ch.value / ch.visits for ch in qual), reverse=True)
    best = max(visited, key=lambda ch: ch.visits)
    return {"qs": qs, "v_root": v_root, "n_qual": len(qual),
            "n_legal": len(root.children), "best_move": _uci(best.move)}


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pct(xs, ps=(10, 25, 50, 75, 90)):
    return {p: float(np.percentile(xs, p)) for p in ps}


def _ascii_hist(values, lo=0.0, hi=1.0, bins=20, width=40, label="F_gap"):
    counts, edges = np.histogram(values, bins=bins, range=(lo, hi))
    peak = max(1, counts.max())
    print(f"\n{label} distribution ({len(values)} positions)")
    for i, n in enumerate(counts):
        bar = "#" * int(round(width * n / peak))
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f} {bar} {n}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Forgiveness probe over "
                                             "positions with a checkpoint")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--positions", type=int, default=300,
                    help="positions to harvest (ignored with --fens)")
    ap.add_argument("--fens", default=None,
                    help="file of FENs (one per line) to probe instead of "
                         "harvesting. NOTE: FENs carry no repetition history.")
    ap.add_argument("--sims", type=int, default=700,
                    help="search budget per position")
    ap.add_argument("--sims-hi", type=int, default=0,
                    help="optional second budget for the budget-sensitivity "
                         "check (e.g. 5000); 0 = skip")
    ap.add_argument("--force-m", type=int, default=8,
                    help="root children (by prior) receiving the visit floor")
    ap.add_argument("--force-n", type=int, default=40,
                    help="visit floor per forced child (auto-shrunk to "
                         "sims // (2*m) if the budget is small)")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="temperature for F_gap and the Q-entropy; 0 = "
                         "auto-calibrate so the median gap maps to F = 0.5")
    ap.add_argument("--gamma", type=float, default=0.85,
                    help="decay in the recursive tree forgiveness "
                         "(F_tree_gap and F_tree_entropy)")
    ap.add_argument("--concurrency", type=int, default=128,
                    help="positions searched simultaneously per chunk")
    ap.add_argument("--min-ply", type=int, default=6)
    ap.add_argument("--max-ply", type=int, default=140)
    ap.add_argument("--harvest-temp", type=float, default=1.0,
                    help="softmax temperature for harvest move sampling")
    ap.add_argument("--out", default="forgiveness_probe.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", action="store_true",
                    help="run the net under fp16 autocast (faster, but "
                         "overtrained checkpoints can overflow to NaN)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    net = load_net(args.checkpoint, device).eval()
    eval_fn = _probe_eval_fn(net, amp=args.amp)

    # ---- positions ----
    if args.fens:
        with open(args.fens) as f:
            fens = [ln.strip() for ln in f if ln.strip()]
        items = [_ProbeItem(env_from_fen(fen), -1) for fen in fens]
        print(f"loaded {len(items)} positions from {args.fens}")
    else:
        print(f"harvesting {args.positions} positions "
              f"(policy playouts, plies {args.min_ply}-{args.max_ply})")
        snaps = harvest_positions(
            eval_fn, args.positions, min_ply=args.min_ply,
            max_ply=args.max_ply, concurrency=args.concurrency,
            temp=args.harvest_temp, seed=args.seed)
        items = [_ProbeItem(env, ply) for env, ply in snaps]

    # ---- searches (chunked to bound the eval batch) ----
    def run_all(budget, tag):
        force_eff = 0
        for i in range(0, len(items), args.concurrency):
            chunk = items[i:i + args.concurrency]
            force_eff = search_positions(
                chunk, eval_fn, sims=budget, force_m=args.force_m,
                force_n=args.force_n, tag=tag)
            stats = [root_q_vector(it.root, force_eff) for it in chunk]
            for it, st in zip(chunk, stats):
                yield it, st
            print(f"  {tag}searched {min(i + args.concurrency, len(items))}"
                  f"/{len(items)} positions at {budget} sims", flush=True)

    print(f"\nprobing at {args.sims} sims "
          f"(floor: top-{args.force_m} by prior, "
          f">= {min(args.force_n, max(1, args.sims // (2 * max(1, args.force_m))))} "
          f"visits each)")
    results = []
    for it, st in run_all(args.sims, ""):
        if st is not None:
            results.append((it, st))

    # ---- tau calibration (median gap -> F = 0.5; shared by gap & entropy) ----
    def _cal(xs, fallback):
        pos = [x for x in xs if x is not None and x > 0]
        return max(1e-4, float(np.median(pos)) / math.log(2.0)) if pos else fallback

    gaps = [st["qs"][0] - st["qs"][1] for _, st in results if len(st["qs"]) > 1]

    tau = args.tau if args.tau > 0 else _cal(gaps, 0.05)
    print(f"\ngap percentiles: " +
          "  ".join(f"p{p}={v:.4f}" for p, v in _pct(gaps).items()))
    print(f"tau (gap/entropy) = {tau:.4f} "
          f"({'given' if args.tau > 0 else 'auto: median gap -> F=0.5'}; "
          f"freeze this for a training run)")

    rows = []
    for it, st in results:
        forgiveness = forgiveness_from_qs(st["qs"], tau)
        # tree/flat statistics must be computed HERE, while it.root still holds
        # the low-budget tree (a --sims-hi re-search resets the roots).
        # Both recursive formulations, on BOTH local statistics:
        f_tree_gap = tree_forgiveness(it.root, args.gamma, tau, stat="gap")
        f_flat_gap = flat_forgiveness(it.root, tau, stat="gap")
        f_tree_ent = tree_forgiveness(it.root, args.gamma, tau, stat="entropy")
        f_flat_ent = flat_forgiveness(it.root, tau, stat="entropy")
        rows.append({
            "ply": it.ply,
            "side": it.env.board.sideToMove,
            "n_legal": st["n_legal"], "n_qualified": st["n_qual"],
            "v_root": round(st["v_root"], 4),
            "q1": round(st["qs"][0], 4),
            "q2": round(st["qs"][1], 4) if len(st["qs"]) > 1 else "",
            "gap": round(forgiveness["gap"], 4),
            "F_gap": round(forgiveness["F_gap"], 4),
            "eff_actions": round(forgiveness["eff_actions"], 3),
            "forgiveness_entropy": round(forgiveness["forgiveness_entropy"], 4),
            "F_tree_gap": round(f_tree_gap, 4) if f_tree_gap is not None else "",
            "F_flat_gap": round(f_flat_gap, 4) if f_flat_gap is not None else "",
            "F_tree_entropy": round(f_tree_ent, 4) if f_tree_ent is not None else "",
            "F_flat_entropy": round(f_flat_ent, 4) if f_flat_ent is not None else "",
            "best_move": st["best_move"],
            "fen": it.fen,
        })

    # ---- cross-statistic rank agreement: do the measures even disagree? ----
    stat_cols = ["F_gap", "forgiveness_entropy",
                 "F_tree_gap", "F_flat_gap",
                 "F_tree_entropy", "F_flat_entropy"]
    complete = [r for r in rows if all(r[c] != "" for c in stat_cols)]
    if len(complete) > 10:
        arrs = {c: np.asarray([float(r[c]) for r in complete]) for c in stat_cols}
        print(f"\ncross-statistic Spearman ({len(complete)} positions):")
        print("            " + "".join(f"{c:>13}" for c in stat_cols))
        for a in stat_cols:
            cells = []
            for b in stat_cols:
                rho = 1.0 if a == b else spearman(arrs[a], arrs[b])
                cells.append(f"{rho:>13.3f}")
            print(f"{a:>12}" + "".join(cells))

    # ---- optional budget-sensitivity check ----
    if args.sims_hi > args.sims:
        print(f"\nre-probing at {args.sims_hi} sims for the budget check")
        hi_gap, hi_ent = {}, {}
        for it, st in run_all(args.sims_hi, "hi-"):
            if st is not None and len(st["qs"]) > 1:
                hi_gap[id(it)] = st["qs"][0] - st["qs"][1]
                hi_ent[id(it)] = forgiveness_from_qs(st["qs"], tau)["forgiveness_entropy"]
        lo, hi, lo_e, hi_e = [], [], [], []
        for (it, st), row in zip(results, rows):
            g = hi_gap.get(id(it))
            row["gap_hi"] = round(g, 4) if g is not None else ""
            if g is not None and len(st["qs"]) > 1:
                lo.append(row["gap"]); hi.append(g)
                lo_e.append(row["forgiveness_entropy"]); hi_e.append(hi_ent[id(it)])
        rho = spearman(np.asarray(lo), np.asarray(hi))
        print(f"Spearman(gap @ {args.sims}, gap @ {args.sims_hi}) = {rho:.3f} "
              f"over {len(lo)} positions")
        if len(lo_e) > 10:
            rho_e = spearman(np.asarray(lo_e), np.asarray(hi_e))
            print(f"Spearman(forgiveness_entropy @ {args.sims}, @ {args.sims_hi}) = "
                  f"{rho_e:.3f} over {len(lo_e)} positions")
        print("  >= ~0.8: cheap-search forgiveness is a good rank proxy -- harvest "
              "forgiveness-head targets from ordinary self-play searches.")
        print("  <  ~0.6: budget matters -- plan a dedicated high-budget "
              "labelling pass.")

    # ---- write CSV ----
    fieldnames = list(rows[0].keys())
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}")

    # ---- summaries ----
    F = [r["F_gap"] for r in rows]
    print("F_gap percentiles: " +
          "  ".join(f"p{p}={v:.3f}" for p, v in _pct(F).items()))
    Fe = [r["forgiveness_entropy"] for r in rows]
    print("forgiveness_entropy percentiles: " +
          "  ".join(f"p{p}={v:.3f}" for p, v in _pct(Fe).items()))
    eff = [r["eff_actions"] for r in rows]
    print(f"effective actions exp(H): mean {np.mean(eff):.2f}, "
          f"median {np.median(eff):.2f}")
    for col in ("F_tree_gap", "F_flat_gap", "F_tree_entropy", "F_flat_entropy"):
        vals = [float(r[col]) for r in rows if r[col] != ""]
        if vals:
            print(f"{col} percentiles: " +
                  "  ".join(f"p{p}={v:.3f}" for p, v in _pct(vals).items()))
    _ascii_hist(F)
    _ascii_hist(Fe, label="forgiveness_entropy")

    by_f = sorted(rows, key=lambda r: r["F_gap"])
    print("\nmost BRITTLE positions (low F_gap):")
    for r in by_f[:5]:
        print(f"  F={r['F_gap']:.3f} gap={r['gap']:.3f} "
              f"effA={r['eff_actions']:.2f} "
              f"v={r['v_root']:+.2f} best={r['best_move']}  {r['fen']}")
    print("most FORGIVING positions (high F_gap):")
    for r in by_f[-5:]:
        print(f"  F={r['F_gap']:.3f} gap={r['gap']:.3f} "
              f"effA={r['eff_actions']:.2f} "
              f"v={r['v_root']:+.2f} best={r['best_move']}  {r['fen']}")


if __name__ == "__main__":
    main()
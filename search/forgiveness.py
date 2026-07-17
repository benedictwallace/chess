"""
Forgiveness statistics over search trees -- the SINGLE source of truth.

Both sides of the project import from here:
  * training/self_play_batched.py  -> forgiveness_target() for the forgiveness-head labels;
  * evaluation/probe_forgiveness.py and play_checkpoint.py -> the same statistics for
    probing, calibration, and in-game display.

This module is dependency-free (math + numpy) and duck-typed: any node with
`.children`, `.visits`, `.value` works. Q values are chooser-POV averages
`value / visits`; sibling Qs share the chooser's sign at every level.

ARCHITECTURE -- two local statistics, two aggregators:

  The LOCAL measures live in the LOCAL_STATS registry, every one a function
      f(node, tau, min_kids, floor) -> F in [0, 1] or None (undefined).
  node_local_F(...) dispatches into the registry (or accepts a callable), and
  the two AGGREGATORS -- tree_forgiveness and flat_forgiveness -- take a
  `stat` argument and aggregate WHICHEVER local measure you choose. So
  "recursive forgiveness built on the Q-entropy" is
      tree_forgiveness(root, gamma, tau, stat="entropy")
  Both local measures work as training targets, inside both aggregators, and
  in every consumer.

LOCAL STATISTICS (both in [0, 1], higher = more forgiving):

  "gap"      F = exp(-(Q1 - Q2) / tau) over qualified children -- the ACTION
             GAP forgiveness. Q1 - Q2 >= 0 is the cost of the single best
             move being denied, so F = 1 when the top two moves are equal
             (nothing to lose) and F -> 0 as the position becomes a
             one-move-only cliff. tau sets how much Q-cost counts as
             brittle: F = 0.5 exactly when the gap equals tau * ln 2.

  "entropy"  F = H(softmax(Q / tau)) / log(n) over the n qualified children
             -- the normalised Q-ENTROPY, a smooth "how many genuinely good
             moves are there?" measure.

             Construction, step by step:
               1. Boltzmann weights over the qualified Qs:
                      p_a = exp(Q_a / tau) / sum_b exp(Q_b / tau).
                  A move whose Q sits Delta below the best gets relative
                  weight exp(-Delta / tau): moves within ~tau of the best
                  count almost fully, moves several tau worse count ~0. tau
                  is therefore the same "how close is near-optimal" knob as
                  in the gap statistic.
               2. Shannon entropy H = -sum_a p_a log p_a of those weights.
               3. exp(H) is the PERPLEXITY of p -- the "effective number of
                  good moves". If exactly k moves are (near-)equally best
                  and the rest are far worse, p is ~uniform over those k and
                  exp(H) ~ k. This is the smooth, tie-break-free analogue of
                  counting the eps-near-optimal moves.
               4. Normalisation to [0, 1]: for a fixed support of n moves,
                  H is minimised at 0 (one move carries all the weight -->
                  maximally brittle) and maximised at log(n) (all n moves
                  equally good --> maximally forgiving, uniform p). Dividing
                  by that attainable maximum,
                      F = H / log(n)  in [0, 1],
                  equivalently F = log(effective moves) / log(available
                  moves). Requires n >= 2 (log(1) = 0), which the shared
                  min_kids qualification already guarantees.

             Caveat of the normalisation: F is RELATIVE to the number of
             qualified moves n. 2 equally-good moves out of 2 scores F = 1,
             the same as 30 equally good out of 30 -- the measure asks "what
             fraction of the available choice is genuinely usable?", not
             "how many usable moves in absolute terms?". Keep exp(H) around
             (forgiveness_from_qs reports it as eff_actions) when the absolute
             count is wanted. Under a forced-visit floor, n is pinned to the
             forced set, which keeps the denominator comparable across
             positions.

AGGREGATORS (the two formulations of the project doc; both local statistics
plug into either one for a recursive / subtree-level forgiveness):

  tree_forgiveness   F(s) = (1-gamma) F_local(s) + gamma sum_a N(a)/sum N F(s_a)
                     recursively; children with undefined F are handled by
                     renormalising visit weights over the defined ones.
  flat_forgiveness   F(s) = sum_{s' in D(s)} F_local(s') N(s') / N(D(s))
                     over the downstream subtree including s; undefined nodes
                     skipped, weights renormalised. Depth decay is implicit.

QUALIFICATION (shared by both statistics): with floor > 0 (a root searched
under a forced-visit floor) only children with visits >= floor qualify --
matched-variance Qs -- falling back to the min_kids most-visited children if
too few qualify; with floor == 0 (interior nodes / unforced trees) all visited
children qualify. Fewer than min_kids usable children -> the statistic is
undefined (None), which forgiveness_target turns into a masked-out training row.

SETTING TAU: tau is the value-unit scale that decides how much cost counts as
"brittle" -- rankings between positions do not depend on it, absolute F values
and training-target contrast do. Calibrate it empirically: probe_forgiveness prints
tau = median(gap) / ln 2, which maps the median position's gap to F = 0.5; the
same tau serves the entropy statistic (it plays the identical near-optimality
role in the softmax). Fix tau for a whole training run: changing it mid-run
rescales the head's targets under its feet.
"""

import math

import numpy as np


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _qualified(node, min_kids=2, floor=0):
    """Children whose Q estimates are trustworthy enough to compare.
    Returns a list (possibly shorter than min_kids -- callers treat that as
    'statistic undefined')."""
    if floor > 0:
        qual = [c for c in node.children if c.visits >= floor]
        if len(qual) < min_kids:
            qual = sorted((c for c in node.children if c.visits > 0),
                          key=lambda c: -c.visits)[:min_kids]
    else:
        qual = [c for c in node.children if c.visits > 0]
    return qual


# --------------------------------------------------------------------------- #
# local statistics -- every entry: f(node, tau, min_kids, floor) -> F | None
# --------------------------------------------------------------------------- #
def _stat_gap(node, tau, min_kids=2, floor=0):
    qual = _qualified(node, min_kids, floor)
    if len(qual) < min_kids:
        return None
    qs = sorted((c.value / c.visits for c in qual), reverse=True)
    return math.exp(-(qs[0] - qs[1]) / tau)


def _stat_entropy(node, tau, min_kids=2, floor=0):
    qual = _qualified(node, min_kids, floor)
    if len(qual) < min_kids:
        return None
    q = np.asarray(sorted((c.value / c.visits for c in qual), reverse=True))
    p = _softmax(q / tau)
    h = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return h / math.log(len(q))


LOCAL_STATS = {
    "gap": _stat_gap,
    "entropy": _stat_entropy,
}


def node_local_F(node, tau, min_kids=2, floor=0, stat="gap"):
    """LOCAL forgiveness of a tree node under the chosen statistic (a
    LOCAL_STATS key or a callable with the same signature). None where
    undefined."""
    fn = stat if callable(stat) else LOCAL_STATS[stat]
    return fn(node, tau, min_kids=min_kids, floor=floor)


# --------------------------------------------------------------------------- #
# aggregators -- take EITHER local statistic via `stat`
# --------------------------------------------------------------------------- #
def tree_forgiveness(node, gamma, tau, min_kids=2, stat="gap"):
    """Recursive visit-weighted forgiveness (formulation 1 of the project
    doc), built on the chosen local statistic. Summing over visited children
    only IS the formula (unvisited actions carry zero visit weight); children
    with undefined F are handled by renormalising the weights over the
    defined ones. Interior nodes are unforced, so the local statistic is
    evaluated with floor=0 throughout. Returns None below unexpanded
    leaves."""
    kids = [c for c in node.children if c.visits > 0]
    if not kids:
        return None
    f_local = node_local_F(node, tau, min_kids, floor=0, stat=stat)
    tot = sum(c.visits for c in kids)
    acc = wsum = 0.0
    for c in kids:
        f_c = tree_forgiveness(c, gamma, tau, min_kids, stat=stat)
        if f_c is not None:
            w = c.visits / tot
            acc += w * f_c
            wsum += w
    down = acc / wsum if wsum > 0 else None
    if f_local is not None and down is not None:
        return (1.0 - gamma) * f_local + gamma * down
    return f_local if f_local is not None else down


def flat_forgiveness(root, tau, min_kids=2, stat="gap"):
    """Flat subtree forgiveness (formulation 2 of the project doc), built on
    the chosen local statistic. Undefined-F nodes are skipped and the visit
    weights renormalised over the rest."""
    acc = wsum = 0.0
    stack = [root]
    while stack:
        n = stack.pop()
        kids = [c for c in n.children if c.visits > 0]
        stack.extend(kids)
        f = node_local_F(n, tau, min_kids, floor=0, stat=stat)
        if f is not None:
            acc += n.visits * f
            wsum += n.visits
    return acc / wsum if wsum > 0 else None


# --------------------------------------------------------------------------- #
# scalar statistics from a Q vector (probe / display convenience)
# --------------------------------------------------------------------------- #
def forgiveness_from_qs(qs, tau):
    """Scalar forgiveness statistics from a DESCENDING chooser-POV Q vector:
      gap          Q1 - Q2
      F_gap        exp(-gap / tau), the action-gap forgiveness
      eff_actions  exp(H), the effective number of good moves (perplexity of
                   softmax(Q / tau))
      forgiveness_entropy H / log(n), the normalised Q-entropy forgiveness in [0,1]
    A single-move position has zero choice: gap pinned to the maximum (2.0),
    one effective action, zero entropy."""
    if len(qs) < 2:
        return {"gap": 2.0, "F_gap": math.exp(-2.0 / tau),
                "eff_actions": 1.0, "forgiveness_entropy": 0.0}
    q = np.asarray(qs, dtype=np.float64)
    gap = float(q[0] - q[1])
    p = _softmax(q / tau)
    h = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
    return {"gap": gap,
            "F_gap": math.exp(-gap / tau),
            "eff_actions": math.exp(h),
            "forgiveness_entropy": h / math.log(len(q))}


# --------------------------------------------------------------------------- #
# training target
# --------------------------------------------------------------------------- #
def forgiveness_target(root, floor, tau, mode="gap", gamma=0.85, stat=None):
    """(target, mask) pair, both np.float32, for training the forgiveness head from
    a finished root search. mask is 1.0 where the statistic was computable
    and 0.0 otherwise (masked rows contribute nothing to the forgiveness loss).

    mode: a LOCAL_STATS key ("gap" or "entropy") -> that local statistic at
    the root with the forced-visit floor; or "tree" / "flat" -> the
    corresponding aggregator, built on `stat` (default "gap") as its local
    measure. Compound strings "tree_gap" / "tree_entropy" / "flat_gap" /
    "flat_entropy" select aggregator and statistic in one token, so the
    combination is expressible through a single config value
    (CONFIG["forgiveness_target_mode"]) with no extra plumbing. Examples:
        forgiveness_target(root, floor, tau)                             # local gap
        forgiveness_target(root, floor, tau, mode="entropy")             # local entropy
        forgiveness_target(root, floor, tau, mode="tree", stat="entropy")# tree-of-entropy
        forgiveness_target(root, floor, tau, mode="flat_entropy")        # same idea, flat
    """
    if stat is None and "_" in mode:
        agg, _, s = mode.partition("_")
        if agg in ("tree", "flat") and s in LOCAL_STATS:
            mode, stat = agg, s
        else:
            raise ValueError(f"unknown forgiveness target mode {mode!r}")
    if mode == "tree":
        f = tree_forgiveness(root, gamma, tau, stat=stat or "gap")
    elif mode == "flat":
        f = flat_forgiveness(root, tau, stat=stat or "gap")
    else:
        f = node_local_F(root, tau, floor=floor, stat=mode)
    if f is None:
        return np.float32(0.0), np.float32(0.0)
    return np.float32(min(1.0, max(0.0, f))), np.float32(1.0)


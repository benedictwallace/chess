"""
ease_cliff.py - normalised action-gap "cliff", recursed over the tree.

At each node the LOCAL ease is 1 - gap/2, where gap = best_Q - second_best_Q is the
action gap (Q in [-1, 1] so gap in [0, 2], /2 -> [0, 1]):
    1.0 -> top moves tied (flat, forgiving)
    ->0 -> one move far ahead of the rest (a cliff: you must find it)

    
Aggregated as an exponentially-weighted average over the visit-weighted future:
    ease(n) = (1 - gamma) * local(n) + gamma * E_{child ~ visits}[ease(child)]
so a position that is flat now but forces a cliff a few plies later is discounted
toward brittle. Recursing is safe without sign care because gaps are
frame-independent.

FRONTIER HANDLING: continuation is taken ONLY over children that are themselves
decision nodes (have >= 1 well-visited grandchild). Bare frontier leaves are not
folded in, so a shallow tree does not get dragged toward a default value; at the
frontier the measure is just the local gap. (The original folded leaves in, which
biased shallow trees optimistic.)

wide=True measures the gap as best minus the MEAN of the rest instead of best minus
second-best - less sensitive to a single strong second move.


Returns float in [0, 1], or None when the root has < 2 well-visited moves.

PUCT Node needs .children, .visits, .value.
"""


def _decision_node(node, min_visits):
    return any(g.visits >= min_visits for g in node.children)


def _cliff_ease(node, gamma, min_visits, wide):
    kids = [c for c in node.children if c.visits >= min_visits]

    if len(kids) >= 2:
        qs = sorted((c.value / c.visits for c in kids), reverse=True)
        if wide:
            gap = qs[0] - (sum(qs[1:]) / len(qs[1:]))
        else:
            gap = qs[0] - qs[1]
        local = 1.0 - gap / 2.0
    else:
        local = 1.0  # no real choice here -> can't blunder -> forgiving

    branched = [c for c in kids if _decision_node(c, min_visits)]
    if not branched:
        return local

    tot = sum(c.visits for c in branched)
    cont = sum((c.visits / tot) * _cliff_ease(c, gamma, min_visits, wide) for c in branched)
    return (1.0 - gamma) * local + gamma * cont


def ease_cliff(root, gamma=0.9, min_visits=5, wide=False):
    kids = [c for c in root.children if c.visits >= min_visits]
    if len(kids) < 2:
        return None
    return float(_cliff_ease(root, gamma, min_visits, wide))



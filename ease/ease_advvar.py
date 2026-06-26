"""
ease_advvar.py - policy-weighted advantage variance.

low variance in the advantage distribution within a state. Since V(s) is constant across actions,
    Var_{a ~ pi}[A(s, a)] = Var_{a ~ pi}[Q(s, a)].

We weight by the search policy pi (visit distribution by default) so blunders,
which carry ~0 policy mass, do not inflate the variance -- the spread reflects only
the moves the policy actually plays. (A raw moment over ALL legal actions would be
dominated by blunders and would mis-rank a many-good-moves state as high-variance;
the weighting is the fix.)

    ease = exp(-var / scale)  in (0, 1]
    1.0 -> played moves equally good (flat advantages, forgiving)
    ->0  -> large spread among played moves (brittle)

Like perplexity, this is a structural single-state measure: it flags the
single-precise-move cliff (mass concentrates, but the few alternatives that do
carry mass are far below best -> high weighted variance -> low ease). No recursion;
the cheapest faithful target.

weight="visits" (default) uses pi = visit distribution; "softmax" uses a
Q-Boltzmann pi (with tau) to decouple from search effort.

Returns float in (0, 1], or None when < 2 moves clear min_visits.

PUCT Node needs .children, .visits, .value.
"""

import numpy as np


def ease_advvar(root, min_visits=5, scale=0.25, weight="visits", tau=0.15):
    kids = [c for c in root.children if c.visits >= min_visits]
    if len(kids) < 2:
        return None

    q = np.array([c.value / c.visits for c in kids], dtype=np.float64)

    if weight == "softmax":
        w = np.exp((q - q.max()) / tau)
    else:
        w = np.array([c.visits for c in kids], dtype=np.float64)
    w /= w.sum()

    mean = float((w * q).sum())
    var = float((w * (q - mean) ** 2).sum())
    return float(np.exp(-var / scale))
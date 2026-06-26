"""
ease_perplexity.py -- effective number of near-optimal moves (a smooth frac_safe).

Builds a Boltzmann policy over the root moves from their search Q-values,
p_i proportional to exp(Q_i / tau), and returns the NORMALISED entropy:
    H(p) / log(n)   in [0, 1]
    1.0 -> all candidate moves equally good (very forgiving)
    ->0 -> one move dominates (brittle)

exp(H(p)) is the interpretable "effective number of good moves" (perplexity);
return that instead with as_count=True (then predict it with a softplus head, not
sigmoid).

Unlike ease_veps, this is a structural measure: it flags the single-precise-move
cliff directly (peaked policy -> low entropy -> low ease), independent of whether
the trained policy would actually deviate. That makes it a good match for the
project's "avoid states needing one precise action".

Returns float in [0, 1] (or a count >= 1 if as_count), or None when < 2 moves
clear min_visits (mask out).

PUCT Node needs .children, .visits, .value.
"""

import numpy as np


def ease_perplexity(root, tau=0.15, min_visits=5, as_count=False):
    kids = [c for c in root.children if c.visits >= min_visits]
    if len(kids) < 2:
        return None

    q = np.array([c.value / c.visits for c in kids], dtype=np.float64)
    p = np.exp((q - q.max()) / tau)
    p /= p.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())

    if as_count:
        return float(np.exp(H))
    return H / np.log(len(kids))
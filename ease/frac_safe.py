"""
Ease signal #1: FRAC-SAFE (forgiveness), with a discounted return over future states.

Concept
-------
"Local frac-safe" at a position = the share of well-visited root moves whose
search value Q is within `delta` of the best move's Q. High => many moves are
roughly as good (a forgiving position); low => one sharp best line.

You asked for frac-safe *with added return for future states*: rather than
training the head on the local value alone, the target at ply t is the
gamma-discounted average of the local frac-safe at t and at the realised future
plies of the same game:

        G_t = ( sum_{j>=t, L_j defined} gamma^(j-t) * L_j )
              / ( sum_{j>=t, L_j defined} gamma^(j-t) )

Because every L_j is in [0, 1] and the weights are a probability distribution,
G_t is a convex combination of values in [0, 1] -> G_t in [0, 1] automatically
(no extra normalisation, matches the sigmoid ease head's range).

This is an ABSOLUTE (unsigned) signal: it describes how forgiving the position
is, independent of who is winning, so no mover-sign is applied. It is
RECORD-ONLY -- trained as a head, never used to steer search.

Interface (shared by every ease signal, so cliff/stability drop in the same way)
-------------------------------------------------------------------------------
    sig.local(root)      -> float | None   # per-position scalar, or None if undefined
    sig.returns(locals_) -> list[(target, mask)]   # post-game aggregation
    sig.name             -> str             # used for logging / column names
"""

from typing import Optional


class FracSafeEase:
    name = "fracsafe"

    def __init__(self, min_visits: int = 5, delta: float = 0.1, gamma: float = 0.9):
        """
        min_visits : ignore barely-explored root moves (noisy Q).
        delta      : a move counts as "safe" if its Q is within delta of the best Q.
        gamma      : discount for the future-state return (0 => local only).
        """
        self.min_visits = min_visits
        self.delta = delta
        self.gamma = gamma

    # ------------------------------------------------------------------ #
    # per-position (called once per ply, on the finished search tree)
    # ------------------------------------------------------------------ #
    def local(self, root) -> Optional[float]:
        """
        Local frac-safe from the root's children Q values. Returns None when
        fewer than two root moves clear `min_visits` (signal undefined).
        root.children[i].value / .visits is already in the side-to-move frame,
        so the comparison across siblings is consistent.
        """
        qs = [c.value / c.visits for c in root.children if c.visits >= self.min_visits]
        if len(qs) < 2:
            return None
        q_best = max(qs)
        return sum(q >= q_best - self.delta for q in qs) / len(qs)

    # ------------------------------------------------------------------ #
    # post-game aggregation into per-ply (target, mask)
    # ------------------------------------------------------------------ #
    def returns(self, locals_):
        """
        Turn the per-ply local values (some may be None) into a discounted
        future-state return per ply. A ply whose own local is None is masked
        out of the loss (mask 0.0) but its discount gap is still respected: the
        running accumulators decay through it so a later defined ply is weighted
        by its true ply distance.

        Returns a list of (target, mask) the same length as `locals_`.
        """
        n = len(locals_)
        out = [(0.0, 0.0)] * n
        num = 0.0   # running  sum_{future defined} gamma^k * L
        den = 0.0   # running  sum_{future defined} gamma^k
        for t in range(n - 1, -1, -1):
            L = locals_[t]
            if L is not None:
                num = L + self.gamma * num
                den = 1.0 + self.gamma * den
                out[t] = (num / den, 1.0)
            else:
                # undefined here: emit no target, but decay so the gap counts
                num = self.gamma * num
                den = self.gamma * den
                out[t] = (0.0, 0.0)
        return out
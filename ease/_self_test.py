"""
Synthetic sanity check for the tree measures -- no engine or net required.

Run from the directory that CONTAINS the ease/ package:
    python -m ease._selftest

It builds two toy roots (a flat position where many moves are equally good, and a
cliff position where one move is far ahead) and prints every measure for each, then
shows all three V_eps perturbation modes on the cliff so you can SEE why the mode
matters. Use it as a template to eyeball calibration on your own trees.
"""

from . import TREE_MEASURES
from .ease_veps import ease_veps


class _FakeNode:
    """Minimal stand-in for puct.Node. `value` is the accumulated sum; Q = value/visits."""
    def __init__(self, q=0.0, visits=0, prior=0.0, moverSign=0, children=None):
        self.value = q * visits          # store as a sum, like the real Node
        self.visits = visits
        self.prior = prior
        self.moverSign = moverSign
        self.children = children or []
        self.move = object()


def _root(qs, mover_sign=1):
    """A root whose children have the given mover-frame Q-values (children are leaves)."""
    n = len(qs)
    kids = [_FakeNode(q=q, visits=20, prior=1.0 / n, moverSign=mover_sign) for q in qs]
    r = _FakeNode(q=0.0, visits=20 * n, prior=0.0, moverSign=0, children=kids)
    return r


def _fmt(x):
    return "None " if x is None else f"{x:5.3f}"


if __name__ == "__main__":
    flat = _root([0.50, 0.49, 0.48])      # many near-equal moves -> should be EASY (~1)
    cliff = _root([0.90, -0.50, -0.50])   # one move far ahead     -> should be BRITTLE (~0)

    print(f"{'measure':<12}{'flat':>8}{'cliff':>8}   (higher = more forgiving)")
    print("-" * 40)
    for name, fn in TREE_MEASURES.items():
        print(f"{name:<12}{_fmt(fn(flat)):>8}{_fmt(fn(cliff)):>8}")

    print()
    print("V_eps on the CLIFF, by perturbation mode (this is the subtlety):")
    print(f"  topk_uniform : {_fmt(ease_veps(cliff, mode='topk_uniform', topk=3))}"
          "   forced deviation -> correctly BRITTLE")
    print(f"  softmax      : {_fmt(ease_veps(cliff, mode='softmax', tau=0.15))}"
          "   peaked policy rarely deviates -> reads forgiving")
    print(f"  near_set     : {_fmt(ease_veps(cliff, mode='near_set', delta=0.10))}"
          "   singleton near-set -> degenerate, reads forgiving")
    print()
    print("Takeaway: for 'avoid states needing one precise action', topk_uniform is")
    print("the V_eps mode that behaves like the structural measures (perplexity/advvar).")
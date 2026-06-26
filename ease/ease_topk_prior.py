"""
ease_topk_prior.py -- de-confounded frac_safe over the top-k moves BY NETWORK PRIOR.

The original frac_safe read its near-optimal set off MCTS visit counts, but a move
only earns visits if PUCT already liked it, so the denominator is shaped by search
effort. Here the candidate set is the k moves the POLICY HEAD ranks highest (by
.prior), chosen independently of how the visit budget was spent; we then count the
fraction whose Q is within delta of the best among them:

    ease = |{ q_i >= q_best - delta }| / (#candidates with a usable Q)
    1.0 -> all top-prior moves are near-best (forgiving)
    low -> only a few plausible moves actually hold up (brittle)

HONEST CAVEAT: this fixes WHICH moves are in the set, but their Q still comes from
visit-based subtree averages, so a top-prior move that search barely explored has a
noisy Q. Two handles:
  - min_visits gates out under-explored candidates (cheap; partially reintroduces a
    visit dependence -- the thing you were trying to escape).
  - pass eval_fn=callable(move)->Q to score candidates DIRECTLY with a one-ply net
    eval instead of trusting the subtree (cleaner, costs k evals per logged
    position). eval_fn must return Q in the ROOT MOVER's frame (higher = better for
    the side to move).

This is a coarse, thresholded measure (it discretises into 1/k, 2/k, ...). If you
want the smooth version of the same idea, use ease_perplexity. Run this mainly as a
control against the original frac_safe to see how much the visit confound cost you.

Returns float in [0, 1], or None when < 2 candidates have a usable Q.

Duck-types on the PUCT Node: needs .children, .prior, .visits, .value, .move.
"""


def ease_topk_prior(root, k=5, delta=0.1, min_visits=1, eval_fn=None):
    ranked = sorted(root.children, key=lambda c: c.prior, reverse=True)[:k]

    qs = []
    for c in ranked:
        if eval_fn is not None:
            qs.append(eval_fn(c.move))
        elif c.visits >= min_visits:
            qs.append(c.value / c.visits)

    if len(qs) < 2:
        return None

    q_best = max(qs)
    return sum(q >= q_best - delta for q in qs) / len(qs)
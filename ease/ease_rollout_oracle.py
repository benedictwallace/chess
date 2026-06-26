"""
ease_rollout_oracle.py -- ground-truth downstream forgiveness via perturbed rollouts.

NOT a training head. This is the reference signal you use to CHOOSE among the cheap
in-tree measures: run it offline on a sample of positions, correlate each cheap
target against it, and commit to whichever tracks it best.

It plays k games from the position where, at every ply, it runs a short search and
then moves UNIFORMLY among the top-k moves by Q (policy-independent deviation -- the
same forced-perturbation idea as ease_veps mode="topk_uniform", which is why this is
the oracle for that measure), to terminal or max_plies, adjudicating by material at
the cap. The mean white-POV outcome is V_eps; V is the greedy search value. The
returned gap is in the ROOT MOVER's frame (<= 0; more negative = less forgiving) and
is left RAW/unsquashed -- correlation is scale-invariant, so don't normalise.

To instead measure realized-noise forgiveness, set select="softmax" (samples a
Q-Boltzmann move); see the ease_veps docstring for why that under-reports the
structural cliff.

COST is k * max_plies * search_iters network evaluations. Offline-only, on a few
hundred sampled positions -- never in the self-play loop.

opponent="greedy" makes only the root side sloppy (opponent always plays its best
searched move); "perturb" (default) perturbs whoever is to move.

Returns float (raw mover-frame gap, <= 0), or None when the root has < 2 moves.
"""

import random


def _root_best(root):
    kids = [c for c in root.children if c.visits > 0]
    if len(kids) < 2:
        return None, kids
    best = max(kids, key=lambda c: c.value / c.visits)
    return best, kids


def _pick(kids, root_sign, select, topk, tau, delta, opponent):
    node_sign = kids[0].moverSign
    sloppy = (node_sign == root_sign) or (opponent == "perturb")
    q = {c: c.value / c.visits for c in kids}

    if not sloppy:
        return max(kids, key=lambda c: q[c]).move

    if select == "softmax":
        import numpy as np
        arr = np.array([q[c] for c in kids], dtype=np.float64)
        w = np.exp((arr - arr.max()) / tau)
        w /= w.sum()
        return kids[int(np.random.choice(len(kids), p=w))].move

    if select == "near_set":
        qb = max(q.values())
        near = [c for c in kids if q[c] >= qb - delta]
        return random.choice(near).move

    # topk_uniform (default)
    top = sorted(kids, key=lambda c: q[c], reverse=True)[:topk]
    return random.choice(top).move


def ease_rollout_oracle(env, net, k=8, search_iters=64, max_plies=80, c=1.5,
                        select="topk_uniform", topk=3, tau=0.15, delta=0.1,
                        opponent="perturb"):
    # imported here so the module loads without the engine on the path
    from search.puct import search

    root, _ = search(env, net, iterations=search_iters, add_noise=False, c=c)
    best, kids = _root_best(root)
    if best is None:
        return None

    root_sign = best.moverSign
    v_greedy_white = (best.value / best.visits) * best.moverSign

    results = []
    for _ in range(k):
        e = env.clone()
        for _ply in range(max_plies):
            if e.isTerminal():
                break
            r, _vc = search(e, net, iterations=search_iters, add_noise=False, c=c)
            ek = [x for x in r.children if x.visits > 0]
            if not ek:
                break
            e.step(_pick(ek, root_sign, select, topk, tau, delta, opponent))
        res = e.result()
        results.append(res if res is not None else e.adjudicate())

    v_eps_white = sum(results) / len(results)
    return (v_eps_white - v_greedy_white) * root_sign
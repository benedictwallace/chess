"""
ease_veps.py -- ease as the value lost to sloppy-but-reasonable play: V_eps(s) - V(s).

V(s)      : value under greedy play (the search Q of the best move).
V_eps(s)  : value when, at each ply, the side to move plays a PERTURBED policy
            instead of always the single best move.

ease = V_eps - V, measured in the ROOT MOVER's frame so it is always <= 0 (being
sloppy never helps the mover), then squashed to [0, 1]:
    1.0  -> fully forgiving (sloppy play costs ~nothing downstream)
    ->0  -> brittle (only the precise line holds up)

Recursive over the tree. Horizon is bounded by how deep the tree was expanded.

---------------------------------
perturbation choice 
----------------------------------

    mode="topk_uniform"  (default, k fixed)
        At each sloppy node, play UNIFORMLY among the top-k moves by Q. Deviation
        probability is independent of how peaked Q is, so a position with one good
        move and several bad ones loses value -> BRITTLE. Need to watch out for forced
        moves since this discourages.
        
    mode="softmax"  (tau)
        Sample from a Q-Boltzmann policy. takes proportional moves from the dist.
        most similar to maxEntRL, makes policy robust to self noise, probably 
        increases the action gap rather than looks for forgiving states.

    mode="near_set"  (delta)
        Uniform over the delta-near-optimal set. Degenerate on the cliff: when only
        the best move is within delta, the set is a one move, V_eps = V, ease =1
        (max forgiving), not what we want, maybe can be paired with perplexity or frac safe?



soft max and near set both like brittle moves, they probably become robust in a more classical sense, robust to action noise
by increasing the action gap.

------------------------------------------------------------------------------

opponent:
    "perturb" (default) - both sides play the perturbed policy.
    "greedy"            - only the root mover is sloppy; the opponent always
                            plays its best reply.

Returns float in [0, 1], or None when the root has < 2 well-visited moves
(target undefined -> mask out).

PUCT Node needs .children, .visits, .value, .moverSign.
"""

import numpy as np


def _white_pov_q(node):
    if node.visits == 0:
        return 0.0
    return (node.value / node.visits) * node.moverSign


def _v_eps_white(node, min_visits, root_sign, mode, opponent, topk, tau, delta):
    kids = [c for c in node.children if c.visits >= min_visits]
    if not kids:
        return _white_pov_q(node)

    node_sign = kids[0].moverSign # side to move at `node`
    sloppy = (node_sign == root_sign) or (opponent == "perturb")
    rec = lambda c: _v_eps_white(c, min_visits, root_sign, mode, opponent, topk, tau, delta)

    if not sloppy:  # sharp opponent: best reply only
        best = max(kids, key=lambda c: c.value / c.visits)
        return rec(best)

    q = [c.value / c.visits for c in kids] # mover frame (higher = better for mover)

    if mode == "softmax":
        qa = np.asarray(q, dtype=np.float64)
        w = np.exp((qa - qa.max()) / tau)
        w /= w.sum()
        return float(sum(wi * rec(c) for wi, c in zip(w, kids)))

    if mode == "near_set":
        q_best = max(q)
        near = [c for c in kids if (c.value / c.visits) >= q_best - delta]
        return sum(rec(c) for c in near) / len(near)

    # topk_uniform (default)
    top = sorted(kids, key=lambda c: c.value / c.visits, reverse=True)[:topk]
    return sum(rec(c) for c in top) / len(top)


def ease_veps(root, mode="topk_uniform", topk=3, tau=0.15, delta=0.1,
                min_visits=5, scale=1.0, opponent="perturb"):
    kids = [c for c in root.children if c.visits >= min_visits]
    if len(kids) < 2:
        return None

    root_sign = kids[0].moverSign
    best = max(kids, key=lambda c: c.value / c.visits)

    v_greedy_white = _white_pov_q(best)  # white-POV
    v_eps_white = _v_eps_white(root, min_visits, root_sign,
                                mode, opponent, topk, tau, delta)  # white-POV

    gap_mover = (v_eps_white - v_greedy_white) * root_sign # <= 0, mover frame
    return max(0.0, 1.0 + gap_mover / scale)





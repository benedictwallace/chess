"""
Each example: (planes, policy_target, value_target,
               ease_target, ease_mask,
               cliff_target, cliff_mask,
               stab_target, stab_mask)

  value_target : game outcome from the mover's POV (signed by mover_sign)
  ease_target  : frac_safe forgiveness in [0,1]   -- ABSOLUTE (unsigned)
  cliff_target : cliff-size discounted return in [0,1] -- ABSOLUTE (unsigned)
  stab_target  : trajectory stability in [0,1]    -- ABSOLUTE (unsigned)
  *_mask       : 1.0 if the target is defined, else 0.0 (masked out of loss)

All three auxiliary signals are RECORD-ONLY: trained as heads, never used to
steer search.
"""

import numpy as np

from gameEnv import Chess
from encoding import encode
from move_encoding import encodeMove, NUM_ACTIONS
from puct import search, select_move


def _policy_target(visit_counts):
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return target
    for move, count in visit_counts.items():
        target[encodeMove(move)] = count / total
    return target


def _forgiveness(root, min_visits=5, delta=0.1):
    """frac_safe: share of well-visited root moves within `delta` of the best Q."""
    qs = [c.value / c.visits for c in root.children if c.visits >= min_visits]
    if len(qs) < 2:
        return None
    q_best = max(qs)
    return sum(q >= q_best - delta for q in qs) / len(qs)


def _cliff_return(node, gamma, min_visits):
    """
    Recursive cliff-size discounted return over the finished search tree.
    local cliff = (best_Q - second_best_Q)/2 in [0,1] (cost of erring here);
    aggregated as an exponentially-weighted average over the visit-weighted
    future:  ease(n) = (1-gamma)*local + gamma * E_children[ease(child)].
    """
    kids = [c for c in node.children if c.visits >= min_visits]
    if len(kids) >= 2:
        qs = sorted((c.value / c.visits for c in kids), reverse=True)
        local = (qs[0] - qs[1]) / 2.0
    else:
        local = 0.0
    if not kids:
        return local
    tot = sum(c.visits for c in kids)
    cont = sum((c.visits / tot) * _cliff_return(c, gamma, min_visits) for c in kids)
    return (1.0 - gamma) * local + gamma * cont


def _cliff_target(root, gamma=0.9, min_visits=5):
    """Root cliff-return, or None when forgiveness is undefined at the root."""
    kids = [c for c in root.children if c.visits >= min_visits]
    if len(kids) < 2:
        return None
    return _cliff_return(root, gamma, min_visits)


def _position_value_white(root, mover_sign):
    """Search-policy-weighted value (mover frame) converted to white's POV."""
    tot = sum(c.visits for c in root.children if c.visits > 0)
    if tot == 0:
        return 0.0
    v_mover = sum(c.value for c in root.children if c.visits > 0) / tot
    return v_mover * mover_sign


def _trajectory_stability(values, t, horizon, scale=2.0):
    """
    Stability of position t = how little the white-POV value swings over the
    next `horizon` plies of the ACTUAL game:  1 - mean|delta v| / scale, in
    [0,1]. None when fewer than two values are available (near game end).
    """
    window = values[t: t + horizon + 1]
    if len(window) < 2:
        return None
    deltas = [abs(window[i + 1] - window[i]) for i in range(len(window) - 1)]
    mac = sum(deltas) / len(deltas)            # mean abs consecutive change, [0,2]
    return max(0.0, 1.0 - mac / scale)


def play_game(net, iterations, max_plies=200, temp_moves=30, c=1.5,
              ease_min_visits=5, ease_delta=0.1,
              cliff_gamma=0.9, stab_horizon=8):
    """Play one self-game; return training examples (3-tuples)."""
    env = Chess()
    env.reset()

    # history rows: (planes, policy, mover_sign, v_white)
    history = []

    ply = 0
    while ply < max_plies:
        if env.isTerminal():
            break

        mover_sign = 1 if env.board.sideToMove == "white" else -1
        planes = encode(env.board)

        root, visit_counts = search(env, net, iterations=iterations, c=c, add_noise=True)
        if not visit_counts:
            break

        policy_target = _policy_target(visit_counts)

        #ez = _forgiveness(root, ease_min_visits, ease_delta)
        #ease_target, ease_mask = (ez, 1.0) if ez is not None else (0.0, 0.0)

        #cl = _cliff_target(root, cliff_gamma, ease_min_visits)
        #cliff_target, cliff_mask = (cl, 1.0) if cl is not None else (0.0, 0.0)

        v_white = _position_value_white(root, mover_sign)

        history.append((planes, policy_target, mover_sign, v_white))

        temperature = 1.0 if ply < temp_moves else 0.0
        move = select_move(visit_counts, temperature)
        env.step(move)
        ply += 1

    result = env.result()
    if result is None:                 # hit the ply cap -> adjudicate by material
        result_white_pov = env.adjudicate()
    else:
        result_white_pov = result

    # FIX: Point index to 3 instead of 7 since history tuple was shortened
    values = [row[3] for row in history]        # white-POV value per ply

    examples = []
    for t, (planes, policy_target, mover_sign, _v) in enumerate(history):
        value_target = result_white_pov * mover_sign            # signed (mover POV)

        # st = _trajectory_stability(values, t, stab_horizon)
        # stab_target, stab_mask = (st, 1.0) if st is not None else (0.0, 0.0)

        examples.append((
            planes,
            policy_target,
            np.float32(value_target),
            # np.float32(ease_target),  np.float32(ease_mask),
            # np.float32(cliff_target), np.float32(cliff_mask),
            # np.float32(stab_target),  np.float32(stab_mask),
        ))

    return examples


def generate_games(net, num_games, iterations=100, max_plies=200,
                   temp_moves=30, c=1.5, verbose=True):
    all_examples = []
    for g in range(num_games):
        examples = play_game(net, iterations, max_plies, temp_moves, c)
        all_examples.extend(examples)
        if verbose:
            print(f"Game {g + 1}/{num_games}: {len(examples)} positions.")
    return all_examples
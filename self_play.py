"""
Each example gives (planes, policy_target, value_target, ease_target, ease_mask)

  value_target : game outcome from the mover's POV (signed by mover_sign)
  ease_target  : forgiveness of the position in [0, 1] -- ABSOLUTE (a property of
                 the position, NOT signed by mover_sign). It is the share of
                 well-visited root moves whose Q is within `delta` of the best.
  ease_mask    : 1.0 if ease_target is defined, 0.0 for forced/near-forced
                 positions where forgiveness is meaningless (masked out of loss).
"""

import numpy as np

from gameEnv import Chess
from encoding import encode
from move_encoding import encodeMove, NUM_ACTIONS
from puct import search, select_move


def _policy_target(visit_counts):
    """
    Map moves to logit encoding and normalise.

    Returns:
        target: encoded move mapping with visit counts.
    """
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())

    if total == 0:
        return target
    for move, count in visit_counts.items():
        target[encodeMove(move)] = count/total
    return target


def _forgiveness(root, min_visits=5, delta=0.1):
    """
    Absolute forgiveness ('ease') of the root position, computed from the
    finished search tree.

    All root children were entered by the same player (the side to move at the
    root), so their Q = value/visits are in one shared frame and directly
    comparable -- no sign flip needed. We only trust children with enough
    visits, since a thinly-visited child's Q is noise.

    Returns:
        frac_safe in [0, 1]: share of well-visited moves within `delta` of the
        best move's Q (high -> flat/forgiving, low -> only-move position), or
        None when fewer than two moves are well-visited (forgiveness undefined).
    """
    qs = [c.value / c.visits for c in root.children if c.visits >= min_visits]
    if len(qs) < 2:
        return None
    q_best = max(qs)
    return sum(q >= q_best - delta for q in qs) / len(qs)


def play_game(net, iterations, max_plies=200, temp_moves=30, c=1.5,
              ease_min_visits=5, ease_delta=0.1):
    """
    Play one self-game. Return training examples.

    Args:
        net: network.
        temp_moves: number of opening plies (one side moves) played at temperature 1,
                    after that temperature drops to 0.
        max_plies: hard cap on moves, game considered draw if met.
        ease_min_visits / ease_delta: forgiveness-target hyperparameters.
    """

    env = Chess()
    env.reset()

    # list of (planes, policy_target, mover_sign, ease_target, ease_mask)
    history = []

    ply = 0
    while ply < max_plies:
        if env.isTerminal():
            break

        mover_sign = 1 if env.board.sideToMove == "white" else -1
        planes = encode(env.board)

        root, visit_counts = search(env, net, iterations=iterations, c=c, add_noise=True)

        # no legal moves, terminal
        if not visit_counts:
            break

        policy_target = _policy_target(visit_counts)

        # ease target is a property of THIS position, known now (no game result needed)
        ez = _forgiveness(root, ease_min_visits, ease_delta)
        if ez is None:
            ease_target, ease_mask = 0.0, 0.0      # undefined -> masked out of loss
        else:
            ease_target, ease_mask = ez, 1.0

        history.append((planes, policy_target, mover_sign, ease_target, ease_mask))

        temperature = 1.0 if ply < temp_moves else 0.0
        move = select_move(visit_counts, temperature)
        env.step(move)
        ply += 1

    result = env.result()
    result_white_pov = result if result is not None else 0.0

    examples = []
    for planes, policy_target, mover_sign, ease_target, ease_mask in history:
        value_target = result_white_pov * mover_sign            # mover's POV (signed)
        examples.append((
            planes,
            policy_target,
            np.float32(value_target),
            np.float32(ease_target),                            # absolute (unsigned)
            np.float32(ease_mask),
        ))

    return examples


def generate_games(net, num_games, iterations=100, max_plies=200,
                   temp_moves=30, c=1.5, verbose=True):
    """
    Loops games, returns all examples.
    """

    rng = np.random.default_rng()
    all_examples = []
    for g in range(num_games):
        examples = play_game(net, iterations, max_plies, temp_moves, c)
        all_examples.extend(examples)

        if verbose:
            print(f"Game {g + 1}/{num_games}: {len(examples)} positions.")

    return all_examples
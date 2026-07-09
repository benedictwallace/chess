"""
Each example: (planes, policy_target, value_target)

  value_target : game outcome from the mover's POV (signed by mover_sign)

Legacy sequential self-play, policy/value only. Ease/forgiveness targets
(action-gap and Q-entropy statistics from search/ease.py) are produced by
training/self_play_batched.py, which is the maintained path.
"""

import numpy as np

from engine.gameEnv import Chess
from model.encoding import encode
from model.move_encoding import encodeMove, encodeMovePOV, NUM_ACTIONS
from search.puct import search, select_move


def _policy_target(visit_counts, sideToMove):
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return target
    for move, count in visit_counts.items():
        target[encodeMovePOV(move, sideToMove)] = count / total
    return target


def _position_value_white(root, mover_sign):
    """Search-policy-weighted value (mover frame) converted to white's POV."""
    tot = sum(c.visits for c in root.children if c.visits > 0)
    if tot == 0:
        return 0.0
    v_mover = sum(c.value for c in root.children if c.visits > 0) / tot
    return v_mover * mover_sign


def play_game(net, iterations, max_plies=200, temp_moves=30, c=1.5,
              adj_margin=5.0, adj_plies=20):
    """Play one self-game; return training examples (3-tuples).

    adj_margin/adj_plies control early adjudication ("resignation"): a game in
    which one side holds a material lead of >= adj_margin for adj_plies
    consecutive plies is stopped early and scored for that side. Set adj_plies<=0
    to disable and always play to a natural result or the ply cap.
    """
    env = Chess()
    env.reset()

    # history rows: (planes, policy, mover_sign, v_white)
    history = []

    ply = 0
    adj_streak = 0          # consecutive plies one side has held a decisive lead
    early_result = None     # white-POV result if the game is adjudicated early
    while ply < max_plies:
        if env.isTerminal():
            break

        mover_sign = 1 if env.board.sideToMove == "white" else -1
        planes = encode(env.board)

        root, visit_counts = search(env, net, iterations=iterations, c=c, add_noise=True)
        if not visit_counts:
            break

        # env.board.sideToMove is still the mover here: search() runs on clones
        # and never mutates the env passed in.
        policy_target = _policy_target(visit_counts, env.board.sideToMove)

        v_white = _position_value_white(root, mover_sign)

        history.append((planes, policy_target, mover_sign, v_white))

        temperature = 1.0 if ply < temp_moves else 0.0
        move = select_move(visit_counts, temperature)
        env.step(move)
        ply += 1

        # EARLY ADJUDICATION ("resignation"). If one side has held a decisive
        # material lead (>= adj_margin) for adj_plies consecutive plies, end the
        # game and score it for that side. Two payoffs: (a) clearly-decided games
        # get a clean +/-1 value target instead of being dragged to the ply cap
        # by weak endgame conversion -- where a single material snapshot is a far
        # noisier label; and (b) it removes the biggest self-play cost, grinding
        # out already-won games. Sustained material lead is a conservative proxy
        # (low false-positive at self-play strength); the trade-off is somewhat
        # less endgame-conversion training data. adj_plies<=0 disables it.
        if adj_plies > 0:
            diff = env.material_diff()       # white-POV; O(1) popcounts
            if abs(diff) >= adj_margin:
                adj_streak += 1
                if adj_streak >= adj_plies:
                    early_result = 1.0 if diff > 0 else -1.0
                    break
            else:
                adj_streak = 0

    if early_result is not None:
        result_white_pov = early_result
    else:
        result = env.result()
        if result is None:             # hit the ply cap -> adjudicate by material
            result_white_pov = env.adjudicate()
        else:
            result_white_pov = result

    examples = []
    for planes, policy_target, mover_sign, _v in history:
        value_target = result_white_pov * mover_sign            # signed (mover POV)
        examples.append((
            planes,
            policy_target,
            np.float32(value_target),
        ))

    return examples


def generate_games(net, num_games, iterations=100, max_plies=200,
                   temp_moves=30, c=1.5, adj_margin=5.0, adj_plies=20,
                   verbose=True):
    all_examples = []
    for g in range(num_games):
        examples = play_game(net, iterations, max_plies, temp_moves, c,
                             adj_margin=adj_margin, adj_plies=adj_plies)
        all_examples.extend(examples)
        if verbose:
            print(f"Game {g + 1}/{num_games}: {len(examples)} positions.")
    return all_examples
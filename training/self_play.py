"""
Self-play game generation.

Each training example is:
  - WITHOUT an ease signal:  (planes, policy_target, value_target)
  - WITH an ease signal:     (planes, policy_target, value_target, ease_target, ease_mask)

  value_target : game outcome from the mover's POV (signed by mover_sign), [-1, 1]
  ease_target  : an absolute (unsigned) record-only signal in [0, 1] produced by the
                 chosen ease module (e.g. training.ease_fracsafe.FracSafeEase)
  ease_mask    : 1.0 if the ease target is defined at this ply, else 0.0

The ease head is RECORD-ONLY: it is trained, but never used to steer search.
Pass `ease_signal=None` (the default) to reproduce the original 3-tuple output
exactly.
"""

import numpy as np

from engine.gameEnv import Chess
from model.encoding import encode
from model.move_encoding import encodeMove, NUM_ACTIONS
from search.puct import search, select_move


def _policy_target(visit_counts):
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return target
    for move, count in visit_counts.items():
        target[encodeMove(move)] = count / total
    return target


def _position_value_white(root, mover_sign):
    """Search-policy-weighted value (mover frame) converted to white's POV."""
    tot = sum(c.visits for c in root.children if c.visits > 0)
    if tot == 0:
        return 0.0
    v_mover = sum(c.value for c in root.children if c.visits > 0) / tot
    return v_mover * mover_sign


def play_game(net, iterations, max_plies=200, temp_moves=30, c=1.5, ease_signal=None):
    """
    Play one self-game and return its training examples.

    ease_signal : an object exposing
                    .local(root)      -> float | None
                    .returns(locals_) -> list[(target, mask)]
                  (see training.ease_fracsafe.FracSafeEase). When None, output is
                  the original (planes, policy, value) 3-tuples.
    """
    env = Chess()
    env.reset()

    # history rows: (planes, policy, mover_sign, v_white, local_ease)
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
        v_white = _position_value_white(root, mover_sign)
        local_ease = ease_signal.local(root) if ease_signal is not None else None

        history.append((planes, policy_target, mover_sign, v_white, local_ease))

        temperature = 1.0 if ply < temp_moves else 0.0
        move = select_move(visit_counts, temperature)
        env.step(move)
        ply += 1

    result = env.result()
    if result is None:                 # hit the ply cap -> adjudicate by material
        result_white_pov = env.adjudicate()
    else:
        result_white_pov = result

    # post-game: aggregate per-ply local ease into future-state returns
    if ease_signal is not None:
        locals_ = [row[4] for row in history]
        ease_pairs = ease_signal.returns(locals_)   # [(target, mask), ...]

    examples = []
    for t, (planes, policy_target, mover_sign, _v, _le) in enumerate(history):
        value_target = result_white_pov * mover_sign            # signed (mover POV)

        if ease_signal is None:
            examples.append((
                planes,
                policy_target,
                np.float32(value_target),
            ))
        else:
            ease_target, ease_mask = ease_pairs[t]
            examples.append((
                planes,
                policy_target,
                np.float32(value_target),
                np.float32(ease_target),
                np.float32(ease_mask),
            ))

    return examples


def generate_games(net, num_games, iterations=100, max_plies=200,
                   temp_moves=30, c=1.5, verbose=True, ease_signal=None):
    all_examples = []
    for g in range(num_games):
        examples = play_game(net, iterations, max_plies, temp_moves, c,
                             ease_signal=ease_signal)
        all_examples.extend(examples)
        if verbose:
            print(f"Game {g + 1}/{num_games}: {len(examples)} positions.")
    return all_examples
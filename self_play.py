
"""
Each example gives (planes, policy_target, value_target)

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


def play_game(net, iterations, max_plies=200, temp_moves=30, c=1.5):
    """
    Play one self-game. Return training examples.

    Args:
        net: network.
        temp_moves: number of opening plies (one side moves) played at temperature 1,
                    after that temperature drops to 0.
        max_plies: hard cap on moves, game considered draw if met.
    """

    env = Chess()
    env.reset()

    # list of (planes, policy_target, mover_sign)
    history = []    

    ply = 0
    while ply < max_plies:
        if env.isTerminal():
            break
        
        mover_sign = 1 if env.board.sideToMove == "white" else -1
        planes = encode(env.board)

        _, visit_counts = search(env, net, iterations=iterations, c=c, add_noise=True)

        # no legal moves, terminal
        if not visit_counts:
            break

        policy_target = _policy_target(visit_counts)
        history.append(planes, policy_target, mover_sign)

        temperature = 1.0 if ply < temp_moves else 0.0
        move = select_move(visit_counts, temperature)
        env.step(move)
        ply += 1
    
    result = env.result
    result_white_pov = result if result is not None else 0.0

    examples = []
    for planes, policy_target, mover_sign in history:
        value_target = result_white_pov * mover_sign  # mover's POV
        examples.append((planes, policy_target, np.float32(value_target)))

    return examples


def generate_games(net, num_games, iterations=100, max_plies=200,
                   temp_moves=30, c=1.5, verbose=True):
    """
    Loops games, returns all examples.
    """
    
    rng = np.random.default_rng()
    all_examples=[]
    for g in range(num_games):
        examples = play_game(net, iterations, max_plies, temp_moves, c)
        all_examples.extend(examples)


        if verbose:
            print(f"Game {g + 1}/{num_games}: {len(examples)} positions.")

    return all_examples


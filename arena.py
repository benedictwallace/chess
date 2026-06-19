import argparse
import math
import random

import torch

from gameEnv import Chess
from network import ChessNet
from puct import search, select_move

PIECE_VALUES = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}


# --------------------------------------------------------------------------- #
# Players
# --------------------------------------------------------------------------- #
class NeuralAgent:
    """
    PUCT player. add_noise is OFF for evaluation (no Dirichlet exploration);
    diversity between games comes from sampling the opening plies.
    """

    def __init__(self, net, iterations=50, c=1.5,
                 opening_plies=10, opening_temp=1.0):
        self.net = net
        self.iterations = iterations
        self.c = c
        self.opening_plies = opening_plies
        self.opening_temp = opening_temp

    def select(self, env, ply):
        _, visit_counts = search(
            env, self.net, iterations=self.iterations, c=self.c, add_noise=False
        )
        if not visit_counts:
            return None
        temp = self.opening_temp if ply < self.opening_plies else 0.0
        return select_move(visit_counts, temp)


class RandomAgent:
    def __init__(self, rng):
        self.rng = rng

    def select(self, env, ply):
        moves = env.legalMoves()
        return self.rng.choice(moves) if moves else None


class MaterialAgent:
    """
    Greedy: pick the move maximising (own material : opponent material) one
    ply ahead. Grabs free pieces and recaptures; blind to opponent replies, so
    it's only modestly above random -- which is exactly what a fixed anchor wants.
    """

    def __init__(self, rng):
        self.rng = rng

    @staticmethod
    def _material(board, colour):
        return sum(
            PIECE_VALUES[p] * bin(board.bb[colour, p]).count("1")
            for p in PIECE_VALUES
        )

    def select(self, env, ply):
        moves = env.legalMoves()
        if not moves:
            return None
        mover = env.board.sideToMove
        opp = "black" if mover == "white" else "white"
        best_score, best = None, []
        for m in moves:
            trial = env.clone()
            trial.step(m)
            score = self._material(trial.board, mover) - self._material(trial.board, opp)
            if best_score is None or score > best_score:
                best_score, best = score, [m]
            elif score == best_score:
                best.append(m)
        return self.rng.choice(best)


# --------------------------------------------------------------------------- #
# Game / match
# --------------------------------------------------------------------------- #
def play_game(white_agent, black_agent, max_plies=200):
    """
    Return white's score: 1.0 win, 0.5 draw, 0.0 loss.
    Hitting the ply cap without a terminal position counts as a draw.
    """
    env = Chess()
    env.reset()
    ply = 0
    while ply < max_plies:
        if env.isTerminal():
            break
        agent = white_agent if env.board.sideToMove == "white" else black_agent
        move = agent.select(env, ply)
        if move is None:
            break
        env.step(move)
        ply += 1

    r = env.result()           # +1 white, -1 black, 0 draw, None if not terminal
    if r is None:              # ply cap reached -> adjudicate by material
        r = env.adjudicate()
    if r == 0:
        return 0.5
    return 1.0 if r > 0 else 0.0


def _to_elo(p):
    """
    Expected score p -> Elo difference. Clamped so a clean sweep reports a
    large finite bound rather than infinity.
    """
    p = min(max(p, 1e-9), 1 - 1e-9)
    return 400.0 * math.log10(p / (1 - p))


def match(agent_a, agent_b, games=20, max_plies=200, alternate=True, verbose=True):
    """
    Play games games of A vs B (colours alternating by default).
    Returns a stats dict from A's perspective.
    """
    scores_a = []
    wins = draws = losses = 0

    for g in range(games):
        a_white = (g % 2 == 0) if alternate else True
        if a_white:
            s_a = play_game(agent_a, agent_b, max_plies)
        else:
            s_a = 1.0 - play_game(agent_b, agent_a, max_plies)
        scores_a.append(s_a)

        if s_a == 1.0:
            wins += 1
        elif s_a == 0.0:
            losses += 1
        else:
            draws += 1

        if verbose:
            running = (wins + 0.5 * draws) / (g + 1)
            print(f"  game {g + 1:>3}/{games}  "
                  f"A {'W' if a_white else 'B'}  "
                  f"+{wins} ={draws} -{losses}  score={running:.3f}", flush=True)

    n = len(scores_a)
    score = sum(scores_a) / n
    if n > 1:
        var = sum((x - score) ** 2 for x in scores_a) / (n - 1)
        se = math.sqrt(var / n)
    else:
        se = 0.0

    elo = _to_elo(score)
    elo_lo = _to_elo(score - 1.96 * se)
    elo_hi = _to_elo(score + 1.96 * se)

    return {
        "games": n, "wins": wins, "draws": draws, "losses": losses,
        "score": score, "se": se,
        "elo": elo, "elo_lo": elo_lo, "elo_hi": elo_hi,
        "swept": losses == 0 or wins == 0,
    }


# --------------------------------------------------------------------------- #
# Player construction
# --------------------------------------------------------------------------- #
def load_net(path, device):
    # weights_only=False: these are your own checkpoints (they carry a config
    # dict and optimiser state, not just tensors).
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    net = ChessNet(channels=cfg.get("channels", 64),
                   num_blocks=cfg.get("num_blocks", 5))
    net.load_state_dict(ckpt["model_state"], strict=False)
    net.to(device).eval()
    return net


def make_agent(spec, device, rng, iterations, c, opening_plies):
    if spec == "random":
        return RandomAgent(rng)
    if spec == "material":
        return MaterialAgent(rng)
    if spec == "untrained":
        net = ChessNet().to(device).eval()
        return NeuralAgent(net, iterations, c, opening_plies)
    # otherwise treat as a checkpoint path
    net = load_net(spec, device)
    return NeuralAgent(net, iterations, c, opening_plies)




def main():
    ap = argparse.ArgumentParser(description="Arena / Elo harness")
    ap.add_argument("--a", default="untrained", help="player A: ckpt path | untrained | random | material")
    ap.add_argument("--b", default="random", help="player B: ckpt path | untrained | random | material")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--iterations", type=int, default=50, help="PUCT sims/move for neural players")
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--opening-plies", type=int, default=10, help="plies sampled at temp=1 for game diversity")
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    agent_a = make_agent(args.a, device, rng, args.iterations, args.c, args.opening_plies)
    agent_b = make_agent(args.b, device, rng, args.iterations, args.c, args.opening_plies)

    print(f"device: {device}")
    print(f"A = {args.a}   vs   B = {args.b}")
    print(f"{args.games} games, {args.iterations} sims/move, max {args.max_plies} plies\n")

    stats = match(agent_a, agent_b, games=args.games,
                  max_plies=args.max_plies, alternate=True)

    print("\n" + "=" * 48)
    print(f"A: +{stats['wins']} ={stats['draws']} -{stats['losses']}  "
          f"(score {stats['score']:.3f} +/- {stats['se']:.3f})")
    if stats["swept"]:
        sign = "+" if stats["wins"] else "-"
        print(f"Elo (A - B): {sign}{abs(stats['elo']):.0f}  "
              f"(bound only -- no {'losses' if stats['wins'] else 'wins'} in {stats['games']} games)")
    else:
        print(f"Elo (A - B): {stats['elo']:+.0f}  "
              f"[95% CI {stats['elo_lo']:+.0f}, {stats['elo_hi']:+.0f}]")
    print("=" * 48)


if __name__ == "__main__":
    main()
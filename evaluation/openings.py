"""
A small opening book, in UCI, for the EVALUATION harnesses.

Why this exists
---------------
score_elo_external.make_ckpt_agent builds NeuralAgent(..., opening_plies=8),
whose default opening_temp is 1.0. So every rated game began with eight plies
sampled at temperature 1 from raw visit counts. That buys game diversity by
making the net play deliberately worse for the first four moves of every game
-- the rating you measure is of a handicapped player, and the handicap is
asymmetric when the opponent (Stockfish) plays its best move throughout.

The standard fix is a shared opening book: both seats are handed the SAME
starting position, so diversity costs nobody anything, and each line is played
once with each colour so a line that favours White cancels out.

Usage (in score_elo_external.play_game / evaluation.arena.play_game):

    from openings import BOOK, apply_opening

    def play_game(white_agent, black_agent, max_plies=300, opening=None):
        env = Chess(); env.reset()
        white, black = _Seat(white_agent), _Seat(black_agent)
        white.new_game(); black.new_game()
        ply = 0
        if opening:
            ply = apply_opening(env, opening, [white, black])
        ...  # then the normal loop, starting from `ply`

and in match(), pick the line ONCE per pair of games so colours are balanced:

    for g in range(games):
        line = BOOK[(g // 2) % len(BOOK)]
        ...

Then build the agents with opening_plies=0 (fully deterministic play).
"""

# Mainline openings, 4-8 plies each. All are legal from the start position;
# validated against engine.gameEnv in this project.
BOOK = [
    "e2e4 e7e5 g1f3 b8c6 f1b5",              # Ruy Lopez
    "e2e4 e7e5 g1f3 b8c6 f1c4",              # Italian
    "e2e4 e7e5 g1f3 b8c6 d2d4",              # Scotch
    "e2e4 e7e5 b1c3",                        # Vienna
    "e2e4 c7c5 g1f3 d7d6 d2d4",              # Sicilian, open
    "e2e4 c7c5 g1f3 b8c6 d2d4",              # Sicilian, Nc6
    "e2e4 c7c5 b1c3",                        # Closed Sicilian
    "e2e4 e7e6 d2d4 d7d5 b1c3",              # French
    "e2e4 e7e6 d2d4 d7d5 e4e5",              # French, advance
    "e2e4 c7c6 d2d4 d7d5 b1c3",              # Caro-Kann
    "e2e4 d7d5 e4d5 d8d5 b1c3",              # Scandinavian
    "e2e4 g8f6",                             # Alekhine
    "e2e4 d7d6 d2d4 g8f6",                   # Pirc
    "d2d4 d7d5 c2c4 e7e6",                   # QGD
    "d2d4 d7d5 c2c4 c7c6",                   # Slav
    "d2d4 d7d5 c2c4 d5c4",                   # QGA
    "d2d4 g8f6 c2c4 e7e6 g1f3",              # Indian
    "d2d4 g8f6 c2c4 g7g6 b1c3",              # King's Indian
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",         # Nimzo-Indian
    "d2d4 f7f5",                             # Dutch
    "d2d4 d7d5 g1f3 g8f6 c1f4",              # London
    "c2c4 e7e5 b1c3",                        # English
    "c2c4 g8f6 b1c3 e7e6",                   # English, Indian
    "g1f3 d7d5 g2g3 g8f6 f1g2",              # Reti / KIA
    "e2e4 e7e5 f2f4",                        # King's Gambit
    "d2d4 d7d5 c1f4 g8f6 e2e3",              # London, black Nf6
    "e2e4 c7c5 g1f3 e7e6 d2d4",              # Sicilian, Kan
    "d2d4 e7e6 c2c4 f8b4",                   # Bogo-ish
]


def apply_opening(env, line, seats=()):
    """Play `line` (space-separated UCI) on `env`, notifying every seat via its
    .observe() hook so UCI mirrors stay in sync. Returns the number of plies
    played. Raises if a book move is not legal -- a silent skip would make two
    agents start from different positions."""
    from engine.fen import square_to_alg

    def uci(m):
        p = m.promotion.lower() if m.promotion else ""
        return f"{square_to_alg(m.fromSq)}{square_to_alg(m.toSq)}{p}"

    n = 0
    for want in line.split():
        for m in env.legalMoves():
            if uci(m) == want:
                for s in seats:
                    s.observe(m)
                env.step(m)
                n += 1
                break
        else:
            raise ValueError(f"book move {want!r} illegal at ply {n} of {line!r}")
    return n

    
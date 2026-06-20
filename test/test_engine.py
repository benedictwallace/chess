"""
Functionality tests for the optimised chess engine.

Covers:
  - perft vs published reference node counts (the gold-standard correctness test)
  - make/unmake round-trip integrity: board state AND occupancy masks restored
  - incremental occupancy invariant: masks match a from-scratch recompute at
    every interior node
  - precomputed attack tables vs known geometry
  - tactical correctness: pins, checkmate, stalemate, en passant, promotion
  - arena Elo math + result accounting (skipped if torch unavailable)

Run:  python test_engine.py
"""

import random
import traceback

from engine.board import Board, perft
from engine.moves import Move, knightMoves, kingMoves, KNIGHT_ATTACKS, KING_ATTACKS, PAWN_ATTACKS


# --------------------------------------------------------------------------- #
# tiny test runner
# --------------------------------------------------------------------------- #
_PASS = _FAIL = 0

def check(cond, msg):
    global _PASS, _FAIL
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
        print(f"  FAIL: {msg}")

def section(name):
    print(f"\n=== {name} ===")


# --------------------------------------------------------------------------- #
# FEN loader (their square layout: a1=0, h1=7, a8=56, h8=63)
# --------------------------------------------------------------------------- #
_PIECE = {"p": "pawn", "n": "knight", "b": "bishop",
          "r": "rook", "q": "queen", "k": "king"}

def board_from_fen(fen: str) -> Board:
    b = Board()
    for k in b.bb:
        b.bb[k] = 0
    parts = fen.split()
    placement, side = parts[0], parts[1]
    castling = parts[2] if len(parts) > 2 else "-"
    ep = parts[3] if len(parts) > 3 else "-"

    for i, row in enumerate(placement.split("/")):   # row 0 = rank 8
        rank = 7 - i
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                colour = "white" if ch.isupper() else "black"
                b.bb[colour, _PIECE[ch.lower()]] |= (1 << (rank * 8 + file))
                file += 1

    b.sideToMove = "white" if side == "w" else "black"
    b.whiteKCastle, b.whiteQCastle = "K" in castling, "Q" in castling
    b.blackKCastle, b.blackQCastle = "k" in castling, "q" in castling
    if ep == "-":
        b.enPassantSq = -1
    else:
        b.enPassantSq = (int(ep[1]) - 1) * 8 + (ord(ep[0]) - ord("a"))
    b.history = []
    b.updatePieces()
    return b


# --------------------------------------------------------------------------- #
# 1. perft vs published references
# --------------------------------------------------------------------------- #
# (name, fen, [(depth, expected), ...])   depths kept modest for pure-Python speed
PERFT_CASES = [
    ("startpos",
     "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
     [(1, 20), (2, 400), (3, 8902), (4, 197281)]),
    ("kiwipete (castling-heavy)",
     "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
     [(1, 48), (2, 2039), (3, 97862)]),
    ("position 3 (pins / e.p.)",
     "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
     [(1, 14), (2, 191), (3, 2812), (4, 43238)]),
    ("position 4 (promotion)",
     "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
     [(1, 6), (2, 264), (3, 9467)]),
    ("position 5 (promotion)",
     "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
     [(1, 44), (2, 1486), (3, 62379)]),
    ("position 6",
     "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - -",
     [(1, 46), (2, 2079)]),
]

def test_perft():
    section("perft vs reference node counts")
    for name, fen, expected in PERFT_CASES:
        for depth, want in expected:
            b = board_from_fen(fen)
            got = perft(b, b.sideToMove, depth)
            ok = got == want
            tag = "ok" if ok else f"MISMATCH (got {got}, want {want})"
            print(f"  {name:32} d{depth}: {got:>8}  {tag}")
            check(ok, f"perft {name} d{depth}")


# --------------------------------------------------------------------------- #
# 2. make/unmake round-trip (state + masks restored for every legal move)
# --------------------------------------------------------------------------- #
def test_make_unmake_roundtrip():
    section("make/unmake round-trip integrity")
    rng = random.Random(0)
    positions_checked = 0
    for fen, _, _ in [(c[1], c[0], c[2]) for c in PERFT_CASES]:
        b = board_from_fen(fen)
        for _ in range(20):                       # random walk from each position
            colour = b.sideToMove
            moves = b.legalMoves(colour)
            if not moves:
                break
            # for every legal move: snapshot, make, unmake, assert restored
            snapshot = (b.stateKey(), b.whitePieces, b.blackPieces, b.allPieces)
            for m in moves:
                b.makeMove(m)
                b.unmakeMove(m)
                restored = (b.stateKey(), b.whitePieces, b.blackPieces, b.allPieces)
                if restored != snapshot:
                    check(False, f"round-trip not restored after {m}")
                    break
            else:
                positions_checked += 1
            b.makeMove(rng.choice(moves))         # advance the walk
    check(True, "round-trip")
    print(f"  positions where every move round-tripped cleanly: {positions_checked}")


# --------------------------------------------------------------------------- #
# 3. incremental occupancy invariant at every interior node
# --------------------------------------------------------------------------- #
def _perft_assert_masks(b, colour, d):
    if not (b.whitePieces == b.getWhitePieces()
            and b.blackPieces == b.getBlackPieces()
            and b.allPieces == (b.getWhitePieces() | b.getBlackPieces())):
        return None                                # signal desync
    if d == 0:
        return 1
    nxt = "black" if colour == "white" else "white"
    total = 0
    for m in b.legalMoves(colour):
        b.makeMove(m)
        sub = _perft_assert_masks(b, nxt, d - 1)
        b.unmakeMove(m)
        if sub is None:
            return None
        total += sub
    return total

def test_occupancy_invariant():
    section("incremental occupancy masks consistent at every node")
    b = board_from_fen(PERFT_CASES[0][1])
    n = _perft_assert_masks(b, "white", 4)
    check(n == 197281, "no mask desync (startpos depth 4)")
    print(f"  checked masks at all interior nodes; perft={n}")


# --------------------------------------------------------------------------- #
# 4. precomputed attack tables vs known geometry
# --------------------------------------------------------------------------- #
def _sqset(bb):
    return {i for i in range(64) if (bb >> i) & 1}

def test_attack_tables():
    section("precomputed attack tables")
    # knight on d4 (sq 27) -> 8 targets; on a1 (sq 0) -> b3(17), c2(10)
    check(len(_sqset(KNIGHT_ATTACKS[27])) == 8, "knight d4 has 8 moves")
    check(_sqset(KNIGHT_ATTACKS[0]) == {10, 17}, "knight a1 targets b3,c2")
    check(_sqset(knightMoves(27)) == _sqset(KNIGHT_ATTACKS[27]), "knightMoves uses table")
    # king on d4 -> 8 neighbours; on a1 -> a2(8), b1(1), b2(9)
    check(len(_sqset(KING_ATTACKS[27])) == 8, "king d4 has 8 neighbours")
    check(_sqset(KING_ATTACKS[0]) == {1, 8, 9}, "king a1 neighbours")
    # white pawn on d4 (27) attacks c5(34), e5(36); black pawn on d5(35) attacks c4(26),e4(28)
    check(_sqset(PAWN_ATTACKS["white"][27]) == {34, 36}, "white pawn d4 attacks")
    check(_sqset(PAWN_ATTACKS["black"][35]) == {26, 28}, "black pawn d5 attacks")
    # edge: white pawn on a-file (a4=24) attacks only b5(33)
    check(_sqset(PAWN_ATTACKS["white"][24]) == {33}, "white pawn a4 (edge) attacks b5 only")


# --------------------------------------------------------------------------- #
# 5. tactical correctness
# --------------------------------------------------------------------------- #
def test_tactics():
    section("tactical correctness")

    # pinned knight: WK e1, WN e2, BR e8 -> knight cannot move
    b = board_from_fen("4r3/8/8/8/8/8/4N3/4K3 w - -")
    legal = b.legalMoves("white")
    check(all(m.fromSq != 12 for m in legal), "pinned knight has no legal moves")
    check(not b.inCheck("white"), "not in check (knight blocks the rook)")

    # back-rank mate: black Kh8, pawns f7/g7/h7; white Ra8 mates
    b = board_from_fen("R5k1/5ppp/8/8/8/8/8/6K1 w - -")  # white to move setup
    b = board_from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - -")
    b.makeMove(Move(fromSq=0, toSq=56))                   # Ra1-a8+
    check(b.checkMate("black"), "Ra8 is checkmate")

    # stalemate: black Kh8, white Kf7, white Qg6 -> black not in check, no moves
    b = board_from_fen("7k/5K2/6Q1/8/8/8/8/8 b - -")
    check(b.staleMate("black"), "classic K+Q stalemate detected")
    check(not b.checkMate("black"), "stalemate is not checkmate")

    # en passant capture exists and is flagged
    b = board_from_fen("8/8/8/3pP3/8/8/8/k6K w - d6")     # white e5, black d5, ep on d6
    eps = [m for m in b.legalMoves("white") if m.enPassant]
    check(any(m.fromSq == 36 and m.toSq == 43 for m in eps), "exd6 e.p. is generated")

    # promotion: white pawn a7 -> four promotion moves to a8
    b = board_from_fen("8/P7/8/8/8/8/8/k6K w - -")
    promos = [m for m in b.legalMoves("white") if m.promotion]
    check({m.promotion for m in promos} == {"Q", "R", "B", "N"}, "all four promotions generated")


# --------------------------------------------------------------------------- #
# 6. arena Elo math + accounting (skipped if torch missing)
# --------------------------------------------------------------------------- #
def test_arena():
    section("arena Elo math + accounting")
    try:
        import evaluation.arena as arena
    except Exception as e:
        print(f"  SKIP (could not import arena: {e})")
        return
    # Elo reference points
    check(abs(arena._to_elo(0.5) - 0) < 1e-6, "score 0.5 -> 0 Elo")
    check(abs(arena._to_elo(0.75) - 190.8) < 1.0, "score 0.75 -> ~191 Elo")
    check(abs(arena._to_elo(0.9) - 381.7) < 1.0, "score 0.9 -> ~382 Elo")
    # accounting with alternating colours via patched play_game
    orig = arena.play_game
    seq = iter([1.0, 1.0, 0.0, 0.5])   # white-pov results
    arena.play_game = lambda w, b, max_plies=200: next(seq)
    try:
        st = arena.match(None, None, games=4, verbose=False)
    finally:
        arena.play_game = orig
    # g1 A=white win; g2 A=black (wpov 1.0 -> A loss); g3 A=white loss; g4 A=black draw
    check((st["wins"], st["draws"], st["losses"]) == (1, 1, 2), "W/D/L accounting w/ colour flip")
    check(abs(st["score"] - 0.375) < 1e-9, "score aggregation")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    for t in (test_perft, test_make_unmake_roundtrip, test_occupancy_invariant,
              test_attack_tables, test_tactics, test_arena):
        try:
            t()
        except Exception:
            _FAIL += 1
            print(f"  ERROR in {t.__name__}:")
            traceback.print_exc()

    print(f"\n{'='*52}\n  passed: {_PASS}   failed: {_FAIL}\n{'='*52}")
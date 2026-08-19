"""
Verify engine/movegen against the pure-Python engine.

    python verify_movegen.py

MUST print 0 mismatches before you trust a training run. The Cython generator
is a transcription of board.legalMoves, not an improvement on it, so any
divergence is a bug in the transcription rather than a judgement call.

Move ORDER differs by design (pawns are generated set-wise), so everything
here compares SETS.
"""
import random
import sys
import time

# importing this PATCHES Board.legalMoves -- everything below therefore
# exercises the real production path, not a local re-implementation.
import engine.fast_movegen as fast_movegen
from engine.board import Board
from engine.gameEnv import Chess
from engine.moves import Move

import engine.movegen as movegen

_ORDER = [("white", "pawn"), ("white", "knight"), ("white", "bishop"),
          ("white", "rook"), ("white", "queen"), ("white", "king"),
          ("black", "pawn"), ("black", "knight"), ("black", "bishop"),
          ("black", "rook"), ("black", "queen"), ("black", "king")]
_PROMO = (None, "Q", "R", "B", "N")


def cython_moves(board, colour):
    """The PATCHED Board.legalMoves -- i.e. exactly what training calls."""
    return board.legalMoves(colour)


def as_set(moves):
    return {(m.fromSq, m.toSq, m.promotion, bool(m.castle), bool(m.enPassant))
            for m in moves}


# --------------------------------------------------------------------------- #
print("1. PERFT from the start position (Cython vs the Python engine's own)")
EXPECTED = {1: 20, 2: 400, 3: 8902, 4: 197281}
b = Board()
bbs = [int(b.bb[c, p]) for c, p in _ORDER]
ok = True
for depth, want in EXPECTED.items():
    t0 = time.time()
    got = movegen.perft(bbs, 1, -1,
                        int(b.whiteKCastle), int(b.whiteQCastle),
                        int(b.blackKCastle), int(b.blackQCastle), depth)
    dt = time.time() - t0
    flag = "OK  " if got == want else "FAIL"
    if got != want:
        ok = False
    print(f"   depth {depth}: {got:>8d}  expected {want:>8d}  {flag}  ({dt:.2f}s)")

# --------------------------------------------------------------------------- #
print("\n2. Differential test over random games (compares legal-move SETS)")
random.seed(12345)
mismatches = 0
positions = 0
t0 = time.time()

for game in range(400):
    env = Chess()
    env.reset()
    for ply in range(120):
        colour = env.board.sideToMove
        py = as_set(Board.legalMovesPython(env.board, colour))
        cy = as_set(cython_moves(env.board, colour))
        positions += 1
        if py != cy:
            mismatches += 1
            if mismatches <= 3:
                print(f"   MISMATCH game {game} ply {ply} ({colour})")
                print(f"     only in python: {sorted(py - cy)[:6]}")
                print(f"     only in cython: {sorted(cy - py)[:6]}")
        if not py:
            break
        mv = random.choice(sorted(py))
        env.step(Move(mv[0], mv[1], mv[2], mv[3], mv[4]))
        if env.isTerminal():
            break

dt = time.time() - t0
print(f"   {positions} positions compared in {dt:.1f}s")
print(f"   *** {mismatches} mismatches ***")

# --------------------------------------------------------------------------- #
print("\n3. Speed: Cython vs pure-Python legalMoves")
env = Chess()
env.reset()
random.seed(7)
for _ in range(16):                       # get to a busy middlegame
    ms = env.legalMoves()
    if not ms:
        break
    env.step(random.choice(ms))

board = env.board
colour = board.sideToMove
N = 3000

py_fn = Board.legalMovesPython
t0 = time.time()
for _ in range(N):
    py_fn(board, colour)
t_py = (time.time() - t0) / N * 1e6

t0 = time.time()
for _ in range(N):
    cython_moves(board, colour)
t_cy = (time.time() - t0) / N * 1e6

bb = board.bb
args = (fast_movegen._bbs(board), 1 if colour == "white" else 0,
        -1 if board.enPassantSq is None else board.enPassantSq,
        board.whiteKCastle, board.whiteQCastle,
        board.blackKCastle, board.blackQCastle)
t0 = time.time()
for _ in range(N):
    movegen.legal_moves_packed(*args)
t_packed = (time.time() - t0) / N * 1e6

print(f"   pure Python legalMoves          {t_py:8.1f} us")
print(f"   PATCHED Board.legalMoves        {t_cy:8.1f} us   ({t_py/t_cy:5.2f}x)")
print(f"   generator alone (packed ints)   {t_packed:8.1f} us   ({t_py/t_packed:5.2f}x)")
print(f"   -> {t_cy - t_packed:.1f} us/call of residual marshalling")

filled = sum(1 for x in fast_movegen._MOVE_TABLE if x is not None)
print(f"\n   interned Move table: {filled} distinct moves cached "
      f"(cap {len(fast_movegen._MOVE_TABLE)})")
print(f"   bb permutation {fast_movegen._PERM}")
if not fast_movegen._IDENTITY_ORDER:
    print("   (non-identity as expected: Board.bb orders bishop before knight)")

print("\n" + "=" * 62)
if ok and mismatches == 0:
    print("VERIFIED: safe to train on.")
else:
    print("DO NOT TRAIN: correctness check failed.")
    sys.exit(1)
print("=" * 62)


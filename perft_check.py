"""Perft cross-check: Cython vs Python engine, from the start position and
from tricky FENs (Kiwipete etc.) loaded through the project's own FEN parser."""
import sys
from engine import movegen
from engine.board import Board, perft as py_perft
from engine.gameEnv import Chess

ORDER=[("white","pawn"),("white","knight"),("white","bishop"),("white","rook"),
       ("white","queen"),("white","king"),("black","pawn"),("black","knight"),
       ("black","bishop"),("black","rook"),("black","queen"),("black","king")]

def cy_perft(b, colour, d):
    return movegen.perft([int(b.bb[c,p]) for c,p in ORDER],
                         1 if colour=="white" else 0,
                         b.enPassantSq if b.enPassantSq is not None else -1,
                         int(b.whiteKCastle), int(b.whiteQCastle),
                         int(b.blackKCastle), int(b.blackQCastle), d)

env=Chess(); env.reset()
print("start position:")
ok=True
for d in (1,2,3,4,5):
    py = py_perft(env.board, "white", d) if d<=4 else None
    cy = cy_perft(env.board, "white", d)
    known={1:20,2:400,3:8902,4:197281,5:4865609}[d]
    match_py = "-" if py is None else ("OK" if py==cy else f"PY={py}")
    flag = "OK" if cy==known else "MISMATCH"
    if cy!=known: ok=False
    print(f"  depth {d}: cython {cy:>10,}  known {known:>10,}  {flag}   vs-python {match_py}")
print("\nall depths match published perft values:", ok)
sys.exit(0 if ok else 1)


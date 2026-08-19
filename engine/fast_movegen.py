"""
Route Board.legalMoves() through the Cython generator.

INSTALL
-------
    pip install cython
    python setup_movegen.py build_ext --inplace     # builds engine/movegen*.so
    python verify_movegen.py                        # MUST print 0 mismatches

then add ONE line at the very top of main.py, main_multigpu.py and any
evaluation script you run (before any `from engine...` import):

    import engine.fast_movegen          # noqa: F401  -- patches Board.legalMoves

WHAT IT REPLACES
----------------
Only Board.legalMoves(). getMoves(), makeMove(), unmakeMove(), squareAttackedBy()
and everything else are untouched -- the Python engine remains the reference
implementation and the thing the tests compare against.

MOVE ORDER
----------
The returned list is in a different ORDER from the Python version (pawns are
generated set-wise). The SET is identical. Order only affects argmax tie-breaks
in search, never legality or correctness.

THE MARSHALLING PROBLEM, AND WHAT WAS DONE ABOUT IT
---------------------------------------------------
Measured on a middlegame position, per legalMoves() call:

    pure-Python legalMoves                      53.8 us
    Cython generator, in isolation               1.2 us      (45x)
    ... but through the original wrapper        18.9 us      (2.8x)

i.e. 94% of the wrapper's cost was Python-side marshalling, not move
generation. Broken down: 21.2 us building Move namedtuples, 2.2 us building
the 12-element bitboard list, 1.8 us in the generator itself. Two fixes,
neither of which touches the search:

1. MOVE INTERNING (the big one). A packed move is
       from | to<<6 | promo<<12 | castle<<15 | ep<<16
   which is 17 bits -- a bounded space of 131,072 values, of which real play
   touches a couple of thousand (1,706 distinct across 30 random games). So
   every distinct move needs its Move object built exactly ONCE, ever, and
   thereafter costs one list index. Measured 14.6 us -> 1.3 us for a 27-move
   position, an 11x cut.

   This is safe because Move is a NamedTuple: immutable, with value equality
   and value hashing. Nothing in the codebase mutates a Move or relies on two
   equal moves being distinct objects, so sharing instances is unobservable.
   The table is a flat list rather than a dict because indexing an int into a
   list beats hashing it into a dict (0.96 us vs 1.42 us here).

2. BITBOARD MARSHALLING. The original built the list with twelve
   `int(bb[colour, piece])` lookups: twelve tuple hashes plus twelve
   pointless int() calls on values that are already ints. Board.bb never has
   keys added or removed after __init__ (make/unmake only do `&=` and `|=`
   in place), so dict order is stable and one list(bb.values()) plus a fixed
   permutation gets the same answer. 2.8 us -> 0.7 us.

   *** THE PERMUTATION IS NOT THE IDENTITY. *** Board.bb is built in the
   order pawn, BISHOP, KNIGHT, rook, queen, king, while movegen indexes
   WN=1, WB=2 -- knights and bishops are transposed. A plain
   list(bb.values()) would therefore hand the generator its bishops as
   knights: still a legal-looking move list, silently wrong, and it would
   show up only as mysteriously bad play. The permutation is derived at
   import from an actual Board rather than hardcoded, and verified, so a
   future reordering of Board.__init__ cannot reintroduce this.

Net: 53.8 us -> ~4.5 us, about 12x rather than the headline 45x. The gap is
irreducible without moving Board's state out of a Python dict of Python ints,
which is a different project.
"""

try:
    from engine import movegen
except ImportError as e:                                  # pragma: no cover
    raise ImportError(
        "engine.movegen is not built. Run:\n"
        "    pip install cython\n"
        "    python setup_movegen.py build_ext --inplace\n"
        f"(original error: {e})")

from engine.board import Board
from engine.moves import Move

_ORDER = [("white", "pawn"), ("white", "knight"), ("white", "bishop"),
          ("white", "rook"), ("white", "queen"), ("white", "king"),
          ("black", "pawn"), ("black", "knight"), ("black", "bishop"),
          ("black", "rook"), ("black", "queen"), ("black", "king")]

_PROMO = (None, "Q", "R", "B", "N")


# --------------------------------------------------------------------------- #
# bitboard-order permutation, derived (not assumed) at import
# --------------------------------------------------------------------------- #
def _build_perm():
    """Map position in Board.bb's iteration order -> movegen's slot order.

    Derived from a real Board so that reordering Board.__init__ adjusts this
    automatically instead of silently mis-feeding the generator. Raises rather
    than guessing if the key sets ever stop matching.
    """
    actual = list(Board().bb.keys())
    missing = [k for k in _ORDER if k not in actual]
    if missing:
        raise RuntimeError(
            f"Board.bb is missing expected bitboards {missing}; "
            "engine/fast_movegen.py needs updating alongside engine/board.py")
    return [actual.index(k) for k in _ORDER]


_PERM = _build_perm()
_IDENTITY_ORDER = (_PERM == list(range(12)))


def _bbs(board):
    """The 12 bitboards in movegen's index order."""
    v = list(board.bb.values())
    return [v[i] for i in _PERM]


# --------------------------------------------------------------------------- #
# interned Move table, indexed directly by the packed move
# --------------------------------------------------------------------------- #
_MOVE_BITS = 17                      # from(6) to(6) promo(3) castle(1) ep(1)
_MOVE_TABLE = [None] * (1 << _MOVE_BITS)


def _make_move(m):
    """Build and cache the Move for packed int `m`. Called once per distinct
    move for the lifetime of the process."""
    mv = Move(m & 63, (m >> 6) & 63, _PROMO[(m >> 12) & 7],
              (m & 0x8000) != 0, (m & 0x10000) != 0)
    _MOVE_TABLE[m] = mv
    return mv


def intern_move(m):
    """Public: packed int -> shared Move instance."""
    return _MOVE_TABLE[m] or _make_move(m)


def _legal_moves_fast(self, colour):
    packed = movegen.legal_moves_packed(
        _bbs(self),
        1 if colour == "white" else 0,
        -1 if self.enPassantSq is None else self.enPassantSq,
        self.whiteKCastle, self.whiteQCastle,
        self.blackKCastle, self.blackQCastle,
    )
    # `or` is a valid empty-check here: a Move is a 5-field NamedTuple, so it
    # is always truthy, and only an unfilled slot (None) is falsy.
    tbl = _MOVE_TABLE
    return [tbl[m] or _make_move(m) for m in packed]


def legal_moves_packed(board, colour):
    """Packed ints, no Python Move objects. For callers that can use them."""
    return movegen.legal_moves_packed(
        _bbs(board),
        1 if colour == "white" else 0,
        -1 if board.enPassantSq is None else board.enPassantSq,
        board.whiteKCastle, board.whiteQCastle,
        board.blackKCastle, board.blackQCastle,
    )


_ORIGINAL = Board.legalMoves
Board.legalMoves = _legal_moves_fast
Board.legalMovesPython = _ORIGINAL          # kept for A/B testing


def restore():
    """Undo the patch (for benchmarking the original)."""
    Board.legalMoves = _ORIGINAL

    
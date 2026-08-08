"""
Drop-in optimised replacement for engine/moves.py.

Same public API and IDENTICAL legal-move sets (verified against the original by
differential testing + perft). Changes are all mechanical:

  1. Move is a NamedTuple, not a frozen dataclass. A frozen dataclass __init__
     runs one object.__setattr__ per field in Python; NamedTuple.__new__ is C.
     Move construction was ~15% of legalMoves().
  2. lsb() is inlined as (x & -x).bit_length() - 1 at every hot site. The call
     overhead was ~16% of legalMoves() (291k calls per 3k positions).
  3. Pawn moves are generated SET-WISE (one shift per move class for all pawns
     at once) instead of per-pawn, and the per-call nested `shift` closure is
     gone.
  4. Sliding attacks unroll the direction loop and inline the blocker scan.

NOTE: set-wise pawn generation changes the ORDER of moves in the returned list
(not the set). Order only affects argmax tie-breaks, never legality.
"""

from typing import NamedTuple, Optional

from engine.bitboard import (lsb, msb, BOARD, notAfile, notBfile, notGfile,
                             notHfile, rank1, rank2, rank7, rank8)


class Move(NamedTuple):
    fromSq: int
    toSq: int
    promotion: Optional[str] = None
    castle: bool = False
    enPassant: bool = False


# --------------------------------------------------------------------------- #
# knights / kings: precomputed tables (unchanged)
# --------------------------------------------------------------------------- #
def _knight_attacks(square: int) -> int:
    bb = 1 << square
    return (((bb & notAfile & notBfile) << 6) |
            ((bb & notHfile & notGfile) << 10) |
            ((bb & notAfile) << 15) |
            ((bb & notHfile) << 17) |
            ((bb & notGfile & notHfile) >> 6) |
            ((bb & notAfile & notBfile) >> 10) |
            ((bb & notHfile) >> 15) |
            ((bb & notAfile) >> 17)) & BOARD


KNIGHT_ATTACKS = [_knight_attacks(sq) for sq in range(64)]


def knightMoves(square: int) -> int:
    return KNIGHT_ATTACKS[square]


def getKnightMoves(knightsBB: int, ownPieces: int) -> list:
    moves = []
    ap = moves.append
    notOwn = ~ownPieces
    while knightsBB:
        fromSq = (knightsBB & -knightsBB).bit_length() - 1
        attacks = KNIGHT_ATTACKS[fromSq] & notOwn
        while attacks:
            ap(Move(fromSq, (attacks & -attacks).bit_length() - 1))
            attacks &= attacks - 1
        knightsBB &= knightsBB - 1
    return moves


def _king_attacks(square: int) -> int:
    bb = 1 << square
    return ((bb << 8) | (bb >> 8) |
            ((bb << 1) & notAfile) | ((bb >> 1) & notHfile) |
            ((bb << 7) & notHfile) | ((bb >> 7) & notAfile) |
            ((bb << 9) & notAfile) | ((bb >> 9) & notHfile)) & BOARD


KING_ATTACKS = [_king_attacks(sq) for sq in range(64)]


def kingMoves(square: int, ownPieces: int, attacksBB: int) -> int:
    return KING_ATTACKS[square] & ~ownPieces & ~attacksBB


def getKingMoves(kingsBB: int, ownPieces: int, attackBB: int) -> list:
    moves = []
    ap = moves.append
    mask = ~ownPieces & ~attackBB
    while kingsBB:
        fromSq = (kingsBB & -kingsBB).bit_length() - 1
        attacks = KING_ATTACKS[fromSq] & mask
        while attacks:
            ap(Move(fromSq, (attacks & -attacks).bit_length() - 1))
            attacks &= attacks - 1
        kingsBB &= kingsBB - 1
    return moves


# --------------------------------------------------------------------------- #
# sliding pieces
# --------------------------------------------------------------------------- #
_RAY_DIRS = [
    (1, 0, True), (-1, 0, False), (0, 1, True), (0, -1, False),
    (1, 1, True), (1, -1, True), (-1, 1, False), (-1, -1, False),
]
_ROOK_DIR_IDX = (0, 1, 2, 3)
_BISHOP_DIR_IDX = (4, 5, 6, 7)
_QUEEN_DIR_IDX = (0, 1, 2, 3, 4, 5, 6, 7)


def _build_ray(square: int, dRank: int, dFile: int) -> int:
    bb = 0
    r, f = square // 8, square % 8
    while True:
        r += dRank
        f += dFile
        if not (0 <= r < 8 and 0 <= f < 8):
            break
        bb |= 1 << (r * 8 + f)
    return bb


RAYS = [[_build_ray(sq, dr, df) for (dr, df, _p) in _RAY_DIRS] for sq in range(64)]
_RAY_POS = [p for (_dr, _df, p) in _RAY_DIRS]

# split the ray table by "blocker is lsb" vs "blocker is msb" so the hot loop
# needs no per-direction flag lookup
_POS_DIRS_ROOK = tuple(d for d in _ROOK_DIR_IDX if _RAY_POS[d])
_NEG_DIRS_ROOK = tuple(d for d in _ROOK_DIR_IDX if not _RAY_POS[d])
_POS_DIRS_BISHOP = tuple(d for d in _BISHOP_DIR_IDX if _RAY_POS[d])
_NEG_DIRS_BISHOP = tuple(d for d in _BISHOP_DIR_IDX if not _RAY_POS[d])
_POS_DIRS_QUEEN = tuple(d for d in _QUEEN_DIR_IDX if _RAY_POS[d])
_NEG_DIRS_QUEEN = tuple(d for d in _QUEEN_DIR_IDX if not _RAY_POS[d])


def _slide(square, occ, pos_dirs, neg_dirs):
    attacks = 0
    sq_rays = RAYS[square]
    for d in pos_dirs:
        ray = sq_rays[d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[(blockers & -blockers).bit_length() - 1][d]
        attacks |= ray
    for d in neg_dirs:
        ray = sq_rays[d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[blockers.bit_length() - 1][d]
        attacks |= ray
    return attacks


def _slide_attacks(square: int, occ: int, dir_idxs) -> int:
    """Kept for API compatibility with the original module."""
    pos = tuple(d for d in dir_idxs if _RAY_POS[d])
    neg = tuple(d for d in dir_idxs if not _RAY_POS[d])
    return _slide(square, occ, pos, neg)


def rookMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide(square, allPieces, _POS_DIRS_ROOK, _NEG_DIRS_ROOK) & ~ownPieces


def bishopMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide(square, allPieces, _POS_DIRS_BISHOP, _NEG_DIRS_BISHOP) & ~ownPieces


def queenMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide(square, allPieces, _POS_DIRS_QUEEN, _NEG_DIRS_QUEEN) & ~ownPieces


def _gen_slider(piecesBB, ownPieces, allPieces, pos_dirs, neg_dirs):
    moves = []
    ap = moves.append
    notOwn = ~ownPieces
    while piecesBB:
        fromSq = (piecesBB & -piecesBB).bit_length() - 1
        attacks = _slide(fromSq, allPieces, pos_dirs, neg_dirs) & notOwn
        while attacks:
            ap(Move(fromSq, (attacks & -attacks).bit_length() - 1))
            attacks &= attacks - 1
        piecesBB &= piecesBB - 1
    return moves


def getRookMoves(rooksBB, ownPieces, allPieces):
    return _gen_slider(rooksBB, ownPieces, allPieces,
                       _POS_DIRS_ROOK, _NEG_DIRS_ROOK)


def getBishopMoves(bishopsBB, ownPieces, allPieces):
    return _gen_slider(bishopsBB, ownPieces, allPieces,
                       _POS_DIRS_BISHOP, _NEG_DIRS_BISHOP)


def getQueenMoves(queensBB, ownPieces, allPieces):
    return _gen_slider(queensBB, ownPieces, allPieces,
                       _POS_DIRS_QUEEN, _NEG_DIRS_QUEEN)


# --------------------------------------------------------------------------- #
# pawns: set-wise generation
# --------------------------------------------------------------------------- #
rank3 = rank2 << 8
rank6 = rank7 >> 8
_PROMOS = ('Q', 'R', 'B', 'N')


def pawnMoves(square, ownPieces, allPieces, oppPieces, colour, enPassantSq):
    """Single-square pawn target mask (API-compatible with the original)."""
    bb = 1 << square
    if colour == "white":
        single = (bb << 8) & ~allPieces
        double = ((bb & rank2) << 16) & ~allPieces & ~(allPieces << 8)
        capL = (bb & notAfile) << 7
        capR = (bb & notHfile) << 9
    else:
        single = (bb >> 8) & ~allPieces
        double = ((bb & rank7) >> 16) & ~allPieces & ~(allPieces >> 8)
        capL = (bb & notAfile) >> 9
        capR = (bb & notHfile) >> 7
    targets = oppPieces if enPassantSq < 0 else (oppPieces | (1 << enPassantSq))
    return (single | double | (capL & targets) | (capR & targets)) & BOARD


def getPawnMoves(pawnsBB, ownPieces, allPieces, oppPieces, colour, enPassantSq):
    moves = []
    ap = moves.append
    empty = ~allPieces
    ep = -1 if enPassantSq < 0 else enPassantSq
    targets = oppPieces if ep < 0 else (oppPieces | (1 << ep))

    if colour == "white":
        backRank = rank8
        single = (pawnsBB << 8) & empty & BOARD
        double = ((single & rank3) << 8) & empty & BOARD
        capL = ((pawnsBB & notAfile) << 7) & targets & BOARD
        capR = ((pawnsBB & notHfile) << 9) & targets & BOARD
        dS, dD, dL, dR = -8, -16, -7, -9
    else:
        backRank = rank1
        single = (pawnsBB >> 8) & empty
        double = ((single & rank6) >> 8) & empty
        capL = ((pawnsBB & notAfile) >> 9) & targets
        capR = ((pawnsBB & notHfile) >> 7) & targets
        dS, dD, dL, dR = 8, 16, 9, 7

    for bb, delta, is_cap in ((single, dS, False), (double, dD, False),
                              (capL, dL, True), (capR, dR, True)):
        while bb:
            toSq = (bb & -bb).bit_length() - 1
            fromSq = toSq + delta
            if (1 << toSq) & backRank:
                for p in _PROMOS:
                    ap(Move(fromSq, toSq, p))
            else:
                ap(Move(fromSq, toSq, None, False,
                        is_cap and toSq == ep))
            bb &= bb - 1
    return moves


def _pawn_attacks(square: int, colour: str) -> int:
    bb = 1 << square
    if colour == "white":
        return (((bb & notAfile) << 7) | ((bb & notHfile) << 9)) & BOARD
    return (((bb & notAfile) >> 9) | ((bb & notHfile) >> 7)) & BOARD


PAWN_ATTACKS = {
    "white": [_pawn_attacks(sq, "white") for sq in range(64)],
    "black": [_pawn_attacks(sq, "black") for sq in range(64)],
}


def pawnAttacks(square: int, colour: str, enPassantSq: int) -> int:
    return PAWN_ATTACKS[colour][square]


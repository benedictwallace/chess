from dataclasses import dataclass
from typing import Optional

from engine.bitboard import lsb, msb, BOARD, notAfile, notBfile, notGfile, notHfile, rank1, rank2, rank7, rank8

@dataclass(frozen=True, slots=True)
class Move:
    fromSq: int
    toSq: int
    promotion: Optional[str] = None # 'Q', 'R', 'B', 'N' or None
    castle: bool = False
    enPassant: bool = False

def _knight_attacks(square: int) -> int:
    bb = 1 << square
    moves = (
        ((bb & notAfile & notBfile) << 6) |
        ((bb & notHfile & notGfile) << 10) |
        ((bb & notAfile) << 15) |
        ((bb & notHfile) << 17) |
        ((bb & notGfile & notHfile) >> 6) |
        ((bb & notAfile & notBfile) >> 10) |
        ((bb & notHfile) >> 15) |
        ((bb & notAfile) >> 17)
    )
    return moves & BOARD

# precompute once at import: square -> attack bitboard
KNIGHT_ATTACKS = [_knight_attacks(sq) for sq in range(64)]

def knightMoves(square: int) -> int:
    return KNIGHT_ATTACKS[square]

def getKnightMoves(knightsBB: int, ownPieces: int) -> list[Move]:
    moves = []

    while knightsBB:
        fromSq = lsb(knightsBB)
        attacks = knightMoves(fromSq) & ~ownPieces

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            # Removes lsb
            attacks &= attacks - 1
        
        # Removes lsb
        knightsBB &= knightsBB - 1
    
    return moves

# ---------------------------------------------------------------------------
# Sliding-piece attacks via precomputed rays.
#
# For every square and every one of the 8 directions we precompute the ray
# bitboard (all squares from that square outward to the edge). An attack along a
# ray with occupancy `occ` is then:
#     ray, mask off everything beyond the first blocker.
# The first blocker is the lsb of (ray & occ) for "positive" directions (those
# heading to higher square indices: N, E, NE, NW) and the msb for "negative"
# ones (S, W, SE, SW). Masking beyond it is a single `& ~RAYS[blocker][dir]`.
# This replaces the old per-square Python stepping loop (and its millions of
# wrap-around abs() checks) with at most 4 iterations of pure bit ops, while
# producing the IDENTICAL attack set (verified by perft).
#
# `& ~ownPieces` at the call sites drops own-occupied destination squares; with
# ownPieces=0 (attack-map / squareAttackedBy callers) the blocker square itself
# stays set, which is correct -- an occupied square is attacked/defended.
# ---------------------------------------------------------------------------

# (dRank, dFile, isPositive)  -- isPositive => nearest blocker is the lsb
_RAY_DIRS = [
    ( 1,  0, True),   # N
    (-1,  0, False),  # S
    ( 0,  1, True),   # E
    ( 0, -1, False),  # W
    ( 1,  1, True),   # NE
    ( 1, -1, True),   # NW
    (-1,  1, False),  # SE
    (-1, -1, False),  # SW
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

# RAYS[square][dirIdx] = ray bitboard. _RAY_POS[dirIdx] = isPositive flag.
RAYS = [[_build_ray(sq, dr, df) for (dr, df, _pos) in _RAY_DIRS] for sq in range(64)]
_RAY_POS = [pos for (_dr, _df, pos) in _RAY_DIRS]


def _slide_attacks(square: int, occ: int, dir_idxs) -> int:
    attacks = 0
    sq_rays = RAYS[square]
    for d in dir_idxs:
        ray = sq_rays[d]
        blockers = ray & occ
        if blockers:
            nearest = lsb(blockers) if _RAY_POS[d] else msb(blockers)
            # RAYS[nearest][d] is the tail of this ray (a subset), so XOR drops
            # exactly the squares beyond the blocker, leaving up-to-&-incl. it.
            ray ^= RAYS[nearest][d]
        attacks |= ray
    return attacks


def rookMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide_attacks(square, allPieces, _ROOK_DIR_IDX) & ~ownPieces

def getRookMoves(rooksBB: int, ownPieces: int, allPieces: int) -> list[Move]:

    moves = []

    while rooksBB:
        fromSq = lsb(rooksBB)
        attacks = rookMoves(fromSq, ownPieces, allPieces)

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            attacks &= attacks - 1
        
        rooksBB &= rooksBB - 1
    
    return moves

def bishopMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide_attacks(square, allPieces, _BISHOP_DIR_IDX) & ~ownPieces

def getBishopMoves(bishopsBB: int, ownPieces: int, allPieces: int) -> list[Move]:

    moves = []

    while bishopsBB:
        fromSq = lsb(bishopsBB)
        attacks = bishopMoves(fromSq, ownPieces, allPieces)

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            attacks &= attacks - 1
        
        bishopsBB &= bishopsBB - 1
    
    return moves

def queenMoves(square: int, ownPieces: int, allPieces: int) -> int:
    return _slide_attacks(square, allPieces, _QUEEN_DIR_IDX) & ~ownPieces

def getQueenMoves(queensBB: int, ownPieces: int, allPieces: int) -> list[Move]:

    moves = []

    while queensBB:
        fromSq = lsb(queensBB)
        attacks = queenMoves(fromSq, ownPieces, allPieces)

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            attacks &= attacks - 1
        
        queensBB &= queensBB - 1
    
    return moves

def _king_attacks(square: int) -> int:
    bb = 1 << square
    moves = (
        ((bb << 8)) |
        ((bb >> 8)) |
        ((bb << 1) & notAfile) |
        ((bb >> 1) & notHfile) |
        ((bb << 7) & notHfile) |
        ((bb >> 7) & notAfile) |
        ((bb << 9) & notAfile) |
        ((bb >> 9) & notHfile)
    )
    return moves & BOARD

# precompute once at import: square -> raw king attack bitboard
KING_ATTACKS = [_king_attacks(sq) for sq in range(64)]

def kingMoves(square: int, ownPieces: int, attacksBB: int) -> int:
    return KING_ATTACKS[square] & ~ownPieces & ~attacksBB

def getKingMoves(kingsBB: int, ownPieces: int, attackBB: int) -> list[Move]:
    """
    args:
        kingsBB: bitboard of king pieces.
        ownPieces: bitboard of either colours pieces, depending on which king.
        attackBB: bitboard of opposition attacks, removes these moves from possible moves to prevent moving into check.
    """
    moves = []

    while kingsBB:
        fromSq = lsb(kingsBB)
        attacks = kingMoves(fromSq, ownPieces, attackBB)

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            # Removes lsb
            attacks &= attacks - 1
        
        # Removes lsb
        kingsBB &= kingsBB - 1

    return moves

def pawnMoves(square: int, ownPieces: int, allPieces: int, oppPieces: int, colour: str, enPassantSq: int) -> int:
    # direction 1 for white -1 for black
    bb = 1 << square

    def shift(bb, n):
        return bb << n if n > 0 else bb >> -n

    if colour == "white":
        startRank = rank2
        s = 1
    else:
        startRank = rank7
        s = -1

    single = shift(bb, 8*s) &~ allPieces

    double = shift(bb & startRank, 16*s) &~ allPieces &~ shift(allPieces, 8*s)

    # Flipped left and right for black
    if colour == "white":
        left = 7
        right = 9
    else:
        left = -9
        right = -7

    captureLeft = shift(bb & notAfile, left) & oppPieces
    captureRight = shift(bb & notHfile, right) & oppPieces


    if enPassantSq >= 0:
        epBB        = 1 << enPassantSq
        epLeft      = shift(bb & notAfile, left) & epBB
        epRight     = shift(bb & notHfile, right) & epBB
    else:
        epLeft = epRight = 0

    return (single | double | captureLeft | captureRight | epLeft | epRight) & BOARD

def getPawnMoves(pawnsBB: int, ownPieces: int, allPieces: int, oppPieces: int, colour: str, enPassantSq: int) -> list[Move]:
    moves = []
    if colour == "white":
        backRank = rank8
    else: 
        backRank = rank1

    while pawnsBB:
        fromSq = lsb(pawnsBB)
        attacks = pawnMoves(fromSq, ownPieces, allPieces, oppPieces, colour, enPassantSq)
    
        while attacks:
            toSq = lsb(attacks)
            isEn = (enPassantSq >= 0 and toSq == enPassantSq)

            if (1 << toSq) & backRank:
                for piece in ['Q', 'R', 'B', 'N']:
                    moves.append(Move(fromSq = fromSq, toSq = toSq, promotion=piece))
            else:
                moves.append(Move(fromSq = fromSq, toSq = toSq, enPassant=isEn))
            attacks &= attacks - 1
    
        pawnsBB &= pawnsBB - 1
    
    return moves

def _pawn_attacks(square: int, colour: str) -> int:
    bb = 1 << square
    def shift(bb, n):
        return bb << n if n > 0 else bb >> -n
    if colour == "white":
        left, right = 7, 9
    else:
        left, right = -9, -7
    captureLeft = shift(bb & notAfile, left)
    captureRight = shift(bb & notHfile, right)
    # the old en-passant terms were subsets of these and added nothing
    return (captureLeft | captureRight) & BOARD

# precompute once at import: colour -> square -> attack bitboard
PAWN_ATTACKS = {
    "white": [_pawn_attacks(sq, "white") for sq in range(64)],
    "black": [_pawn_attacks(sq, "black") for sq in range(64)],
}

def pawnAttacks(square: int, colour: str, enPassantSq: int) -> int:
    # enPassantSq kept for signature compatibility; it never affected the result
    return PAWN_ATTACKS[colour][square]
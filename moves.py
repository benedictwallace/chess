from dataclasses import dataclass
from typing import Optional

from bitboard import lsb, BOARD, notAfile, notBfile, notGfile, notHfile, rank1, rank2, rank7, rank8

@dataclass(frozen=True)
class Move:
    fromSq: int
    toSq: int
    promotion: Optional[str] = None # 'Q', 'R', 'B', 'N' or None
    castle: bool = False
    enPassant: bool = False

def knightMoves(square: int) -> int:
    # Square index number not bb
    # Grid numbers 0 to 63
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

def rookMoves(square: int, ownPieces: int, allPieces: int) -> int:
    attacks = 0

    for direction in [1, -1, 8, -8]:

        current = square

        while True:
            prevFile = current % 8
            current += direction

            if current > 63 or current < 0:
                break

            currFile = current % 8
            if abs(currFile - prevFile) > 1: # wrap around on edges
                break

            if (ownPieces >> current) & 1: # Moves own pieces bb and checks if this is a 1 if there is a piece on current
                break
            
            attacks |= (1 << current)

            if (allPieces >> current) & 1:
                break
            
    return attacks

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
    attacks = 0

    for direction in [7, -7, 9, -9]:

        current = square

        while True:
            prevFile = current % 8
            current += direction

            if current > 63 or current < 0:
                break

            currFile = current % 8
            if abs(currFile - prevFile) > 1: # wrap around on edges
                break

            if (ownPieces >> current) & 1: # Moves own pieces bb and checks if this is a 1 if there is a piece on current
                break
            
            attacks |= (1 << current)

            if (allPieces >> current) & 1:
                break
            
    return attacks

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

def queenMoves(square: int,ownPieces: int, allPieces: int) -> int:
    return rookMoves(square, ownPieces, allPieces) | bishopMoves(square, ownPieces, allPieces)

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

def kingMoves(square: int, ownPieces: int, attacksBB: int) -> int:
    bb = 1 << square

    moves = (
        ((bb << 8)) |
        ((bb >> 8)) |
        ((bb << 1) & notHfile) |
        ((bb >> 1) & notAfile) |
        ((bb << 7) & notHfile) |
        ((bb >> 7) & notAfile) |
        ((bb << 9) & notAfile) |
        ((bb >> 9) & notHfile)
    )

    return moves & ~ownPieces & ~attacksBB & BOARD

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

def pawnAttacks(square: int, colour: str, enPassantSq: int) -> int:
    bb = 1 << square

    def shift(bb, n):
        return bb << n if n > 0 else bb >> -n
    
    # Flipped left and right for black
    if colour == "white":
        left = 7
        right = 9
    else:
        left = -9
        right = -7
    
    captureLeft = shift(bb & notAfile, left) 
    captureRight = shift(bb & notHfile, right)

    if enPassantSq >= 0:
        epBB        = 1 << enPassantSq
        epLeft      = shift(bb & notAfile, left) & epBB
        epRight     = shift(bb & notHfile, right) & epBB
    else:
        epLeft = epRight = 0

    return (captureLeft | captureRight | epLeft | epRight) & BOARD



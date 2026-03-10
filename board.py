from dataclasses import dataclass
from typing import Optional



# Useful functions
def bb_from_string(s: str) -> int:
    rows = [row.split() for row in s.strip().split("\n")]
    result = 0
    for rank_idx, row in enumerate(reversed(rows)):  # reverse so rank 1 = bottom row
        for file_idx, ch in enumerate(row):
            if ch == "1":
                square = rank_idx * 8 + file_idx
                result |= (1 << square)
    return result

def bb_to_string(bb: int) -> str:
    rows = []
    for rank in range(7, -1, -1):  # rank 8 down to rank 1
        row = []
        for file in range(8):
            square = rank * 8 + file
            row.append("1" if (bb >> square) & 1 else "0")
        rows.append(" ".join(row))
    return "\n".join(rows)

def lsb(bb: int) -> int:
    # lowest set bit, returns the index of the first piece
    return (bb & -bb).bit_length() - 1


# Classes

@dataclass
class Move:
    fromSq: int
    toSq: int
    promotion: Optional[str] = None # 'Q', 'R', 'B', 'N' or None
    castle: bool = False
    enPassant: bool = False


class Board:
    def __init__(self):
        self.whitePawn = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 1
    0 0 0 0 0 0 0 0
""")
        self.whiteBishop = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 1 0 0 1 0 0
""")
        self.whiteKnight = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 1 0 0 0 0 1 0
""")
        self.whiteRook = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    1 0 0 0 0 0 0 1
""")
        self.whiteQueen = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 1 0 0 0 0
""")
        self.whiteKing = bb_from_string("""
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 1 0 0 0
""")
        self.whitePieces = self.getWhitePieces()

        self.blackPawn = bb_from_string("""
    0 0 0 0 0 0 0 0
    1 1 1 1 1 1 1 1
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackBishop = bb_from_string("""
    0 0 1 0 0 1 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackKnight = bb_from_string("""
    0 1 0 0 0 0 1 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackRook = bb_from_string("""
    1 0 0 0 0 0 0 1
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackQueen = bb_from_string("""
    0 0 0 1 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackKing = bb_from_string("""
    0 0 0 0 1 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
    0 0 0 0 0 0 0 0
""")
        self.blackPieces = self.getBlackPieces()

        self.allPieces = self.getWhitePieces() | self.getBlackPieces()
        self.enPassantSq = -1

    def getWhitePieces(self) -> int:
        return self.whitePawn | self.whiteBishop | self.whiteKnight | self.whiteRook | self.whiteQueen | self.whiteKing

    def getBlackPieces(self) -> int:
        return self.blackPawn | self.blackBishop | self.blackKnight | self.blackRook | self.blackQueen | self.blackKing
    
    def whiteMoves(self) -> list[Move]:
        moves = []
        moves += getKnightMoves(self.whiteKnight, self.whitePieces)
        moves += getRookMoves(self.whiteRook, ownPieces=self.whitePieces, allPieces=self.allPieces)
        moves += getBishopMoves(self.whiteBishop, ownPieces=self.whitePieces, allPieces=self.allPieces)
        moves += getQueenMoves(self.whiteQueen, ownPieces=self.whitePieces, allPieces=self.allPieces)
        moves += getKingMoves(self.whiteKing, ownPieces=self.whitePieces)
        moves += getPawnMoves(self.whitePawn, ownPieces=self.whitePieces, allPieces=self.allPieces, oppPieces=self.blackPieces, direction=1, enPassantSq=self.enPassantSq)
        return moves

    def blackMoves(self) -> list[Move]:
        moves = []
        moves += getKnightMoves(self.blackKnight, self.blackPieces)
        moves += getRookMoves(self.blackRook, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getBishopMoves(self.blackBishop, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getQueenMoves(self.blackQueen, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getKingMoves(self.blackKing, ownPieces=self.blackPieces)
        moves += getPawnMoves(self.blackPawn, ownPieces=self.blackPieces, allPieces=self.allPieces, oppPieces=self.whitePieces, direction=-1, enPassantSq=self.enPassantSq)
        return moves
    
    def updatePieces(self):
        self.whitePieces = self.getWhitePieces()
        self.blackPieces = self.getBlackPieces()
        self.allPieces = self.whitePieces | self.blackPieces

    def getAttackedSq(self, colour: str) -> int:
        """
        colour: "white" or "black", for the colour of the attacker ("white" gives all of whites attacks).
        """
        attacked = 0
        if colour == "black":

            bb = self.blackKnight
            while bb:
                fromSq = lsb(bb)
                attacked |= knightMoves(fromSq)
                bb &= bb - 1
            
            bb = self.blackBishop
            while bb:
                fromSq = lsb(bb)
                attacked |= bishopMoves(fromSq, ownPieces=self.blackPieces, allPieces=self.allPieces)
                bb &= bb - 1

            bb = self.blackRook
            while bb:
                fromSq = lsb(bb)
                attacked |= rookMoves(fromSq, ownPieces=self.blackPieces, allPieces=self.allPieces)
                bb &= bb - 1
            
            bb = self.blackQueen
            while bb:
                fromSq = lsb(bb)
                attacked |= queenMoves(fromSq, ownPieces=self.blackPieces, allPieces=self.allPieces)
                bb &= bb - 1

            bb = self.blackKing
            while bb:
                fromSq = lsb(bb)
                attacked |= kingMoves(fromSq, ownPieces=self.blackPieces)
                bb &= bb - 1

            bb = self.blackPawn
            while bb:
                fromSq = lsb(bb)
                attacked |= pawnAttacks(fromSq, direction=-1, enPassantSq=self.enPassantSq)
                bb &= bb - 1           
        else:
            bb = self.whiteKnight
            while bb:
                fromSq = lsb(bb)
                attacked |= knightMoves(fromSq)
                bb &= bb - 1
            
            bb = self.whiteBishop
            while bb:
                fromSq = lsb(bb)
                attacked |= bishopMoves(fromSq, ownPieces=self.whitePieces, allPieces=self.allPieces)
                bb &= bb - 1

            bb = self.whiteRook
            while bb:
                fromSq = lsb(bb)
                attacked |= rookMoves(fromSq, ownPieces=self.whitePieces, allPieces=self.allPieces)
                bb &= bb - 1
            
            bb = self.whiteQueen
            while bb:
                fromSq = lsb(bb)
                attacked |= queenMoves(fromSq, ownPieces=self.whitePieces, allPieces=self.allPieces)
                bb &= bb - 1

            bb = self.whiteKing
            while bb:
                fromSq = lsb(bb)
                attacked |= kingMoves(fromSq, ownPieces=self.whitePieces)
                bb &= bb - 1

            bb = self.whitePawn
            while bb:
                fromSq = lsb(bb)
                attacked |= pawnAttacks(fromSq, direction=1, enPassantSq=self.enPassantSq)
                bb &= bb - 1     
        
        return attacked



def knightMoves(square: int) -> int:
    # Square index number not bb
    # Grid numbers 0 to 63
    bb = 1 << square
    # 6, 10, 15, 17, -6, -10, -15, -17

    notAfile = bb_from_string("""
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
    """)
    notBfile = bb_from_string("""
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
        1 0 1 1 1 1 1 1
    """)
    notGfile = bb_from_string("""
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
        1 1 1 1 1 1 0 1
    """)
    notHfile = bb_from_string("""
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
    """)


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

    return moves

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

def kingMoves(square: int, ownPieces: int) -> int:
    bb = 1 << square

    notAfile = bb_from_string("""
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
    """)
    notHfile = bb_from_string("""
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
    """)

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

    return moves & ~ownPieces

def getKingMoves(kingsBB: int, ownPieces: int) -> list[Move]:
    moves = []

    while kingsBB:
        fromSq = lsb(kingsBB)
        attacks = kingMoves(fromSq) & ~ownPieces

        while attacks:
            toSq = lsb(attacks)
            moves.append(Move(fromSq = fromSq, toSq = toSq))
            # Removes lsb
            attacks &= attacks - 1
        
        # Removes lsb
        kingsBB &= kingsBB - 1
    
    return moves

def pawnMoves(square: int, ownPieces: int, allPieces: int, oppPieces: int, direction: int, enPassantSq: int) -> int:
    # direction 1 for white -1 for black
    bb = 1 << square

    def shift(bb, n):
        return bb << n if n > 0 else bb >> -n

    rank2 = bb_from_string("""
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        1 1 1 1 1 1 1 1
        0 0 0 0 0 0 0 0
    """)
    rank7 = bb_from_string("""
        0 0 0 0 0 0 0 0
        1 1 1 1 1 1 1 1                   
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
    """)
    notAfile = bb_from_string("""
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
    """)
    notHfile = bb_from_string("""
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
    """)
    
    s = direction
    startRank = rank2 if direction == 1 else rank7

    single = shift(bb, 8*s) &~ allPieces

    double = shift(bb & startRank, 16*s) &~ allPieces &~ shift(allPieces, 8*s)

    captureLeft = shift(bb & notAfile & oppPieces, 7*s)
    captureRight = shift(bb & notHfile & oppPieces, 9*s)

    if enPassantSq >= 0:
        epBB        = 1 << enPassantSq
        epLeft      = shift(bb & notHfile, s * 7) & epBB
        epRight     = shift(bb & notAfile, s * 9) & epBB
    else:
        epLeft = epRight = 0

    return single | double | captureLeft | captureRight | epLeft | epRight

def getPawnMoves(pawnsBB: int, ownPieces: int, allPieces: int, oppPieces: int, direction: int, enPassantSq: int) -> list[Move]:
    moves = []
    if direction == 1:
        backRank = bb_from_string("""
            1 1 1 1 1 1 1 1
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
        """)  
    else: 
        backRank = bb_from_string("""
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            0 0 0 0 0 0 0 0
            1 1 1 1 1 1 1 1
        """)

    while pawnsBB:
        fromSq = lsb(pawnsBB)
        attacks = pawnMoves(fromSq, ownPieces, allPieces, oppPieces, direction, enPassantSq)
    
        while attacks:
            toSq = lsb(attacks)
            isEn = (enPassantSq >= 0 and toSq == enPassantSq)

            if (1 << toSq) & backRank:
                for piece in ['Q', 'R', 'B', 'N']:
                    moves.append(Move(fromSq = fromSq, toSq = toSq, promotion=piece))
            else:
                moves.append(Move(fromSq = fromSq, toSq = toSq))
            attacks &= attacks - 1
    
        pawnsBB &= pawnsBB - 1
    
    return moves

def pawnAttacks(square: int, direction: int, enPassantSq: int) -> int:
    bb = 1 << square

    def shift(bb, n):
        return bb << n if n > 0 else bb >> -n
    
    s = direction

    notAfile = bb_from_string("""
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
        0 1 1 1 1 1 1 1
    """)
    notHfile = bb_from_string("""
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
        1 1 1 1 1 1 1 0
    """)
    
    captureLeft = shift(bb & notAfile, 7*s) 
    captureRight = shift(bb & notHfile, 9*s)
    if enPassantSq >= 0:
        epBB        = 1 << enPassantSq
        epLeft      = shift(bb & notHfile, s * 7) & epBB
        epRight     = shift(bb & notAfile, s * 9) & epBB
    else:
        epLeft = epRight = 0

    return captureLeft | captureRight | epLeft | epRight



notAfile = bb_from_string("""
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
    0 1 1 1 1 1 1 1
""")
notBfile = bb_from_string("""
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
    1 0 1 1 1 1 1 1
""")
notGfile = bb_from_string("""
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
    1 1 1 1 1 1 0 1
""")
notHfile = bb_from_string("""
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
    1 1 1 1 1 1 1 0
""")



if __name__ == "__main__":
    assert bb_from_string(bb_to_string(1)) == 1

    board = Board()


    #print(bb_to_string(board.getAttackedSq("white")))
    print(bb_to_string(queenMoves(3, board.whitePieces, board.allPieces)))


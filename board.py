
from bitboard import bb_from_string, bb_to_string, lsb
from moves import (
    Move,
    knightMoves, getKnightMoves,
    rookMoves, getRookMoves,
    bishopMoves, getBishopMoves,
    queenMoves, getQueenMoves,
    kingMoves, getKingMoves,
    pawnMoves, getPawnMoves, pawnAttacks,
)


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
        # track rook/king moves from start sq.
        self.whiteKCastle = True
        self.whiteQCastle = True
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
        # track rook/king moves from start sq.
        self.blackKCastle = True
        self.blackQCastle = True
        self.blackPieces = self.getBlackPieces()

        self.allPieces = self.blackPieces | self.whitePieces
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
        moves += getKingMoves(self.whiteKing, ownPieces=self.whitePieces, attackBB=self.getAttackedSq("black"))
        moves += getPawnMoves(self.whitePawn, ownPieces=self.whitePieces, allPieces=self.allPieces, oppPieces=self.blackPieces, direction=1, enPassantSq=self.enPassantSq)
        moves += self.getCastles(colour="white", attacksBB=self.getAttackedSq("black"))
        return moves

    def blackMoves(self) -> list[Move]:
        moves = []
        moves += getKnightMoves(self.blackKnight, self.blackPieces)
        moves += getRookMoves(self.blackRook, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getBishopMoves(self.blackBishop, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getQueenMoves(self.blackQueen, ownPieces=self.blackPieces, allPieces=self.allPieces)
        moves += getKingMoves(self.blackKing, ownPieces=self.blackPieces, attackBB=self.getAttackedSq("white"))
        moves += getPawnMoves(self.blackPawn, ownPieces=self.blackPieces, allPieces=self.allPieces, oppPieces=self.whitePieces, direction=-1, enPassantSq=self.enPassantSq)
        moves += self.getCastles(colour="black", attacksBB=self.getAttackedSq("white"))
        return moves
    
    def updatePieces(self):
        self.whitePieces = self.getWhitePieces()
        self.blackPieces = self.getBlackPieces()
        self.allPieces = self.whitePieces | self.blackPieces

    def getAttackedSq(self, colour: str) -> int:
        """
        colour: "white" or "black", for the colour of the attacker ("white" gives all of whites attacks).
            kingMoves is passed attackBB as 0 since we need to know king surrounding squares.
        """
        attacked = 0
        if colour == "black":
            # Remove king to allow attacked spaces behind king to be added.
            allPiecesNoKing = self.allPieces & ~self.whiteKing

            bb = self.blackKnight
            while bb:
                fromSq = lsb(bb)
                attacked |= knightMoves(fromSq)
                bb &= bb - 1
            
            bb = self.blackBishop
            while bb:
                fromSq = lsb(bb)
                attacked |= bishopMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1

            bb = self.blackRook
            while bb:
                fromSq = lsb(bb)
                attacked |= rookMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1
            
            bb = self.blackQueen
            while bb:
                fromSq = lsb(bb)
                attacked |= queenMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1

            bb = self.blackKing
            while bb:
                fromSq = lsb(bb)
                attacked |= kingMoves(fromSq, ownPieces=0, attacksBB=0)
                bb &= bb - 1

            bb = self.blackPawn
            while bb:
                fromSq = lsb(bb)
                attacked |= pawnAttacks(fromSq, direction=-1, enPassantSq=self.enPassantSq)
                bb &= bb - 1           
        else:
            # Remove king to allow attacked spaces behind king to be added.
            allPiecesNoKing = self.allPieces & ~self.blackKing

            bb = self.whiteKnight
            while bb:
                fromSq = lsb(bb)
                attacked |= knightMoves(fromSq)
                bb &= bb - 1
            
            bb = self.whiteBishop
            while bb:
                fromSq = lsb(bb)
                attacked |= bishopMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1

            bb = self.whiteRook
            while bb:
                fromSq = lsb(bb)
                attacked |= rookMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1
            
            bb = self.whiteQueen
            while bb:
                fromSq = lsb(bb)
                attacked |= queenMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
                bb &= bb - 1

            bb = self.whiteKing
            while bb:
                fromSq = lsb(bb)
                attacked |= kingMoves(fromSq, ownPieces=0, attacksBB=0)
                bb &= bb - 1

            bb = self.whitePawn
            while bb:
                fromSq = lsb(bb)
                attacked |= pawnAttacks(fromSq, direction=1, enPassantSq=self.enPassantSq)
                bb &= bb - 1     
        
        return attacked

    def getCastles(self, colour: str, attacksBB) -> list[Move]:
        moves = []


        if colour == "white":
            
            mustBeEmptyKwhite = bb_from_string("""
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 1 1 0
            """) 
            cantBeAttackedKwhite = bb_from_string("""
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 1 1 1 0
            """)         
            mustBeEmptyQwhite = bb_from_string("""
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 1 1 1 0 0 0 0
            """) 
            cantBeAttackedQwhite = bb_from_string("""
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 1 1 1 0 0 0
            """)      
            
            
            if self.whiteKCastle and not (self.allPieces & mustBeEmptyKwhite) and not (attacksBB & cantBeAttackedKwhite):
                moves.append(Move(fromSq = 4, toSq = 6, castle = True))

            if self.whiteQCastle and not (self.allPieces & mustBeEmptyQwhite) and not (attacksBB & cantBeAttackedQwhite):
                moves.append(Move(fromSq = 4, toSq = 2, castle = True))

        else:
            
            mustBeEmptyKblack = bb_from_string("""
                                           0 0 0 0 0 1 1 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
            """) 
            cantBeAttackedKblack = bb_from_string("""
                                           0 0 0 0 1 1 1 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
            """)         
            mustBeEmptyQblack = bb_from_string("""
                                           0 1 1 1 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
            """) 
            cantBeAttackedQblack = bb_from_string("""
                                           0 0 1 1 1 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
                                           0 0 0 0 0 0 0 0
            """)
            
            if self.blackKCastle and not (self.allPieces & mustBeEmptyKblack) and not (attacksBB & cantBeAttackedKblack):
                moves.append(Move(fromSq = 4, toSq = 6, castle = True))

            if self.blackQCastle and not (self.allPieces & mustBeEmptyQblack) and not (attacksBB & cantBeAttackedQblack):
                moves.append(Move(fromSq = 4, toSq = 2, castle = True))

        return moves

    def makeMove(self, move):

        # find which piece it is
        if (self.whitePawn >> move.fromSq) & 1:
            pass


        # adjust that bit board
        # remove a piece if it's on that square




if __name__ == "__main__":
    assert bb_from_string(bb_to_string(1)) == 1

    # Clear the board
    board = Board()
    board.whitePawn = 0
    board.whiteBishop = 0
    board.whiteKnight = 0
    board.whiteRook = 0
    board.whiteQueen = 0
    board.blackPawn = 0
    board.blackBishop = 0
    board.blackKnight = 0
    board.blackRook = 0
    board.blackQueen = 0

    # White king on e1 (sq 4), black rook on e5 (sq 36)
    board.whiteKing = bb_from_string("""
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
    """)
    board.blackRook = bb_from_string("""
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 1 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
        0 0 0 0 0 0 0 0
    """)
    board.updatePieces()

    moves = board.whiteMoves()
    king_moves = [m for m in moves]
    king_destinations = [m.toSq for m in king_moves]
    print(king_moves)
    print(king_destinations)



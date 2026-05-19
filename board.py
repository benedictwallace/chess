
from bitboard import (
    bb_from_string, bb_to_string, lsb, 
    mustBeEmptyKwhite, mustBeEmptyQwhite, 
    cantBeAttackedKwhite, cantBeAttackedQwhite, 
    mustBeEmptyKblack, cantBeAttackedKblack,
    mustBeEmptyQblack, cantBeAttackedQblack,
)
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
        self.bb = {
            ('white', 'pawn') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                1 1 1 1 1 1 1 1
                                                0 0 0 0 0 0 0 0
                                            """),
            ('white', 'bishop') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 1 0 0 1 0 0
                                            """),
            ('white', 'knight') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 1 0 0 0 0 1 0
                                            """),
            ('white', 'rook') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                1 0 0 0 0 0 0 1
                                            """),
            ('white', 'queen') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 1 0 0 0 0
                                            """),
            ('white', 'king') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 1 0 0 0
                                            """),
            ('black', 'pawn') : bb_from_string("""
                                                0 0 0 0 0 0 0 0
                                                1 1 1 1 1 1 1 1
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """),
            ('black', 'bishop') : bb_from_string("""
                                                0 0 1 0 0 1 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """),
            ('black', 'knight') : bb_from_string("""
                                                0 1 0 0 0 0 1 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """),
            ('black', 'rook') : bb_from_string("""
                                                1 0 0 0 0 0 0 1
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """),
            ('black', 'queen') : bb_from_string("""
                                                0 0 0 1 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """),
            ('black', 'king') : bb_from_string("""
                                                0 0 0 0 1 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                                0 0 0 0 0 0 0 0
                                            """)
        }
        # track rook/king moves from start sq.
        self.whiteKCastle = True
        self.whiteQCastle = True
        self.whitePieces = self.getWhitePieces()

        # track rook/king moves from start sq.
        self.blackKCastle = True
        self.blackQCastle = True
        self.blackPieces = self.getBlackPieces()

        self.allPieces = self.blackPieces | self.whitePieces
        self.enPassantSq = -1

    def getWhitePieces(self) -> int:
        result = 0
        for (colour, _), bb in self.bb.items():
            if colour == "white":
                result |= bb
        return result

    def getBlackPieces(self) -> int:
        result = 0
        for (colour, _), bb in self.bb.items():
            if colour == "black":
                result |= bb
        return result
    
    def getMoves(self, colour: str) -> list[Move]:
        if colour == "white":
            ownPieces = self.whitePieces
            oppPieces = self.blackPieces
            
            oppColour = "black"
        else:
            ownPieces = self.blackPieces
            oppPieces = self.whitePieces
            
            oppColour = "white"

        oppAttacks = self.getAttackedSq(oppColour)
        moves = []
        moves += getKnightMoves(self.bb[colour, "knight"], ownPieces)
        moves += getRookMoves(self.bb[colour, "rook"], ownPieces=ownPieces, allPieces=self.allPieces)
        moves += getBishopMoves(self.bb[colour, "bishop"], ownPieces=ownPieces, allPieces=self.allPieces)
        moves += getQueenMoves(self.bb[colour, "queen"], ownPieces=ownPieces, allPieces=self.allPieces)
        moves += getKingMoves(self.bb[colour, "king"], ownPieces=ownPieces, attackBB=oppAttacks)
        moves += getPawnMoves(self.bb[colour, "pawn"], ownPieces=ownPieces, allPieces=self.allPieces, oppPieces=oppPieces, colour=colour, enPassantSq=self.enPassantSq)
        moves += self.getCastles(colour=colour, attacksBB=oppAttacks)
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

        oppColour = "black" if colour == "white" else "white"
        # Remove king to allow attacked spaces behind king to be added.
        allPiecesNoKing = self.allPieces & ~self.bb[oppColour, "king"]

        bb = self.bb[colour, "knight"]
        while bb:
            fromSq = lsb(bb)
            attacked |= knightMoves(fromSq)
            bb &= bb - 1
        
        bb = self.bb[colour, "bishop"]
        while bb:
            fromSq = lsb(bb)
            attacked |= bishopMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
            bb &= bb - 1

        bb = self.bb[colour, "rook"]
        while bb:
            fromSq = lsb(bb)
            attacked |= rookMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
            bb &= bb - 1
        
        bb = self.bb[colour, "queen"]
        while bb:
            fromSq = lsb(bb)
            attacked |= queenMoves(fromSq, ownPieces=0, allPieces=allPiecesNoKing)
            bb &= bb - 1

        bb = self.bb[colour, "king"]
        while bb:
            fromSq = lsb(bb)
            attacked |= kingMoves(fromSq, ownPieces=0, attacksBB=0)
            bb &= bb - 1

        bb = self.bb[colour, "pawn"]
        while bb:
            fromSq = lsb(bb)
            attacked |= pawnAttacks(fromSq, colour=colour, enPassantSq=self.enPassantSq)
            bb &= bb - 1           
        
        return attacked

    def getCastles(self, colour: str, attacksBB) -> list[Move]:
        moves = []


        if colour == "white":

            if self.whiteKCastle and not (self.allPieces & mustBeEmptyKwhite) and not (attacksBB & cantBeAttackedKwhite):
                moves.append(Move(fromSq = 4, toSq = 6, castle = True))

            if self.whiteQCastle and not (self.allPieces & mustBeEmptyQwhite) and not (attacksBB & cantBeAttackedQwhite):
                moves.append(Move(fromSq = 4, toSq = 2, castle = True))

        else:

            if self.blackKCastle and not (self.allPieces & mustBeEmptyKblack) and not (attacksBB & cantBeAttackedKblack):
                moves.append(Move(fromSq = 60, toSq = 62, castle = True))

            if self.blackQCastle and not (self.allPieces & mustBeEmptyQblack) and not (attacksBB & cantBeAttackedQblack):
                moves.append(Move(fromSq = 60, toSq = 58, castle = True))

        return moves

    def inCheck(self, colour: str) -> bool:
        if colour == "white":
            return bool(self.bb["white", "king"] & self.getAttackedSq("black"))
        else:
            return bool(self.bb["black", "king"] & self.getAttackedSq("white"))
        
    def legalMoves(self, colour: str) -> int:
        # return the list of legal moves, e.g. moves not leading to check.
        pass


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



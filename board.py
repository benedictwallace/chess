import copy

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
        self.sideToMove = "white"

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
        
    def legalMoves(self, colour: str) -> list[Move]:
        """
        Return moves that don't leave our own king in check.

        """
        pseudo = self.getMoves(colour)
        legal = []
        for move in pseudo:
            trial = copy.deepcopy(self)
            trial.makeMove(move)
            if not trial.inCheck(colour):
                legal.append(move)
        return legal

    def pieceAt(self, square: Move) -> tuple[str]:
        bit = 1 << square

        for key, bb in self.bb.items():
            if bit & bb:
                return key
        return None
    
    def makeMove(self, move):
        fromBB = 1 << move.fromSq
        toBB = 1 << move.toSq

        # find which piece is moving
        mover = self.pieceAt(move.fromSq)
        if mover is None:
            raise ValueError(f"makeMove: no piece on fromSq={move.fromSq}")
        
        colour, piece = mover
        oppColour = "black" if colour == "white" else "white"
        
        # find if move takes
        taken = self.pieceAt(move.toSq)

        # move mover
        self.bb[mover] &= ~fromBB
        self.bb[mover] |= toBB

        # take piece
        if taken is not None:
            self.bb[taken] &= ~toBB
        
        # promotion
        if move.promotion is not None:
            promoMap = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
            promoted = promoMap[move.promotion]
            self.bb[colour, "pawn"] &= ~toBB # remove the pawn
            self.bb[colour, promoted] |= toBB # add the promoted piece

        # castling
        if move.castle:
            if move.toSq == 6: # white kingside
                self.bb[colour, "rook"] &= ~(1 << 7)
                self.bb[colour, "rook"] |= (1 << 5)
            elif move.toSq == 2: # white queenside
                self.bb[colour, "rook"] &= ~(1 << 0)
                self.bb[colour, "rook"] |= (1 << 3)
            elif move.toSq == 62: # black kingside
                self.bb[colour, "rook"] &= ~(1 << 63)
                self.bb[colour, "rook"] |= (1 << 61)
            elif move.toSq == 58: # black queenside
                self.bb[colour, "rook"] &= ~(1 << 56)
                self.bb[colour, "rook"] |= (1 << 59)

        # enpassant
        if move.enPassant:
            capturedSq = move.toSq - 8 if colour == "white" else move.toSq + 8
            self.bb[oppColour, "pawn"] &= ~(1 << capturedSq)

        # update enpensantsq + castling rights
        if piece == "pawn" and abs(move.toSq - move.fromSq) == 16:
            self.enPassantSq = (move.fromSq + move.toSq) // 2
        else:
            self.enPassantSq = -1

        if move.fromSq == 4 or move.toSq == 4:
            self.whiteKCastle = self.whiteQCastle = False
        if move.fromSq == 60 or move.toSq == 60:
            self.blackKCastle = self.blackQCastle = False
        if move.fromSq == 0 or move.toSq == 0:
            self.whiteQCastle = False
        if move.fromSq == 7 or move.toSq == 7:
            self.whiteKCastle = False
        if move.fromSq == 56 or move.toSq == 56:
            self.blackQCastle = False
        if move.fromSq == 63 or move.toSq == 63:
            self.blackKCastle = False

        # update pieces
        self.updatePieces()
        
        self.sideToMove = "black" if self.sideToMove == "white" else "white"

    def checkMate(self, colour: str) -> bool:
        """
        Is colour in checkmate?
        """
        return self.inCheck(colour) and not self.legalMoves(colour)
    
    def staleMate(self, colour: str) -> bool:
        """
        Is colour in stalemate?
        """
        return not self.inCheck(colour) and not self.legalMoves(colour)

def perft(board, colour, depth):
    if depth == 0:
        return 1
    total = 0
    for move in board.legalMoves(colour):
        trial = copy.deepcopy(board)
        trial.makeMove(move)
        next_colour = "black" if colour == "white" else "white"
        total += perft(trial, next_colour, depth - 1)
    return total


if __name__ == "__main__":
    assert bb_from_string(bb_to_string(1)) == 1

    b = Board()
    print(perft(b, "white", 1))
    print(perft(b, "white", 2))
    print(perft(b, "white", 3))
    print(perft(b, "white", 4))

import copy

from engine.bitboard import (
    bb_from_string, bb_to_string, lsb,
    mustBeEmptyKwhite, mustBeEmptyQwhite,
    mustBeEmptyKblack, mustBeEmptyQblack,
)
from engine.moves import (
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
        self.history = []   # undo stack for make/unmake
        # square -> (colour, piece) key (or None). Maintained incrementally in
        # make/unmake so pieceAt() is O(1) instead of a 12-bitboard scan.
        self._build_mailbox()

    def _build_mailbox(self):
        """(Re)build the square->piece mailbox from the bitboards. Used at
        construction and by updatePieces(); the hot path keeps it in sync
        incrementally and never calls this."""
        mb = [None] * 64
        for key, bb in self.bb.items():
            x = bb
            while x:
                sq = (x & -x).bit_length() - 1
                mb[sq] = key
                x &= x - 1
        self.mailbox = mb

    def stateKey(self) -> tuple:
        return (
            self.bb[("white", "pawn")],
            self.bb[("white", "knight")],
            self.bb[("white", "bishop")],
            self.bb[("white", "rook")],
            self.bb[("white", "queen")],
            self.bb[("white", "king")],
            self.bb[("black", "pawn")],
            self.bb[("black", "knight")],
            self.bb[("black", "bishop")],
            self.bb[("black", "rook")],
            self.bb[("black", "queen")],
            self.bb[("black", "king")],
            self.sideToMove,
            self.whiteKCastle, self.whiteQCastle,
            self.blackKCastle, self.blackQCastle,
            self.enPassantSq,
        )

    def clone(self):
        new = Board.__new__(Board) # skip __init__
        new.bb = dict(self.bb)            
        new.whiteKCastle = self.whiteKCastle
        new.whiteQCastle = self.whiteQCastle
        new.blackKCastle = self.blackKCastle
        new.blackQCastle = self.blackQCastle
        new.whitePieces = self.whitePieces
        new.blackPieces = self.blackPieces
        new.allPieces = self.allPieces
        new.enPassantSq = self.enPassantSq
        new.sideToMove = self.sideToMove
        new.history = []   # fresh undo stack; clones don't share history
        new.mailbox = list(self.mailbox)   # shallow copy of the square->piece map
        return new

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
        else:
            ownPieces = self.blackPieces
            oppPieces = self.whitePieces

        moves = []
        moves += getKnightMoves(self.bb[colour, "knight"], ownPieces)
        moves += getRookMoves(self.bb[colour, "rook"], ownPieces=ownPieces, allPieces=self.allPieces)
        moves += getBishopMoves(self.bb[colour, "bishop"], ownPieces=ownPieces, allPieces=self.allPieces)
        moves += getQueenMoves(self.bb[colour, "queen"], ownPieces=ownPieces, allPieces=self.allPieces)
        # King moves are generated WITHOUT an attack mask (attackBB=0): the
        # make/unmake legality filter in legalMoves() already rejects any king
        # move that walks into check, so pre-masking here just duplicated a
        # full-board getAttackedSq() sweep on every single legalMoves() call.
        moves += getKingMoves(self.bb[colour, "king"], ownPieces=ownPieces, attackBB=0)
        moves += getPawnMoves(self.bb[colour, "pawn"], ownPieces=ownPieces, allPieces=self.allPieces, oppPieces=oppPieces, colour=colour, enPassantSq=self.enPassantSq)
        moves += self.getCastles(colour)
        return moves
    
    def updatePieces(self):
        self.whitePieces = self.getWhitePieces()
        self.blackPieces = self.getBlackPieces()
        self.allPieces = self.whitePieces | self.blackPieces
        # rebuild the mailbox to match (used by the FEN loader, which sets bb
        # directly via __new__ and then calls this).
        self._build_mailbox()

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

    def getCastles(self, colour: str) -> list[Move]:
        """
        Castling moves. Attack checks are done lazily with targeted
        squareAttackedBy() calls (which early-exit) on exactly the king's
        start/transit/destination squares, and ONLY when castling rights plus
        the empty-square requirement are already satisfied. In the common case
        (no rights, or blocked) this returns immediately without touching the
        opponent's attack set at all -- replacing the old full getAttackedSq()
        sweep that ran on every move generation.

        The squares tested match the old cantBeAttacked* bitboards exactly, so
        the legal set (and perft) is unchanged.
        """
        moves = []

        if colour == "white":
            kingside = self.whiteKCastle and not (self.allPieces & mustBeEmptyKwhite)
            queenside = self.whiteQCastle and not (self.allPieces & mustBeEmptyQwhite)
            if not (kingside or queenside):
                return moves
            opp = "black"
            if kingside and not any(self.squareAttackedBy(sq, opp) for sq in (4, 5, 6)):
                moves.append(Move(fromSq=4, toSq=6, castle=True))
            if queenside and not any(self.squareAttackedBy(sq, opp) for sq in (4, 3, 2)):
                moves.append(Move(fromSq=4, toSq=2, castle=True))
        else:
            kingside = self.blackKCastle and not (self.allPieces & mustBeEmptyKblack)
            queenside = self.blackQCastle and not (self.allPieces & mustBeEmptyQblack)
            if not (kingside or queenside):
                return moves
            opp = "white"
            if kingside and not any(self.squareAttackedBy(sq, opp) for sq in (60, 61, 62)):
                moves.append(Move(fromSq=60, toSq=62, castle=True))
            if queenside and not any(self.squareAttackedBy(sq, opp) for sq in (60, 59, 58)):
                moves.append(Move(fromSq=60, toSq=58, castle=True))

        return moves

    def squareAttackedBy(self, square: int, byColour: str) -> bool:
        """
        True if `square` is attacked by any `byColour` piece.
        Casts each piece's pattern OUT from `square` and intersects with the
        relevant enemy bitboard, so it stops at the first hit instead of
        building the whole attack set.
        """
        occ = self.allPieces

        # a byColour pawn attacks `square` from where an opposite-colour pawn
        # placed on `square` would attack -> reuse the pawn-attack pattern
        defColour = "black" if byColour == "white" else "white"
        if pawnAttacks(square, defColour, -1) & self.bb[byColour, "pawn"]:
            return True
        if knightMoves(square) & self.bb[byColour, "knight"]:
            return True
        if kingMoves(square, 0, 0) & self.bb[byColour, "king"]:
            return True
        if bishopMoves(square, 0, occ) & (self.bb[byColour, "bishop"] | self.bb[byColour, "queen"]):
            return True
        if rookMoves(square, 0, occ) & (self.bb[byColour, "rook"] | self.bb[byColour, "queen"]):
            return True
        return False

    def inCheck(self, colour: str) -> bool:
        oppColour = "black" if colour == "white" else "white"
        kingSq = lsb(self.bb[colour, "king"])
        return self.squareAttackedBy(kingSq, oppColour)

    def legalMoves(self, colour: str) -> list[Move]:
        """
        Return moves that don't leave our own king in check.

        """
        oppColour = "black" if colour == "white" else "white"
        legal = []
        for move in self.getMoves(colour):
            self.makeMove(move)
            kingSq = lsb(self.bb[colour, "king"])
            if not self.squareAttackedBy(kingSq, oppColour):
                legal.append(move)
            self.unmakeMove(move)
        return legal

    def pieceAt(self, square: int) -> tuple[str]:
        # O(1) mailbox lookup (kept in sync by make/unmake).
        return self.mailbox[square]
    
    def makeMove(self, move):
        fromBB = 1 << move.fromSq
        toBB = 1 << move.toSq

        # find which piece is moving (O(1) mailbox read)
        mover = self.mailbox[move.fromSq]
        if mover is None:
            raise ValueError(f"makeMove: no piece on fromSq={move.fromSq}")
        
        colour, piece = mover
        oppColour = "black" if colour == "white" else "white"
        
        # find if move takes
        taken = self.mailbox[move.toSq]

        # save everything unmake can't recompute, BEFORE we mutate anything
        self.history.append((
            mover, taken, self.enPassantSq,
            self.whiteKCastle, self.whiteQCastle,
            self.blackKCastle, self.blackQCastle,
        ))

        # move mover (bitboards + mailbox)
        self.bb[mover] &= ~fromBB
        self.bb[mover] |= toBB
        self.mailbox[move.fromSq] = None
        self.mailbox[move.toSq] = mover      # overwrites `taken` if a capture

        # take piece
        if taken is not None:
            self.bb[taken] &= ~toBB
        
        # promotion
        if move.promotion is not None:
            promoMap = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
            promoted = promoMap[move.promotion]
            self.bb[colour, "pawn"] &= ~toBB # remove the pawn
            self.bb[colour, promoted] |= toBB # add the promoted piece
            self.mailbox[move.toSq] = (colour, promoted)

        # castling
        if move.castle:
            if move.toSq == 6: # white kingside
                self.bb[colour, "rook"] &= ~(1 << 7)
                self.bb[colour, "rook"] |= (1 << 5)
                self.mailbox[7] = None; self.mailbox[5] = (colour, "rook")
            elif move.toSq == 2: # white queenside
                self.bb[colour, "rook"] &= ~(1 << 0)
                self.bb[colour, "rook"] |= (1 << 3)
                self.mailbox[0] = None; self.mailbox[3] = (colour, "rook")
            elif move.toSq == 62: # black kingside
                self.bb[colour, "rook"] &= ~(1 << 63)
                self.bb[colour, "rook"] |= (1 << 61)
                self.mailbox[63] = None; self.mailbox[61] = (colour, "rook")
            elif move.toSq == 58: # black queenside
                self.bb[colour, "rook"] &= ~(1 << 56)
                self.bb[colour, "rook"] |= (1 << 59)
                self.mailbox[56] = None; self.mailbox[59] = (colour, "rook")

        # enpassant
        if move.enPassant:
            capturedSq = move.toSq - 8 if colour == "white" else move.toSq + 8
            self.bb[oppColour, "pawn"] &= ~(1 << capturedSq)
            self.mailbox[capturedSq] = None

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

        
        ownClear = fromBB
        ownSet = toBB
        if move.castle:
            if move.toSq == 6:    ownClear |= (1 << 7);  ownSet |= (1 << 5)
            elif move.toSq == 2:  ownClear |= (1 << 0);  ownSet |= (1 << 3)
            elif move.toSq == 62: ownClear |= (1 << 63); ownSet |= (1 << 61)
            elif move.toSq == 58: ownClear |= (1 << 56); ownSet |= (1 << 59)
        oppClear = 0
        if taken is not None:
            oppClear |= toBB
        if move.enPassant:
            capturedSq = move.toSq - 8 if colour == "white" else move.toSq + 8
            oppClear |= (1 << capturedSq)
        if colour == "white":
            self.whitePieces = (self.whitePieces & ~ownClear) | ownSet
            self.blackPieces &= ~oppClear
        else:
            self.blackPieces = (self.blackPieces & ~ownClear) | ownSet
            self.whitePieces &= ~oppClear
        self.allPieces = self.whitePieces | self.blackPieces

        self.sideToMove = "black" if self.sideToMove == "white" else "white"

    def unmakeMove(self, move):
        """
        Reverse the most recent makeMove. `move` must be the same Move that was
        last applied; the rest of the lost state is popped from self.history.
        """
        fromBB = 1 << move.fromSq
        toBB = 1 << move.toSq

        (mover, taken, prevEP,
         wK, wQ, bK, bQ) = self.history.pop()

        colour, piece = mover
        oppColour = "black" if colour == "white" else "white"

        # undo promotion: drop the promoted piece, restore the pawn to fromSq
        if move.promotion is not None:
            promoMap = {"Q": "queen", "R": "rook", "B": "bishop", "N": "knight"}
            promoted = promoMap[move.promotion]
            self.bb[colour, promoted] &= ~toBB
            self.bb[colour, "pawn"] &= ~toBB     # ensure nothing left on toSq
            self.bb[colour, "pawn"] |= fromBB
        else:
            # move the piece back from toSq to fromSq
            self.bb[mover] &= ~toBB
            self.bb[mover] |= fromBB

        # mailbox: piece returns to fromSq, destination restored to whatever was
        # captured (None if the move wasn't a capture).
        self.mailbox[move.fromSq] = mover
        self.mailbox[move.toSq] = taken

        # undo the castling rook hop
        if move.castle:
            if move.toSq == 6:        # white kingside: rook 7->5, put it back
                self.bb[colour, "rook"] &= ~(1 << 5)
                self.bb[colour, "rook"] |= (1 << 7)
                self.mailbox[5] = None; self.mailbox[7] = (colour, "rook")
            elif move.toSq == 2:      # white queenside: rook 0->3
                self.bb[colour, "rook"] &= ~(1 << 3)
                self.bb[colour, "rook"] |= (1 << 0)
                self.mailbox[3] = None; self.mailbox[0] = (colour, "rook")
            elif move.toSq == 62:     # black kingside: rook 63->61
                self.bb[colour, "rook"] &= ~(1 << 61)
                self.bb[colour, "rook"] |= (1 << 63)
                self.mailbox[61] = None; self.mailbox[63] = (colour, "rook")
            elif move.toSq == 58:     # black queenside: rook 56->59
                self.bb[colour, "rook"] &= ~(1 << 59)
                self.bb[colour, "rook"] |= (1 << 56)
                self.mailbox[59] = None; self.mailbox[56] = (colour, "rook")

        # restore a normally-captured piece on toSq
        if taken is not None:
            self.bb[taken] |= toBB

        # restore an en-passant-captured pawn (it never sat on toSq)
        if move.enPassant:
            capturedSq = move.toSq - 8 if colour == "white" else move.toSq + 8
            self.bb[oppColour, "pawn"] |= (1 << capturedSq)
            self.mailbox[capturedSq] = (oppColour, "pawn")

        # restore scalar state and side to move
        self.enPassantSq = prevEP
        self.whiteKCastle, self.whiteQCastle = wK, wQ
        self.blackKCastle, self.blackQCastle = bK, bQ

        # incremental occupancy update (replaces updatePieces)
        ownClear = toBB
        ownSet = fromBB
        if move.castle:
            if move.toSq == 6:    ownClear |= (1 << 5);  ownSet |= (1 << 7)
            elif move.toSq == 2:  ownClear |= (1 << 3);  ownSet |= (1 << 0)
            elif move.toSq == 62: ownClear |= (1 << 61); ownSet |= (1 << 63)
            elif move.toSq == 58: ownClear |= (1 << 59); ownSet |= (1 << 56)
        oppSet = 0
        if taken is not None:
            oppSet |= toBB
        if move.enPassant:
            capturedSq = move.toSq - 8 if colour == "white" else move.toSq + 8
            oppSet |= (1 << capturedSq)
        if colour == "white":
            self.whitePieces = (self.whitePieces & ~ownClear) | ownSet
            self.blackPieces |= oppSet
        else:
            self.blackPieces = (self.blackPieces & ~ownClear) | ownSet
            self.whitePieces |= oppSet
        self.allPieces = self.whitePieces | self.blackPieces

        self.sideToMove = colour

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
    next_colour = "black" if colour == "white" else "white"
    for move in board.legalMoves(colour):
        board.makeMove(move)
        total += perft(board, next_colour, depth - 1)
        board.unmakeMove(move)
    return total


if __name__ == "__main__":
    assert bb_from_string(bb_to_string(1)) == 1

    b = Board()
    print(perft(b, "white", 1))
    print(perft(b, "white", 2))
    print(perft(b, "white", 3))
    print(perft(b, "white", 4))
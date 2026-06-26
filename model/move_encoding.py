from engine.moves import Move

DIRECTIONS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
]


KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

UNDERPROMO_PIECES = ["N", "B", "R"] 

UNDERPROMO_DIRS = [0, -1, 1]

NUM_ACTIONS = 64 * 73 # 64 From squares, 73 move types

def _rc(square):
    """
    Returns row, column
    """
    return square // 8, square % 8


def _flip_sq_vertical(square: int) -> int:
    """
    Rank-only (vertical) flip of a square: rank r -> 7-r, FILE UNCHANGED.
    Involution: _flip_sq_vertical(_flip_sq_vertical(s)) == s.
    """
    return (7 - (square >> 3)) * 8 + (square & 7)


def flip_move_vertical(move: Move) -> Move:
    """
    Rank-only (vertical) flip of a move's squares (file preserved). Promotion,
    castle and en-passant flags are carried through unchanged -- a vertical flip
    maps a promotion to a promotion of the same piece, a castle to a castle, etc.
    Used to move a Black move into / out of the side-to-move canonical frame.
    """
    return Move(
        fromSq=_flip_sq_vertical(move.fromSq),
        toSq=_flip_sq_vertical(move.toSq),
        promotion=move.promotion,
        castle=move.castle,
        enPassant=move.enPassant,
    )


def encodeMovePOV(move: Move, sideToMove: str) -> int:
    """
    Encode a move in the MOVER's canonical frame, matching the rank-only flip
    that model.encoding.encode applies to the input planes.

      white -> identity (board already in absolute = mover frame)
      black -> vertical-flip the squares first, so the mover's pawns advance
               "up" the board and a policy index means the same thing it does
               for white.

    Because the board is only ever flipped on the rank axis, the existing
    absolute encodeMove handles the flipped squares correctly: a vertical flip
    negates the rank component of every queen/knight direction (each of which has
    its negation already in the table) and leaves the file component (dC, used by
    the underpromotion branch) untouched. The result is a bijection on legal
    moves with no collisions -- see the tests.
    """
    if sideToMove == "black":
        move = flip_move_vertical(move)
    return encodeMove(move)


def encodeMove(move: Move) -> int:
    fromR, fromC = _rc(move.fromSq)
    toR, toC = _rc(move.toSq)
    dR, dC = toR - fromR, toC - fromC

    # under promo
    if move.promotion is not None and move.promotion != "Q":
        pieceIdx = UNDERPROMO_PIECES.index(move.promotion)
        dirIdx = UNDERPROMO_DIRS.index(dC)
        moveType = 56 + 8 + pieceIdx * 3 + dirIdx
        return move.fromSq * 73 + moveType
    # knight moves
    if (dR, dC) in KNIGHT_MOVES:
        moveType = 56 + KNIGHT_MOVES.index((dR, dC))
        return move.fromSq * 73 + moveType
    # all other moves
    stepR = (dR > 0) - (dR < 0)
    stepC = (dC > 0) - (dC < 0)
    distance = max(abs(dR), abs(dC))
    dirIdx = DIRECTIONS.index((stepR, stepC))
    moveType = dirIdx * 7 + (distance - 1)
    return move.fromSq * 73 + moveType




def decodeMove(index: int) -> Move:
    """
    From the 64 * 73 moves, to a Move object.
    """
    fromSq = index // 73
    moveType = index % 73
    fromR, fromC = _rc(fromSq)

    if moveType < 56:
        # queen-style slide. NOTE: queen PROMOTIONS live here too -- only
        # under-promotions get their own planes, so a pawn promoting to a queen
        # was encoded as a 1-step forward move onto the back rank.
        dirIdx = moveType // 7
        distance = (moveType % 7) + 1
        dR, dC = DIRECTIONS[dirIdx]
        toR = fromR + dR * distance
        toC = fromC + dC * distance
        toSq = toR * 8 + toC

        # A single forward step landing on the last rank is a queen promotion.
        # "Forward" depends on colour/frame exactly as in the under-promotion
        # branch below (rank 7 -> promoting upward, rank 2 -> promoting downward).
        # Marking it makes encodeMove <-> decodeMove an exact round-trip for queen
        # promotions instead of silently dropping them. (A real queen sliding one
        # square onto the back rank shares this index; in practice decode is only
        # ever matched against the legal-move list, and only a pawn can promote,
        # so the promotion reading is the intended one.)
        promotion = None
        if distance == 1 and ((fromR == 6 and dR == 1) or (fromR == 1 and dR == -1)):
            promotion = "Q"

        return Move(fromSq=fromSq, toSq=toSq, promotion=promotion)

    elif moveType < 64:
        # knight move
        dR, dC = KNIGHT_MOVES[moveType - 56]
        toR, toC = fromR + dR, fromC + dC
        return Move(fromSq=fromSq, toSq=toR * 8 + toC)

    else:
        # underpromotion
        idx = moveType - 64
        pieceIdx = idx // 3
        dirIdx = idx % 3
        piece = UNDERPROMO_PIECES[pieceIdx]
        dC = UNDERPROMO_DIRS[dirIdx]
        # a pawn underpromoting always moves one rank forward; direction
        # depends on colour, so decode forward-rank from the fromSq
        dR = 1 if fromR == 6 else -1   # rank 7 -> white promoting, rank 2 -> black
        toR, toC = fromR + dR, fromC + dC
        return Move(fromSq=fromSq, toSq=toR * 8 + toC, promotion=piece)
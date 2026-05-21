from moves import Move

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
        # queen move
        dirIdx = moveType // 7
        distance = (moveType % 7) + 1
        dR, dC = DIRECTIONS[dirIdx]
        toR = fromR + dR * distance
        toC = fromC + dC * distance
        toSq = toR * 8 + toC

        return Move(fromSq=fromSq, toSq=toSq)

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




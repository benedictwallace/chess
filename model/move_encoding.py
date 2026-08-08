"""
Drop-in optimised replacement for model/move_encoding.py.

encodeMovePOV was 3.0M calls (~9% of self-play wall time) in the profile: once
per legal move, per leaf evaluation. The mapping is a pure function of
(fromSq, toSq, promotion, sideToMove) over a tiny finite domain, so it is
precomputed into flat lists at import and the hot path becomes one list index.

Everything else (DIRECTIONS, KNIGHT_MOVES, decodeMove, NUM_ACTIONS,
flip_move_vertical) is byte-identical to the original; the tables are BUILT by
calling the original reference implementation, so they cannot drift from it.
"""

from engine.moves import Move

DIRECTIONS = [
    (1, 0), (1, 1), (0, 1), (-1, 1),
    (-1, 0), (-1, -1), (0, -1), (1, -1),
]

KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

UNDERPROMO_PIECES = ["N", "B", "R"]
UNDERPROMO_DIRS = [0, -1, 1]

NUM_ACTIONS = 64 * 73

_DIR_TO_IDX = {d: i for i, d in enumerate(DIRECTIONS)}
_KNIGHT_TO_IDX = {m: i for i, m in enumerate(KNIGHT_MOVES)}
_UNDERPROMO_PIECE_IDX = {p: i for i, p in enumerate(UNDERPROMO_PIECES)}
_UNDERPROMO_DIR_IDX = {d: i for i, d in enumerate(UNDERPROMO_DIRS)}


def _rc(square):
    return square // 8, square % 8


def _flip_sq_vertical(square: int) -> int:
    return (7 - (square >> 3)) * 8 + (square & 7)


def flip_move_vertical(move: Move) -> Move:
    return Move(
        fromSq=_flip_sq_vertical(move.fromSq),
        toSq=_flip_sq_vertical(move.toSq),
        promotion=move.promotion,
        castle=move.castle,
        enPassant=move.enPassant,
    )


def _encode_raw(fromSq: int, toSq: int, promotion) -> int:
    """Reference implementation -- unchanged. Used to BUILD the tables below."""
    fromR, fromC = fromSq >> 3, fromSq & 7
    toR, toC = toSq >> 3, toSq & 7
    dR, dC = toR - fromR, toC - fromC

    if promotion is not None and promotion != "Q":
        moveType = 64 + _UNDERPROMO_PIECE_IDX[promotion] * 3 + _UNDERPROMO_DIR_IDX[dC]
        return fromSq * 73 + moveType
    ki = _KNIGHT_TO_IDX.get((dR, dC))
    if ki is not None:
        return fromSq * 73 + 56 + ki
    stepR = (dR > 0) - (dR < 0)
    stepC = (dC > 0) - (dC < 0)
    distance = abs(dR) if abs(dR) > abs(dC) else abs(dC)
    dirIdx = _DIR_TO_IDX[(stepR, stepC)]
    return fromSq * 73 + dirIdx * 7 + (distance - 1)


def encodeMove(move: Move) -> int:
    return _encode_raw(move.fromSq, move.toSq, move.promotion)


# --------------------------------------------------------------------------- #
# precomputed tables
#   _PLAIN_W[frm * 64 + to]  -> index for white, promotion in (None, 'Q')
#   _PLAIN_B[frm * 64 + to]  -> same for black (squares vertically flipped)
#   _UP_W[(frm, to, promo)]  -> index for under-promotions
# Entries for geometrically impossible (frm, to) pairs are None and can never
# be looked up: every caller passes a legal move.
# --------------------------------------------------------------------------- #
def _build():
    plain_w = [None] * 4096
    plain_b = [None] * 4096
    up_w, up_b = {}, {}
    for frm in range(64):
        ffw, ffb = frm, _flip_sq_vertical(frm)
        for to in range(64):
            if frm == to:
                continue
            ttw, ttb = to, _flip_sq_vertical(to)
            try:
                plain_w[frm * 64 + to] = _encode_raw(ffw, ttw, None)
            except KeyError:
                pass
            try:
                plain_b[frm * 64 + to] = _encode_raw(ffb, ttb, None)
            except KeyError:
                pass
            for p in ("N", "B", "R"):
                dcw = (ttw & 7) - (ffw & 7)
                dcb = (ttb & 7) - (ffb & 7)
                if dcw in _UNDERPROMO_DIR_IDX:
                    up_w[(frm, to, p)] = _encode_raw(ffw, ttw, p)
                if dcb in _UNDERPROMO_DIR_IDX:
                    up_b[(frm, to, p)] = _encode_raw(ffb, ttb, p)
    return plain_w, plain_b, up_w, up_b


_PLAIN_W, _PLAIN_B, _UP_W, _UP_B = _build()


def encodeMovePOV(move: Move, sideToMove: str) -> int:
    """One list index on the hot path (promotion is None or 'Q'), one dict
    lookup for the rare under-promotion."""
    promo = move.promotion
    if sideToMove == "black":
        if promo is None or promo == "Q":
            return _PLAIN_B[move.fromSq * 64 + move.toSq]
        return _UP_B[(move.fromSq, move.toSq, promo)]
    if promo is None or promo == "Q":
        return _PLAIN_W[move.fromSq * 64 + move.toSq]
    return _UP_W[(move.fromSq, move.toSq, promo)]


def decodeMove(index: int) -> Move:
    fromSq = index // 73
    moveType = index % 73
    fromR, fromC = _rc(fromSq)

    if moveType < 56:
        dirIdx = moveType // 7
        distance = (moveType % 7) + 1
        dR, dC = DIRECTIONS[dirIdx]
        toR = fromR + dR * distance
        toC = fromC + dC * distance
        toSq = toR * 8 + toC
        promotion = None
        if distance == 1 and ((fromR == 6 and dR == 1) or (fromR == 1 and dR == -1)):
            promotion = "Q"
        return Move(fromSq=fromSq, toSq=toSq, promotion=promotion)

    elif moveType < 64:
        dR, dC = KNIGHT_MOVES[moveType - 56]
        toR, toC = fromR + dR, fromC + dC
        return Move(fromSq=fromSq, toSq=toR * 8 + toC)

    else:
        idx = moveType - 64
        piece = UNDERPROMO_PIECES[idx // 3]
        dC = UNDERPROMO_DIRS[idx % 3]
        dR = 1 if fromR == 6 else -1
        toR, toC = fromR + dR, fromC + dC
        return Move(fromSq=fromSq, toSq=toR * 8 + toC, promotion=piece)

    
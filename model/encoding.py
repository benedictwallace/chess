import numpy as np


# Plane Layout (17 planes):
#   0-5  : current player pieces (pawn, bishop, knight, rook, queen, king)
#   6-11 : opponent pieces       (pawn, bishop, knight, rook, queen, king)
#   12-15: castling rights (own kingside, own queenside, opp kingside, opp queenside)
#   16   : en passant target sq
#
# ORIENTATION: the board is canonicalised to the MOVER's point of view with a
# RANK-ONLY (vertical) flip -- rank r <-> rank 7-r, FILE UNCHANGED. This mirrors
# the position across the central rank so the side to move always looks "up" the
# board from the bottom, exactly as White does. Files are preserved, so the king
# stays on the e-file and the queen on the d-file for both colours. (A 180-degree
# rotation, which also flips files, would wrongly swap them.)
#
# CRITICAL: the move/policy space MUST be canonicalised with the SAME rank-only
# flip, or the input the net sees and the action it is graded on disagree. That
# is done in model.move_encoding.encodeMovePOV, which vertical-flips a Black
# move's squares before encoding.
#
# NO SIDE-TO-MOVE PLANE: because the position is always presented from the mover's
# POV (own pieces in planes 0-5, advancing up-board), a side-to-move plane carries
# no information and is omitted. The payoff is PERFECT colour symmetry: a
# white-to-move position and its colour-swapped vertical mirror now encode to
# byte-identical tensors AND identical legal-move policy indices, so every example
# trains both colours at once. (If you ever re-add it, bump NUM_PLANES here AND
# network.NUM_PLANES together -- they must match the conv stem's in-channels.)

PIECE_ORDER = ["pawn", "bishop", "knight", "rook", "queen", "king"]
NUM_PLANES = 17

def _square_to_rc(square: int, flip: bool) -> tuple[int, int]:
    """
    Convert a 0-63 square index to (row, col) in the plane.
    if flip=True apply a RANK-ONLY (vertical) flip so the mover's side is at the
    bottom. The file is deliberately left UNCHANGED.
    """
    rank = square // 8
    file = square % 8
    if flip:
        rank = 7 - rank
        # file is intentionally NOT flipped (rank-only / vertical mirror)
    return rank, file


def _fill_plane(plane: np.ndarray, bb: int, flip: bool) -> None:
    """
    Set plane[rank][file] = 1 for every set bit in the bitboard.
    """
    while bb:
        sq = (bb & -bb).bit_length() - 1 # lsb index
        rank, file = _square_to_rc(sq, flip)
        plane[rank, file] = 1.0
        bb &= bb - 1

def encode(board) -> np.ndarray:
    """
    Encode the board onto a 17 x 8 x 8 float32 tensor,
    oriented from the perspective of the side to move (rank-only flip for black).
    """

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    us = board.sideToMove # colour
    them = "black" if us =="white" else "white"

    # flip for black so "us" is looking from bottom (rank-only / vertical)
    flip = (us == "black")

    # piece planes
    for i, piece in enumerate(PIECE_ORDER):
        _fill_plane(planes[i], board.bb[us, piece], flip=flip) # 0-5
        _fill_plane(planes[6+i], board.bb[them, piece], flip=flip) # 6-11

    # castling rights, mover's POV (no side-to-move plane: position is already
    # presented from the mover's POV, so it would carry no information)
    if us == "white":
        ownK, ownQ = board.whiteKCastle, board.whiteQCastle
        oppK, oppQ = board.blackKCastle, board.blackQCastle
    else:
        oppK, oppQ = board.whiteKCastle, board.whiteQCastle
        ownK, ownQ = board.blackKCastle, board.blackQCastle

    if ownK: planes[12, :, :] = 1.0
    if ownQ: planes[13, :, :] = 1.0
    if oppK: planes[14, :, :] = 1.0
    if oppQ: planes[15, :, :] = 1.0

    # en passant (square is vertical-flipped for black, same as the pieces)
    if board.enPassantSq >= 0:
        rank, file = _square_to_rc(board.enPassantSq, flip)
        planes[16, rank, file] = 1.0

    return planes
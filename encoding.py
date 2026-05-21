import numpy as np


# Plane Layout:
#   0-5  : current player pieces (pawn, bishop, knight, rook, queen, king)
#   6-11 : oppenents pieces (pawn, bishop, knight, rook, queen, king)
#   12   : side to move (all 1s for white, 0s for black)
#   13-16: castling rights (own kingside, own queenside, opponents kingside, opponents queenside)
#   17   : en passant target sq

PIECE_ORDER = ["pawn", "bishop", "knight", "rook", "queen", "king"]
NUM_PLANES = 18

def _square_to_rc(square: int, flip: bool) -> tuple[int, int]:
    """
    Convert a 0-63 square index to (row, col) in the plane.
    if flip=True rotate 180 deg so mover's side is at the bottom.
    """
    rank = square // 8
    file = square % 8
    if flip:
        rank = 7 - rank
        file = 7 - file
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
    Encode the board onto an 18 x 8 x 8 float32 tensor,
    oriented from perspective of side to move.
    """

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    us = board.sideToMove # colour
    them = "black" if us =="white" else "white"

    # flip for black so "us" is looking from bottom
    flip = (us == "black")

    # piece planes
    for i, piece in enumerate(PIECE_ORDER):
        _fill_plane(planes[i], board.bb[us, piece], flip=flip) # 0-5
        _fill_plane(planes[6+i], board.bb[them, piece], flip=flip) # 6-11

    # store colour
    if us == "white":
        planes[12, :, :] = 1.0

    # castling rights, movers POV

    if us == "white":
        ownK, ownQ = board.whiteKCastle, board.whiteQCastle
        oppK, oppQ = board.blackKCastle, board.blackQCastle
    else:
        oppK, oppQ = board.whiteKCastle, board.whiteQCastle
        ownK, ownQ = board.blackKCastle, board.blackQCastle

    if ownK: planes[13, :, :] = 1.0
    if ownQ: planes[14, :, :] = 1.0
    if oppK: planes[15, :, :] = 1.0
    if oppQ: planes[16, :, :] = 1.0

    # en passant
    if board.enPassantSq >= 0:
        rank, file = _square_to_rc(board.enPassantSq, flip)
        planes[17, rank, file] = 1.0
    
    return planes
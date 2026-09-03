import numpy as np


# Plane Layout (19 planes):
#   0-5  : current player pieces (pawn, bishop, knight, rook, queen, king)
#   6-11 : opponent pieces       (pawn, bishop, knight, rook, queen, king)
#   12-15: castling rights (own kingside, own queenside, opp kingside, opp queenside)
#   16   : en passant target sq
#   17   : halfmove clock / 100 (uniform fill; 1.0 == fifty-move draw imminent)
#   18   : repetition count, (occurrences-1)/2 (uniform fill; 0.5 == this is the
#          2nd occurrence, 1.0 == threefold is one repeat away / already claimed)
#
# Planes 17-18 exist because SEARCH scores repetitions and the fifty-move rule
# as hard terminal draws, but a clock-blind net cannot see either coming: the
# value head was systematically wrong near rule-draws (grinding endgames,
# perpetual-check defenses). Both are env-level counters, not Board state, so
# encode() takes them as arguments -- use encode_env(env) whenever you hold the
# Chess env (every search/self-play/eval path does).
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
NUM_PLANES = 19

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

    Kept as the REFERENCE implementation. _piece_planes below is the one the
    hot path uses; this stays so the two can be diffed.
    """
    while bb:
        sq = (bb & -bb).bit_length() - 1 # lsb index
        rank, file = _square_to_rc(sq, flip)
        plane[rank, file] = 1.0
        bb &= bb - 1


def _piece_planes(bbs, flip: bool) -> np.ndarray:
    """
    All 12 piece bitboards -> (12, 8, 8) float32, in one pass.

    The per-bit Python loop in _fill_plane ran once per SET BIT per plane, at
    every leaf evaluation on every search path -- measured 20.1 us per encode()
    against 6.9 us here (2.9x), for bit-identical output.

    The trick is that a bitboard's little-endian byte layout IS its rank layout:
    byte r holds rank r, and bit f within that byte holds file f. So
    unpackbits(bitorder="little") on the 8 bytes yields an (8, 8) rank-major
    grid directly, with no index arithmetic at all.

    The rank-only (vertical) flip is then just [::-1] on the rank axis --
    files untouched, exactly as the module docstring requires. A 180-degree
    rotation would also mirror files and put the king on the d-file.
    """
    raw = b"".join(int(bb).to_bytes(8, "little") for bb in bbs)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")
    grid = bits.reshape(len(bbs), 8, 8).astype(np.float32)
    return grid[:, ::-1, :] if flip else grid

def encode(board, halfmove_clock: int = 0, repetitions: int = 1) -> np.ndarray:
    """
    Encode the board onto a 19 x 8 x 8 float32 tensor, oriented from the
    perspective of the side to move (rank-only flip for black).

    halfmove_clock : plies since the last capture/pawn move (fifty-move rule).
    repetitions    : how many times this exact position has occurred INCLUDING
                     the current occurrence (the Chess env's counts value, so
                     1 == first time). Defaults (0, 1) reproduce a "fresh"
                     position; prefer encode_env(env) which fills both.
    """

    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)

    us = board.sideToMove # colour
    them = "black" if us =="white" else "white"

    # flip for black so "us" is looking from bottom (rank-only / vertical)
    flip = (us == "black")

    # piece planes 0-11: mover's pieces then the opponent's, one vectorised pass
    planes[:12] = _piece_planes(
        [board.bb[us, p] for p in PIECE_ORDER]
        + [board.bb[them, p] for p in PIECE_ORDER], flip)

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

    # rule-draw counters (uniform planes, like castling rights). Clamped so the
    # encoding stays in [0, 1] even for malformed inputs.
    planes[17, :, :] = min(max(int(halfmove_clock), 0), 100) / 100.0
    planes[18, :, :] = min(max(int(repetitions) - 1, 0), 2) / 2.0

    return planes


def repetition_count(env) -> int:
    """Occurrences of the env's CURRENT position (>= 1), from its threefold
    bookkeeping."""
    return env.counts.get(env.board.stateKey(), 1)


def encode_env(env) -> np.ndarray:
    """encode() with the halfmove clock and repetition count read off a Chess
    env. This is the canonical entry point for every search / self-play / eval
    path -- calling encode(board) directly zeroes the rule-draw planes and the
    net will misjudge positions near a repetition or fifty-move draw."""
    return encode(env.board, env.halfmove_clock, repetition_count(env))


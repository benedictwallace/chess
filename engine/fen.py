"""
FEN <-> project board/env conversion.

The engine never needed arbitrary positions before (games always start from the
standard array), so this adds the missing piece: load any position from a FEN
string into a Board / Chess, and serialise back for display.

Square convention matches the rest of the engine (little-endian rank-file):
square = rank * 8 + file, with square 0 = a1, 63 = h8.
"""

from engine.board import Board
from engine.gameEnv import Chess

_FEN_TO_PIECE = {
    "p": "pawn", "n": "knight", "b": "bishop",
    "r": "rook", "q": "queen", "k": "king",
}
_PIECE_TO_FEN = {
    ("white", "pawn"): "P", ("white", "knight"): "N", ("white", "bishop"): "B",
    ("white", "rook"): "R", ("white", "queen"): "Q", ("white", "king"): "K",
    ("black", "pawn"): "p", ("black", "knight"): "n", ("black", "bishop"): "b",
    ("black", "rook"): "r", ("black", "queen"): "q", ("black", "king"): "k",
}


def square_to_alg(sq: int) -> str:
    """0..63 -> 'a1'..'h8'."""
    return "abcdefgh"[sq % 8] + str(sq // 8 + 1)


def alg_to_square(s: str) -> int:
    """'e3' -> 20."""
    file = ord(s[0].lower()) - ord("a")
    rank = int(s[1]) - 1
    return rank * 8 + file


def board_from_fen(fen: str):
    """
    Parse a FEN into a Board. Returns (board, halfmove_clock).
    Only the piece placement, side, castling, and en-passant fields affect the
    board; the halfmove clock is returned separately (the Chess env owns it),
    and the fullmove number is ignored (the engine doesn't track it).
    """
    parts = fen.split()
    if len(parts) < 4:
        raise ValueError(f"FEN needs at least 4 fields, got {len(parts)}: {fen!r}")
    placement, side, castling, ep = parts[0], parts[1], parts[2], parts[3]
    halfmove = int(parts[4]) if len(parts) >= 5 else 0

    board = Board.__new__(Board)              # skip __init__ (no start array)
    board.bb = {(c, p): 0
                for c in ("white", "black")
                for p in ("pawn", "knight", "bishop", "rook", "queen", "king")}

    ranks = placement.split("/")
    if len(ranks) != 8:
        raise ValueError(f"FEN placement needs 8 ranks, got {len(ranks)}")
    for r_fen, row in enumerate(ranks):       # r_fen=0 is rank 8 (top)
        board_rank = 7 - r_fen
        file = 0
        for ch in row:
            if ch.isdigit():
                file += int(ch)
            else:
                colour = "white" if ch.isupper() else "black"
                piece = _FEN_TO_PIECE[ch.lower()]
                board.bb[(colour, piece)] |= (1 << (board_rank * 8 + file))
                file += 1
        if file != 8:
            raise ValueError(f"FEN rank {8 - r_fen} does not sum to 8 files: {row!r}")

    board.whiteKCastle = "K" in castling
    board.whiteQCastle = "Q" in castling
    board.blackKCastle = "k" in castling
    board.blackQCastle = "q" in castling
    board.enPassantSq = -1 if ep == "-" else alg_to_square(ep)
    board.sideToMove = "white" if side == "w" else "black"
    board.history = []
    board.updatePieces()                      # fills white/black/allPieces from bb
    return board, halfmove


def env_from_fen(fen: str) -> Chess:
    """Build a Chess env positioned at `fen` (counts/clock initialised correctly)."""
    board, halfmove = board_from_fen(fen)
    env = Chess.__new__(Chess)
    env.board = board
    env.counts = {board.stateKey(): 1}
    env.halfmove_clock = halfmove
    return env


def board_to_fen(board, halfmove: int = 0, fullmove: int = 1) -> str:
    """Serialise a Board back to FEN (for echoing positions)."""
    rows = []
    for board_rank in range(7, -1, -1):       # rank 8 -> rank 1
        row, empty = "", 0
        for file in range(8):
            sq = board_rank * 8 + file
            here = None
            for key in _PIECE_TO_FEN:
                if (board.bb[key] >> sq) & 1:
                    here = _PIECE_TO_FEN[key]
                    break
            if here is None:
                empty += 1
            else:
                if empty:
                    row += str(empty); empty = 0
                row += here
        if empty:
            row += str(empty)
        rows.append(row)
    placement = "/".join(rows)

    side = "w" if board.sideToMove == "white" else "b"
    castling = ("K" if board.whiteKCastle else "") + ("Q" if board.whiteQCastle else "") \
             + ("k" if board.blackKCastle else "") + ("q" if board.blackQCastle else "")
    castling = castling or "-"
    ep = "-" if board.enPassantSq < 0 else square_to_alg(board.enPassantSq)
    return f"{placement} {side} {castling} {ep} {halfmove} {fullmove}"
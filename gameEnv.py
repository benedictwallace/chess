from moves import Move
from board import Board

State = tuple

# material values for ply-cap adjudication
_ADJ_VALUES = {"pawn": 1, "knight": 3, "bishop": 3, "rook": 5, "queen": 9, "king": 0}


class Chess:
    def __init__(self):
        self.board = Board()
        # threefold repetition: times each position (stateKey) has occurred
        self.counts = {self.board.stateKey(): 1}
        # 50-move rule: plies since the last capture or pawn move
        self.halfmove_clock = 0

    def clone(self):
        new = Chess.__new__(Chess)
        new.board = self.board.clone()
        new.counts = dict(self.counts)
        new.halfmove_clock = self.halfmove_clock
        return new

    def reset(self) -> State:
        """Start new game. Returns the initial state."""
        self.board = Board()
        self.counts = {self.board.stateKey(): 1}
        self.halfmove_clock = 0
        return self.board.stateKey()

    def step(self, move: Move) -> State:
        """
        Apply a move and update repetition / 50-move bookkeeping. Returns the
        new state key.

        NOTE: this deliberately does NOT compute terminality or a reward.
        During MCTS selection `step` is called for every node on the descent
        path, and the old checkmate/stalemate probe ran a full legal-move
        generation on each of those -- the single biggest self-play cost --
        only for the result to be discarded (every caller ignores step's return
        and derives terminality via isTerminal()/result() where it actually
        matters). Returning only the key (rather than a stale done=False) means
        any future caller expecting the old 3-tuple fails loudly instead of
        silently reading a wrong flag.
        """
        mover = self.board.sideToMove

        # 50-move clock inputs, read BEFORE the move mutates the board (O(1) bitboard tests)
        is_pawn = bool((self.board.bb[mover, "pawn"] >> move.fromSq) & 1)
        is_capture = bool((self.board.allPieces >> move.toSq) & 1) or move.enPassant

        self.board.makeMove(move)

        if is_pawn or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        key = self.board.stateKey()
        self.counts[key] = self.counts.get(key, 0) + 1
        return key

    def legalMoves(self) -> list[Move]:
        return self.board.legalMoves(self.board.sideToMove)

    def isRepetition(self) -> bool:
        """True if the current position has occurred three times (threefold)."""
        return self.counts.get(self.board.stateKey(), 0) >= 3

    def isFiftyMove(self) -> bool:
        """True if 50 full moves (100 plies) have passed with no capture or pawn move."""
        return self.halfmove_clock >= 100

    def _terminal_for_key(self, key) -> bool:
        # cheap draw-by-rule checks first; they short-circuit the expensive
        # checkmate/stalemate (which recompute legal moves)
        if self.counts.get(key, 0) >= 3:        # threefold repetition
            return True
        if self.halfmove_clock >= 100:          # fifty-move rule
            return True
        side = self.board.sideToMove
        return self.board.checkMate(side) or self.board.staleMate(side)

    def isTerminal(self) -> bool:
        return self._terminal_for_key(self.board.stateKey())

    def result(self) -> float | None:
        """
        +1 white win, -1 black win, 0 draw (stalemate / threefold / fifty-move),
        None if not finished. Checkmate takes priority over a draw.
        """
        if not self.isTerminal():
            return None
        side = self.board.sideToMove
        if self.board.checkMate(side):
            return -1.0 if side == "white" else 1.0
        return 0.0   # stalemate, threefold, or fifty-move

    def adjudicate(self, margin: float = 5.0) -> float:
        """
        Score an unfinished (ply-cap) position by material, from white's POV:
        +1 if white leads by >= margin, -1 if black does, else 0 (draw).
        margin=5 means 'up at least a rook'. Used only when a game hits the
        hard ply cap, so reaching a won position still earns a win signal even
        if the engine cannot deliver mate in time.
        """
        white = sum(_ADJ_VALUES[p] * bin(self.board.bb["white", p]).count("1")
                    for p in _ADJ_VALUES)
        black = sum(_ADJ_VALUES[p] * bin(self.board.bb["black", p]).count("1")
                    for p in _ADJ_VALUES)
        diff = white - black
        if diff >= margin:
            return 1.0
        if diff <= -margin:
            return -1.0
        return 0.0
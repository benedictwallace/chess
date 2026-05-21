from moves import Move
from board import Board

State = tuple

class Chess:
    def __init__(self):
        self.board = Board()

    def clone(self):
        new = Chess.__new__(Chess)
        new.board = self.board.clone()
        return new
    
    def reset(self) -> State:
        """
        Start new game.
        Returns:
            initial state
        """
        self.board = Board()
        return self.board.stateKey()

    def step(self, move: Move) -> tuple[State, float, bool]:
        """
        Play a move, reward from perspective of player just moved, 1 if checkmate, 0 otherwise incl. draws.
        Returns:
            (next state, reward, done)
        """
        mover = self.board.sideToMove
        self.board.makeMove(move)
        
        done = self.isTerminal()
        if done:
            opponent = "black" if mover == "white" else "white"
            if self.board.checkMate(opponent):
                reward = 1.0
            else:
                reward = 0.0
        else:
            reward = 0.0

        return self.board.stateKey(), reward, done
    

        
    def legalMoves(self) -> list[Move]:
        return self.board.legalMoves(self.board.sideToMove)
    

    def isTerminal(self) -> bool:
        side = self.board.sideToMove
        return self.board.checkMate(side) or self.board.staleMate(side)
    
    def result(self) -> float | None:
        """
        +1 -> white win
        -1 -> black win
        0 draw
        None if not finished.
        """
        if not self.isTerminal():
            return None
        side = self.board.sideToMove
        if self.board.checkMate(side):
            return -1.0 if side == "white" else 1.0
        return 0.0 

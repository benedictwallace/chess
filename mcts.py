
import math
import random
from gameEnv import Chess



class Node:
    def __init__(self, parent=None, move=None):
        self.parent = parent
        self.move = move
        self.children = []
        self.legalMoves = None
        self.untriedMoves = None
        self.visits = 0
        self.value = 0.0
        self.moverSign = 0
        self.terminal = False  

    def isLeaf(self) -> bool:
        return self.untriedMoves is not None and len(self.untriedMoves) > 0
    
    def isFullExpanded(self) -> bool:
        return self.untriedMoves is not None and len(self.untriedMoves) == 0

def makeNode(env, parent=None, move=None, moverSign=0):
    node = Node(parent=parent, move=move)
    node.moverSign = moverSign
    moves = env.legalMoves()
    node.legalMoves = moves
    node.untriedMoves = list(moves)
    node.terminal = (len(moves) == 0)
    return node

def ucb(child, parent, c=1.4):
    """
    confidence upper bound
    """
    if child.visits == 0:
        return float("inf")
    
    exploit = child.value / child.visits
    explore = c * math.sqrt(math.log(parent.visits) / child.visits)
    return exploit + explore

def selectChild(node):
    """
    Pick the child with the highest UCB score.
    """
    return max(node.children, key=lambda c: ucb(c, node))

def rollout(env, maxPlies=300) -> float:
    """
    Play random moves until terminal. Return result from white's POV.
    """
    plies = 0
    while plies < maxPlies:
        moves = env.legalMoves()
        if not moves:
            break
        move = random.choice(moves)
        env.step(move)
        plies += 1

    r = env.result()
    return r if r is not None else 0.0

def backprop(node, result_white_pov):
    """
    Walk back up the tree, updating visits and value.
    Each node's value is from the perspective of the player who moved
    into that node i.e. the parent's side-to-move.
    """
    while node is not None:
        node.visits += 1

        node.value += result_white_pov * node.moverSign
        node = node.parent


def mctsSearch(rootEnv, iterations=1000):
    root = makeNode(rootEnv)

    for _ in range(iterations):
        node = root
        env = rootEnv.clone()

        # 1. SELECTION: descend through fully-expanded nodes
        while node.isFullExpanded() and not node.terminal:
            node = selectChild(node)
            env.step(node.move)

        # 2. EXPANSION: if non-terminal and has untried moves, expand one
        if not node.terminal and node.untriedMoves:

            move = node.untriedMoves.pop()
            mover = env.board.sideToMove
            env.step(move)
            child = makeNode(env, parent=node, move=move, moverSign = 1 if mover == "white" else -1)

            node.children.append(child)
            node = child

        # 3. ROLLOUT: random play to end
        result = rollout(env.clone())   # white's POV

        # 4. BACKPROP
        backprop(node, result)

    if not root.children:
        return None
    # Pick the move from root with the highest visit count
    bestChild = max(root.children, key=lambda c: c.visits)
    return bestChild.move


if __name__ == "__main__":
    env = Chess()
    env.reset()

    # Run MCTS for a single move
    move = mctsSearch(env, iterations=500)
    print(f"MCTS chose: {move}")
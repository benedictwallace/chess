import math
import torch
from encoding import encode
from move_encoding import encodeMove, NUM_ACTIONS


def evaluate(net, env):
    """Run the network on the current position.
    Returns (priors, value):
        priors: dict {Move: probability} over LEGAL moves only
        value:  float in [-1, 1], from the mover's perspective
    """
    board = env.board
    legal = env.legalMoves()

    if not legal:
        return {}, 0.0 # terminal

    planes = encode(board)
    x = torch.from_numpy(planes).unsqueeze(0) # (1, 18, 8, 8)

    net.eval()
    with torch.no_grad():
        policy_logits, value = net(x)

    logits = policy_logits[0] # (4672,)

    # Mask: keep only legal-move logits, softmax on them
    legal_indices = [encodeMove(m) for m in legal]
    legal_logits = torch.tensor([logits[i] for i in legal_indices])
    probs = torch.softmax(legal_logits, dim=0)

    priors = {m: probs[k].item() for k, m in enumerate(legal)}

    return priors, value.item()

class Node:
    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.moverSign = 0
        self.terminal = False
        self.expanded = False 

def puctScore(child, parent, c=1.5):
    """
    PUCT: exploitation + prior-weighted exploration.
    """
    if child.visits == 0:
        q = 0.0
    else:
        q = child.value / child.visits
    u = c * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return q + u



def search(rootEnv, net, iterations=400, c=1.5):
    root = Node()
    root.moverSign = 0

    for _ in range(iterations):
        node = root
        env = rootEnv.clone()
        path = [node]

        # 1. SELECTION: descend via PUCT until an unexpanded or terminal node
        while node.expanded and not node.terminal:
            node = max(node.children, key=lambda ch: puctScore(ch, node, c))
            env.step(node.move)
            path.append(node)

        # 2. EXPANSION + EVALUATION
        if node.terminal:
            # use the true game result, not the network
            r = env.result()
            leaf_value_white_pov = r if r is not None else 0.0
        else:
            priors, value = evaluate(net, env)
            leaf_value_white_pov = value if env.board.sideToMove == "white" else -value

            mover = env.board.sideToMove
            for m, p in priors.items():
                child = Node(parent=node, move=m, prior=p)
                child.moverSign = 1 if mover == "white" else -1
                child.terminal = False  # determined lazily when first selected
                node.children.append(child)
            node.expanded = True

        # 3. BACKPROP
        for n in path:
            n.visits += 1
            n.value += leaf_value_white_pov * n.moverSign

    if not root.children:
        return None
    best = max(root.children, key=lambda ch: ch.visits)
    return best.move
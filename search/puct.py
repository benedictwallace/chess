import math

import numpy as np

from model.encoding import encode_env
from model.move_encoding import encodeMove, encodeMovePOV, NUM_ACTIONS
from engine.moves import Move

def _add_dirichlet_noise(root, alpha=0.3, frac=0.25, rng=None):
    """
    Mix Dirichlet noise into priors
        p' = (1 - frac) * p + frac * noise
    applied after root expanded.
    """
    if not root.children:
        return
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.dirichlet([alpha] * len(root.children))
    for child, n in zip(root.children, noise):
        child.prior = (1 - frac) * child.prior + frac*n


def evaluate(net, env, legal=None):
    """Run the network on the current position.
    Supports standard PyTorch Modules or RemoteEvaluator proxies interchangeably.
    Returns (priors, value):
        priors: dict {Move: probability} over LEGAL moves only
        value:  float in [-1, 1], from the mover's perspective
    """
    import torch   # lazy: keeps this module torch-free to IMPORT, so the
                   # torch-free consumers (score_elo_batched's core runner,
                   # probe/robustness fakes) that pull node_fpu_q from here
                   # stay unit-testable on a machine without torch.

    board = env.board
    if legal is None:
        legal = env.legalMoves()
    if not legal:
        return {}, 0.0

    planes = encode_env(env)   # includes the halfmove-clock / repetition planes

    # Intercept evaluation if utilizing centralized process batching
    if hasattr(net, "is_proxy") and net.is_proxy:
        policy_logits_np, value = net.evaluate_remote(planes)
        logits = torch.from_numpy(policy_logits_np)
    else:
        # Standard structural fallback execution path
        x = torch.from_numpy(planes).unsqueeze(0) # (1, 19, 8, 8)
        x = x.to(next(net.parameters()).device)
        net.eval()
        with torch.no_grad():
            policy_logits, value_tensor = net(x)
        logits = policy_logits.squeeze(0).cpu()
        value = float(value_tensor.item())

    legal_indices = [encodeMovePOV(m, board.sideToMove) for m in legal]

    # Advanced vectorized advanced indexing optimization
    legal_logits = logits[legal_indices]
    probs = torch.softmax(legal_logits, dim=0)

    priors = {m: float(probs[i].item()) for i, m in enumerate(legal)}
    return priors, value


class Node:
    __slots__ = ("parent", "move", "prior", "children",
                 "visits", "value", "moverSign", "terminal", "expanded")

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


def node_fpu_q(node, fpu_reduction):
    """First-play urgency: the Q an UNVISITED child of `node` is assumed to
    have, from the perspective of the player choosing at `node`.

    Plain PUCT's q=0 for unvisited children is a hidden bias: when the chooser
    is losing (parent Q < 0) every unexplored move looks better than reality
    and simulations get sprayed across refuted alternatives. Instead, assume an
    untried move is slightly WORSE (by fpu_reduction) than the running value of
    the moves already explored from this node (LC0/KataGo-style FPU).

    node.value is accumulated from the perspective of the player who moved
    INTO the node -- the chooser's opponent -- so for internal nodes the
    chooser-POV average is just its negation. Root nodes carry moverSign == 0
    (their .value stays 0 / goes stale under subtree reuse), so the root falls
    back to a visit-weighted average over its children, whose values are
    already chooser-POV. That O(children) pass runs only at the root.
    """
    if node.moverSign != 0 and node.visits > 0:
        return -(node.value / node.visits) - fpu_reduction
    vsum = 0.0
    nsum = 0
    for ch in node.children:
        if ch.visits:
            vsum += ch.value
            nsum += ch.visits
    return (vsum / nsum - fpu_reduction) if nsum else 0.0


def puctScore(child, parent, c=1.5, fpu_q=0.0):
    """
    PUCT: exploitation + prior-weighted exploration. `fpu_q` is the assumed Q
    for an unvisited child (see node_fpu_q); pass 0.0 for the legacy behavior.
    """
    if child.visits == 0:
        q = fpu_q
    else:
        q = child.value / child.visits
    u = c * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return q + u


def search(rootEnv, net, iterations=400, c=1.5, add_noise = True, dirichlet_alpha=0.3, 
               noise_frac=0.25, fpu_reduction=0.25,
               rng=None) -> tuple[Node, dict[Move, int]]:
    """
    Executes MCTS iteration steps. fpu_reduction: first-play-urgency penalty
    for unvisited children (see node_fpu_q); 0 restores the legacy q=0 init.
    rng: optional np.random.Generator for the Dirichlet noise -- pass a seeded
    one for reproducible searches (default: fresh nondeterministic entropy).
    """
    root = Node()
    root.moverSign = 0

    for _ in range(iterations):
        node = root
        env = rootEnv.clone()
        path = [node]

        # 1. SELECTION: descend via PUCT until an unexpanded or terminal node
        while node.expanded and not node.terminal and node.children:
            fpu_q = node_fpu_q(node, fpu_reduction)   # once per node, not per child
            node = max(node.children,
                       key=lambda ch: puctScore(ch, node, c, fpu_q))
            env.step(node.move)
            path.append(node)

        # 2. EXPANSION + EVALUATION
        if node.terminal:
            r = env.result()
            leaf_value_white_pov = r if r is not None else 0.0
        else:
            legal = env.legalMoves()
            if not legal:
                node.terminal = True
                r = env.result()
                leaf_value_white_pov = r if r is not None else 0.0
            elif env.isRepetition() or env.isFiftyMove():
                node.terminal = True
                leaf_value_white_pov = 0.0
            else:
                priors, value = evaluate(net, env, legal)
                leaf_value_white_pov = value if env.board.sideToMove == "white" else -value
                mover = env.board.sideToMove
                for m, p in priors.items():
                    child = Node(parent=node, move=m, prior=p)
                    child.moverSign = 1 if mover == "white" else -1
                    node.children.append(child)
                node.expanded = True
                if add_noise and node is root:
                    _add_dirichlet_noise(root, dirichlet_alpha, noise_frac, rng)

        # 3. BACKPROPATION
        for n in path:
            n.visits += 1
            n.value += leaf_value_white_pov * n.moverSign

    visit_counts = {ch.move: ch.visits for ch in root.children}
    return root, visit_counts


def select_move(visit_counts, temp=1.0, rng=None):
    """
    Choose move from root visit counts. rng: optional seeded Generator for
    reproducible temperature sampling.
    """
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)

    if temp <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    
    logits = counts ** (1.0 / temp)
    probs = logits / logits.sum()
    if rng is None:
        rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=probs)]


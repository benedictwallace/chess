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


# --------------------------------------------------------------------------- #
# Gumbel-AlphaZero root move selection:  argmax  logits + sigma(q_hat)
# (Danihelka et al. 2022, "Policy Improvement by Planning with Gumbel")
# --------------------------------------------------------------------------- #
def gumbel_scores(root, c_visit=50.0, c_scale=1.0, min_visits=1,
                  include_unvisited=False):
    """Per-child Gumbel selection scores at a finished root:

        score(a) = log prior(a) + (c_visit + max_b N(b)) * c_scale * q_hat(a)

    log prior(a) is the policy logit up to a softmax-invariant constant (the
    root stores renormalised priors, possibly post-Dirichlet -- the shift
    cancels inside any argmax/softmax over these scores). q_hat(a) is the
    chooser-POV search Q, value/visits. The (c_visit + max_N) factor is the
    paper's monotone transform sigma: with a small search the prior dominates,
    with a large one the Qs do -- visit counts appear ONLY as this trust
    scale, never as the selection statistic.

    CANDIDATES: only children with visits >= min_visits are scored. The
    paper's guarantee assumes candidate Qs of comparable variance (its
    sequential halving arranges that; our forced-root-visit floor is the
    analogue), and sigma multiplies Q noise by hundreds in logit space, so a
    lucky 2-visit Q must not compete with a 400-visit one. Pass the forced
    floor as min_visits on forced roots; on unforced roots pick a small guard
    (callers here use ~budget/100).

    include_unvisited=True additionally scores every below-floor child with a
    COMPLETED Q -- the visit-weighted chooser-POV root value, this module's
    stand-in for the paper's v_mix interpolation. That makes softmax(scores)
    the paper's improved policy pi' over ALL actions (useful as a policy
    target); for move selection leave it False, a completed Q is a
    placeholder, not evidence the move is good.

    Returns {Move: score}; empty if the root has no visited children.
    Duck-typed: any node with .children / .visits / .value / .prior / .move.
    """
    kids = root.children
    if not kids:
        return {}
    max_n = max(ch.visits for ch in kids)
    if max_n == 0:
        return {}
    scale = (c_visit + max_n) * c_scale
    floor = max(1, int(min_visits))
    v_mix = None
    if include_unvisited:
        nsum = sum(ch.visits for ch in kids)
        v_mix = sum(ch.value for ch in kids) / nsum   # chooser-POV root value
    scores = {}
    for ch in kids:
        if ch.visits >= floor:
            q = ch.value / ch.visits
        elif include_unvisited:
            q = v_mix
        else:
            continue
        scores[ch.move] = math.log(max(ch.prior, 1e-12)) + scale * q
    return scores


def select_move_gumbel(root, temp=0.0, rng=None, c_visit=50.0, c_scale=1.0,
                       min_visits=1, include_unvisited=False,
                       fallback_counts=None):
    """Choose the root move by the Gumbel score logits + sigma(q_hat).

    temp <= 0: argmax over the candidate scores (the paper's acting rule).
    temp > 0 : sample from softmax(scores / temp) -- a tempered version of the
    improved policy pi', the drop-in replacement for opening-phase
    visit-temperature sampling.

    If fewer than TWO children qualify (min_visits too high for this tree --
    e.g. an unforced root that PUCT starved), there is nothing for the Q term
    to compare, so selection falls back to `fallback_counts` via
    select_move(visit_counts, temp) when given, else to the single qualifier /
    most-visited child. rng: optional seeded Generator (reproducible runs).
    """
    scores = gumbel_scores(root, c_visit=c_visit, c_scale=c_scale,
                           min_visits=min_visits,
                           include_unvisited=include_unvisited)
    if len(scores) < 2:
        if fallback_counts:
            return select_move(fallback_counts, temp, rng)
        if scores:
            return next(iter(scores))
        visited = [ch for ch in root.children if ch.visits > 0]
        return max(visited, key=lambda ch: ch.visits).move if visited else None

    moves = list(scores.keys())
    s = np.array([scores[m] for m in moves], dtype=np.float64)
    if temp <= 1e-6:
        return moves[int(s.argmax())]
    s = s / temp
    s -= s.max()
    p = np.exp(s)
    p /= p.sum()
    if rng is None:
        rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=p)]


"""
Single-process, leaf-batched self-play.

This replaces the per-eval cross-process round trip in self_play_parallel.py.
The old design had each worker block on its pipe after submitting ONE position,
so at most `workers` evaluations were ever in flight and the GPU saw batches of
size <= workers (tiny) while pickling an 8.7 KB array per evaluation. Profiling
showed the system running at a few percent of what the engine alone can feed --
the binding constraint was that synchronous one-eval-at-a-time path, not move
generation or the GPU.

Here we instead run many games CONCURRENTLY in one process and batch their leaf
evaluations together. Every "round" advances each live game by exactly one MCTS
simulation; the leaves that need the network are collected and evaluated in a
SINGLE forward pass (batch ~= number of concurrent games), and the GPU stays
busy with large batches. There is no IPC and nothing is pickled.

Within a single game this is identical in semantics to search.puct.search: one
pending leaf per tree per round (so no virtual loss is needed), `iterations`
simulations per move, Dirichlet noise at the root, the same PUCT formula, the
same value-sign / backprop convention, and the same early-adjudication and
ply-cap handling as training.self_play.play_game. Examples are the same
(planes, policy_target, value_target) 3-tuples.

A small within-phase evaluation cache (keyed by the exact position) skips the
network for transpositions and repetitions. It stores the legal-move priors and
value, NOT raw logits, so each entry is a few KB. It is rebuilt every call
because the network changes between training iterations -- caching across
iterations would serve stale outputs.
"""

import math
import numpy as np

from engine.gameEnv import Chess
from model.encoding import encode
from model.move_encoding import encodeMovePOV, NUM_ACTIONS


# --------------------------------------------------------------------------- #
# small pure-numpy helpers (kept local so this module needs no torch to import;
# the torch evaluator below imports torch lazily)
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _policy_target(visit_counts, sideToMove):
    target = np.zeros(NUM_ACTIONS, dtype=np.float32)
    total = sum(visit_counts.values())
    if total == 0:
        return target
    for move, count in visit_counts.items():
        target[encodeMovePOV(move, sideToMove)] = count / total
    return target


def _position_value_white(root, mover_sign):
    tot = sum(c.visits for c in root.children if c.visits > 0)
    if tot == 0:
        return 0.0
    v_mover = sum(c.value for c in root.children if c.visits > 0) / tot
    return v_mover * mover_sign


def select_move(visit_counts, temp=1.0):
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temp <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    logits = counts ** (1.0 / temp)
    probs = logits / logits.sum()
    rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=probs)]


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


def _puct_score(child, parent, c):
    q = 0.0 if child.visits == 0 else child.value / child.visits
    u = c * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return q + u


def _add_dirichlet_noise(root, alpha, frac):
    if not root.children:
        return
    rng = np.random.default_rng()
    noise = rng.dirichlet([alpha] * len(root.children))
    for child, n in zip(root.children, noise):
        child.prior = (1 - frac) * child.prior + frac * n


def _expand(node, priors, mover, add_noise, is_root,
            dirichlet_alpha, noise_frac):
    """Attach children to `node` from a {Move: prior} dict."""
    sign = 1 if mover == "white" else -1
    for m, p in priors.items():
        child = Node(parent=node, move=m, prior=p)
        child.moverSign = sign
        node.children.append(child)
    node.expanded = True
    if add_noise and is_root:
        _add_dirichlet_noise(node, dirichlet_alpha, noise_frac)


def _backprop(path, leaf_value_white):
    for n in path:
        n.visits += 1
        n.value += leaf_value_white * n.moverSign


# --------------------------------------------------------------------------- #
# per-game state
# --------------------------------------------------------------------------- #
class _GameState:
    __slots__ = ("env", "root", "ply", "adj_streak", "early_result",
                 "sims_done", "done", "history")

    def __init__(self):
        self.env = Chess()
        self.env.reset()
        self.root = Node()
        self.root.moverSign = 0
        self.ply = 0
        self.adj_streak = 0
        self.early_result = None
        self.sims_done = 0
        self.done = False
        self.history = []   # (planes, policy_target, mover_sign, v_white)


# --------------------------------------------------------------------------- #
# core batched self-play loop (eval_fn injected -> unit-testable without torch)
# --------------------------------------------------------------------------- #
def run_selfplay(eval_fn, num_games, *, iterations=400, concurrency=64,
                 max_plies=200, temp_moves=30, c=1.5,
                 add_noise=True, dirichlet_alpha=0.3, noise_frac=0.25,
                 adj_margin=5.0, adj_plies=20,
                 use_cache=True, cache_cap=200_000,
                 verbose=True):
    """
    Play `num_games` games, keeping up to `concurrency` of them running at once
    and batching their leaf evaluations.

    eval_fn(planes_list) -> (logits, values)
        planes_list : list of (17,8,8) float32 arrays
        logits      : array-like [B, NUM_ACTIONS]   (mover-POV policy logits)
        values      : array-like [B]                (mover-POV value in [-1,1])

    Returns a flat list of (planes, policy_target, value_target) examples.
    """
    concurrency = max(1, min(concurrency, num_games))
    cache = {} if use_cache else None

    active = [_GameState() for _ in range(concurrency)]
    started = len(active)
    finished = 0
    all_examples = []

    def finalize(g):
        nonlocal finished
        if g.early_result is not None:
            rwp = g.early_result
        else:
            r = g.env.result()
            rwp = g.env.adjudicate() if r is None else r
        out = []
        for (planes, policy_target, mover_sign, _v) in g.history:
            out.append((planes, policy_target, np.float32(rwp * mover_sign)))
        finished += 1
        if verbose:
            print(f"  game {finished}/{num_games}: {len(out)} positions "
                  f"(plies {g.ply}, result {rwp:+.0f}, total {len(all_examples)+len(out)})",
                  flush=True)
        return out

    def do_move(g):
        root, env = g.root, g.env
        visit_counts = {ch.move: ch.visits for ch in root.children}
        if not visit_counts:               # terminal at root (no legal moves)
            g.done = True
            return
        mover = env.board.sideToMove
        mover_sign = 1 if mover == "white" else -1
        planes = encode(env.board)
        policy_target = _policy_target(visit_counts, mover)
        v_white = _position_value_white(root, mover_sign)
        g.history.append((planes, policy_target, mover_sign, v_white))

        temp = 1.0 if g.ply < temp_moves else 0.0
        move = select_move(visit_counts, temp)
        env.step(move)
        g.ply += 1

        if env.isTerminal() or g.ply >= max_plies:
            g.done = True
            return
        if adj_plies > 0:
            diff = env.material_diff()
            if abs(diff) >= adj_margin:
                g.adj_streak += 1
                if g.adj_streak >= adj_plies:
                    g.early_result = 1.0 if diff > 0 else -1.0
                    g.done = True
                    return
            else:
                g.adj_streak = 0

    while active:
        # ---- one simulation per game; collect leaves needing the network ----
        batch_planes = []
        batch_meta = []
        for g in active:
            node = g.root
            env = g.env.clone()
            path = [node]
            while node.expanded and not node.terminal and node.children:
                node = max(node.children, key=lambda ch: _puct_score(ch, node, c))
                env.step(node.move)
                path.append(node)

            if node.terminal:
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                g.sims_done += 1
                continue

            legal = env.legalMoves()
            if not legal:
                node.terminal = True
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                g.sims_done += 1
                continue
            if env.isRepetition() or env.isFiftyMove():
                node.terminal = True
                _backprop(path, 0.0)
                g.sims_done += 1
                continue

            mover = env.board.sideToMove
            if cache is not None:
                key = env.board.stateKey()
                hit = cache.get(key)
                if hit is not None:
                    priors, value = hit
                    _expand(node, priors, mover, add_noise, node is g.root,
                            dirichlet_alpha, noise_frac)
                    _backprop(path, value if mover == "white" else -value)
                    g.sims_done += 1
                    continue

            batch_planes.append(encode(env.board))
            batch_meta.append((g, node, env, legal, path, mover))

        # ---- single batched network forward ----
        if batch_planes:
            logits_b, values_b = eval_fn(batch_planes)
            for (g, node, env, legal, path, mover), logits, value in zip(
                    batch_meta, logits_b, values_b):
                idxs = [encodeMovePOV(m, mover) for m in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {m: float(p) for m, p in zip(legal, probs)}
                value = float(value)
                if cache is not None and len(cache) < cache_cap:
                    cache[env.board.stateKey()] = (priors, value)
                _expand(node, priors, mover, add_noise, node is g.root,
                        dirichlet_alpha, noise_frac)
                _backprop(path, value if mover == "white" else -value)
                g.sims_done += 1

        # ---- games that finished their sims pick a move; refill the pool ----
        still = []
        for g in active:
            if g.sims_done < iterations:
                still.append(g)
                continue
            do_move(g)
            if g.done:
                all_examples.extend(finalize(g))
                if started < num_games:
                    still.append(_GameState())
                    started += 1
            else:
                g.root = Node()
                g.root.moverSign = 0
                g.sims_done = 0
                still.append(g)
        active = still

    return all_examples


# --------------------------------------------------------------------------- #
# torch evaluator + public entry point
# --------------------------------------------------------------------------- #
def _make_torch_eval_fn(net):
    """Wrap a ChessNet into eval_fn(planes_list) -> (logits[B,A], values[B])."""
    import torch
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def eval_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    policy_logits, value = net(x)
            else:
                policy_logits, value = net(x)
        # .float() so half-precision autocast outputs survive the numpy cast
        return (policy_logits.float().cpu().numpy(),
                value.float().cpu().numpy().reshape(-1))

    return eval_fn


def generate_games_batched(net, num_games, iterations=400, max_plies=200,
                           temp_moves=30, c=1.5, concurrency=64,
                           adj_margin=5.0, adj_plies=20,
                           use_cache=True, cache_cap=200_000, verbose=True):
    """
    Drop-in replacement for generate_games_parallel: same return type (flat list
    of (planes, policy_target, value_target) examples) and the same adjudication
    knobs. `concurrency` is the number of games run/batched simultaneously and is
    the main lever on GPU batch size -- set it as high as GPU memory for the
    forward pass allows.
    """
    if num_games <= 0:
        return []
    eval_fn = _make_torch_eval_fn(net)
    return run_selfplay(
        eval_fn, num_games,
        iterations=iterations, concurrency=concurrency,
        max_plies=max_plies, temp_moves=temp_moves, c=c,
        adj_margin=adj_margin, adj_plies=adj_plies,
        use_cache=use_cache, cache_cap=cache_cap, verbose=verbose,
    )
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

Within a single game this is, with the default settings, identical in semantics
to search.puct.search: one pending leaf per tree per round (so no virtual loss is
needeed), `iterations` simulations per move, Dirichlet noise at the root, the same
PUCT formula, the same value-sign / backprop convention, and the same
early-adjudication and ply-cap handling as training.self_play.play_game.

EXAMPLES (changed): each recorded row is now a 5-tuple
    (planes, policy_target, value_target, ease_target, ease_mask)
matching training.train's aux_ease path. ease_target is a search-derived
forgiveness statistic of the ROOT position in [0, 1] (see below); ease_mask is
1.0 where the statistic was computable and 0.0 otherwise (masked rows still
train policy/value, contribute nothing to the ease loss).

EASE TARGETS (new): the ease head amortises a search statistic, so every
recorded root computes one. Definition is configurable via ease_target_mode
("gap" or "entropy" -- the two local forgiveness statistics of search/ease.py
-- or a recursive aggregate of either):
  * "gap"  (default): F = exp(-(Q1 - Q2) / ease_tau) over the root children
    that met the forced-visit floor -- the local action-gap forgiveness.
  * "tree": the recursive formulation F(s) = (1-g) F_local + g * sum
    N(a)/sum N * F(s_a) over the search tree (ease_gamma).
  * "flat": the subtree formulation, F_local of every subtree node weighted by
    its visit share.
Set ease_tau from a probe_ease.py calibration of a comparable checkpoint.
All ease statistics are imported from search/ease.py -- the SAME functions the
probe and the play GUI use, so head targets and displayed/probed values can
never drift apart.

MORE EXHAUSTIVE ROOT SEARCH FOR EASE (new): trustworthy ease targets need more
balanced root Qs than move selection does, so on full-search moves the budget
is extended by `ease_extra_sims` and the forced-visit floor is WIDENED to
`ease_force_m` children (if larger than root_force_m). With the defaults, full
moves run 700+300 = 1000 sims with the top 12 prior moves floored -- Q1..Q12
all carry matched standard errors. Because forced visits are PRUNED from the
recorded policy target (see below), the wider floor sharpens the ease target
WITHOUT blurring the policy target. Cost: full moves are `full_search_prob` of
plies, so +300 sims on them is roughly +30% self-play compute at the default
0.25. With ease_targets=False, rows are still 5-tuples but carry ease_mask=0.0
everywhere, so the training path never needs to branch on format.

FORCED ROOT VISITS (Gumbel/KataGo-style): on full-search moves, the top
`root_force_m` (or `ease_force_m`, whichever is larger when ease targets are
on) root children BY PRIOR are each guaranteed a floor of visits before
ordinary PUCT selection resumes. Plain PUCT starves everything but its
favourite, which makes root Q values unusable for anything that compares
actions. The floor auto-shrinks to move_cap // (2*m) when the budget is small,
and fast (playout-capped) moves are never forced. Forced visits are SUBTRACTED
from every non-best child when building the policy target and when sampling
the move to play (KataGo's policy-target pruning), so training targets keep
the sharpness of plain PUCT while the tree retains balanced Q statistics.
Set root_force_m=0 AND ease_targets=False to restore plain PUCT roots.

VALUE LABELS FOR PLY-CAP GAMES: a game that hits `max_plies` without a
terminal result is scored by material adjudication first -- a lead of
>= adj_margin is a win for that side. A game that is materially BALANCED at
the cap is NOT labelled a hard 0.0 draw (that taught the value head that any
advantage smaller than the margin, and any win needing more than max_plies,
is worthless); its value target is bootstrapped from the LAST recorded
full-search root value (white POV, continuous in [-1, 1]). True terminal
draws (stalemate, threefold, fifty-move) are still 0.0.

Two optional throughput features cut the number of network evaluations (the
dominant cost) -- see run_selfplay:
  * subtree reuse (reuse_tree, on by default);
  * playout-cap randomization (full_search_prob<1 + fast_iterations), the
    KataGo trick: only full-budget moves are recorded as training rows.

A small within-phase evaluation cache (keyed by the exact position) skips the
network for transpositions and repetitions. It is rebuilt every call because
the network changes between training iterations.
"""

import math
import numpy as np

from engine.gameEnv import Chess
from model.encoding import encode
from model.move_encoding import encodeMovePOV, NUM_ACTIONS
from search.ease import ease_target


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
                 "sims_done", "done", "history",
                 "move_cap", "is_full_move", "chosen",
                 "force_m", "force_n", "forced_set", "forced_counts")

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
        self.history = []   # (planes, policy_t, mover_sign, v_white, ease_t, ease_m)
        self.move_cap = 0         # per-move simulation budget (set by _begin_move)
        self.is_full_move = True  # full-search move? only these become training rows
        self.chosen = None        # child node do_move picked (for subtree reuse)
        self.force_m = 0          # per-move width of the forced set
        self.force_n = 0          # per-move forced-visit floor (0 = no forcing)
        self.forced_set = None    # top-m root children by prior (computed lazily
                                  # after root expansion, i.e. post-noise)
        self.forced_counts = {}   # Move -> visits given by forcing this move


# --------------------------------------------------------------------------- #
# core batched self-play loop (eval_fn injected -> unit-testable without torch)
# --------------------------------------------------------------------------- #
def run_selfplay(eval_fn, num_games, *, iterations=400, concurrency=64,
                 max_plies=200, temp_moves=30, c=1.5,
                 add_noise=True, dirichlet_alpha=0.3, noise_frac=0.25,
                 adj_margin=5.0, adj_plies=20,
                 use_cache=True, cache_cap=200_000,
                 reuse_tree=True, full_search_prob=1.0, fast_iterations=None,
                 root_force_m=8, root_force_visits=40,
                 ease_targets=True, ease_tau=0.05, ease_target_mode="gap",
                 ease_gamma=0.85, ease_extra_sims=300, ease_force_m=12,
                 verbose=True):
    """
    Play `num_games` games, keeping up to `concurrency` of them running at once
    and batching their leaf evaluations.

    eval_fn(planes_list) -> (logits, values)
        planes_list : list of (17,8,8) float32 arrays
        logits      : array-like [B, NUM_ACTIONS]   (mover-POV policy logits)
        values      : array-like [B]                (mover-POV value in [-1,1])

    Returns a flat list of (planes, policy_target, value_target, ease_target,
    ease_mask) examples -- the aux_ease format of training.train.

    Throughput options: reuse_tree (carry the chosen child's searched subtree
    over as the next root; sound because the net is fixed for the whole call)
    and playout-cap randomization (full_search_prob < 1 with fast_iterations:
    only full-budget moves are recorded and get root noise).

    Forced root visits: on full-search moves, each of the top
    max(root_force_m, ease_force_m if ease_targets else 0) root children by
    prior is selected directly (bypassing PUCT) until it has
    min(root_force_visits, move_cap // (2*m)) visits. Equalises the top
    actions' Q standard errors (prerequisite for gap/variance/ease statistics
    and clean policy tails). Forced visits are pruned from non-best moves in
    the recorded policy target and the move-selection distribution, so targets
    keep PUCT's sharpness. Visits carried in by subtree reuse count toward the
    floor.

    Ease targets: on full-search moves the budget is iterations +
    ease_extra_sims and after the search an ease statistic of the root
    (ease_target_mode: "gap" | "entropy" | "tree" | "flat"; temperature
    ease_tau; "tree" decay ease_gamma) is recorded with mask 1.0. Rows where the statistic is
    uncomputable, or all rows when ease_targets=False, carry mask 0.0 -- the
    ease loss ignores them, policy/value train as normal.
    """
    concurrency = max(1, min(concurrency, num_games))
    cache = {} if use_cache else None

    full_cap = iterations + (int(ease_extra_sims) if ease_targets else 0)
    fast_cap = iterations if fast_iterations is None else max(1, int(fast_iterations))
    full_search_prob = min(1.0, max(0.0, full_search_prob))
    root_force_m = max(0, int(root_force_m))
    full_force_m = max(root_force_m, int(ease_force_m)) if ease_targets \
        else root_force_m
    move_rng = np.random.default_rng()

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
            if r is not None:
                rwp = r                       # true terminal: win/loss/draw as-is
            else:
                # Ply-cap hit without a terminal result. Material adjudication
                # first: a >= margin lead is a win signal. Materially BALANCED
                # cap games bootstrap the label from the last recorded
                # full-search root value (white POV) instead of a hard draw --
                # see the module docstring.
                rwp = g.env.adjudicate()
                if rwp == 0.0 and g.history:
                    rwp = float(g.history[-1][3])
        out = []
        for (planes, policy_t, mover_sign, _v, ease_t, ease_m) in g.history:
            out.append((planes, policy_t, np.float32(rwp * mover_sign),
                        ease_t, ease_m))
        finished += 1
        if verbose:
            print(f"  game {finished}/{num_games}: {len(out)} positions "
                  f"(plies {g.ply}, result {rwp:+.2f}, total {len(all_examples)+len(out)})",
                  flush=True)
        return out

    def _pruned_visit_counts(g, root):
        """Visit counts with this move's FORCED visits subtracted from every
        child except the most-visited one (KataGo's policy-target pruning).
        Forcing exists to firm up Q statistics, not to claim those moves
        deserve probability mass."""
        raw = {ch.move: ch.visits for ch in root.children}
        if not g.forced_counts or not raw:
            return raw
        best_move = max(raw, key=raw.get)
        counts = {}
        for m, n in raw.items():
            if m != best_move:
                n = n - g.forced_counts.get(m, 0)
            if n > 0:
                counts[m] = n
        if not counts:
            counts = {best_move: raw[best_move]}
        return counts

    def do_move(g):
        root, env = g.root, g.env
        g.chosen = None
        visit_counts = _pruned_visit_counts(g, root)
        if not visit_counts:               # terminal at root (no legal moves)
            g.done = True
            return
        mover = env.board.sideToMove
        mover_sign = 1 if mover == "white" else -1
        # Only full-search moves become training targets; fast (capped) moves
        # still advance the game but emit no row.
        if g.is_full_move:
            planes = encode(env.board)
            policy_target = _policy_target(visit_counts, mover)
            v_white = _position_value_white(root, mover_sign)
            if ease_targets:
                ease_t, ease_m = ease_target(root, g.force_n, ease_tau,
                                              ease_target_mode, ease_gamma)
            else:
                ease_t, ease_m = np.float32(0.0), np.float32(0.0)
            g.history.append((planes, policy_target, mover_sign, v_white,
                              ease_t, ease_m))

        temp = 1.0 if g.ply < temp_moves else 0.0
        move = select_move(visit_counts, temp)
        for ch in root.children:           # remember picked child -> subtree reuse
            if ch.move == move:
                g.chosen = ch
                break
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

    def _begin_move(g):
        """Start a new move: pick its simulation budget (playout-cap
        randomization; full moves carry the ease_extra_sims extension) and the
        forced-visit configuration. sims_done starts at the (possibly reused)
        root's visit count so the budget counts what the tree already holds."""
        full = move_rng.random() < full_search_prob
        g.is_full_move = full
        g.move_cap = full_cap if full else fast_cap
        g.sims_done = g.root.visits
        g.forced_set = None
        g.forced_counts = {}
        g.force_m = 0
        g.force_n = 0
        if full and full_force_m > 0 and root_force_visits > 0:
            g.force_m = full_force_m
            g.force_n = min(int(root_force_visits),
                            max(1, g.move_cap // (2 * full_force_m)))
        # A reused root is already expanded, so _expand never re-fires on it;
        # apply its root noise here instead (fresh roots get noise in _expand).
        if g.root.expanded and full and add_noise:
            _add_dirichlet_noise(g.root, dirichlet_alpha, noise_frac)

    for g in active:
        _begin_move(g)

    while active:
        # ---- one simulation per game; collect leaves needing the network ----
        batch_planes = []
        batch_meta = []
        for g in active:
            node = g.root
            env = g.env.clone()
            path = [node]
            at_root = True
            while node.expanded and not node.terminal and node.children:
                best = None
                # forced root visits: the top-m-by-prior children each get a
                # floor of visits before PUCT resumes (full-search moves only).
                # forced_set is computed lazily on first descent AFTER the root
                # expands, so it sees post-Dirichlet priors.
                if at_root and g.force_n > 0:
                    if g.forced_set is None:
                        g.forced_set = sorted(node.children,
                                              key=lambda ch: ch.prior,
                                              reverse=True)[:g.force_m]
                    for ch in g.forced_set:
                        if ch.visits < g.force_n:
                            best = ch
                            g.forced_counts[ch.move] = \
                                g.forced_counts.get(ch.move, 0) + 1
                            break
                if best is None:
                    sqrt_pv = math.sqrt(node.visits)
                    best_score = -1e30
                    for ch in node.children:
                        v = ch.visits
                        q = ch.value / v if v else 0.0
                        s = q + c * ch.prior * sqrt_pv / (1 + v)
                        if s > best_score:
                            best_score = s
                            best = ch
                node = best
                env.step(node.move)
                path.append(node)
                at_root = False

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
                    _expand(node, priors, mover, add_noise and g.is_full_move,
                            node is g.root, dirichlet_alpha, noise_frac)
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
                _expand(node, priors, mover, add_noise and g.is_full_move,
                        node is g.root, dirichlet_alpha, noise_frac)
                _backprop(path, value if mover == "white" else -value)
                g.sims_done += 1

        # ---- games that finished their sims pick a move; refill the pool ----
        still = []
        for g in active:
            if g.sims_done < g.move_cap:
                still.append(g)
                continue
            do_move(g)
            if g.done:
                all_examples.extend(finalize(g))
                if started < num_games:
                    ng = _GameState()
                    _begin_move(ng)
                    still.append(ng)
                    started += 1
            else:
                child = g.chosen
                if reuse_tree and child is not None and child.expanded \
                        and not child.terminal:
                    child.parent = None      # carry the searched subtree over as
                    child.moverSign = 0      # the new root (net is fixed this call)
                    g.root = child
                else:
                    g.root = Node()
                    g.root.moverSign = 0
                _begin_move(g)               # budget + warm-start sims_done
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
                           use_cache=True, cache_cap=200_000,
                           reuse_tree=True, full_search_prob=1.0,
                           fast_iterations=None,
                           root_force_m=8, root_force_visits=40,
                           ease_targets=True, ease_tau=0.05,
                           ease_target_mode="gap", ease_gamma=0.85,
                           ease_extra_sims=300, ease_force_m=12,
                           verbose=True):
    """
    Drop-in replacement for generate_games_parallel. Returns a flat list of
    (planes, policy_target, value_target, ease_target, ease_mask) examples --
    train with train_epoch(..., aux_ease=True). `concurrency` is the number of
    games run/batched simultaneously (the main GPU batch-size lever).

    See run_selfplay for reuse_tree / playout-cap randomization, forced root
    visits (root_force_m / root_force_visits) and the ease-target options
    (ease_targets, ease_tau, ease_target_mode, ease_gamma, ease_extra_sims,
    ease_force_m). Ease targets default ON with a "gap" statistic from a
    widened 12-move floor and a +300-sim extended full-move budget; set
    ease_tau from a probe_ease.py calibration.
    """
    if num_games <= 0:
        return []
    eval_fn = _make_torch_eval_fn(net)
    return run_selfplay(
        eval_fn, num_games,
        iterations=iterations, concurrency=concurrency,
        max_plies=max_plies, temp_moves=temp_moves, c=c,
        adj_margin=adj_margin, adj_plies=adj_plies,
        use_cache=use_cache, cache_cap=cache_cap,
        reuse_tree=reuse_tree, full_search_prob=full_search_prob,
        fast_iterations=fast_iterations,
        root_force_m=root_force_m, root_force_visits=root_force_visits,
        ease_targets=ease_targets, ease_tau=ease_tau,
        ease_target_mode=ease_target_mode, ease_gamma=ease_gamma,
        ease_extra_sims=ease_extra_sims, ease_force_m=ease_force_m,
        verbose=verbose,
    )
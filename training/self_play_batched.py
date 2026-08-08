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
    (planes, policy_target, value_target, forgiveness_target, forgiveness_mask)
matching training.train's aux_forgiveness path -- with policy_target stored SPARSE as
an (action_indices, probs) pair (an empty pair marks a value-only row from a
playout-capped fast move; see record_fast_rows in run_selfplay). Value targets
are blended lam*z + (1-lam)*Q_root (see value_target_lambda). forgiveness_target is a search-derived
forgiveness statistic of the ROOT position in [0, 1] (see below); forgiveness_mask is
1.0 where the statistic was computable and 0.0 otherwise (masked rows still
train policy/value, contribute nothing to the forgiveness loss).

FORGIVENESS TARGETS (new): the forgiveness head amortises a search statistic, so every
recorded root computes one. Definition is configurable via forgiveness_target_mode
("gap" or "entropy" -- the two local forgiveness statistics of search/forgiveness.py
-- or a recursive aggregate of either):
  * "gap"  (default): F = exp(-(Q1 - Q2) / forgiveness_tau) over the root children
    that met the forced-visit floor -- the local action-gap forgiveness.
  * "tree": the recursive formulation F(s) = (1-g) F_local + g * sum
    N(a)/sum N * F(s_a) over the search tree (forgiveness_gamma).
  * "flat": the subtree formulation, F_local of every subtree node weighted by
    its visit share.
Set forgiveness_tau from a probe_forgiveness.py calibration of a comparable checkpoint.
All forgiveness statistics are imported from search/forgiveness.py -- the SAME functions the
probe and the play GUI use, so head targets and displayed/probed values can
never drift apart.

MORE EXHAUSTIVE ROOT SEARCH FOR FORGIVENESS (new): trustworthy forgiveness targets need more
balanced root Qs than move selection does, so on full-search moves the budget
is extended by `forgiveness_extra_sims` and the forced-visit floor is WIDENED to
`forgiveness_force_m` children (if larger than root_force_m). With the defaults, full
moves run 700+300 = 1000 sims with the top 12 prior moves floored -- Q1..Q12
all carry matched standard errors. Because forced visits are PRUNED from the
recorded policy target (see below), the wider floor sharpens the forgiveness target
WITHOUT blurring the policy target. Cost: full moves are `full_search_prob` of
plies, so +300 sims on them is roughly +30% self-play compute at the default
0.25. With forgiveness_targets=False, rows are still 5-tuples but carry forgiveness_mask=0.0
everywhere, so the training path never needs to branch on format.

FORCED ROOT VISITS (Gumbel/KataGo-style): on full-search moves, the top
`root_force_m` (or `forgiveness_force_m`, whichever is larger when forgiveness targets are
on) root children BY PRIOR are each guaranteed a floor of visits before
ordinary PUCT selection resumes. Plain PUCT starves everything but its
favourite, which makes root Q values unusable for anything that compares
actions. The floor auto-shrinks to move_cap // (2*m) when the budget is small,
and fast (playout-capped) moves are never forced. Forced visits are SUBTRACTED
from every non-best child when building the policy target and when sampling
the move to play (KataGo's policy-target pruning), so training targets keep
the sharpness of plain PUCT while the tree retains balanced Q statistics.
Set root_force_m=0 AND forgiveness_targets=False to restore plain PUCT roots.

VALUE LABELS FOR PLY-CAP GAMES: a game that hits `max_plies` without a
terminal result is scored by material adjudication first -- a lead of
>= adj_margin is a win for that side. A game that is materially BALANCED at
the cap is NOT labelled a hard 0.0 draw (that taught the value head that any
advantage smaller than the margin, and any win needing more than max_plies,
is worthless); its value target is bootstrapped from the LAST recorded
FULL-SEARCH root value (white POV, continuous in [-1, 1]) -- fast rows are
skipped when scanning back, since their playout-capped root values are far
noisier. True terminal draws (stalemate, threefold, fifty-move) are still 0.0.

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
from model.encoding import encode, encode_env
from model.move_encoding import encodeMovePOV, NUM_ACTIONS
from search.forgiveness import forgiveness_target, select_move_forgiving
from search.puct import select_move_gumbel   # torch-free to import


# --------------------------------------------------------------------------- #
# small pure-numpy helpers (kept local so this module needs no torch to import;
# the torch evaluator below imports torch lazily)
# --------------------------------------------------------------------------- #
def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


_EMPTY_POLICY = (np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32))


def _policy_target_sparse(visit_counts, sideToMove):
    """Policy target as (action_indices int32, probs float32) -- ~40 entries
    instead of a dense NUM_ACTIONS (4672) float32 vector. This is what makes a
    600k-row buffer (fast value-only rows included) cost ~3 GB instead of
    ~14 GB. training.train._collate densifies per batch on the fly."""
    total = sum(visit_counts.values())
    if total == 0:
        return _EMPTY_POLICY
    idx = np.empty(len(visit_counts), dtype=np.int32)
    pr = np.empty(len(visit_counts), dtype=np.float32)
    for i, (move, count) in enumerate(visit_counts.items()):
        idx[i] = encodeMovePOV(move, sideToMove)
        pr[i] = count / total
    return idx, pr


def _position_value_white(root, mover_sign):
    tot = sum(c.visits for c in root.children if c.visits > 0)
    if tot == 0:
        return 0.0
    v_mover = sum(c.value for c in root.children if c.visits > 0) / tot
    return v_mover * mover_sign


def select_move(visit_counts, temp=1.0, rng=None):
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temp <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    logits = counts ** (1.0 / temp)
    probs = logits / logits.sum()
    if rng is None:
        rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=probs)]


class Node:
    __slots__ = ("parent", "move", "prior", "children",
                 "visits", "value", "value_sh", "moverSign", "terminal",
                 "expanded")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0         # RAW accumulator: targets, stats, selection
        self.value_sh = 0.0      # SHAPED accumulator: PUCT descent only
        self.moverSign = 0
        self.terminal = False
        self.expanded = False


def _add_dirichlet_noise(root, alpha, frac, rng=None):
    if not root.children:
        return
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.dirichlet([alpha] * len(root.children))
    for child, n in zip(root.children, noise):
        child.prior = (1 - frac) * child.prior + frac * n


def _expand(node, priors, mover, add_noise, is_root,
            dirichlet_alpha, noise_frac, rng=None):
    """Attach children to `node` from a {Move: prior} dict."""
    sign = 1 if mover == "white" else -1
    for m, p in priors.items():
        child = Node(parent=node, move=m, prior=p)
        child.moverSign = sign
        node.children.append(child)
    node.expanded = True
    if add_noise and is_root:
        _add_dirichlet_noise(node, dirichlet_alpha, noise_frac, rng)


def _backprop(path, leaf_value_white, leaf_value_white_sh=None):
    """Accumulate the RAW leaf value into .value and the SHAPED one into
    .value_sh (same number when shaping is off / at terminals). The search
    DESCENDS on .value_sh; every recorded quantity -- policy/value/
    forgiveness targets, move-selection Qs -- reads .value, so shaping
    steers visit allocation without corrupting any label."""
    if leaf_value_white_sh is None:
        leaf_value_white_sh = leaf_value_white
    for n in path:
        n.visits += 1
        n.value += leaf_value_white * n.moverSign
        n.value_sh += leaf_value_white_sh * n.moverSign


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
        self.history = []   # (planes, sparse_policy, mover_sign, v_white, forgiveness_t, forgiveness_m)
                            # sparse_policy = (idx, probs); EMPTY pair -> value-only row
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
                 root_force_m=6, root_force_visits=80,
                 forgiveness_targets=True, forgiveness_tau=0.0313, forgiveness_target_mode="gap",
                 forgiveness_gamma=0.85, forgiveness_extra_sims=100, forgiveness_force_m=6,
                 fpu_reduction=0.25, value_target_lambda=0.7,
                 record_fast_rows=True,
                 gumbel_select=False, gumbel_c_visit=50.0, gumbel_c_scale=1.0,
                 forgiving_select=False, forgiving_delta=0.05,
                 forgiving_stat="gap", forgiving_agg="tree",
                 forgiving_parity=1,
                 forgiveness_shaping_beta=0.0,
                 seed=None,
                 verbose=True):
    """
    Play `num_games` games, keeping up to `concurrency` of them running at once
    and batching their leaf evaluations.

    FORGIVENESS-SHAPED SEARCH (forgiveness_shaping_beta > 0): every leaf
    evaluation backs up
        v' = clip(v + beta * (2*F_hat - 1), -1, 1)     [mover POV]
    where F_hat in [0, 1] is the net's own forgiveness-head output on the
    leaf position -- eval_fn must then return a THIRD array (see
    _make_torch_eval_fn(return_forgiveness=True)). Centred (2F-1) means
    forgiving continuations (F > 0.5) are boosted for the player to move
    there and brittle ones penalised, symmetrically for both players --
    use a parity-BLENDED head (e.g. flat_entropy) for this. Because the
    bonus flows through backup, the SEARCH ITSELF prefers forgiving lines,
    so the visit-count policy targets -- and hence the trained policy --
    absorb the preference: this is the training-time mechanism, in contrast
    to post-search selection. Terminal values are never shaped.

    DUAL-TRACK VALUES: the shaped value steers ONLY the PUCT descent
    (Node.value_sh). Every recorded quantity -- value targets, forgiveness
    labels, move-selection Qs, root values -- reads the RAW accumulator
    (Node.value), so the labels the head trains on are statistics of TRUE
    values rather than of the head's own bonus (no self-referential
    feedback), and the value targets carry no shaping bias. Shaping reaches
    the policy exclusively through visit allocation -- exactly the
    policy-target channel. Use beta ~ typical Q gap (0.02-0.05).

    FORGIVING MOVE SELECTION (forgiving_select, default off): on POST-OPENING
    (ply >= temp_moves) full-search moves, form the near-optimal set
        S = { a : Q1 - Q(a) <= forgiving_delta }
    over the root children that met the forced-visit floor and play the
    member whose subtree is most forgiving -- search.forgiveness.
    select_move_forgiving with forgiveness_tau/forgiveness_gamma, statistic
    forgiving_stat, aggregator forgiving_agg, and forgiving_parity=1 by
    default ("MY future slack": on a root child the root player's decision
    nodes sit at odd depths; see the parity section of search/forgiveness).
    Every move sacrifices at most forgiving_delta of Q versus the greedy
    choice -- delta is the return-vs-robustness knob, in the same value units
    as forgiveness_tau. Opening plies keep visit-temperature sampling; fast
    moves and floor-starved roots fall back to visit selection. Takes
    precedence over gumbel_select post-opening. Policy targets UNCHANGED.

    GUMBEL MOVE SELECTION (gumbel_select, default off): on FULL-SEARCH moves,
    pick the move played by argmax of the Gumbel-AlphaZero score
        log prior + (gumbel_c_visit + max_visits) * gumbel_c_scale * Q
    over the root children that met the forced-visit floor (see
    search.puct.select_move_gumbel), instead of argmax pruned visits; during
    the temp_moves opening the same scores are SAMPLED at the usual
    temperature. This decides on prior-regularised Q values -- a late Q-flip
    the visit counts have not caught up with gets played -- and it is sound
    here precisely because the forced floor gives the candidates
    matched-variance Qs. Fast (playout-capped) moves have no floor and junk
    Qs, so they keep visit-based selection regardless. POLICY TARGETS ARE
    UNCHANGED (still pruned visit counts): this flag changes which move is
    played, not what is recorded. If fewer than two children met the floor
    (e.g. root_force_m=0), selection falls back to pruned-visit selection.

    seed: seeds ONE np.random.Generator that drives every random draw in this
    call (playout-cap coin flips, Dirichlet noise, temperature sampling), so a
    fixed seed plus a deterministic eval_fn reproduces the games exactly.
    None (default) keeps the fresh-entropy behavior for training diversity.

    eval_fn(planes_list) -> (logits, values)
        planes_list : list of (19,8,8) float32 arrays
        logits      : array-like [B, NUM_ACTIONS]   (mover-POV policy logits)
        values      : array-like [B]                (mover-POV value in [-1,1])

    Returns a flat list of (planes, policy_target, value_target, forgiveness_target,
    forgiveness_mask) examples -- the aux_forgiveness format of training.train -- where
    policy_target is SPARSE: an (action_indices, probs) pair. Rows with an
    EMPTY pair are value-only rows; training.train._collate gives them policy
    weight 0 (value/forgiveness train as normal).

    VALUE-ONLY FAST ROWS (record_fast_rows, default on): playout-capped fast
    moves -- 1-full_search_prob of all plies -- previously emitted nothing,
    discarding ~75% of the game's positions. Their visit counts are useless as
    policy targets (tiny budget, no noise, no forcing) but the position is
    labelled by the SAME game outcome and its root value, so each is now
    recorded as a value-only row (empty policy, forgiveness_mask 0). ~4x value-head
    data at the cost of one encode() per fast ply. Size the replay buffer for
    the extra volume (rows/game grows ~4x).

    VALUE TARGET BLENDING (value_target_lambda): every row's value target is
        lam * z + (1 - lam) * Q_root      (both mover-POV)
    where z is the final game outcome and Q_root the recorded position's own
    search root value. lam=1 restores pure-outcome labels; 0.5-0.75 cuts the
    dominant variance term of the value loss in a data-starved run (the KataGo
    trick). The ply-cap bootstrap for materially balanced capped games is
    unchanged (its z IS a root value).

    FIRST-PLAY URGENCY (fpu_reduction): unvisited children score
    parent-running-Q minus fpu_reduction instead of a flat 0, so search stops
    over-exploring refuted moves whenever the side to move is worse. 0
    restores legacy behavior. Applied at every tree node incl. the root.

    Throughput options: reuse_tree (carry the chosen child's searched subtree
    over as the next root; sound because the net is fixed for the whole call)
    and playout-cap randomization (full_search_prob < 1 with fast_iterations:
    only full-budget moves are recorded and get root noise).

    Forced root visits: on full-search moves, each of the top
    max(root_force_m, forgiveness_force_m if forgiveness_targets else 0) root children by
    prior is selected directly (bypassing PUCT) until it has
    min(root_force_visits, move_cap // (2*m)) visits. Equalises the top
    actions' Q standard errors (prerequisite for gap/variance/forgiveness statistics
    and clean policy tails). Forced visits are pruned from non-best moves in
    the recorded policy target and the move-selection distribution, so targets
    keep PUCT's sharpness. Visits carried in by subtree reuse count toward the
    floor.

    Forgiveness targets: on full-search moves the budget is iterations +
    forgiveness_extra_sims and after the search an forgiveness statistic of the root
    (forgiveness_target_mode: "gap" | "entropy" | "tree" | "flat"; temperature
    forgiveness_tau; "tree" decay forgiveness_gamma) is recorded with mask 1.0. Rows where the statistic is
    uncomputable, or all rows when forgiveness_targets=False, carry mask 0.0 -- the
    forgiveness loss ignores them, policy/value train as normal.
    """
    concurrency = max(1, min(concurrency, num_games))
    cache = {} if use_cache else None

    full_cap = iterations + (int(forgiveness_extra_sims) if forgiveness_targets else 0)
    fast_cap = iterations if fast_iterations is None else max(1, int(fast_iterations))
    full_search_prob = min(1.0, max(0.0, full_search_prob))
    root_force_m = max(0, int(root_force_m))
    full_force_m = max(root_force_m, int(forgiveness_force_m)) if forgiveness_targets \
        else root_force_m
    rng = np.random.default_rng(seed)   # single source of randomness (see `seed`)

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
                    # Bootstrap from the last FULL-SEARCH root value. With
                    # record_fast_rows on, history[-1] is usually a fast
                    # (playout-capped) row whose ~100-sim root value is a much
                    # noisier label; scan back for the last row with a
                    # non-empty policy target (= full-search row) and only
                    # fall back to the final row if the game had none.
                    v_boot = None
                    for row in reversed(g.history):
                        pol = row[1]
                        if isinstance(pol, tuple) and len(pol[0]):
                            v_boot = row[3]
                            break
                    rwp = float(v_boot if v_boot is not None
                                else g.history[-1][3])
        out = []
        lam = value_target_lambda
        for (planes, policy_t, mover_sign, v_white, forgiveness_t, forgiveness_m) in g.history:
            # blend the game outcome with the position's own search root value
            # (both converted to the recorded mover's POV). lam=1 -> pure z.
            z_mover = rwp * mover_sign
            q_mover = v_white * mover_sign
            target = lam * z_mover + (1.0 - lam) * q_mover
            out.append((planes, policy_t, np.float32(target), forgiveness_t, forgiveness_m))
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
        # Full-search moves carry a real policy target (+ forgiveness when enabled).
        # Fast (playout-capped) moves are recorded as VALUE-ONLY rows when
        # record_fast_rows: empty policy (their tiny no-noise search is not a
        # policy target), forgiveness mask 0, but the position still trains the value
        # head on the blended outcome/root-value label.
        if g.is_full_move or record_fast_rows:
            planes = encode_env(env)
            v_white = _position_value_white(root, mover_sign)
            if g.is_full_move:
                policy_target = _policy_target_sparse(visit_counts, mover)
                if forgiveness_targets:
                    forgiveness_t, forgiveness_m = forgiveness_target(root, g.force_n, forgiveness_tau,
                                                  forgiveness_target_mode, forgiveness_gamma)
                else:
                    forgiveness_t, forgiveness_m = np.float32(0.0), np.float32(0.0)
            else:
                policy_target = _EMPTY_POLICY
                forgiveness_t, forgiveness_m = np.float32(0.0), np.float32(0.0)
            g.history.append((planes, policy_target, mover_sign, v_white,
                              forgiveness_t, forgiveness_m))

        temp = 1.0 if g.ply < temp_moves else 0.0
        move = None
        if forgiving_select and g.is_full_move and g.force_n > 0 \
                and temp <= 1e-6:
            # delta-constrained forgiveness selection over the floored
            # candidates: play the most forgiving member of
            # { a : Q1 - Q(a) <= delta }. Falls through to visit selection
            # when the floor starved the candidate set (returns the Q-best
            # move itself otherwise, so None only means "no visited child").
            move = select_move_forgiving(
                root, forgiving_delta, forgiveness_tau, floor=g.force_n,
                gamma=forgiveness_gamma, stat=forgiving_stat,
                agg=forgiving_agg, parity=forgiving_parity)
        if move is None and gumbel_select and g.is_full_move \
                and g.force_n > 0:
            # Gumbel-AZ selection: logits + sigma(Q) over the floored
            # candidates; pruned visit counts remain the FALLBACK (and the
            # policy target, untouched above).
            move = select_move_gumbel(
                root, temp=temp, rng=rng,
                c_visit=gumbel_c_visit, c_scale=gumbel_c_scale,
                min_visits=g.force_n, fallback_counts=visit_counts)
        if move is None:
            move = select_move(visit_counts, temp, rng)
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
        randomization; full moves carry the forgiveness_extra_sims extension) and the
        forced-visit configuration. sims_done starts at the (possibly reused)
        root's visit count so the budget counts what the tree already holds."""
        full = rng.random() < full_search_prob
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
            _add_dirichlet_noise(g.root, dirichlet_alpha, noise_frac, rng)

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
                    # first-play urgency: unvisited children assume the node's
                    # running Q minus a penalty, not a flat 0 (see
                    # search.puct.node_fpu_q for the sign/derivation; the
                    # O(children) fallback runs only at roots, whose
                    # moverSign==0 .value is 0/stale under subtree reuse).
                    if node.moverSign != 0:
                        fpu_q = -(node.value_sh / node.visits) - fpu_reduction
                    else:
                        vsum = 0.0
                        nsum = 0
                        for ch in node.children:
                            if ch.visits:
                                vsum += ch.value_sh
                                nsum += ch.visits
                        fpu_q = (vsum / nsum - fpu_reduction) if nsum else 0.0
                    best_score = -1e30
                    for ch in node.children:
                        v = ch.visits
                        # descend on the SHAPED Q; .value stays raw for all
                        # recorded statistics and for move selection
                        q = ch.value_sh / v if v else fpu_q
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
            # The net input now includes the halfmove clock and repetition
            # count, so the eval cache MUST key on them too -- two visits to
            # the same board with different counters are different net inputs.
            # (This costs cache hit-rate on repetition revisits; that loss is
            # correctness, not waste: the net SHOULD re-evaluate rep=2 lower.)
            bkey = env.board.stateKey()
            rep = env.counts.get(bkey, 1)
            if cache is not None:
                key = (bkey, env.halfmove_clock, rep)
                hit = cache.get(key)
                if hit is not None:
                    priors, value, v_sh = hit
                    _expand(node, priors, mover, add_noise and g.is_full_move,
                            node is g.root, dirichlet_alpha, noise_frac, rng)
                    sgn = 1.0 if mover == "white" else -1.0
                    _backprop(path, sgn * value, sgn * v_sh)
                    g.sims_done += 1
                    continue

            batch_planes.append(encode(env.board, env.halfmove_clock, rep))
            batch_meta.append((g, node, env, legal, path, mover, bkey, rep))

        # ---- single batched network forward ----
        if batch_planes:
            out = eval_fn(batch_planes)
            if forgiveness_shaping_beta > 0.0:
                if len(out) != 3:
                    raise ValueError(
                        "forgiveness_shaping_beta > 0 needs an eval_fn that "
                        "returns (logits, values, forgiveness) -- build it "
                        "with _make_torch_eval_fn(net, return_forgiveness="
                        "True)")
                logits_b, values_b, f_b = out
                shaped_b = np.clip(
                    values_b + forgiveness_shaping_beta * (2.0 * f_b - 1.0),
                    -1.0, 1.0)
            else:
                logits_b, values_b = out[0], out[1]
                shaped_b = values_b
            for (g, node, env, legal, path, mover, bkey, rep), logits, value, \
                    v_sh in zip(batch_meta, logits_b, values_b, shaped_b):
                idxs = [encodeMovePOV(m, mover) for m in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {m: float(p) for m, p in zip(legal, probs)}
                value = float(value)
                v_sh = float(v_sh)
                if cache is not None and len(cache) < cache_cap:
                    # raw AND shaped cached (beta and the net are fixed for
                    # the whole call, so hits agree)
                    cache[(bkey, env.halfmove_clock, rep)] = \
                        (priors, value, v_sh)
                _expand(node, priors, mover, add_noise and g.is_full_move,
                        node is g.root, dirichlet_alpha, noise_frac, rng)
                sgn = 1.0 if mover == "white" else -1.0
                _backprop(path, sgn * value, sgn * v_sh)
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
def _make_torch_eval_fn(net, return_forgiveness=False):
    """Wrap a ChessNet into eval_fn(planes_list) -> (logits[B,A], values[B]),
    or (logits, values, forgiveness[B]) with return_forgiveness=True -- the
    form forgiveness-shaped search requires."""
    import torch
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def eval_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    out = (net(x, return_forgiveness=True)
                           if return_forgiveness else net(x))
            else:
                out = (net(x, return_forgiveness=True)
                       if return_forgiveness else net(x))
        # .float() so half-precision autocast outputs survive the numpy cast
        if return_forgiveness:
            policy_logits, value, f = out
            return (policy_logits.float().cpu().numpy(),
                    value.float().cpu().numpy().reshape(-1),
                    f.float().cpu().numpy().reshape(-1))
        policy_logits, value = out
        return (policy_logits.float().cpu().numpy(),
                value.float().cpu().numpy().reshape(-1))

    return eval_fn


def generate_games_batched(net, num_games, iterations=400, max_plies=200,
                           temp_moves=30, c=1.5, concurrency=64,
                           adj_margin=5.0, adj_plies=20,
                           use_cache=True, cache_cap=200_000,
                           reuse_tree=True, full_search_prob=1.0,
                           fast_iterations=None,
                           root_force_m=6, root_force_visits=80,
                           forgiveness_targets=True, forgiveness_tau=0.0313,
                           forgiveness_target_mode="gap", forgiveness_gamma=0.85,
                           forgiveness_extra_sims=100, forgiveness_force_m=6,
                           fpu_reduction=0.25, value_target_lambda=0.7,
                           record_fast_rows=True,
                           gumbel_select=False, gumbel_c_visit=50.0,
                           gumbel_c_scale=1.0,
                           forgiving_select=False, forgiving_delta=0.05,
                           forgiving_stat="gap", forgiving_agg="tree",
                           forgiving_parity=1,
                           forgiveness_shaping_beta=0.0,
                           seed=None,
                           verbose=True):
    """
    Drop-in replacement for generate_games_parallel. Returns a flat list of
    (planes, policy_target, value_target, forgiveness_target, forgiveness_mask) examples --
    train with train_epoch(..., aux_forgiveness=True). `concurrency` is the number of
    games run/batched simultaneously (the main GPU batch-size lever).

    See run_selfplay for reuse_tree / playout-cap randomization, forced root
    visits (root_force_m / root_force_visits) and the forgiveness-target options
    (forgiveness_targets, forgiveness_tau, forgiveness_target_mode, forgiveness_gamma, forgiveness_extra_sims,
    forgiveness_force_m). Forgiveness targets default ON with a "gap" statistic. Defaults
    (probe_forgiveness-calibrated tau=0.0313; DEEP-NOT-WIDE floor: 6 forced children,
    80-visit ceiling; +100-sim budget) follow the noise analysis: the gap
    statistic only needs the TOP-2 Qs to be trustworthy, so the forced budget
    buys more per sim as depth-per-child than as width. The effective floor is
    min(root_force_visits, (iterations+forgiveness_extra_sims) // (2*m)) -- with the
    defaults min(80, 800//12) = 66 visits/child. Judge label quality by the
    forgiveness_R2 column train_epoch now reports, not by the raw forgiveness MSE.
    """
    if num_games <= 0:
        return []
    eval_fn = _make_torch_eval_fn(
        net, return_forgiveness=(forgiveness_shaping_beta > 0.0))
    return run_selfplay(
        eval_fn, num_games,
        iterations=iterations, concurrency=concurrency,
        max_plies=max_plies, temp_moves=temp_moves, c=c,
        adj_margin=adj_margin, adj_plies=adj_plies,
        use_cache=use_cache, cache_cap=cache_cap,
        reuse_tree=reuse_tree, full_search_prob=full_search_prob,
        fast_iterations=fast_iterations,
        root_force_m=root_force_m, root_force_visits=root_force_visits,
        forgiveness_targets=forgiveness_targets, forgiveness_tau=forgiveness_tau,
        forgiveness_target_mode=forgiveness_target_mode, forgiveness_gamma=forgiveness_gamma,
        forgiveness_extra_sims=forgiveness_extra_sims, forgiveness_force_m=forgiveness_force_m,
        fpu_reduction=fpu_reduction, value_target_lambda=value_target_lambda,
        record_fast_rows=record_fast_rows,
        gumbel_select=gumbel_select, gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
        forgiving_select=forgiving_select, forgiving_delta=forgiving_delta,
        forgiving_stat=forgiving_stat, forgiving_agg=forgiving_agg,
        forgiving_parity=forgiving_parity,
        forgiveness_shaping_beta=forgiveness_shaping_beta,
        seed=seed,
        verbose=verbose,
    )


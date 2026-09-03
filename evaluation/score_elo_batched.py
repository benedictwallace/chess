"""
Single-process, leaf-batched checkpoint Elo scoring.

Same schedule, same Bradley-Terry Elo fit, and same resumable cache as
score_elo.py / score_elo_parallel.py -- only the way games are PLAYED changes.

score_elo.py plays one game at a time (batch-1 GPU forwards). score_elo_parallel
spreads games over CPU worker processes feeding a central GPU batcher, but each
worker blocks on its pipe after submitting ONE position, so at most `workers`
evaluations are ever in flight and the cores idle on the queue/GPU/pipe round
trip -- it is latency-bound, not compute-bound.

Here, as in training/self_play_batched.py, many games run CONCURRENTLY in one
process and their MCTS leaf evaluations are batched into single GPU forwards.
No IPC, nothing pickled. Two things differ from self-play:

  * Two nets per game. When an agent searches for its move it uses ITS OWN net
    for every leaf in that search (exactly as arena.NeuralAgent does). Different
    concurrent games -- and the two colours within one game -- may be searching
    with different nets, so each round we GROUP the pending leaves by the
    searching net and run one batched forward per net. Because the schedule
    emits games grouped by pairing, only ~2 nets are ever in flight, so the
    batches stay large (just like score_elo_parallel's per-net batching, minus
    the IPC).

  * Anchor moves (random / material) take no network evaluation, so a game
    alternates between "neural search" rounds (batched) and instant anchor
    moves (played inline). A game is only in the active batched set while it is
    waiting on a neural search.

Eval cache: checkpoints are FIXED for the whole run, so the cache is keyed by
(net_id, exact position) and persists across every game and pairing -- openings
recur constantly. It stores legal-move priors + value (a few KB/entry), bounded
by cache_cap.

ROOT VISIT ALLOCATION: by default the root is plain PUCT, which is fine when
all you want is a win rate -- selection is by visit count and the ranking
survives unequal Q precision. It is NOT fine for anything that compares root Q
values ACROSS actions, because PUCT concentrates the budget and leaves
second-tier children holding a handful of visits and a Q that is essentially
one leaf evaluation. Pass sequential_halving=True to allocate the root budget
by Gumbel top-m + sequential halving instead, exactly as full-search self-play
moves do: every survivor of a phase carries the same visit count by
construction, the top sh_stat_width actions are snapshotted at each phase
advance as a matched-variance statistics set (g.sh.stat_children()), and the
move played comes from pi' rather than from visit counts. Interior nodes stay
on PUCT in both modes.

The core run_elo_matches_batched() takes injected eval_fns and anchor agents and
imports no torch, so it is unit-testable on CPU with fakes; main() lazy-imports
the torch / arena / score_elo pieces.
"""

# MUST be first: patches Board.legalMoves() with the Cython generator
# before anything imports engine.board. The search is move-generation
# bound, so this is worth roughly 3x end to end.
try:
    from evaluation.fast_movegen_boot import ensure_fast_movegen
except ImportError:
    from fast_movegen_boot import ensure_fast_movegen
ensure_fast_movegen()

import math
import numpy as np
import os
import glob
import re
import csv

from engine.gameEnv import Chess
from model.encoding import encode_env
from search.puct import node_fpu_q, select_move_gumbel
from model.move_encoding import encodeMovePOV

# Sequential halving is OPTIONAL here: the module still imports (and the plain
# PUCT path still runs) if search/sequential_halving.py is unavailable. Asking
# for sequential_halving=True without it is an error, not a silent fallback.
try:                                        # repo-layout tolerant
    from search.sequential_halving import SHState, improved_policy
except ImportError:                         # pragma: no cover
    try:
        from sequential_halving import SHState, improved_policy
    except ImportError:
        SHState = None
        improved_policy = None


# --------------------------------------------------------------------------- #
# MCTS primitives (torch-free; mirror search.puct with add_noise always off)
# --------------------------------------------------------------------------- #
class Node:
    # net_value: the RAW network evaluation of THIS position in the POV of the
    # player to move here, stored at expansion. Roots carry moverSign == 0 so
    # their .value accumulator stays 0 and cannot stand in for it; the Gumbel
    # v_mix completion (sequential_halving.root_v_mix) needs it, and without
    # the slot it degraded silently to the visit-weighted child average.
    # value / value_sh: DUAL-TRACK values, mirroring
    # training/self_play_batched.py. .value is the RAW backed-up network
    # value; .value_sh carries the forgiveness-shaped one. With shaping off
    # they hold identical numbers and every code path below is unchanged.
    __slots__ = ("parent", "move", "prior", "children",
                 "visits", "value", "value_sh", "moverSign", "terminal",
                 "expanded", "net_value")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.value_sh = 0.0
        self.moverSign = 0
        self.terminal = False
        self.expanded = False
        self.net_value = None


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _expand(node, priors, mover, net_value=None):
    sign = 1 if mover == "white" else -1
    for m, p in priors.items():
        child = Node(parent=node, move=m, prior=p)
        child.moverSign = sign
        node.children.append(child)
    node.expanded = True
    if net_value is not None:
        node.net_value = float(net_value)


class _ZeroGumbel:
    """np.random.Generator stand-in exposing only what SHState uses, with the
    Gumbel variables pinned to ZERO.

    Gumbel top-m sampling is EXPLORATION: it is what replaces Dirichlet noise
    in self-play. In a rated arena game exploration is a handicap and a source
    of variance -- two arms differing only in their selection rule would also
    differ in which actions each even considered. With zeros, top-m is exactly
    the top m priors and select_action is the deterministic
    argmax(logit + sigma(q)), so a match is reproducible and diversity comes
    from the opening book instead. Pass sh_gumbel=True to restore sampling.
    """

    @staticmethod
    def gumbel(loc=0.0, scale=1.0, size=None):
        return np.zeros(() if size is None else size, dtype=np.float64)


def _backprop(path, leaf_value_white, leaf_value_white_sh=None):
    """Accumulate the RAW leaf value into .value and the SHAPED one into
    .value_sh (same number when shaping is off, and at terminals, which are
    never shaped). Mirrors training/self_play_batched.py::_backprop.

    The split matters for what each track is allowed to influence. The shaped
    track steers every quantity that DECIDES something: the PUCT descent, the
    halving eliminations, the completed Qs behind pi', and the move played.
    The raw track feeds everything RECORDED -- here the per-decision state
    statistics -- so the forgiveness numbers reported for an arm are never a
    statistic of that arm's own bonus."""
    if leaf_value_white_sh is None:
        leaf_value_white_sh = leaf_value_white
    for n in path:
        n.visits += 1
        n.value += leaf_value_white * n.moverSign
        n.value_sh += leaf_value_white_sh * n.moverSign


def _fpu_q_sh(node, fpu_reduction):
    """node_fpu_q against the SHAPED accumulator. Same logic as
    search.puct.node_fpu_q -- negate for internal nodes, visit-weighted child
    average at a root (moverSign == 0) -- but reading .value_sh, so an
    unvisited child is compared on the same track as its siblings."""
    if node.moverSign != 0 and node.visits > 0:
        return -(node.value_sh / node.visits) - fpu_reduction
    vsum = 0.0
    nsum = 0
    for ch in node.children:
        if ch.visits:
            vsum += ch.value_sh
            nsum += ch.visits
    return (vsum / nsum - fpu_reduction) if nsum else 0.0


def select_move(visit_counts, temp, rng=None):
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
# per-game state
# --------------------------------------------------------------------------- #
def select_move_sh(root, sh, temp, rng=None, c_visit=50.0, c_scale=0.02,
                   shaped=False):
    """The move to PLAY at a sequential-halving root.

    Visit counts are NOT a ranking under halving -- every survivor of a phase
    holds the same count by construction, so argmax-by-visits reads the
    elimination schedule rather than action quality. The ranking is pi', the
    improved policy softmax(log prior + sigma(qhat)), restricted to the
    candidates halving left alive (pi' assigns mass to every legal move,
    including ones eliminated in phase 0 and completed with v_mix, and we do
    not want to play those).

    shaped=True reads the shaped value track. Under halving this is the ONLY
    channel by which value shaping reaches the played move: the visit counts
    are fixed by the schedule, so unlike plain AlphaZero the preference
    cannot arrive through visit allocation.

    temp <= 0 -> argmax (the acting rule for rated games). temp > 0 -> sample
    from pi'^(1/temp) over the survivors, for opening diversity when no book
    is in use. Deterministic given the tree: unlike SHState.select_action this
    carries no Gumbel term.
    """
    if improved_policy is None:
        return None
    kw = {}
    if shaped:
        try:
            import inspect
            if "shaped" in inspect.signature(improved_policy).parameters:
                kw["shaped"] = True
        except (TypeError, ValueError):
            pass
    probs = improved_policy(root, c_visit=c_visit, c_scale=c_scale, **kw)
    if not probs:
        return None
    if sh is not None and sh.candidates:
        alive = {c.move for c in sh.candidates}
        restricted = {m: p for m, p in probs.items() if m in alive}
        if restricted:
            probs = restricted
    moves = list(probs.keys())
    p = np.array([probs[m] for m in moves], dtype=np.float64)
    if temp <= 1e-6 or p.sum() <= 0.0:
        return moves[int(p.argmax())]
    w = p ** (1.0 / temp)
    if rng is None:
        rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=w / w.sum())]


class _AGame:
    """One arena game. `white`/`black` are each either a net_id (str -> neural,
    searched with that net) or an anchor agent object exposing .select(env, ply)."""
    __slots__ = ("pidx", "a_is_white", "white", "black",
                 "env", "ply", "done", "a_score",
                 "root", "sims_done", "search_net", "sh")

    def __init__(self, pidx, a_is_white, white, black):
        self.pidx = pidx
        self.a_is_white = a_is_white
        self.white = white
        self.black = black
        self.env = Chess()
        self.env.reset()
        self.ply = 0
        self.done = False
        self.a_score = None
        self.root = None
        self.sims_done = 0
        self.search_net = None
        self.sh = None                  # SHState for the CURRENT move, or None


# --------------------------------------------------------------------------- #
# core batched runner
# --------------------------------------------------------------------------- #
def run_elo_matches_batched(tickets, eval_fns, *, iterations=400, c=1.5,
                            fpu_reduction=0.25,
                            opening_plies=8, opening_temp=1.0, max_plies=160,
                            concurrency=128, use_cache=True, cache_cap=250_000,
                            on_game_done=None, decide_move=None,
                            gumbel_select=False, gumbel_c_visit=50.0,
                            gumbel_c_scale=1.0, gumbel_min_visits=None,
                            sequential_halving=False, sh_m=16, sh_stat_width=4,
                            sh_c_visit=50.0, sh_c_scale=0.02, sh_gumbel=False,
                            sh_seed=0, shape_beta=None,
                            rng=None):
    """
    tickets : list of (pidx, a_is_white, white_mover, black_mover)
              white_mover/black_mover is a net_id str (neural) or an anchor agent.
    iterations: an int (same budget for every searcher), OR a dict
              {net_id: sims} so two agents can search the SAME net at
              DIFFERENT budgets. That is what makes the search-scaling
              diagnostic possible: play a net against itself at N and 2N sims
              and the score difference IS the value of doubling search, with
              no external anchor and no assumption about the opponent.
    eval_fns: dict net_id -> callable(planes_list) -> (logits[B,A], values[B]),
              mover-POV policy logits and mover-POV value in [-1, 1].
    on_game_done(pidx, a_score) is called as each game finishes (for resumable
              caching / progress).
    decide_move(g, visit_counts) -> move, optional: overrides the default
              move choice (opening temp then argmax-by-visits). g exposes
              .search_net (the mover's net_id), .env, .ply, .root and .sh --
              enough for perturbed / forgiveness-aware selection. The callback
              owns ALL temperature handling, including the opening.
              g.sh is the move's SHState when sequential_halving is on (None
              otherwise), so a callback can take the matched-variance
              statistics set with g.sh.stat_children() / g.sh.stat_floor()
              and the acting rule with select_move_sh(g.root, g.sh, 0.0).
              Under halving, `visit_counts` encodes the ELIMINATION SCHEDULE,
              not action quality: do not rank by it.
    sequential_halving: allocate ROOT visits by Gumbel top-m + sequential
              halving (search/sequential_halving.py) instead of PUCT, exactly
              as full-search self-play moves do. Interior nodes stay on PUCT
              either way. This is what makes root Q estimates comparable
              ACROSS actions -- PUCT roots concentrate visits, so a
              second-best child can hold three visits and a Q that is one
              noisy leaf evaluation. Any statistic that is a function of
              differences BETWEEN root Qs (an action gap, a Q-entropy, a
              delta-set membership test) needs this on.
    sh_m      : root actions considered (top-m by Gumbel-perturbed prior).
              Capped by the branching factor and by the budget. At small
              budgets prefer a smaller m: plan_phases(300, 16) gives per
              candidate 4/13/31/77 across its four phases, while
              plan_phases(800, 8) gives 33/99/235 -- the second puts the
              statistics set at 99 visits, near the 108 of the 1000-sim
              training configuration.
    sh_stat_width: how many actions to snapshot for the statistics set at
              each phase advance (default 4). The final pair is too narrow
              for a Q-entropy -- over two actions it is a monotone function
              of the action gap. Read the set back with g.sh.stat_children().
    sh_c_visit / sh_c_scale: the sigma transform in pi' and in the acting
              rule. Match the training configuration (0.02, not the 1.0
              default that sigma_scale carries for other callers).
    sh_gumbel : keep the Gumbel exploration variables (default False -> a
              deterministic, reproducible arena; see _ZeroGumbel).
    sh_seed   : seed for the Gumbel draws when sh_gumbel is True.
    shape_beta: {net_id: beta} -- forgiveness VALUE SHAPING at search time,
              per player. At each non-terminal leaf the backed-up value
              becomes clip(v + beta*(2*F-1), -1, 1), with F the net's own
              forgiveness head on that leaf, mirroring
              training/self_play_batched.py. This is the same mechanism the
              shaped checkpoints trained under, so switching it on here
              evaluates the METHOD; leaving it off evaluates only the policy
              those checkpoints learned. Per-player because in a head-to-head
              only one arm should be shaped.
              Requires eval_fns entries built with return_forgiveness=True
              for the shaped ids, and a parity-BLENDED head: the bonus is
              applied symmetrically at both players' leaves, so a _me or _opp
              head would boost one side's slack and penalise the other's.
              Terminal values are never shaped.
    gumbel_select: replace the default POST-OPENING argmax-by-visits with
              Gumbel-AlphaZero selection, argmax of
                  log prior + (gumbel_c_visit + max_visits) * gumbel_c_scale * Q
              over root children with visits >= gumbel_min_visits (default
              max(1, iterations // 100) -- arena searches have NO forced
              floor, so a guard keeps lucky low-visit Qs out of a score that
              multiplies Q by hundreds in logit space; see
              search.puct.select_move_gumbel). Opening plies keep the usual
              visit-temperature sampling. Ignored when decide_move is given.
    rng     : optional seeded np.random.Generator for the default opening-
              temperature sampling (reproducible evaluation runs). Ignored
              when decide_move is given -- the callback owns its randomness.

    Returns list of (pidx, a_score) for every ticket (a_score in {0.0,0.5,1.0}).
    """
    concurrency = max(1, min(concurrency, len(tickets)))
    cache = {} if use_cache else None
    ticket_iter = iter(tickets)
    results = []

    sh_on = bool(sequential_halving)
    if sh_on and SHState is None:
        raise ImportError(
            "sequential_halving=True but search.sequential_halving could not "
            "be imported -- refusing to fall back to PUCT silently, because "
            "the difference is invisible in the results and changes what the "
            "root Q estimates mean.")
    sh_rng = np.random.default_rng(sh_seed) if sh_gumbel else _ZeroGumbel()

    shape_beta = {k: float(v) for k, v in (shape_beta or {}).items()
                  if float(v) != 0.0}
    shaping_on = bool(shape_beta)
    # Does SHState steer on the shaped track? If not, shaping still reaches
    # the search through the interior PUCT descent on .value_sh -- which
    # changes WHICH leaves are visited, so the raw Qs differ too -- but the
    # root eliminations and pi' read raw values. That is a weaker channel,
    # not a broken one, and it is what a training run whose
    # sequential_halving.py also lacked the parameter would have used. Match
    # training rather than refusing.
    sh_shaped_ok = False
    ip_shaped_ok = False
    if shaping_on:
        if not sh_on:
            raise ValueError(
                "shape_beta needs sequential_halving=True. Under plain PUCT "
                "the shaped preference would reach the played move only "
                "through visit allocation, and the arena selects by pi' or "
                "visit argmax -- the mechanism would be half-connected.")
        import inspect
        sh_shaped_ok = ("shaped"
                        in inspect.signature(SHState.__init__).parameters)
        ip_shaped_ok = ("shaped"
                        in inspect.signature(improved_policy).parameters)
        if not (sh_shaped_ok and ip_shaped_ok):
            print("=" * 70)
            print("NOTE: search.sequential_halving does not support shaped=; "
                  "value shaping will")
            print("run in INTERIOR-ONLY mode. The PUCT descent below the root "
                  "steers on the")
            print("shaped track, but the root eliminations and pi' -- and so "
                  "the move played --")
            print("read raw values. Shaping still biases which leaves are "
                  "visited, so it is a")
            print("real but weaker channel.")
            print("")
            print("This MATCHES training if the run that produced the "
                  "checkpoint used the same")
            print("sequential_halving.py. Check with:")
            print("    grep -n 'shaped' training/self_play_batched.py")
            print("If training DID pass shaped=, update sequential_halving.py "
                  "before trusting")
            print("these numbers -- otherwise evaluation is running a weaker "
                  "mechanism than")
            print("the net trained under.")
            print("=" * 70)

    # per-game simulation budget (an int for everyone, or {net_id: sims})
    _iters_map = iterations if isinstance(iterations, dict) else None
    _iters_default = (max(iterations.values()) if _iters_map else iterations)

    def game_budget(g):
        return (_iters_map.get(g.search_net, _iters_default)
                if _iters_map else iterations)

    def finalize(g):
        r = g.env.result()
        if r is None:
            r = g.env.adjudicate()
        white_score = 0.5 if r == 0 else (1.0 if r > 0 else 0.0)
        g.a_score = white_score if g.a_is_white else (1.0 - white_score)
        g.done = True

    def advance_until_neural(g):
        """Play instant anchor moves / detect game end until it's a neural
        mover's turn (then set up its search root) or the game finishes."""
        while True:
            if g.env.isTerminal() or g.ply >= max_plies:
                finalize(g)
                return
            side = g.env.board.sideToMove
            mover = g.white if side == "white" else g.black
            if isinstance(mover, str):          # neural -> start a search
                g.root = Node()
                g.root.moverSign = 0
                g.sims_done = 0
                g.search_net = mover
                # Built lazily on the first descent after the root expands
                # (SHState needs children to sample top-m from). The arena
                # does not reuse subtrees, so every move starts from scratch.
                g.sh = None
                return
            move = mover.select(g.env, g.ply)   # anchor -> instant move
            if move is None:
                finalize(g)
                return
            g.env.step(move)
            g.ply += 1

    def start_next():
        """Pull the next ticket, fast-forward to its first neural move. Returns
        an active (awaiting-search) game, or None when tickets are exhausted.
        Games that finish before any neural move are recorded immediately."""
        while True:
            try:
                pidx, a_is_white, w, b = next(ticket_iter)
            except StopIteration:
                return None
            g = _AGame(pidx, a_is_white, w, b)
            advance_until_neural(g)
            if g.done:
                results.append((g.pidx, g.a_score))
                if on_game_done is not None:
                    on_game_done(g.pidx, g.a_score)
                continue
            return g

    active = []
    while len(active) < concurrency:
        g = start_next()
        if g is None:
            break
        active.append(g)

    while active:
        # ---- one simulation per active game; group leaves by searching net ----
        batches = {}
        for g in active:
            node = g.root
            env = g.env.clone()
            path = [node]
            at_root = True
            g_shaped = g.search_net in shape_beta
            while node.expanded and not node.terminal and node.children:
                best = None
                # ---- root: sequential halving decides who gets this visit ----
                # Interior nodes stay on PUCT, as in training: the root is
                # where both the played move and the statistics come from, and
                # replacing the interior selection is a larger change with its
                # own parameters.
                if at_root and sh_on:
                    if g.sh is None:
                        kw = ({"shaped": True}
                              if (g_shaped and sh_shaped_ok) else {})
                        g.sh = SHState(node,
                                       budget=game_budget(g) - node.visits,
                                       m=sh_m, rng=sh_rng,
                                       c_visit=sh_c_visit, c_scale=sh_c_scale,
                                       stat_width=sh_stat_width, **kw)
                    best = g.sh.next_child()
                    # None once the schedule is spent -- PUCT below soaks up
                    # any residual budget (only happens if the plan underspent).
                if best is None:
                    sqrt_pv = math.sqrt(node.visits)
                    best_score = -1e30
                    # Descend on whichever track this player's search runs on:
                    # shaping only steers if the exploitation term reads it.
                    if g_shaped:
                        fpu_q = _fpu_q_sh(node, fpu_reduction)
                        for ch in node.children:
                            v = ch.visits
                            q = ch.value_sh / v if v else fpu_q
                            s = q + c * ch.prior * sqrt_pv / (1 + v)
                            if s > best_score:
                                best_score = s
                                best = ch
                    else:
                        fpu_q = node_fpu_q(node, fpu_reduction)
                        for ch in node.children:
                            v = ch.visits
                            q = ch.value / v if v else fpu_q
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
                # net inputs include the halfmove clock / repetition count now,
                # so the cache key must too (see self_play_batched)
                key = (g.search_net, env.board.stateKey(),
                       env.halfmove_clock, env.counts.get(env.board.stateKey(), 1))
                hit = cache.get(key)
                if hit is not None:
                    # Cache entries carry BOTH tracks: beta and the net are
                    # fixed for the whole call, so a hit agrees on both.
                    priors, value, v_sh = hit
                    _expand(node, priors, mover, net_value=value)
                    sgn = 1.0 if mover == "white" else -1.0
                    _backprop(path, sgn * value, sgn * v_sh)
                    g.sims_done += 1
                    continue

            batches.setdefault(g.search_net, []).append(
                (g, node, env, legal, path, mover))

        # ---- one batched forward per distinct searching net ----
        for net_id, items in batches.items():
            planes = [encode_env(it[2]) for it in items]
            beta = shape_beta.get(net_id, 0.0)
            out = eval_fns[net_id](planes)
            if beta:
                if len(out) != 3:
                    raise ValueError(
                        f"shape_beta set for {net_id!r} but its eval_fn "
                        "returns 2 arrays -- build it with "
                        "_make_eval_fn(net, return_forgiveness=True)")
                logits_b, values_b, f_b = out
                # v' = clip(v + beta*(2F-1), -1, 1), F in the POV of the
                # player to move at the leaf, exactly as in training.
                shaped_b = np.clip(
                    values_b + beta * (2.0 * f_b - 1.0), -1.0, 1.0)
            else:
                logits_b, values_b = out[0], out[1]
                shaped_b = values_b
            for (g, node, env, legal, path, mover), logits, value, v_sh in zip(
                    items, logits_b, values_b, shaped_b):
                idxs = [encodeMovePOV(m, mover) for m in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {m: float(p) for m, p in zip(legal, probs)}
                value = float(value)
                v_sh = float(v_sh)
                if cache is not None and len(cache) < cache_cap:
                    cache[(net_id, env.board.stateKey(), env.halfmove_clock,
                           env.counts.get(env.board.stateKey(), 1))] = (
                        priors, value, v_sh)
                _expand(node, priors, mover, net_value=value)
                sgn = 1.0 if mover == "white" else -1.0
                _backprop(path, sgn * value, sgn * v_sh)
                g.sims_done += 1

        # ---- games whose search is complete pick a move; refill the pool ----
        still = []
        for g in active:
            g_iters = game_budget(g)
            if g.sims_done < g_iters:
                still.append(g)
                continue
            visit_counts = {ch.move: ch.visits for ch in g.root.children}
            if not visit_counts:
                finalize(g)
            else:
                if decide_move is not None:
                    move = decide_move(g, visit_counts)
                elif sh_on:
                    # pi' over the survivors, NOT argmax visits (see
                    # select_move_sh). Opening plies still get a temperature,
                    # applied to pi' rather than to the halving schedule.
                    move = select_move_sh(
                        g.root, g.sh,
                        opening_temp if g.ply < opening_plies else 0.0,
                        rng, c_visit=sh_c_visit, c_scale=sh_c_scale,
                        shaped=(ip_shaped_ok
                                and g.search_net in shape_beta))
                    if move is None:
                        move = select_move(visit_counts, 0.0, rng)
                elif gumbel_select and g.ply >= opening_plies:
                    move = select_move_gumbel(
                        g.root, temp=0.0, rng=rng,
                        c_visit=gumbel_c_visit, c_scale=gumbel_c_scale,
                        min_visits=(gumbel_min_visits if gumbel_min_visits
                                    else max(1, g_iters // 100)),
                        fallback_counts=visit_counts)
                else:
                    temp = opening_temp if g.ply < opening_plies else 0.0
                    move = select_move(visit_counts, temp, rng)
                g.env.step(move)
                g.ply += 1
                if g.env.isTerminal() or g.ply >= max_plies:
                    finalize(g)
                else:
                    advance_until_neural(g)     # play anchor moves / set next search
            if g.done:
                results.append((g.pidx, g.a_score))
                if on_game_done is not None:
                    on_game_done(g.pidx, g.a_score)
                ng = start_next()
                if ng is not None:
                    still.append(ng)
            else:
                still.append(g)
        active = still

    return results


# --------------------------------------------------------------------------- #
# torch evaluator (lazy import)
# --------------------------------------------------------------------------- #
def _make_eval_fn(net, return_forgiveness=False):
    """planes list -> (logits, values), or (logits, values, forgiveness) when
    return_forgiveness=True.

    The third output costs nothing extra: the forgiveness head hangs off the
    same trunk, so net(x, return_forgiveness=True) is ONE forward returning
    all three. Value shaping therefore adds no network calls to the search."""
    import torch
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def eval_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    out = net(x, return_forgiveness=True) \
                        if return_forgiveness else net(x)
            else:
                out = net(x, return_forgiveness=True) \
                    if return_forgiveness else net(x)
        if return_forgiveness:
            policy_logits, value, f = out
            return (policy_logits.float().cpu().numpy(),
                    value.float().cpu().numpy().reshape(-1),
                    f.float().cpu().numpy().reshape(-1))
        policy_logits, value = out
        return (policy_logits.float().cpu().numpy(),
                value.float().cpu().numpy().reshape(-1))

    return eval_fn


# --------------------------------------------------------------------------- #
# pairing -> tickets, and a run-a-set-of-pairings helper (torch-free; the eval
# functions and anchor builder are injected, so this is unit-testable on CPU and
# is the single body shared by the serial path and each parallel worker)
# --------------------------------------------------------------------------- #
def _build_tickets(pairings, games, mover_for, seed_base):
    """pairings: list of (global_pidx, a_name, b_name). Returns ticket list with
    colours alternating per game; seeds derived from the GLOBAL pidx so they (and
    therefore the games, and the cache) are identical regardless of worker count."""
    tickets = []
    for (gpidx, a, b) in pairings:
        for gi in range(games):
            a_is_white = (gi % 2 == 0)
            seed = seed_base + gpidx * games + gi
            white_name, black_name = (a, b) if a_is_white else (b, a)
            tickets.append((gpidx, a_is_white,
                            mover_for(white_name, seed),
                            mover_for(black_name, seed)))
    return tickets


def _run_pairings(pairings, games, eval_fns, mover_for, on_pairing, *,
                  iterations, c, opening_plies, opening_temp, max_plies,
                  concurrency, use_cache, cache_cap, seed_base,
                  on_each_game=None, sh=None):
    """Play every game of `pairings` with the batched runner, aggregate per
    pairing, and call on_pairing(a, b, w, d, l, n) as each pairing completes.

    `sh`: optional dict of sequential-halving kwargs for the runner
    (sequential_halving / sh_m / sh_stat_width / sh_c_visit / sh_c_scale /
    sh_gumbel / sh_seed), passed straight through."""
    names = {gpidx: (a, b) for (gpidx, a, b) in pairings}
    agg = {gpidx: {"w": 0, "d": 0, "l": 0, "n": 0} for (gpidx, a, b) in pairings}
    received = {gpidx: 0 for (gpidx, a, b) in pairings}

    def on_game_done(gpidx, a_score):
        rec = agg[gpidx]
        if a_score == 1.0:
            rec["w"] += 1
        elif a_score == 0.0:
            rec["l"] += 1
        else:
            rec["d"] += 1
        rec["n"] += 1
        received[gpidx] += 1
        if on_each_game is not None:
            on_each_game()
        if received[gpidx] == games:
            a, b = names[gpidx]
            on_pairing(a, b, rec["w"], rec["d"], rec["l"], rec["n"])

    tickets = _build_tickets(pairings, games, mover_for, seed_base)
    run_elo_matches_batched(
        tickets, eval_fns,
        iterations=iterations, c=c,
        opening_plies=opening_plies, opening_temp=opening_temp,
        max_plies=max_plies, concurrency=concurrency,
        use_cache=use_cache, cache_cap=cache_cap,
        on_game_done=on_game_done,
        **(sh or {}),
    )


# --------------------------------------------------------------------------- #
# parallel worker (one process; loads only the nets its pairings reference)
# --------------------------------------------------------------------------- #
def _elo_worker(pairings, spec, games, cfg, seed_base, result_queue):
    import random
    import torch
    from evaluation.arena import RandomAgent, MaterialAgent, load_net
    from model.network import ChessNet

    device = torch.device(cfg["device"])
    # load just the nets this worker's pairings need (deduped)
    needed = {nm for (_p, a, b) in pairings for nm in (a, b) if nm in spec}
    eval_fns = {}
    for nm in needed:
        s = spec[nm]
        net = ChessNet().to(device).eval() if s == "untrained" else load_net(s, device)
        eval_fns[nm] = _make_eval_fn(net)

    def mover_for(name, seed):
        if name in spec:
            return name
        if name == "random":
            return RandomAgent(random.Random(seed))
        if name == "material":
            return MaterialAgent(random.Random(seed))
        raise ValueError(f"unknown player {name!r}")

    def on_pairing(a, b, w, d, l, n):
        result_queue.put(("pairing", a, b, w, d, l, n))

    def on_each_game():
        result_queue.put(("game",))

    try:
        _run_pairings(
            pairings, games, eval_fns, mover_for, on_pairing,
            iterations=cfg["iterations"], c=cfg["c"],
            opening_plies=cfg["opening_plies"], opening_temp=cfg["opening_temp"],
            max_plies=cfg["max_plies"], concurrency=cfg["concurrency"],
            use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"],
            seed_base=seed_base, on_each_game=on_each_game,
            sh=cfg.get("sh"),
        )
    finally:
        result_queue.put(("worker_done",))


def _chunk(seq, n):
    """Split into n roughly-equal CONTIGUOUS chunks (keeps each worker's pairings
    on a nearby range of checkpoints, so it loads few distinct nets)."""
    n = max(1, min(n, len(seq)))
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for w in range(n):
        size = k + (1 if w < r else 0)
        out.append(seq[i:i + size])
        i += size
    return [c for c in out if c]

# --------------------------------------------------------------------------- #
# checkpoint discovery
# --------------------------------------------------------------------------- #
def discover_checkpoints(ckpt_dir):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "net_iter*.pt")):
        m = re.search(r"net_iter(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)            # [(iteration, path), ...] ascending


# --------------------------------------------------------------------------- #
# Elo fit  (Bradley-Terry MM, ties as half-wins, light prior, random pinned 0)
# --------------------------------------------------------------------------- #
def fit_elo(names, results, pin="random", prior_games=2.0, steps=400):
    """
    names    : list of player names (index = player id)
    results  : list of (i, j, score_i, n)  -- score_i = i's points over n games
    Returns  : dict name -> Elo, with `pin` at 0.
    """
    P = len(names)
    gamma = [1.0] * P                      # BT strengths; Elo = 400*log10(gamma)
    wins = [0.0] * P                       # total points (wins + 0.5*draws)
    pairs = {}                             # i -> list of (j, n)
    for i in range(P):
        pairs[i] = []
    for (i, j, s_i, n) in results:
        wins[i] += s_i
        wins[j] += (n - s_i)
        pairs[i].append((j, n))
        pairs[j].append((i, n))

    # MM iterations: gamma_i = (W_i + prior/2) / ( sum_j n_ij/(gamma_i+gamma_j) + prior/(gamma_i+1) )
    for _ in range(steps):
        new = list(gamma)
        for i in range(P):
            denom = prior_games / (gamma[i] + 1.0)        # virtual draws vs rating 0
            for (j, n) in pairs[i]:
                denom += n / (gamma[i] + gamma[j])
            if denom > 0:
                new[i] = (wins[i] + 0.5 * prior_games) / denom
        gamma = new

    pin_idx = names.index(pin) if pin in names else 0
    ref = gamma[pin_idx]
    elo = {}
    for i, name in enumerate(names):
        elo[name] = 400.0 * math.log10(gamma[i] / ref) if gamma[i] > 0 else float("-inf")
    return elo


# --------------------------------------------------------------------------- #
# match cache (resumable)
# --------------------------------------------------------------------------- #
def load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["a"], row["b"])
            cache[key] = (int(row["a_wins"]), int(row["draws"]),
                          int(row["b_wins"]), int(row["games"]))
    return cache


def append_cache(path, a, b, wins, draws, losses, games):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["a", "b", "a_wins", "draws", "b_wins", "games"])
        w.writerow([a, b, wins, draws, losses, games])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    import argparse
    import csv
    import os
    import random
    import sys
    import time
    import multiprocessing as mp

    import torch

    from evaluation.arena import RandomAgent, MaterialAgent, load_net

    from model.network import ChessNet

    ap = argparse.ArgumentParser(description="Single/multi-process batched checkpoint Elo scoring")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--anchors", default="random,material")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=400,
                    help="PUCT sims/move. For RANKING, 64-100 is plenty (training uses 400).")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--opening-temp", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=128,
                    help="games run/batched simultaneously PER PROCESS")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes to split pairings over (1 = single process). "
                         "Each worker loads only the nets its pairings need.")
    ap.add_argument("--stride", type=int, default=1,
                    help="test every Nth checkpoint (1 = all); final always kept")
    ap.add_argument("--every-iters", type=int, default=0,
                    help="keep ~1 checkpoint per this many ITERATIONS "
                         "(0 = off; overrides --stride). Final always kept.")
    ap.add_argument("--last-iters", type=int, default=0,
                    help="only score checkpoints within this many iterations of "
                         "the final one (0 = all). Applied before --every-iters/--stride.")
    ap.add_argument("--sequential-halving", action="store_true",
                    help="allocate ROOT visits by Gumbel top-m + sequential "
                         "halving instead of PUCT, as full-search self-play "
                         "moves do, and play the pi' argmax rather than the "
                         "visit argmax. Required for any measurement that "
                         "compares root Q values across actions.")
    ap.add_argument("--sh-m", type=int, default=16,
                    help="root actions considered (top-m). At small budgets "
                         "prefer 8: plan_phases(800, 8) puts the statistics "
                         "set at 99 visits vs 31 for (300, 16).")
    ap.add_argument("--sh-stat-width", type=int, default=4,
                    help="actions snapshotted for the statistics set")
    ap.add_argument("--sh-c-visit", type=float, default=50.0)
    ap.add_argument("--sh-c-scale", type=float, default=0.02,
                    help="match the training config (0.02)")
    ap.add_argument("--sh-gumbel", action="store_true",
                    help="keep the Gumbel exploration draws (default off: "
                         "deterministic, reproducible rated games)")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-cap", type=int, default=250_000)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpts = discover_checkpoints(args.ckpt_dir)
    if not ckpts:
        print(f"no net_iter*.pt found in {args.ckpt_dir}")
        return
    if args.last_iters > 0:
        cutoff = ckpts[-1][0] - args.last_iters    # relative to final checkpoint
        ckpts = [(it, p) for it, p in ckpts if it >= cutoff]
    if args.every_iters > 0:
        kept, last = [], None
        for it, p in ckpts:                      # ascending by iteration
            if last is None or it - last >= args.every_iters:
                kept.append((it, p)); last = it
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept
    elif args.stride > 1:
        kept = ckpts[::args.stride]
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept

    anchors = [a for a in args.anchors.split(",") if a]
    ckpt_names = [f"iter{it}" for it, _ in ckpts]
    spec = {f"iter{it}": os.path.join(args.ckpt_dir, f"net_iter{it}.pt") for it, _ in ckpts}
    players = anchors + ckpt_names
    name_idx = {n: i for i, n in enumerate(players)}

    schedule = []
    for cn in ckpt_names:
        for a in anchors:
            schedule.append((cn, a))
    if args.round_robin:
        for x in range(len(ckpt_names)):
            for y in range(x + 1, len(ckpt_names)):
                schedule.append((ckpt_names[x], ckpt_names[y]))
    else:
        for x in range(len(ckpt_names) - 1):
            schedule.append((ckpt_names[x], ckpt_names[x + 1]))

    cache_path = os.path.join(args.ckpt_dir, "elo_matches.csv")
    match_cache = load_cache(cache_path)

    cached, pending = [], []
    for (a, b) in schedule:
        if (a, b) in match_cache:
            aw, dr, bw, g = match_cache[(a, b)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        elif (b, a) in match_cache:
            bw, dr, aw, g = match_cache[(b, a)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        else:
            pending.append((a, b))

    print(f"{len(ckpts)} checkpoints, anchors={anchors}, {args.games} games/match, "
          f"{args.iterations} sims/move, concurrency={args.concurrency}, "
          f"workers={args.workers}, device={device}")
    print(f"{len(cached)} pairings cached, {len(pending)} to play "
          f"({len(pending) * args.games} games)")

    results = list(cached)
    # global pidx per pending pairing (keeps seeds/cache identical across worker counts)
    pending_idx = [(i, a, b) for i, (a, b) in enumerate(pending)]
    total_games = len(pending) * args.games
    t0 = time.time()

    def progress(done):
        frac = done / total_games if total_games else 1.0
        rate = done / (time.time() - t0) if time.time() > t0 else 0.0
        eta = (total_games - done) / rate if rate > 0 else 0.0
        sys.stdout.write(f"\r  games {done}/{total_games} ({frac*100:3.0f}%)  "
                         f"{rate:.1f}/s  eta {int(eta)}s   ")
        sys.stdout.flush()

    def record_pairing(a, b, w, d, l, n):
        append_cache(cache_path, a, b, w, d, l, n)
        results.append((name_idx[a], name_idx[b], w + 0.5 * d, n))

    if pending:
        cfg = dict(iterations=args.iterations, c=args.c,
                   opening_plies=args.opening_plies, opening_temp=args.opening_temp,
                   max_plies=args.max_plies, concurrency=args.concurrency,
                   use_cache=not args.no_cache, cache_cap=args.cache_cap,
                   device=str(device),
                   sh=dict(sequential_halving=args.sequential_halving,
                           sh_m=args.sh_m, sh_stat_width=args.sh_stat_width,
                           sh_c_visit=args.sh_c_visit,
                           sh_c_scale=args.sh_c_scale,
                           sh_gumbel=args.sh_gumbel, sh_seed=args.seed))

        if args.workers <= 1:
            # ---- single process ----
            needed = {nm for (_p, a, b) in pending_idx for nm in (a, b) if nm in spec}
            eval_fns = {}
            for nm in needed:
                s = spec[nm]
                eval_fns[nm] = _make_eval_fn(
                    ChessNet().to(device).eval() if s == "untrained" else load_net(s, device))
            print(f"loaded {len(eval_fns)} distinct nets onto {device}")

            def mover_for(name, seed):
                if name in spec:
                    return name
                if name == "random":
                    return RandomAgent(random.Random(seed))
                if name == "material":
                    return MaterialAgent(random.Random(seed))
                raise ValueError(f"unknown player {name!r}")

            done = [0]
            def on_each():
                done[0] += 1; progress(done[0])
            progress(0)
            _run_pairings(pending_idx, args.games, eval_fns, mover_for, record_pairing,
                          iterations=cfg["iterations"], c=cfg["c"],
                          opening_plies=cfg["opening_plies"], opening_temp=cfg["opening_temp"],
                          max_plies=cfg["max_plies"], concurrency=cfg["concurrency"],
                          use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"],
                          seed_base=args.seed, on_each_game=on_each,
                          sh=cfg["sh"])
            print()
        else:
            # ---- multi process: contiguous pairing chunks, each worker self-contained ----
            chunks = _chunk(pending_idx, args.workers)
            print(f"splitting {len(pending)} pairings over {len(chunks)} workers")
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            procs = []
            for ch in chunks:
                p = ctx.Process(target=_elo_worker,
                                args=(ch, spec, args.games, cfg, args.seed, result_queue))
                p.start()
                procs.append(p)

            done = 0
            finished_workers = 0
            progress(0)
            while finished_workers < len(chunks):
                msg = result_queue.get()
                if msg[0] == "game":
                    done += 1
                    progress(done)
                elif msg[0] == "pairing":
                    _, a, b, w, d, l, n = msg
                    record_pairing(a, b, w, d, l, n)
                elif msg[0] == "worker_done":
                    finished_workers += 1
            for p in procs:
                p.join()
            print()

    pin = "random" if "random" in players else players[0]
    elo = fit_elo(players, results, pin=pin)

    print(f"\n{'='*46}\n  Elo ratings ({pin} = 0)\n{'='*46}")
    for a in anchors:
        print(f"  {a:14} {elo[a]:+7.0f}")
    print("  " + "-" * 30)
    rows = []
    for it, _ in ckpts:
        name = f"iter{it}"
        print(f"  {name:14} {elo[name]:+7.0f}")
        rows.append((it, elo[name]))

    out_csv = os.path.join(args.ckpt_dir, "elo_ratings.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "elo"])
        for it, e in rows:
            w.writerow([it, f"{e:.1f}"])
    print(f"\n  wrote {out_csv}")

    if rows:
        es = [e for _, e in rows]
        lo, hi = min(es), max(es)
        span = (hi - lo) or 1.0
        print(f"\n  Elo vs iteration  ({lo:+.0f} .. {hi:+.0f})")
        for it, e in rows:
            bar = "#" * int(1 + 40 * (e - lo) / span)
            print(f"  {it:>4} {bar}")


if __name__ == "__main__":
    main()

    
"""
Robustness arena: measure how two models' strength DEGRADES under injected
mistakes -- the evaluation the forgiveness project is aiming at ("policies
tolerant to noise, approximation error, or small distribution shifts").

Two match designs
-----------------
HEAD-TO-HEAD (default):    A vs B, both perturbed IDENTICALLY at each noise
    level. Isolates *relative* robustness: if A's play collapses faster under
    noise, its score drops as epsilon rises even if A wins the clean match.
BENCHMARK (--benchmark C.pt):  A vs C and B vs C at each level, where the
    benchmark C always plays CLEAN (no perturbation, argmax after the shared
    opening). A common, fixed yardstick: you get two independent degradation
    curves score_A(eps), score_B(eps) against the same opponent. This is the
    cleaner design for the report -- head-to-head confounds "A got worse"
    with "B got better at punishing".

Mistake mechanisms (combinable; all activate only AFTER the shared
opening-randomisation plies so they don't confound opening diversity)
---------------------------------------------------------------------
eps + --mode random    with prob eps play a UNIFORM RANDOM legal move
                       (worst-case noise -- an actuator fault, not a chess
                       mistake).
eps + --mode blunder   with prob eps play a searched-but-inferior move:
                       sampled from the root's NON-best children by visits.
                       A chess-shaped mistake -- plausible but suboptimal.
--vnoise sigma         Gaussian noise added to the value head during search.
                       NOTE: the runner caches evals per position, so this is
                       a FIXED PER-POSITION evaluation error, not iid noise
                       per visit -- i.e. a consistently-wrong evaluator,
                       which is precisely "approximation error / distribution
                       shift" rather than jitter that averages out over
                       visits.
--temp-noise t         sample every post-opening move from visits^(1/t)
                       instead of argmax (diffuse imprecision). NOTE:
                       bypasses --select-* (it replaces the base selection).

ROOT VISIT ALLOCATION (new, on by default)
------------------------------------------
The root runs Gumbel top-m + SEQUENTIAL HALVING, as full-search self-play
moves do, instead of PUCT. This is a precondition for everything below, not a
refinement: the delta-window and the forgiveness statistics are functions of
DIFFERENCES between root Q values, and a PUCT root leaves second-tier children
holding a handful of visits and a Q that is one leaf evaluation. Against a
delta of 0.05 the near-optimal set would then be populated by sampling noise.

Halving gives every survivor of a phase the same visit count by construction.
The top --sh-stat-width (default 4) actions are snapshotted at the last phase
advance and are what the delta-window is taken over -- the final PAIR is too
narrow, since a Q-entropy over two actions is a monotone function of the
action gap it was introduced to improve on. The startup banner prints the
realised schedule and how many visits back that set; put that number in the
write-up next to delta.

Two consequences, both handled here: visit counts no longer rank actions (they
encode the elimination schedule), so greedy play, the opening temperature and
the injected blunders all go through pi' instead; and Gumbel exploration is
OFF by default, so two arms differing in selection rule consider the same
actions and the run is reproducible. --no-sequential-halving restores the old
PUCT behaviour for ablation.

Selection policies (--select-a / --select-b; the benchmark always plays
greedy) -- THE forgiveness experiment: same checkpoint, different move rule
---------------------------------------------------------------------------
greedy            the pi' argmax after the opening (argmax visits under
                  --no-sequential-halving).
maxent            sample from the Boltzmann policy exp(Q/alpha) over the
                  searched candidates -- the optimal policy of the
                  maximum-entropy objective (S2.1.6). THE BASELINE THIS WORK
                  IS ARGUED AGAINST, available without training a MaxEnt
                  agent because MaxEnt's acting rule is a function of Q. Same
                  Boltzmann construction the forgiveness statistic uses, put
                  to the other use: MaxEnt samples from the distribution,
                  forgiveness measures its entropy and acts greedily
                  (S2.2.2). Calibrate --maxent-alpha to cost the same clean
                  Elo as --delta, then compare degradation: that asks whether
                  acting stochastically or preferring flat states buys more
                  robustness per point of strength surrendered. Caveat for
                  the write-up: a MaxEnt-TRAINED agent would also learn a
                  soft value function rewarding it for reaching states where
                  uncertainty is cheap; this reproduces the acting half only.
forgiving_tree    delta-constrained forgiveness selection from SUBTREE
                  statistics: among { a : Q1 - Q(a) <= --delta } play the
                  candidate with the highest child_forgiveness (--sel-stat /
                  --sel-agg / --sel-parity; parity=1 = my future slack).
forgiving_head    same delta-set, but scored by the checkpoint's TRAINED
                  FORGIVENESS HEAD evaluated on each candidate's SUCCESSOR
                  position (one batched forward per decision).

PERSPECTIVE FLIP for forgiving_head -- read this before picking a head
checkpoint: the head sees the successor s_a, where the side to move is the
OPPONENT. A head trained on a *_me target therefore predicts the OPPONENT's
slack at s_a, and a *_opp head predicts YOUR post-move slack. So to select
for your own robustness, load the *_opp* head checkpoint (e.g.
net_iter7500_forgiveness_flat_entropy_opp.pt); unfiltered heads
(flat_entropy, tree_entropy) predict the parity-blended slack of the
continuation and need no flip.

Typical selection experiment (same trunk, three rules, clean benchmark):
    python robustness_arena.py \
        forgiveness_heads_iter7500/net_iter7500_forgiveness_flat_entropy_opp.pt \
        forgiveness_heads_iter7500/net_iter7500_forgiveness_flat_entropy_opp.pt \
        --benchmark checkpoints/net_iter2400.pt \
        --select-a greedy --select-b forgiving_head --delta 0.05 \
        --games 600 --sims 800 --eps 0,0.05,0.1,0.2 --mode blunder \
        --out sel_head.csv
(A and B are the SAME checkpoint -- only the selection rule differs, so the
two degradation curves isolate the selector. Run again with
--select-b forgiving_tree for the subtree-statistic agent.)

Sweeps: --eps and --vnoise each take a comma list; levels are their cartesian
product. Both sides of a perturbed pairing get the SAME level.

Output: a printed table and a CSV with one row per (level, pairing):
games, W/D/L, score, Elo diff with a 95% CI, and injected-mistakes-per-game
(the realised mistake rate -- e.g. eps=0.1 over ~40 post-opening moves is ~4
forced mistakes a game; report this, reviewers will ask).

Examples
--------
# sanity run (small, fast):
python robustness_arena.py checkpoints/net_iter2400.pt checkpoints/net_iter1698.pt \
    --games 40 --sims 200 --eps 0,0.1

# the real sweep, vs a clean benchmark:
python robustness_arena.py A.pt B.pt --benchmark checkpoints/net_iter2400.pt \
    --games 200 --sims 300 --eps 0,0.02,0.05,0.1,0.2 --mode blunder \
    --out robustness.csv

# evaluator-error axis instead of move-error axis:
python robustness_arena.py A.pt B.pt --games 200 --sims 300 \
    --eps 0 --vnoise 0,0.05,0.1,0.2
"""

# MUST be first: patches Board.legalMoves() with the Cython generator
# before anything imports engine.board. The search is move-generation
# bound, so this is worth roughly 3x end to end.
try:
    from evaluation.fast_movegen_boot import (ensure_fast_movegen,
                                              status as movegen_status)
except ImportError:
    from fast_movegen_boot import (ensure_fast_movegen,
                                   status as movegen_status)
ensure_fast_movegen()

import argparse
import csv
import json
import math
import os
import time
import zlib

import numpy as np
import torch

try:                                    # repo-layout tolerant imports
    from evaluation.score_elo_batched import (run_elo_matches_batched,
                                              _make_eval_fn, select_move,
                                              select_move_sh)
except ImportError:
    from score_elo_batched import (run_elo_matches_batched,
                                   _make_eval_fn, select_move,
                                   select_move_sh)
try:
    from evaluation.arena import load_net
except ImportError:
    from arena import load_net
try:
    from search.forgiveness import (select_move_forgiving,
                                    flat_forgiveness_hybrid,
                                    forgiveness_from_qs)
except ImportError:
    from forgiveness import (select_move_forgiving, flat_forgiveness_hybrid,
                             forgiveness_from_qs)
try:
    from model.encoding import encode_env
except ImportError:
    from encoding import encode_env
try:
    from search.sequential_halving import improved_policy
except ImportError:
    try:
        from sequential_halving import improved_policy
    except ImportError:
        improved_policy = None

# Does this copy of improved_policy steer on the shaped value track? If not,
# shaping runs interior-only (see score_elo_batched's note) and passing the
# kwarg would just raise.
def _probe_ip_shaped():
    if improved_policy is None:
        return False
    try:
        import inspect
        return "shaped" in inspect.signature(improved_policy).parameters
    except (TypeError, ValueError):
        return False


_IP_SHAPED = _probe_ip_shaped()


# --------------------------------------------------------------------------- #
# perturbations
# --------------------------------------------------------------------------- #
def wrap_value_noise(eval_fn, sigma, seed):
    """Add N(0, sigma) to the value head, clipped to [-1, 1]. Combined with
    the runner's per-position eval cache this realises a fixed per-position
    evaluation error (see module docstring).

    Passes a third (forgiveness) output through untouched: this axis models
    error in the VALUE estimate, and noising the forgiveness head too would
    confound two different error sources in one knob."""
    if sigma <= 0:
        return eval_fn
    rng = np.random.default_rng(seed)

    def noisy(planes_list):
        out = eval_fn(planes_list)
        logits, values = out[0], out[1]
        v = np.asarray(values, dtype=np.float64)
        v = np.clip(v + rng.normal(0.0, sigma, size=v.shape), -1.0, 1.0)
        return (logits, v, out[2]) if len(out) == 3 else (logits, v)
    return noisy


def make_head_F_fn(net):
    """planes list -> forgiveness-head outputs in [0, 1] (one batched
    forward, return_forgiveness=True)."""
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def head_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    _logits, _value, f = net(x, return_forgiveness=True)
            else:
                _logits, _value, f = net(x, return_forgiveness=True)
        return f.float().cpu().numpy().reshape(-1)
    return head_fn


class Perturber:
    """decide_move callback for run_elo_matches_batched. Applies the shared
    opening temperature to EVERYONE, then per-player BASE SELECTION
    (greedy / forgiving_tree / forgiving_head, see `selectors`) with
    per-player mistakes layered on top for perturbed ids. Counts decisions
    and injected mistakes per player for the realised-rate report, plus
    forgiveness-selection stats (delta-set size, switch rate, Q sacrifice)."""

    def __init__(self, perturbed_ids, eps, mode, temp_noise,
                 opening_plies, opening_temp, seed, selectors=None,
                 diag_delta=None, sh=False, sh_c_visit=50.0, sh_c_scale=0.02,
                 record_rows=False, stat_tau=0.0178, shaped_ids=()):
        self.perturbed = set(perturbed_ids)
        self.eps = float(eps)
        self.mode = mode
        self.temp_noise = float(temp_noise)
        self.opening_plies = opening_plies
        self.opening_temp = opening_temp
        self.sh = bool(sh)              # runner is in sequential-halving mode
        self.sh_c_visit = sh_c_visit
        self.sh_c_scale = sh_c_scale
        self.rng = np.random.default_rng(seed)
        self.selectors = selectors or {}    # net_id -> selection spec dict
        self.diag_delta = diag_delta        # log greedy |S| at this delta
        self.decisions = {}             # net_id -> post-opening decisions
        self.mistakes = {}              # net_id -> injected mistakes
        self.sel_stats = {}             # net_id -> dict(n, switched, sacrifice,
                                        #               ndelta_sum)
        # ---- per-GAME bookkeeping -------------------------------------- #
        # eps is a per-MOVE rate, so a longer game absorbs more injected
        # mistakes at the same eps. If one arm draws out longer games -- which
        # is exactly what a robustness-seeking selector might do -- its
        # mistakes-per-game rises for reasons that have nothing to do with
        # robustness. Without game length on the record the two stories are
        # indistinguishable, so length is tracked per game and reported.
        self.game = {}                  # (pidx, net_id) -> per-game counters
        self.game_plies = {}            # pidx -> plies seen (both sides)
        # ---- per-DECISION dataset -------------------------------------- #
        self.rows = []                  # one dict per post-opening decision
        self.record_rows = bool(record_rows)
        self.stat_tau = stat_tau
        # Players whose search runs on shaped values. Their DECISIONS read
        # .value_sh; the state statistics RECORDED for them still read the
        # raw .value, so an arm's reported forgiveness is never a statistic
        # of its own bonus.
        self.shaped_ids = set(shaped_ids)

    # ---- what the root offers: candidate set and the unperturbed best move -- #
    def _candidates(self, g):
        """The children whose Qs may be compared. Under halving this is the
        statistics set snapshotted at the last phase advance -- matched in
        visits by construction. Under PUCT there is no such set and the
        caller falls back to select_move_forgiving's floor logic."""
        sh = getattr(g, "sh", None)
        return sh.stat_children() if sh is not None else None

    def _greedy(self, g, visit_counts):
        """The move an UNMODIFIED agent plays. Under halving that is the pi'
        argmax over the survivors, NOT argmax visits: every survivor of a
        phase holds the same count, so visit counts rank the elimination
        schedule rather than the actions. For a shaped player pi' reads the
        shaped track -- under halving that is the only route by which the
        bonus reaches the played move."""
        if self.sh and getattr(g, "sh", None) is not None:
            mv = select_move_sh(g.root, g.sh, 0.0,
                                c_visit=self.sh_c_visit,
                                c_scale=self.sh_c_scale,
                                shaped=g.search_net in self.shaped_ids)
            if mv is not None:
                return mv
        return max(visit_counts, key=visit_counts.get)

    def _q_best(self, g, visit_counts):
        """Q-argmax over the candidate set -- the reference point the
        delta-window is measured from, and what a forgiving selector falls
        back to. Under halving the pi' argmax and the Q argmax can differ
        (pi' also carries the prior), so they are kept separate."""
        cands = self._candidates(g)
        if not cands:
            cands = [c for c in g.root.children if c.visits > 0]
        if not cands:
            return self._greedy(g, visit_counts)
        shaped = g.search_net in self.shaped_ids
        key = ((lambda c: c.value_sh / c.visits) if shaped
               else (lambda c: c.value / c.visits))
        return max(cands, key=key).move

    def _game_rec(self, g, pid):
        key = (g.pidx, pid)
        rec = self.game.get(key)
        if rec is None:
            rec = dict(decisions=0, mistakes=0, switched=0, sacrifice=0.0,
                       ndelta_sum=0, f_sum=0.0, f_n=0, gap_sum=0.0)
            self.game[key] = rec
        return rec

    def _local_stats(self, g):
        """The forgiveness statistics of the state the player is standing in,
        read off the root's candidate set -- free, since the search already
        computed the Qs.

        THIS IS THE MECHANISM CHECK. A difference in degradation slope between
        two arms says the scores moved; it does not say the agent reached
        flatter states, which is what the objective actually claims. Comparing
        these distributions across arms is what connects the two."""
        cands = self._candidates(g)
        if not cands:
            cands = [c for c in g.root.children if c.visits > 0]
        if len(cands) < 2:
            return None
        qs = sorted((c.value / c.visits for c in cands), reverse=True)
        st = forgiveness_from_qs(qs, self.stat_tau)
        st["q1"] = qs[0]
        st["q2"] = qs[1]
        st["n_cand"] = len(qs)
        return st

    def _head_score_children(self, g, head_fn):
        """score_children hook: forgiveness-head value of each candidate's
        SUCCESSOR position, one batched forward. See the module docstring's
        PERSPECTIVE FLIP note for which head that value belongs to."""
        def score(children):
            planes = []
            for c in children:
                env2 = g.env.clone()
                env2.step(c.move)
                planes.append(encode_env(env2))
            return list(head_fn(planes))
        return score

    def _hybrid_score_children(self, g, spec):
        """score_children hook: flat subtree statistic with the head imputing
        the undefined frontier (flat_forgiveness_hybrid). Load a *_me flat
        head for parity=1 -- imputation nodes are positions where the
        aggregate's owner is to move, so no perspective flip (module
        docstring)."""
        head_fn = spec["head_fn"]

        def score(children):
            out = []
            for c in children:
                base = g.env.clone()
                base.step(c.move)

                def impute(paths, _base=base):
                    planes = []
                    for path in paths:
                        env2 = _base.clone()
                        for mv in path:
                            env2.step(mv)
                        planes.append(encode_env(env2))
                    return list(head_fn(planes))

                f, info = flat_forgiveness_hybrid(
                    c, spec["tau"], impute, stat=spec["stat"],
                    parity=spec["parity"],
                    impute_min_visits=spec["impute_min_visits"],
                    max_impute=spec["max_impute"], return_info=True)
                st = self.sel_stats.setdefault(
                    g.search_net, dict(n=0, switched=0, sacrifice=0.0,
                                       ndelta_sum=0))
                st["imputed_mass"] = (st.get("imputed_mass", 0.0)
                                      + info["imputed_mass"])
                st["imputed_calls"] = st.get("imputed_calls", 0) + 1
                out.append(f)
            return out
        return score

    def _maxent(self, g, spec):
        """MaxEnt / soft-Q ACTING rule: sample from the Boltzmann policy

            pi(a) proportional to exp(Q(a) / alpha)

        over the searched candidate set. This is the optimal policy of the
        maximum-entropy objective (report S2.1.6, Eysenbach & Levine), so it
        gives a MaxEnt comparison WITHOUT training a MaxEnt agent -- the
        acting half of the method, instantiated on a fixed checkpoint.

        WHAT THIS IS AND IS NOT. MaxEnt changes the objective, so a genuinely
        MaxEnt-trained agent would also learn a different (soft) value
        function -- one whose entropy bonus is discounted along the
        trajectory, rewarding the agent for REACHING states where being
        uncertain is cheap. Sampling a standard-trained net reproduces the
        action distribution but not that value function, so this tests the
        acting half only. Say so in the write-up; it is still the right
        control, because the delta-tiebreak is also a deployment-time rule
        over the same Qs, so the two are compared like for like.

        Note what alpha is NOT: it is not forgiveness_tau. The same Boltzmann
        construction appears in both -- report S2.2.2 -- but MaxEnt SAMPLES
        from the distribution (its temperature therefore sets how decisively
        the agent acts) while forgiveness measures its ENTROPY and acts
        greedily (tau only sets what counts as near-optimal). Keeping them
        separate parameters is the whole point of that section, so they are
        separate flags here.

        Restricted to the candidate set rather than all legal moves: an
        unsearched action has no Q to exponentiate, and including one on a
        v_mix completion would put imputed and measured values in the same
        softmax."""
        cands = self._candidates(g)
        if not cands:
            cands = [c for c in g.root.children if c.visits > 0]
        if len(cands) < 2:
            return (cands[0].move if cands else None), None
        shaped = g.search_net in self.shaped_ids
        qs = np.array([(c.value_sh if shaped else c.value) / c.visits
                       for c in cands], dtype=np.float64)
        alpha = max(float(spec["alpha"]), 1e-9)
        z = (qs - qs.max()) / alpha
        p = np.exp(z)
        p /= p.sum()
        i = int(self.rng.choice(len(cands), p=p))
        # Perplexity of the ACTING distribution: the effective number of moves
        # the agent is choosing between. Directly comparable to the |S| the
        # delta-tiebreak reports, which is what makes the two arms' amounts of
        # randomness comparable at all.
        with np.errstate(divide="ignore", invalid="ignore"):
            H = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
        best = int(qs.argmax())
        return cands[i].move, dict(switched=(i != best),
                                   q1=float(qs[best]),
                                   q_played=float(qs[i]),
                                   n_delta=float(np.exp(H)))

    def _base_move(self, g, visit_counts, pid, best):
        """The player's UNPERTURBED move under its selection policy."""
        spec = self.selectors.get(pid)
        cands = self._candidates(g)
        if not spec or spec["kind"] == "greedy":
            if self.diag_delta is not None:      # |S| DIAGNOSTIC only: how
                kids = cands or [c for c in g.root.children if c.visits > 0]
                if len(kids) >= 2:               # flat are the positions this
                    qs = [c.value / c.visits for c in kids]   # player reaches
                    q1 = max(qs)                 # (steering control) -- the
                    nd = sum(q1 - q <= self.diag_delta for q in qs)
                    st = self.sel_stats.setdefault(
                        pid, dict(n=0, switched=0, sacrifice=0.0,
                                  ndelta_sum=0))
                    st["n"] += 1
                    st["ndelta_sum"] += nd       # move played stays `best`
                    self._game_rec(g, pid)["ndelta_sum"] += nd
            return best
        if spec["kind"] == "maxent":
            move, info = self._maxent(g, spec)
            if move is None or info is None:
                return best
            st = self.sel_stats.setdefault(
                pid, dict(n=0, switched=0, sacrifice=0.0,
                          ndelta_sum=0))
            st["n"] += 1
            st["switched"] += bool(info["switched"])
            st["sacrifice"] += info["q1"] - info["q_played"]
            st["ndelta_sum"] += info["n_delta"]
            rec = self._game_rec(g, pid)
            rec["switched"] += bool(info["switched"])
            rec["sacrifice"] += info["q1"] - info["q_played"]
            rec["ndelta_sum"] += info["n_delta"]
            return move
        score_children = None
        if spec["kind"] == "forgiving_head":
            score_children = self._head_score_children(g, spec["head_fn"])
        elif spec["kind"] == "forgiving_hybrid":
            score_children = self._hybrid_score_children(g, spec)
        move, info = select_move_forgiving(
            g.root, spec["delta"], spec["tau"], floor=spec.get("floor", 0),
            gamma=spec["gamma"], stat=spec["stat"], agg=spec["agg"],
            parity=spec["parity"], score_children=score_children,
            candidates=cands, return_info=True)
        st = self.sel_stats.setdefault(
            pid, dict(n=0, switched=0, sacrifice=0.0, ndelta_sum=0))
        st["n"] += 1
        st["switched"] += bool(info.get("switched"))
        st["sacrifice"] += info.get("q1", 0.0) - info.get("q_played", 0.0)
        st["ndelta_sum"] += info.get("n_delta", 1)
        rec = self._game_rec(g, pid)
        rec["switched"] += bool(info.get("switched"))
        rec["sacrifice"] += info.get("q1", 0.0) - info.get("q_played", 0.0)
        rec["ndelta_sum"] += info.get("n_delta", 1)
        return move if move is not None else best

    def _blunder(self, g, visit_counts, best):
        """A searched-but-inferior move: sample among the non-best searched
        actions, weighted so that plausible mistakes are likelier than absurd
        ones.

        Under PUCT the weight is the visit count. Under halving it CANNOT be:
        the survivors of a phase all hold the same count, and a strong move
        eliminated in phase 0 holds fewer visits than a weak one that survived
        to the semi-final, so visit-weighting would make the injected mistake
        distribution a function of the elimination schedule. The weights come
        from pi' instead -- the same ranking the agent acts on."""
        alts = None
        if self.sh and getattr(g, "sh", None) is not None:
            kw = {}
            if g.search_net in self.shaped_ids and _IP_SHAPED:
                kw["shaped"] = True
            probs = improved_policy(g.root, c_visit=self.sh_c_visit,
                                    c_scale=self.sh_c_scale, **kw)
            alive = {c.move for c in g.sh.candidates}
            alts = [(m, p) for m, p in probs.items()
                    if m != best and m in alive and p > 0.0]
            if not alts:                 # everything eliminated but `best`
                alts = [(m, p) for m, p in probs.items()
                        if m != best and p > 0.0]
        if not alts:
            alts = [(m, v) for m, v in visit_counts.items()
                    if m != best and v > 0]
        if not alts:
            alts = [(m, 1) for m in visit_counts if m != best]
        if not alts:
            return best
        moves, w = zip(*alts)
        w = np.asarray(w, dtype=np.float64)
        return moves[self.rng.choice(len(moves), p=w / w.sum())]

    def __call__(self, g, visit_counts):
        # Track length for EVERY ply, both sides, opening included: the
        # denominator of the realised mistake rate is post-opening decisions,
        # but the confound is total game length.
        self.game_plies[g.pidx] = max(self.game_plies.get(g.pidx, 0), g.ply)

        if g.ply < self.opening_plies:                    # shared opening
            if self.sh and getattr(g, "sh", None) is not None:
                mv = select_move_sh(g.root, g.sh, self.opening_temp, self.rng,
                                    c_visit=self.sh_c_visit,
                                    c_scale=self.sh_c_scale)
                if mv is not None:
                    return mv
            return select_move(visit_counts, self.opening_temp, self.rng)

        pid = g.search_net
        greedy = self._greedy(g, visit_counts)             # what it would play
        q_best = self._q_best(g, visit_counts)             # delta-window origin
        state = self._local_stats(g)
        rec = self._game_rec(g, pid)
        if state is not None:
            rec["f_sum"] += state["forgiveness_entropy"]
            rec["gap_sum"] += state["gap"]
            rec["f_n"] += 1

        n_before = self.sel_stats.get(pid, {}).get("n", 0)
        base = self._base_move(g, visit_counts, pid, greedy)
        # _base_move appended at most one entry to sel_stats; read back what
        # it decided for THIS move so the per-decision row carries it.
        st = self.sel_stats.get(pid)
        fired = st is not None and st.get("n", 0) > n_before

        if pid not in self.perturbed:
            self._record(g, pid, state, base, base, False, fired)
            return base                                    # clean benchmark

        self.decisions[pid] = self.decisions.get(pid, 0) + 1
        rec["decisions"] += 1
        move, mistake = base, False
        if self.eps > 0 and self.rng.random() < self.eps:
            self.mistakes[pid] = self.mistakes.get(pid, 0) + 1
            rec["mistakes"] += 1
            mistake = True
            if self.mode == "random":
                legal = g.env.legalMoves()
                move = legal[self.rng.integers(len(legal))]
            else:
                move = self._blunder(g, visit_counts, q_best)
        elif self.temp_noise > 0:
            move = None
            if self.sh and getattr(g, "sh", None) is not None:
                move = select_move_sh(g.root, g.sh, self.temp_noise, self.rng,
                                      c_visit=self.sh_c_visit,
                                      c_scale=self.sh_c_scale,
                                      shaped=g.search_net in self.shaped_ids)
            if move is None:
                move = select_move(visit_counts, self.temp_noise, self.rng)
        self._record(g, pid, state, base, move, mistake, fired)
        return move

    def _record(self, g, pid, state, base, played, mistake, fired):
        """One row per post-opening decision. Rows are what make the
        aggregates re-derivable: a mean tells you nothing about a
        distribution, and a switch rate of 30% could be uniform across
        positions or concentrated in a handful of them."""
        if not self.record_rows:
            return
        st = self.sel_stats.get(pid, {})
        row = dict(pidx=str(g.pidx), player=pid, ply=g.ply,
                   mistake=int(mistake), switched=int(fired and base != played),
                   played_base=int(base == played))
        if state is not None:
            row.update(q1=round(state["q1"], 5), q2=round(state["q2"], 5),
                       gap=round(state["gap"], 5),
                       f_entropy=round(state["forgiveness_entropy"], 5),
                       f_gap=round(state["F_gap"], 5),
                       eff_actions=round(state["eff_actions"], 4),
                       n_cand=state["n_cand"])
        self.rows.append(row)


# --------------------------------------------------------------------------- #
# match plumbing
# --------------------------------------------------------------------------- #
def make_tickets(pairing_name, tracked_id, opp_id, games):
    """`games` tickets with alternating colours; a_score in the results is
    always the TRACKED player's score."""
    out = []
    for i in range(games):
        a_white = (i % 2 == 0)
        w, b = (tracked_id, opp_id) if a_white else (opp_id, tracked_id)
        out.append(((pairing_name, i), a_white, w, b))
    return out


def elo_ci(scores):
    """Mean score -> Elo difference with a 95% CI from the per-game sample
    std (scores in {0, 0.5, 1}). Probabilities are clamped away from 0/1 so
    a whitewash maps to a finite bound rather than +-inf."""
    n = len(scores)
    s = np.asarray(scores, dtype=np.float64)
    m = s.mean()
    se = s.std(ddof=1) / math.sqrt(n) if n > 1 else 0.5
    lo, hi = m - 1.96 * se, m + 1.96 * se
    eps = 1.0 / (4.0 * max(n, 1))

    def to_elo(p):
        p = min(max(p, eps), 1.0 - eps)
        return -400.0 * math.log10(1.0 / p - 1.0)
    return m, to_elo(m), to_elo(lo), to_elo(hi)


def run_level(nets, pairings, level, args, seed):
    """One noise level: fresh (possibly value-noised) evaluators, one batched
    run over all pairings' tickets, per-pairing score lists back."""
    eps, vnoise = level
    shape_beta = {pid: b for pid, b in
                  (("A", args.shape_a), ("B", args.shape_b)) if b}
    eval_fns = {}
    for pid, net in nets.items():
        # A shaped player needs the forgiveness head in the same forward.
        fn = _make_eval_fn(net, return_forgiveness=pid in shape_beta)
        if pid in pairings["perturbed"]:
            # crc32, NOT hash(): str hashes are salted per process, so hash(pid)
            # made value-noise streams irreproducible across runs -- the one
            # place (paper-facing degradation curves) reproducibility matters.
            fn = wrap_value_noise(fn, vnoise,
                                  seed + zlib.crc32(pid.encode()) % 10_000)
        eval_fns[pid] = fn

    selectors = {}
    for pid, spec in pairings.get("selectors", {}).items():
        spec = dict(spec)
        if spec["kind"] in ("forgiving_head", "forgiving_hybrid"):
            spec["head_fn"] = make_head_F_fn(nets[pid])
        selectors[pid] = spec

    perturb = Perturber(pairings["perturbed"], eps, args.mode,
                        args.temp_noise, args.opening_plies,
                        args.opening_temp, seed, selectors=selectors,
                        # |S| diagnostic for greedy players: log at the
                        # widest delta actually in use, so the greedy
                        # column is comparable with the forgiving arm it
                        # is being contrasted against.
                        diag_delta=(max(args.delta_of.values())
                                    if pairings.get("selectors") else None),
                        sh=not args.no_sequential_halving,
                        sh_c_visit=args.sh_c_visit,
                        sh_c_scale=args.sh_c_scale,
                        record_rows=bool(args.dataset_out),
                        stat_tau=args.sel_tau or 0.0178,
                        shaped_ids=set(shape_beta))

    tickets = []
    for name, tracked, opp in pairings["pairs"]:
        tickets.extend(make_tickets(name, tracked, opp, args.games))

    done = [0]
    t0 = time.time()

    def progress(pidx, a_score):
        done[0] += 1
        if done[0] % 50 == 0:
            print(f"    {done[0]}/{len(tickets)} games "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    results = run_elo_matches_batched(
        tickets, eval_fns, iterations=args.sims, c=args.c,
        fpu_reduction=args.fpu_reduction,
        opening_plies=args.opening_plies, opening_temp=args.opening_temp,
        max_plies=args.max_plies, concurrency=args.concurrency,
        use_cache=True, cache_cap=args.cache_cap,
        decide_move=perturb, on_game_done=progress,
        sequential_halving=not args.no_sequential_halving,
        sh_m=args.sh_m, sh_stat_width=args.sh_stat_width,
        sh_c_visit=args.sh_c_visit, sh_c_scale=args.sh_c_scale,
        sh_gumbel=args.sh_gumbel, sh_seed=seed,
        shape_beta=shape_beta,
        rng=np.random.default_rng(seed))

    by_pairing = {name: [] for name, _, _ in pairings["pairs"]}
    by_game = {}
    for (name, i), score in results:
        by_pairing[name].append(score)
        by_game[(name, i)] = score
    return by_pairing, perturb, by_game


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def save_decision_dataset(path, rows, args, pairings):
    """Per-decision dataset, npz, in the same spirit as the forgiveness-head
    training datasets (train_forgiveness_heads.save_dataset): columns as
    arrays plus a JSON meta blob, so a later analysis never has to guess how
    the numbers were produced.

    One row per post-opening decision, carrying the forgiveness statistics of
    the state the player stood in. This is the file that answers the question
    the score curves cannot: did the forgiving arm actually REACH flatter
    states, or did it merely score differently? Compare the f_entropy
    distributions between selectors at eps = 0 -- the clean condition, where
    no injected noise confounds the comparison.

    Planes are deliberately NOT stored: they would dominate the file size and
    the arena is not a training-data generator. If you need them, the pidx +
    ply pair identifies the position for a replay."""
    cols = {}
    keys = set()
    for r in rows:
        keys.update(r.keys())
    for k in sorted(keys):
        vals = [r.get(k, np.nan) for r in rows]
        if all(isinstance(v, (int, float, np.floating, np.integer))
               or v is np.nan for v in vals):
            cols[k] = np.asarray(vals, dtype=np.float32)
        else:
            cols[k] = np.asarray([str(v) for v in vals])
    meta = dict(model_a=os.path.abspath(args.model_a),
                model_b=os.path.abspath(args.model_b),
                benchmark=(os.path.abspath(args.benchmark)
                           if args.benchmark else None),
                select_a=args.select_a, select_b=args.select_b,
                delta=args.delta, tau=args.sel_tau, gamma=args.sel_gamma,
                stat=args.sel_stat, agg=args.sel_agg, parity=args.sel_parity,
                sims=args.sims, mode=args.mode,
                root=("puct" if args.no_sequential_halving
                      else f"sh_m{args.sh_m}"),
                stat_width=args.sh_stat_width,
                games_per_level=args.games,
                movegen=movegen_status(),
                shape_a=args.shape_a, shape_b=args.shape_b,
                opening_plies=args.opening_plies, seed=args.seed)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(meta), **cols)
    print(f"decision dataset -> {os.path.abspath(path)} "
          f"({len(rows)} rows, {os.path.getsize(path)/1e6:.1f} MB)")


def parse_args():
    p = argparse.ArgumentParser(
        description="Robustness arena: two models under injected mistakes.")
    p.add_argument("model_a", help="checkpoint for model A")
    p.add_argument("model_b", help="checkpoint for model B")
    p.add_argument("--benchmark", default=None,
                   help="optional third checkpoint; A and B each play it "
                        "CLEAN instead of playing each other")
    p.add_argument("--games", type=int, default=200,
                   help="games per pairing per level (default 200; CIs at "
                        "40 games are ~+-100 Elo -- fine for a sanity run, "
                        "useless for a curve)")
    p.add_argument("--sims", type=int, default=800,
                   help="simulations per move. Raised from 300: under "
                        "halving the statistics set holds roughly budget/8 "
                        "visits per action, and 300 sims leaves it at ~37 -- "
                        "too few for a Q difference of 0.05 to mean anything")
    p.add_argument("--eps", default="0,0.05,0.1,0.2",
                   help="comma list of mistake probabilities")
    p.add_argument("--vnoise", default="0",
                   help="comma list of value-noise sigmas (levels = cartesian "
                        "product with --eps)")
    p.add_argument("--mode", choices=["random", "blunder"], default="blunder")
    p.add_argument("--temp-noise", type=float, default=0.0,
                   help="post-opening sampling temperature (0 = argmax; "
                        "bypasses --select-*)")
    sel = ["greedy", "forgiving_tree", "forgiving_head",
           "forgiving_hybrid", "maxent"]
    p.add_argument("--select-a", choices=sel, default="greedy",
                   help="model A's post-opening selection policy")
    p.add_argument("--select-b", choices=sel, default="greedy",
                   help="model B's post-opening selection policy")
    p.add_argument("--delta", type=float, default=0.05,
                   help="near-optimal Q window for forgiving selection; "
                        "default for both players")
    # PER-PLAYER DELTA. The tiebreak's contribution can only be isolated
    # against a matched base rule: both arms Q-argmax, one with delta=0 and
    # one with delta>0. A single global --delta makes two forgiving_* arms
    # identical, so that comparison is impossible without these.
    p.add_argument("--delta-a", type=float, default=None,
                   help="override --delta for model A (delta=0 -> the "
                        "Q-argmax control: the near-optimal set is a "
                        "singleton and forgiveness never enters)")
    p.add_argument("--delta-b", type=float, default=None,
                   help="override --delta for model B")
    p.add_argument("--sel-tau", type=float, default=None,
                   help="forgiveness temperature for subtree scoring; "
                        "default: checkpoint config forgiveness_tau, else 0.044")
    p.add_argument("--sel-gamma", type=float, default=0.85)
    p.add_argument("--maxent-alpha", type=float, default=0.02,
                   help="temperature of the MaxEnt Boltzmann acting "
                        "policy exp(Q/alpha) over the searched "
                        "candidates. NOT --sel-tau: alpha sets how "
                        "decisively the agent ACTS, tau only sets "
                        "what counts as near-optimal when MEASURING "
                        "forgiveness. Calibrate alpha so the MaxEnt "
                        "arm gives up the same clean Elo as the "
                        "delta-tiebreak, then compare degradation.")
    p.add_argument("--sel-stat", choices=["gap", "entropy"], default="entropy")
    p.add_argument("--sel-agg", choices=["tree", "flat"], default="tree")
    p.add_argument("--hybrid-impute-min-visits", type=int, default=2,
                   help="forgiving_hybrid: only impute undefined nodes with "
                        "at least this many visits")
    p.add_argument("--hybrid-max-impute", type=int, default=32,
                   help="forgiving_hybrid: head-impute at most this many "
                        "nodes per candidate (heaviest by visits)")
    p.add_argument("--sel-parity", type=int, default=1, choices=[0, 1],
                   help="forgiving_tree only: 1 = my future decision nodes "
                        "(default), 0 = opponent's. forgiving_head "
                        "perspective is set by WHICH head checkpoint you "
                        "load -- see the module docstring")
    p.add_argument("--shape-a", type=float, default=0.0,
                   help="forgiveness VALUE SHAPING beta for player A: at each "
                        "leaf the backed-up value becomes "
                        "clip(v + beta*(2F-1), -1, 1), the same search-time "
                        "mechanism the shaped checkpoints trained under. "
                        "Leave at 0 to evaluate only the POLICY a shaped "
                        "checkpoint learned; set it to the training beta to "
                        "evaluate the method as designed. Needs a "
                        "parity-BLENDED head, since the bonus applies "
                        "symmetrically at both players' leaves.")
    p.add_argument("--shape-b", type=float, default=0.0,
                   help="value-shaping beta for player B (see --shape-a)")
    p.add_argument("--no-sequential-halving", action="store_true",
                   help="revert the root to plain PUCT. The delta-set and the "
                        "forgiveness statistics are then computed over "
                        "children with wildly unequal visit counts, so the "
                        "near-optimal set is largely noise -- for ablation "
                        "only, never for a reported number.")
    p.add_argument("--sh-m", type=int, default=8,
                   help="root actions considered (top-m). 8 rather than the "
                        "training 16 because the arena runs a smaller budget: "
                        "at 800 sims, m=8 puts the statistics set at 99 "
                        "visits vs 87 for m=16, and at 300 sims it is 37 vs 31")
    p.add_argument("--sh-stat-width", type=int, default=4,
                   help="actions retained for the statistics set -- the set "
                        "the delta-window is taken over")
    p.add_argument("--sh-c-visit", type=float, default=50.0)
    p.add_argument("--sh-c-scale", type=float, default=0.02)
    p.add_argument("--sh-gumbel", action="store_true",
                   help="keep the Gumbel exploration draws (default off: "
                        "both arms then consider the same actions, and the "
                        "only difference between them is the selection rule)")
    p.add_argument("--opening-plies", type=int, default=8)
    p.add_argument("--opening-temp", type=float, default=1.0)
    p.add_argument("--max-plies", type=int, default=200)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--cache-cap", type=int, default=250_000)
    p.add_argument("--c", type=float, default=1.5)
    p.add_argument("--fpu-reduction", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="robustness.csv")
    p.add_argument("--per-game-out", default=None,
                   help="write one row per GAME here (score, plies, mistakes "
                        "injected, switches, Q sacrificed, mean state F). "
                        "Needed for paired or bootstrap analysis -- an "
                        "aggregate cannot be un-aggregated afterwards.")
    p.add_argument("--dataset-out", default=None,
                   help="write one row per DECISION to this .npz (the "
                        "forgiveness statistics of every state reached). This "
                        "is the mechanism check: whether the forgiving arm "
                        "reaches flatter states, which the score curves "
                        "cannot show.")
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    nets = {"A": load_net(args.model_a, device),
            "B": load_net(args.model_b, device)}

    # WHICH forgiveness head each checkpoint carries. There is no --head-a
    # flag: make_head_F_fn reads the head off the player's OWN net, so the
    # parity is whatever train_forgiveness_heads.py baked in and nothing
    # downstream would otherwise reveal it. Recorded into both CSVs so a
    # result can be attributed to a head mode months later.
    ckpt_cfg, head_modes = {}, {}
    for pid, path in (("A", args.model_a), ("B", args.model_b)):
        ckpt_cfg[pid] = torch.load(path, map_location="cpu",
                                   weights_only=False).get("config", {})
        head_modes[pid] = ckpt_cfg[pid].get("forgiveness_target_mode",
                                            "unknown")
    print(f"forgiveness heads: A={head_modes['A']}  B={head_modes['B']}")

    # PERSPECTIVE FLIP (module docstring). forgiving_head scores each
    # candidate's SUCCESSOR position, where the OPPONENT is to move, so an
    # "_opp" head is the one that returns the mover's own slack there. A
    # "_me" head selects for the OPPONENT's slack instead, which shows up
    # as mean_state_F falling below the greedy control rather than rising.
    for pid, kind in (("A", args.select_a), ("B", args.select_b)):
        if kind == "forgiving_head" and not head_modes[pid].endswith("_opp"):
            print(f"  WARNING: player {pid} selects with forgiving_head "
                  f"but its checkpoint carries a '{head_modes[pid]}' "
                  f"head. That head reads the successor position as the "
                  f"mover's own slack, i.e. the OPPONENT's. Expect "
                  f"mean_state_F to FALL relative to greedy.")

    # selection policies for the tracked players (benchmark stays greedy)
    # Only the forgiving_* selectors read tau; maxent has its own alpha and
    # must not be handed a forgiveness temperature it never uses.
    if args.sel_tau is None and any(
            k.startswith("forgiving")
            for k in (args.select_a, args.select_b)):
        args.sel_tau = ckpt_cfg["A"].get("forgiveness_tau", 0.0178)
        print(f"--sel-tau from checkpoint config: {args.sel_tau}")
    # Resolved once here so every downstream consumer (specs, diagnostic,
    # CSV columns) reads the same per-player value.
    delta_of = {"A": args.delta if args.delta_a is None else args.delta_a,
                "B": args.delta if args.delta_b is None else args.delta_b}
    args.delta_of = delta_of
    selectors = {}
    for pid, kind in (("A", args.select_a), ("B", args.select_b)):
        if kind != "greedy":
            selectors[pid] = dict(kind=kind, delta=delta_of[pid],
                                  tau=args.sel_tau, gamma=args.sel_gamma,
                                  stat=args.sel_stat, agg=args.sel_agg,
                                  parity=args.sel_parity,
                                  impute_min_visits=args.hybrid_impute_min_visits,
                                  max_impute=args.hybrid_max_impute,
                                  alpha=args.maxent_alpha)
    print(f"selection: A={args.select_a}  B={args.select_b}"
          + (f"  delta A={delta_of['A']} B={delta_of['B']}"
             if selectors else ""))
    if (args.select_a.startswith("forgiving")
            and args.select_b.startswith("forgiving")
            and delta_of["A"] == delta_of["B"]
            and args.select_a == args.select_b):
        print("  WARNING: both players run the same forgiving selector "
              "with the same delta -- the two arms are identical. Set "
              "--delta-a 0 for the Q-argmax control.")
    if args.benchmark:
        nets["BM"] = load_net(args.benchmark, device)
        pairings = {"pairs": [("A_vs_BM", "A", "BM"),
                              ("B_vs_BM", "B", "BM")],
                    "perturbed": {"A", "B"},
                    "selectors": selectors,
                    "head_modes": head_modes}
        print(f"benchmark mode: A and B each play a clean {args.benchmark}")
    else:
        pairings = {"pairs": [("A_vs_B", "A", "B")],
                    "perturbed": {"A", "B"},
                    "selectors": selectors,
                    "head_modes": head_modes}
        print("head-to-head mode: A vs B, both perturbed identically")

    # The visit count backing the statistics set is the number that decides
    # whether a delta of 0.05 is signal or noise, so print it rather than
    # leaving it implicit in the budget.
    if not args.no_sequential_halving:
        try:
            from search.sequential_halving import plan_phases
        except ImportError:
            from sequential_halving import plan_phases
        plan = plan_phases(args.sims, args.sh_m)
        cum, tot = [], 0
        for n_c, per in plan:
            tot += per
            cum.append((n_c, tot))
        stat_n = next((t for n_c, t in cum if n_c <= args.sh_stat_width), None)
        print(f"root: sequential halving, m={args.sh_m}, "
              f"gumbel={'on' if args.sh_gumbel else 'off'}, "
              f"schedule {cum}")
        print(f"      statistics set = top {args.sh_stat_width} at "
              f"{stat_n if stat_n else '?'} visits each")
    else:
        print("root: PLAIN PUCT -- delta-set computed over unequal visit "
              "counts; ablation only")

    # Value shaping and the delta-tiebreak are the PENALISED and CONSTRAINED
    # forms of the same objective (report S5.4), with beta ~ delta/2 making
    # them about equally willing to trade value for slack. Running both on one
    # arm pays for forgiveness twice, at roughly double the intended rate, and
    # nothing downstream would reveal it.
    for pid, beta, sel in (("A", args.shape_a, args.select_a),
                           ("B", args.shape_b, args.select_b)):
        if beta and sel != "greedy":
            raise SystemExit(
                f"player {pid}: --shape-{pid.lower()} {beta} together with "
                f"--select-{pid.lower()} {sel} applies the forgiveness "
                "preference twice (shaped search AND a forgiving selector). "
                "Pick one: shaped search belongs with greedy selection.")
    if args.shape_a or args.shape_b:
        if args.no_sequential_halving:
            raise SystemExit(
                "--shape-a/--shape-b need the halving root. Under plain PUCT "
                "the shaped preference reaches the played move only through "
                "visit allocation, which the arena does not select on.")
        print(f"value shaping: A beta={args.shape_a}  B beta={args.shape_b} "
              f"(leaf values, parity-blended head required)")

    eps_levels = [float(x) for x in args.eps.split(",")]
    vn_levels = [float(x) for x in args.vnoise.split(",")]
    levels = [(e, v) for v in vn_levels for e in eps_levels]
    print(f"{len(levels)} level(s) x {args.games} games/pairing x "
          f"{len(pairings['pairs'])} pairing(s), {args.sims} sims/move, "
          f"mode={args.mode}")

    rows = []
    game_rows = []
    all_rows = []                       # per-decision dataset rows
    for li, level in enumerate(levels):
        eps, vn = level
        print(f"\n== level {li+1}/{len(levels)}: eps={eps} vnoise={vn} ==")
        by_pairing, perturb, by_game = run_level(nets, pairings, level, args,
                                                 seed=args.seed + 1000 * li)
        for r in perturb.rows:
            r["eps"] = eps
            r["vnoise"] = vn
        all_rows.extend(perturb.rows)
        # HEAD-TO-HEAD (no --benchmark) has ONE pair whose two players each
        # have their own selector and diagnostics; emit a row per player so
        # the opponent's switch/sacrifice/imputed stats are not lost. The
        # opponent's row mirrors the result (score/Elo negated) and carries
        # ITS own selection + mistake counters. Benchmark mode is unchanged.
        # ---- per-GAME rows: the unit paired and bootstrap analysis needs.
        # Aggregates cannot be un-aggregated later, and re-running a 600-game
        # sweep to recover a column costs hours.
        if args.per_game_out:
            for (gp, pid), r in perturb.game.items():
                gname = gp[0] if isinstance(gp, tuple) else str(gp)
                sc = by_game.get(gp)
                tracked_of = {nm: tr for nm, tr, _o in pairings["pairs"]}
                if sc is not None and gname in tracked_of \
                        and tracked_of[gname] != pid:
                    sc = 1.0 - sc            # this player's own perspective
                nd = r["ndelta_sum"] / r["decisions"] if r["decisions"] else 0.0
                # Provenance on EVERY row: which move rule produced it and
                # which head parity that rule read. Without these the file
                # cannot be attributed to a configuration after the fact.
                sel_kind_g = (pairings.get("selectors", {})
                              .get(pid, {}).get("kind", "greedy"))
                game_rows.append(dict(
                    eps=eps, vnoise=vn, pairing=gname, player=pid,
                    selector=sel_kind_g,
                    head_mode=pairings.get("head_modes", {}).get(pid, ""),
                    game=str(gp), score=sc,
                    plies=perturb.game_plies.get(gp, 0),
                    decisions=r["decisions"], mistakes=r["mistakes"],
                    switched=r["switched"],
                    q_sacrificed=round(r["sacrifice"], 5),
                    mean_delta_set=round(nd, 3),
                    mean_state_F=(round(r["f_sum"] / r["f_n"], 5)
                                  if r["f_n"] else ""),
                    mean_state_gap=(round(r["gap_sum"] / r["f_n"], 5)
                                    if r["f_n"] else "")))

        report = []
        for name, tracked, opp in pairings["pairs"]:
            report.append((name, tracked, False))
            if not args.benchmark:
                report.append((f"{name}(B)", opp, True))
        for name, tracked, mirrored in report:
            base = name[:-3] if mirrored else name
            scores = by_pairing[base]
            if mirrored:                      # this player's own perspective
                scores = [1.0 - s for s in scores]
            n = len(scores)
            w = sum(s == 1.0 for s in scores)
            d = sum(s == 0.5 for s in scores)
            l = n - w - d
            m, elo, elo_lo, elo_hi = elo_ci(scores)
            dec = perturb.decisions.get(tracked, 0)
            mis = perturb.mistakes.get(tracked, 0)
            mpg = mis / max(n, 1)
            # Realised per-MOVE rate. eps is what we asked for; this is what
            # the agent actually received, and they differ whenever a game
            # ends early or runs long. Report this, not eps.
            realised_eps = mis / dec if dec else 0.0
            dpg = dec / max(n, 1)
            # Per-game records for this player, across this level's games.
            grecs = [r for (gp, pid), r in perturb.game.items()
                     if pid == tracked]
            plies = [perturb.game_plies.get(gp, 0)
                     for (gp, pid) in perturb.game if pid == tracked]
            mean_plies = float(np.mean(plies)) if plies else 0.0
            med_plies = float(np.median(plies)) if plies else 0.0
            f_vals = [r["f_sum"] / r["f_n"] for r in grecs if r["f_n"]]
            mean_F = float(np.mean(f_vals)) if f_vals else float("nan")
            gap_vals = [r["gap_sum"] / r["f_n"] for r in grecs if r["f_n"]]
            mean_gap = float(np.mean(gap_vals)) if gap_vals else float("nan")
            st = perturb.sel_stats.get(tracked)
            sw = st["switched"] / st["n"] if st and st["n"] else 0.0
            sac = st["sacrifice"] / st["n"] if st and st["n"] else 0.0
            nds = st["ndelta_sum"] / st["n"] if st and st["n"] else 1.0
            imass = (st["imputed_mass"] / st["imputed_calls"]
                     if st and st.get("imputed_calls") else None)
            sel_note = (f"  switch {sw:.0%}, sac {sac:.4f}, "
                        f"|S| {nds:.2f}" if st else "")
            if imass is not None:
                sel_note += f", imputed {imass:.0%}"
            print(f"  {name:9s} [{tracked:2s}] {w:3d}W {d:3d}D {l:3d}L  "
                  f"score {m:.3f}  Elo {elo:+7.1f} "
                  f"[{elo_lo:+7.1f}, {elo_hi:+7.1f}]{sel_note}")
            print(f"             {mpg:.2f} mistakes/game over {dpg:.1f} "
                  f"decisions (realised eps {realised_eps:.3f}), "
                  f"{mean_plies:.0f} plies (median {med_plies:.0f})")
            print(f"             states reached: mean F {mean_F:.4f}, "
                  f"mean gap {mean_gap:.4f}")
            sel_kind = (pairings.get("selectors", {})
                        .get(tracked, {}).get("kind", "greedy"))
            rows.append(dict(eps=eps, vnoise=vn, pairing=name,
                             player=tracked,
                             selector=sel_kind,
                             head_mode=pairings.get("head_modes", {})
                                       .get(tracked, ""),
                             delta=args.delta_of.get(tracked, args.delta),
                             sims=args.sims,
                             root=("puct" if args.no_sequential_halving
                                   else f"sh_m{args.sh_m}"),
                             movegen=movegen_status(),
                             shape_a=args.shape_a,
                             shape_b=args.shape_b,
                             stat_width=args.sh_stat_width, games=n,
                             wins=w, draws=d, losses=l,
                             score=round(m, 4), elo=round(elo, 1),
                             elo_lo=round(elo_lo, 1), elo_hi=round(elo_hi, 1),
                             mistakes_per_game=round(mpg, 3),
                             realised_eps=round(realised_eps, 4),
                             decisions_per_game=round(dpg, 2),
                             mean_plies=round(mean_plies, 1),
                             median_plies=round(med_plies, 1),
                             mean_state_F=round(mean_F, 5),
                             mean_state_gap=round(mean_gap, 5),
                             post_opening_decisions=dec,
                             switch_rate=round(sw, 4),
                             mean_q_sacrifice=round(sac, 5),
                             mean_delta_set=round(nds, 3),
                             imputed_mass=(round(imass, 4)
                                           if imass is not None else "")))
        # write incrementally so a long sweep is never lost
        with open(args.out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
    print(f"\nresults -> {os.path.abspath(args.out)}")

    if args.per_game_out and game_rows:
        with open(args.per_game_out, "w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(game_rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(game_rows)
        print(f"per-game rows -> {os.path.abspath(args.per_game_out)} "
              f"({len(game_rows)} rows)")

    if args.dataset_out and all_rows:
        save_decision_dataset(args.dataset_out, all_rows, args, pairings)

    if len(eps_levels) > 1:
        print("\ndegradation summary (score vs opponent):")
        for name in (("A_vs_BM", "B_vs_BM") if args.benchmark
                     else ("A_vs_B", "A_vs_B(B)")):
            pts = [(r["eps"], r["score"]) for r in rows
                   if r["pairing"] == name and r["vnoise"] == vn_levels[0]]
            if len(pts) > 1:
                slope = np.polyfit(*zip(*pts), 1)[0]
                print(f"  {name}: d(score)/d(eps) = {slope:+.3f} "
                      f"(less negative = more robust)")


if __name__ == "__main__":
    main()


    
"""
Gumbel AlphaZero: Gumbel root sampling, sequential halving, and the improved
policy target pi'.   (Danihelka et al. 2022, "Policy Improvement by Planning
with Gumbel")

WHAT THIS REPLACES
------------------
Three root-level mechanisms are swapped out together. They are a package: any
one of them alone is unsound.

  1. Dirichlet noise            ->  Gumbel top-m sampling.
  2. Forced root visits         ->  sequential halving.
  3. Visit-count policy target  ->  softmax(logits + sigma(completed-Q)).

WHY THEY MUST MOVE TOGETHER
---------------------------
The recorded target is  pi' = softmax(logit(a) + sigma(qhat(a))),  read off the
root's PRIORS. Dirichlet noise overwrites `child.prior` in place
(_add_dirichlet_noise), so leaving it on means the target is built from noised
logits -- the net would be trained to reproduce its own exploration noise. So
Dirichlet is OFF at any sequential-halving root, and exploration comes from the
Gumbel variables instead, which are kept OUT of `prior` (see _SHState.gumbel).

Sequential halving and forced visits are two answers to the same question --
"how do I get comparable Q estimates across root actions" -- and they fight if
both run. Forcing is disabled at SH roots.

The visit-count target is not merely suboptimal here, it is unusable: sequential
halving deliberately spends its whole budget on a shrinking candidate set, so
visit counts encode the HALVING SCHEDULE, not action quality. An eliminated
action that was visited in phase 0 and never again would get a training weight
proportional to how early it died. pi' is the paper's replacement and is defined
over ALL children, including never-visited ones, via the completed-Q v_mix.

CHOOSING c_scale  -- READ THIS BEFORE THE FIRST RUN
---------------------------------------------------
The target's sharpness is governed by the single quantity

    sigma_total = (c_visit + max_a N(a)) * c_scale

and sequential halving makes max_N LARGE by design: with 1000 sims and m=16 the
two finalists hold ~240 visits each, so max_N ~ 240 and sigma_total = 290 *
c_scale. Measured target entropy on a synthetic 32-move root (rescaling on, so
these hold regardless of how tactical the position is):

    c_scale   0.02    0.05    0.10    0.25    1.00
    entropy   ~2.1    ~0.7    ~0.15   ~0.005  ~0.000   nats

For reference, this codebase's visit-count targets sit near 1.7 nats
(training/train.py logs it as target_entropy). c_scale = 1.0 -- the value
written in the paper and in this repo's existing gumbel_c_scale -- produces a
ONE-HOT target here. mctx's default of 0.1 is still ~10x too sharp at this
budget. The default below is therefore 0.02, chosen to land near the target
entropy the policy head is already trained against, so switching to pi' changes
the target's SHAPE without also changing its temperature by two orders of
magnitude.

If you change the simulation budget, revisit this: the right c_scale falls as
the budget rises, because max_N rises with it.

BUDGET
------
sequential halving spends n simulations over ceil(log2(m)) phases, each phase
giving every surviving candidate an equal number of visits and then discarding
the worse half by  g(a) + logit(a) + sigma(qhat(a)).  With n=1000 and m=16 that
is 4 phases of roughly 250 visits spread over 16, 8, 4, 2 candidates: the two
finalists end up with ~180 visits each and matched standard errors -- which is
exactly the property the forgiveness statistics need, obtained for free rather
than by pinning half the budget to a forced floor.

Interior nodes keep plain PUCT. The paper uses a deterministic non-PUCT rule
there too; that is a larger change with its own hyperparameters, and the root is
where the policy target and the action both come from.

TORCH-FREE by construction -- numpy only, so it imports in the same
unit-testable contexts as search/puct.py.
"""

import math

import numpy as np

__all__ = ["SHState", "improved_policy", "root_v_mix", "completed_q_map",
           "sigma_scale", "plan_phases"]


# --------------------------------------------------------------------------- #
# budget planning
# --------------------------------------------------------------------------- #
def plan_phases(budget, m):
    """
    Visit schedule for sequential halving.

    Returns a list of (n_candidates, visits_per_candidate), one entry per phase,
    with the candidate count halving each phase down to 2 (the final pair is
    where the action is decided).

    The naive  budget // (phases * m)  underspends badly for small budgets --
    integer division truncates in every phase and the remainder is silently
    dropped. Here the leftover is handed to the LAST phase, which is the one
    whose Q estimates decide the move, so the full budget is always used.
    """
    m = max(1, int(m))
    budget = max(0, int(budget))
    if m <= 1 or budget <= 0:
        return []

    phases = max(1, int(math.ceil(math.log2(m))))
    plan = []
    remaining = budget
    cur = m
    for p in range(phases):
        last = (p == phases - 1)
        if last:
            per = remaining // cur if cur else 0
        else:
            per = max(1, budget // (phases * cur))
            per = min(per, remaining // cur if cur else 0)
        if per <= 0:
            # budget exhausted early: stop planning, caller falls back to PUCT
            break
        plan.append((cur, per))
        remaining -= per * cur
        if cur <= 2:
            break
        cur = max(2, cur // 2)
    return plan


def sigma_scale(root, c_visit=50.0, c_scale=1.0):
    """
    The paper's monotone transform sigma, evaluated as a scalar multiplier:

        sigma(q) = (c_visit + max_b N(b)) * c_scale * q

    With a small search the priors dominate the score; with a large one the Qs
    do. Visit counts enter ONLY here, as a trust scale -- never as the selection
    statistic itself.
    """
    if not root.children:
        return c_visit * c_scale
    max_n = max(ch.visits for ch in root.children)
    return (c_visit + max_n) * c_scale


# --------------------------------------------------------------------------- #
# completed Q  (v_mix)
# --------------------------------------------------------------------------- #
def root_v_mix(root):
    """
    The paper's v_mix: the value assigned to a root action that search never
    visited, interpolating the root's own network value with the prior-weighted
    average Q of the actions that WERE visited.

        v_mix = 1/(1 + sum_a N(a)) * ( v(s) + sum_a N(a) / sum_{a:N>0} pi(a)
                                              * sum_{a:N>0} pi(a) q(a) )

    all in the ROOT MOVER's point of view.

    Using a flat 0, or the visit-weighted child average alone, both bias the
    target: an unvisited action is not "worth zero" and is not "worth the same
    as the moves search actually liked". Early in a search sum N is small and
    v_mix sits near the network value; late it converges to the searched
    average.

    root.net_value is the RAW network evaluation of the root position in the
    root mover's POV, stored at expansion (Node.net_value). Roots carry
    moverSign == 0 so their .value accumulator stays 0 -- it cannot be used
    here. If net_value is missing (a reused subtree from before this field
    existed, say) we degrade to the visit-weighted average.
    """
    kids = root.children
    if not kids:
        return 0.0

    sum_n = 0
    w_pi = 0.0          # sum of pi(a) over visited a
    w_piq = 0.0         # sum of pi(a) q(a) over visited a
    for ch in kids:
        n = ch.visits
        if n > 0:
            sum_n += n
            w_pi += ch.prior
            w_piq += ch.prior * (ch.value / n)

    v_root = getattr(root, "net_value", None)
    if v_root is None:
        if sum_n == 0:
            return 0.0
        return sum(ch.value for ch in kids if ch.visits > 0) / sum_n

    if sum_n == 0 or w_pi <= 0.0:
        return float(v_root)

    return float((v_root + (sum_n / w_pi) * w_piq) / (1.0 + sum_n))


def _rescale(vals):
    """Min-max the completed Qs into [0, 1].

    WITHOUT this the target scale depends on the raw Q spread, which varies
    hugely between quiet positions (all moves within 0.01) and tactical ones
    (0.5+ swings). sigma multiplies that spread by (c_visit + max_N), which
    sequential halving drives to ~250-300 because it concentrates visits on
    two finalists -- so a tactical root produced a target of exp(-1e2) on every
    move but one, i.e. a one-hot. A one-hot target is strictly worse than the
    visit-count target it replaced: it carries no information about the
    relative merit of the other moves, which is the entire reason to prefer
    pi'.

    Rescaling makes the target's sharpness a function of c_scale ALONE, so it
    can be tuned once and holds across positions. This follows DeepMind's mctx
    (`qtransform_completed_by_mix_value`, rescale_values=True).
    """
    lo = min(vals)
    hi = max(vals)
    span = hi - lo
    if span < 1e-8:
        return [0.5] * len(vals)
    return [(v - lo) / span for v in vals]


def completed_q_map(root):
    """
    {child -> qhat} in the root mover's POV: the search Q where visited, v_mix
    where not. Returns (dict, v_mix).
    """
    vm = root_v_mix(root)
    out = {}
    for ch in root.children:
        out[ch] = (ch.value / ch.visits) if ch.visits > 0 else vm
    return out, vm


# --------------------------------------------------------------------------- #
# the improved policy pi'  -- THE TRAINING TARGET
# --------------------------------------------------------------------------- #
def improved_policy(root, c_visit=50.0, c_scale=0.02, rescale=True):
    """
    pi'(a) = softmax_a ( log prior(a) + sigma(qhat(a)) )   over ALL children.

    This is the policy target. Note what is NOT in it:
      * no Gumbel variables -- those are exploration, resampled every move; a
        target must be a deterministic function of the search;
      * no visit counts as mass -- they appear only inside sigma;
      * no Dirichlet -- see the module docstring.

    Every legal move gets mass, including ones sequential halving eliminated in
    phase 0, because qhat completes them with v_mix. That is what makes this a
    valid target despite the budget being concentrated on a handful of actions.

    Returns {Move: prob}; empty dict if the root has no children.
    """
    kids = root.children
    if not kids:
        return {}

    qhat, _ = completed_q_map(root)
    scale = sigma_scale(root, c_visit, c_scale)

    qs = [qhat[ch] for ch in kids]
    if rescale:
        qs = _rescale(qs)

    scores = np.empty(len(kids), dtype=np.float64)
    for i, ch in enumerate(kids):
        scores[i] = math.log(max(ch.prior, 1e-12)) + scale * qs[i]

    scores -= scores.max()
    p = np.exp(scores)
    p /= p.sum()
    return {ch.move: float(p[i]) for i, ch in enumerate(kids)}


# --------------------------------------------------------------------------- #
# sequential halving as a per-simulation state machine
# --------------------------------------------------------------------------- #
class SHState:
    """
    Drives one root's sequential halving, one simulation at a time.

    The batched self-play loop advances every live game by exactly ONE
    simulation per round, so the halving cannot be written as a loop -- it has
    to be a state machine that answers "which root child gets the next visit?".
    That is `next_child()`.

    SUBTREE REUSE: a reused root arrives with visits already distributed over
    its children by the PREVIOUS move's search, in no relation to this move's
    schedule. next_child() therefore picks the surviving candidate that is
    furthest below the current phase target rather than round-robining a
    pointer, so carried visits are simply credited and the phase completes when
    every candidate has reached the target. A root that arrives already past a
    phase target skips straight through it. This keeps reuse (worth 15-20% of
    simulations) compatible with halving.
    """

    __slots__ = ("candidates", "gumbel", "plan", "phase", "target",
                 "c_visit", "c_scale", "exhausted", "_m0")

    def __init__(self, root, budget, m=16, rng=None,
                 c_visit=50.0, c_scale=1.0):
        kids = root.children
        self.c_visit = c_visit
        self.c_scale = c_scale
        self.exhausted = False
        self.phase = 0
        self.gumbel = {}
        self.candidates = []
        self.plan = []
        self._m0 = 0

        if not kids:
            self.exhausted = True
            return

        if rng is None:
            rng = np.random.default_rng()

        # ---- m: how many root actions to consider at all ----
        # Capped by the branching factor and by the budget (a phase must give
        # every candidate at least one visit, so m > budget is meaningless).
        m = min(int(m), len(kids))
        while m > 2 and budget < m:
            m //= 2
        m = max(1, m)
        self._m0 = m

        # ---- Gumbel top-m sampling, WITHOUT replacement ----
        # argtop_m( g(a) + logit(a) ) with g ~ Gumbel(0,1) is an exact sample of
        # m distinct actions from softmax(logit) (Kool et al.). This is the
        # paper's replacement for Dirichlet noise: exploration enters by which
        # actions are CONSIDERED, not by corrupting the priors that the target
        # is later read from.
        logits = np.array([math.log(max(ch.prior, 1e-12)) for ch in kids],
                          dtype=np.float64)
        g = rng.gumbel(0.0, 1.0, size=len(kids))
        perturbed = g + logits
        order = np.argsort(-perturbed)[:m]
        self.candidates = [kids[i] for i in order]
        for i in order:
            self.gumbel[kids[i]] = float(g[i])

        self.plan = plan_phases(budget, m)
        if not self.plan:
            self.exhausted = True
            return
        self.target = self._phase_target(0)

    # -- internals ---------------------------------------------------------- #
    def _phase_target(self, phase):
        """Cumulative per-candidate visit target at the END of `phase`."""
        tot = 0
        for p in range(min(phase + 1, len(self.plan))):
            tot += self.plan[p][1]
        return tot

    def _scores(self):
        """{child -> g(a) + logit(a) + sigma(qhat(a))} over the current
        candidates, all in one pass.

        Computed as a SNAPSHOT rather than as a sort key, deliberately:
        CPython's list.sort() blanks the list for the duration of the sort, so
        a key function that reads self.candidates (which the rescaling needs,
        for min/max) sees an empty list and raises. Snapshot first, sort second.

        Uses the SAME rescaled Q transform as improved_policy, so the action
        played and the target recorded stay consistent.
        """
        cands = self.candidates
        if not cands:
            return {}
        scale = (self.c_visit
                 + max(c.visits for c in cands)) * self.c_scale
        raw = [(c.value / c.visits) if c.visits > 0 else 0.0 for c in cands]
        lo, hi = min(raw), max(raw)
        span = hi - lo
        out = {}
        for c, q in zip(cands, raw):
            qn = 0.5 if span < 1e-8 else (q - lo) / span
            out[c] = (self.gumbel.get(c, 0.0)
                      + math.log(max(c.prior, 1e-12))
                      + scale * qn)
        return out

    def _advance_phase(self):
        """Halve the candidate set by score and move to the next phase."""
        if self.phase + 1 >= len(self.plan) or len(self.candidates) <= 2:
            self.exhausted = True
            return
        self.phase += 1
        keep = self.plan[self.phase][0]
        sc = self._scores()
        self.candidates = sorted(self.candidates, key=lambda c: sc[c],
                                 reverse=True)[:keep]
        self.target = self._phase_target(self.phase)

    # -- public ------------------------------------------------------------- #
    def next_child(self):
        """
        The root child that should receive the next simulation, or None when
        the schedule is spent (caller falls back to plain PUCT for any budget
        left over -- which only happens if the plan under-spent).
        """
        if self.exhausted or not self.candidates:
            return None
        for _ in range(len(self.plan) + 2):
            # candidate furthest below the phase target (reuse-safe; see class
            # docstring). min() over visits does exactly that.
            ch = min(self.candidates, key=lambda c: c.visits)
            if ch.visits < self.target:
                return ch
            self._advance_phase()
            if self.exhausted or not self.candidates:
                return None
        return None

    def final_floor(self):
        """
        Minimum visit count among the surviving candidates -- the analogue of
        the old forced-visit floor `force_n`, for statistics that need to know
        which root Qs are trustworthy (search/forgiveness.py takes this as its
        `floor` argument).
        """
        if not self.candidates:
            return 0
        return min(c.visits for c in self.candidates)

    def select_action(self, root):
        """
        The move to PLAY: argmax over the surviving candidates of
        g(a) + logit(a) + sigma(qhat(a)).

        The Gumbel variables are still in here -- unlike in the target. That is
        the point: they make acting stochastic (so self-play games diverge)
        while keeping the guarantee that the chosen action is a policy
        IMPROVEMENT over the prior in expectation.
        """
        if not self.candidates:
            return None
        sc = self._scores()
        return max(self.candidates, key=lambda c: sc[c]).move

    
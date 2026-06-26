"""
ease/ -- candidate "ease" signals for the robust-policy chess project.

Each tree measure takes a finished PUCT search root and returns a scalar in [0, 1]
(higher = easier/more forgiving) or None when undefined at that position (-> mask
the target out of the loss). They duck-type on the puct.Node (.children, .visits,
.value, .moverSign, .prior, .move), so they import nothing from the engine and are
testable with fakes (see _selftest.py).

  ease_veps        V_eps(s) - V(s): value lost to forced sloppy play, recursive
                   (downstream). READ ITS DOCSTRING -- the perturbation mode decides
                   whether it measures structural vs realized fragility.
  ease_perplexity  effective number of near-optimal moves (smooth frac_safe).
  ease_cliff       re-oriented normalised action gap, recursed (your old _cliff fixed).
  ease_advvar      policy-weighted advantage variance (most literal to motivation.txt).
  ease_topk_prior  de-confounded frac_safe over top-k by network prior.

  ease_rollout_oracle  NOT a head -- ground-truth forgiveness via perturbed rollouts,
                       used offline to pick which cheap measure to commit to.

------------------------------------------------------------------------------
RUNNING THE BAKE-OFF (test several, then commit one)
------------------------------------------------------------------------------
All tree measures are record-only and read the SAME root, so compute them together
in one play_game pass; none of them touch PUCT, so they cannot change self-play
strength. One self-play run feeds every candidate.

1) network.py -- one head per candidate you want to train (they share the trunk):
     self.ease_veps_head = ScalarHead(channels, torch.sigmoid)
     # ... one per measure ...
   and return each head's output from forward().

2) self_play.py -- in play_game, after `root, visit_counts = search(...)`:
     from ease import compute_targets
     ease_t = compute_targets(root)          # {name: (target, mask)}
   append the (target, mask) pairs you want to the history row, and carry them
   through into the example tuple (mirror how value_target is stored).

3) train.py -- in _collate unpack each (target, mask); in train_epoch add
     loss += weight * _masked_mse(pred_head, target, mask)
   for each head (you already have _masked_mse).

4) main.py -- add a weight per head to CONFIG and log each loss.

THEN pick the winner on three axes:
  (a) does the head reach low masked-MSE?  (is it learnable from the planes at all)
  (b) does its target correlate with ease_rollout_oracle on a held-out sample of
      positions?  (is it measuring real forgiveness)
  (c) does shaping with it actually help downstream?
Only commit one to the model after (a)-(c); everything before that is free to run
in parallel.

CALIBRATION: the scale/tau/delta/gamma defaults are starting points for Q in
[-1, 1]. Check on real positions that each target spreads across [0, 1] rather than
piling up near 0 or 1 -- a saturated target teaches the head nothing.
"""

from .ease_veps import ease_veps
from .ease_perplexity import ease_perplexity
from .ease_cliff import ease_cliff
from .ease_advvar import ease_advvar
from .ease_topk_prior import ease_topk_prior
from .ease_rollout_oracle import ease_rollout_oracle

# tree measures only (signature: fn(root) -> float | None). The oracle is excluded
# because it needs (env, net) and is offline-only.
TREE_MEASURES = {
    "veps": ease_veps,
    "perplexity": ease_perplexity,
    "cliff": ease_cliff,
    "advvar": ease_advvar,
    "topk_prior": ease_topk_prior,
}


def compute_targets(root):
    """All tree-measure targets for one root as {name: (target, mask)}.

    target is float in [0, 1]; mask is 1.0 if defined else 0.0 (target 0.0).
    """
    out = {}
    for name, fn in TREE_MEASURES.items():
        t = fn(root)
        out[name] = (0.0, 0.0) if t is None else (float(t), 1.0)
    return out


__all__ = [
    "ease_veps", "ease_perplexity", "ease_cliff", "ease_advvar",
    "ease_topk_prior", "ease_rollout_oracle",
    "TREE_MEASURES", "compute_targets",
]
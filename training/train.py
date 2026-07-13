import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from model.move_encoding import NUM_ACTIONS


class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def add_examples(self, examples):
        self.buffer.extend(examples)

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size):
        n = min(batch_size, len(self.buffer))
        return random.sample(self.buffer, n)


def _collate(batch, device, aux_ease=False):
    """
    Examples are (planes, policy, value) or, with aux_ease,
    (planes, policy, value, ease_target, ease_mask).

    `policy` per row is either a dense (NUM_ACTIONS,) float32 array (legacy
    self_play.py rows) or a SPARSE (action_indices, probs) pair (batched
    self-play). Sparse rows are densified here, per batch. An EMPTY sparse
    pair (or an all-zero dense row) is a VALUE-ONLY row: it gets policy-mask 0
    so it contributes nothing to the policy loss but trains value/ease as
    normal.

    Returns (planes, policy, pmask, value[, ease, emask]) -- pmask is (B,1).
    """
    B = len(batch)
    planes = np.stack([b[0] for b in batch])               # (B,19,8,8)
    policy = np.zeros((B, NUM_ACTIONS), dtype=np.float32)  # (B,NUM_ACTIONS)
    pmask  = np.zeros(B, dtype=np.float32)                 # (B,)
    for i, b in enumerate(batch):
        p = b[1]
        if isinstance(p, tuple):                           # sparse (idx, probs)
            idx, pr = p
            if len(idx):
                policy[i, idx] = pr
                pmask[i] = 1.0
        else:                                              # legacy dense row
            policy[i] = p
            if p.sum() > 0:
                pmask[i] = 1.0
    value  = np.array([b[2] for b in batch], np.float32)   # (B,)

    t = lambda a: torch.from_numpy(a).to(device)
    planes = t(planes)
    policy = t(policy)
    pmask  = t(pmask).unsqueeze(1)
    value  = t(value).unsqueeze(1)

    if not aux_ease:
        return planes, policy, pmask, value

    ease  = np.array([b[3] for b in batch], np.float32)
    emask = np.array([b[4] for b in batch], np.float32)
    ease  = t(ease).unsqueeze(1)
    emask = t(emask).unsqueeze(1)
    return planes, policy, pmask, value, ease, emask


def _masked_mse(pred, target, mask):
    return (mask * (pred - target) ** 2).sum() / mask.sum().clamp(min=1.0)


def train_epoch(net, buffer, optimiser, device, batches=32, batch_size=128,
                aux_ease=False, ease_weight=1.0, ease_optimiser=None,
                scaler=None):
    """
    Run gradient steps. Returns mean losses {total, policy, value, ease}.
    With aux_ease=False this is the original policy+value training (ease 0).

    DECOUPLED EASE TRAINING: when aux_ease is on and the net's ease head reads
    DETACHED trunk features (ChessNet's ease_detach=True default), the ease
    loss is never summed with the policy+value loss. Each batch takes TWO
    independent gradient steps from ONE shared trunk forward:

        step 1:  policy_loss + value_loss      -> `optimiser`
                 (trunk + policy head + value head)
        step 2:  ease_weight * ease_loss       -> `ease_optimiser`
                 (the four ease-head layers, nothing else)

    The detach makes the two computation graphs disjoint, so the second
    backward needs no retain_graph, costs no second forward, and physically
    cannot deposit gradient into the trunk or the other heads. Build the two
    optimisers over DISJOINT parameter sets (the mains split on "ease_" in the
    parameter name); if ease_optimiser is omitted, `optimiser` is stepped a
    second time -- only valid when it actually contains the ease parameters.

    If the net was built with ease_detach=False, the ease subgraph shares the
    trunk graph, so two backwards are impossible; training falls back to the
    coupled single-step summed loss (which is also the semantically consistent
    choice for a coupled architecture).

    `scaler` is the AMP GradScaler and MUST persist across iterations: its
    scale factor is tuned online, so building a new one every call discards
    that calibration. Create one in the training loop and pass it in; both
    decoupled steps share it (each backward/step/update cycle is internally
    consistent).
    """
    if scaler is None:
        scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    use_amp = (device.type == "cuda")

    inner = getattr(net, "_orig_mod", net)
    decoupled = aux_ease and getattr(inner, "ease_detach", True)

    net.train()
    acc = dict(total=0.0, policy=0.0, value=0.0, ease=0.0)
    actual = 0
    # masked ease-target statistics, accumulated over the whole epoch. The raw
    # ease MSE is uninterpretable on its own (its floor is the label noise);
    # what matters is the fit relative to the target spread:
    #     ease_R2 = 1 - MSE / Var(target).
    # R2 ~ 0  -> the head predicts no better than the batch mean: either the
    #            labels are noise at this tau/floor or the head can't read the
    #            needed feature from the (detached) trunk.
    # R2 -> 1 -> the head explains the target variance.
    e_w = e_t = e_t2 = e_se = 0.0

    for _ in range(batches):
        if len(buffer) == 0:
            break

        batch = buffer.sample(batch_size)
        collated = _collate(batch, device, aux_ease=aux_ease)
        if aux_ease:
            planes, policy_t, pmask, value_t, ease_t, emask = collated
        else:
            planes, policy_t, pmask, value_t = collated

        optimiser.zero_grad()
        if ease_optimiser is not None:
            ease_optimiser.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            # single trunk forward serves both steps; return_ease only when
            # the ease path consumes it -- ChessNet.forward keeps its 2-tuple
            # contract for every search/arena/probe caller
            out = net(planes, return_ease=aux_ease)
            policy_logits, value_p = out[0], out[1]

            log_probs = F.log_softmax(policy_logits, dim=1)
            # Per-row cross-entropy, averaged over POLICY rows only (pmask).
            # A plain .mean() would dilute the policy gradient by the fraction
            # of value-only rows in the batch (~75% with record_fast_rows), so
            # normalize by the mask sum -- the policy signal per policy row is
            # then identical to a run without value-only rows.
            ce = -(policy_t * log_probs).sum(dim=1, keepdim=True)     # (B,1)
            policy_loss = (ce * pmask).sum() / pmask.sum().clamp(min=1.0)
            value_loss = F.mse_loss(value_p, value_t)
            loss_pv = policy_loss + value_loss

            ease_loss = None
            if aux_ease:
                ease_loss = _masked_mse(out[2], ease_t, emask)
                with torch.no_grad():
                    w = emask.float()
                    t = ease_t.float()
                    p = out[2].detach().float()
                    e_w += w.sum().item()
                    e_t += (w * t).sum().item()
                    e_t2 += (w * t * t).sum().item()
                    e_se += (w * (p - t) ** 2).sum().item()

        if decoupled:
            # ---- step 1: trunk + policy + value (ease graph untouched) ----
            scaler.scale(loss_pv).backward()
            scaler.step(optimiser)
            scaler.update()
            # ---- step 2: ease head only, its own optimiser ----
            eopt = ease_optimiser if ease_optimiser is not None else optimiser
            scaler.scale(ease_weight * ease_loss).backward()
            scaler.step(eopt)
            scaler.update()
            loss = loss_pv + ease_weight * ease_loss    # combined, logging only
        else:
            loss = loss_pv
            if aux_ease:
                loss = loss + ease_weight * ease_loss
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

        acc["total"] += loss.item()
        acc["policy"] += policy_loss.item()
        acc["value"] += value_loss.item()
        if aux_ease:
            acc["ease"] += ease_loss.item()
        actual += 1

    ease_stats = dict(ease_R2=0.0, ease_tvar=0.0, ease_tmean=0.0)
    if e_w > 0:
        tmean = e_t / e_w
        tvar = max(e_t2 / e_w - tmean * tmean, 0.0)
        mse = e_se / e_w
        ease_stats["ease_tmean"] = tmean
        ease_stats["ease_tvar"] = tvar
        # guard: a (near-)constant target makes R2 meaningless; report 0
        ease_stats["ease_R2"] = (1.0 - mse / tvar) if tvar > 1e-6 else 0.0

    if actual == 0:
        return {**{k: 0.0 for k in acc}, **ease_stats}
    return {**{k: v / actual for k, v in acc.items()}, **ease_stats}

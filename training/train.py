import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler


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
    """
    planes = np.stack([b[0] for b in batch])               # (B,17,8,8)
    policy = np.stack([b[1] for b in batch])               # (B,NUM_ACTIONS)
    value  = np.array([b[2] for b in batch], np.float32)   # (B,)

    t = lambda a: torch.from_numpy(a).to(device)
    planes = t(planes)
    policy = t(policy)
    value  = t(value).unsqueeze(1)

    if not aux_ease:
        return planes, policy, value

    ease  = np.array([b[3] for b in batch], np.float32)
    emask = np.array([b[4] for b in batch], np.float32)
    ease  = t(ease).unsqueeze(1)
    emask = t(emask).unsqueeze(1)
    return planes, policy, value, ease, emask


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

    for _ in range(batches):
        if len(buffer) == 0:
            break

        batch = buffer.sample(batch_size)
        collated = _collate(batch, device, aux_ease=aux_ease)
        if aux_ease:
            planes, policy_t, value_t, ease_t, emask = collated
        else:
            planes, policy_t, value_t = collated

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
            policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_p, value_t)
            loss_pv = policy_loss + value_loss

            ease_loss = None
            if aux_ease:
                ease_loss = _masked_mse(out[2], ease_t, emask)

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

    if actual == 0:
        return {k: 0.0 for k in acc}
    return {k: v / actual for k, v in acc.items()}
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
                aux_ease=False, ease_weight=1.0, scaler=None):
    """
    Run gradient steps. Returns mean losses {total, policy, value, ease}.
    With aux_ease=False this is the original policy+value training (ease stays 0).

    `scaler` is the AMP GradScaler and MUST persist across iterations: its scale
    factor is tuned online (halved on overflow, periodically raised), so building
    a new one every call discards that calibration and re-runs the warmup each
    time. Create one GradScaler once in the training loop and pass it in; the
    fallback below only exists for standalone/one-off use.
    """
    if scaler is None:
        scaler = GradScaler("cuda", enabled=(device.type == "cuda"))
    use_amp = (device.type == "cuda")

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

        with autocast(device_type=device.type, enabled=use_amp):
            out = net(planes)
            policy_logits, value_p = out[0], out[1]

            log_probs = F.log_softmax(policy_logits, dim=1)
            policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(value_p, value_t)
            loss = policy_loss + value_loss

            if aux_ease:
                ease_p = out[2]
                ease_loss = _masked_mse(ease_p, ease_t, emask)
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
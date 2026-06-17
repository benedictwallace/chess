import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

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


def _collate(batch, device):
    """
    Each example:
      (planes, policy, value, ease, ease_mask, cliff, cliff_mask, stab, stab_mask)
    """
    planes = np.stack([b[0] for b in batch])               # (B,18,8,8)
    policy = np.stack([b[1] for b in batch])               # (B,NUM_ACTIONS)
    value  = np.array([b[2] for b in batch], np.float32)   # (B,)
    ease   = np.array([b[3] for b in batch], np.float32)
    emask  = np.array([b[4] for b in batch], np.float32)
    cliff  = np.array([b[5] for b in batch], np.float32)
    cmask  = np.array([b[6] for b in batch], np.float32)
    stab   = np.array([b[7] for b in batch], np.float32)
    smask  = np.array([b[8] for b in batch], np.float32)

    t = lambda a: torch.from_numpy(a).to(device)
    planes = t(planes)
    policy = t(policy)
    value  = t(value).unsqueeze(1)
    ease   = t(ease).unsqueeze(1);  emask = t(emask).unsqueeze(1)
    cliff  = t(cliff).unsqueeze(1);  cmask = t(cmask).unsqueeze(1)
    stab   = t(stab).unsqueeze(1);  smask = t(smask).unsqueeze(1)
    return planes, policy, value, ease, emask, cliff, cmask, stab, smask


def _masked_mse(pred, target, mask):
    return (mask * (pred - target) ** 2).sum() / mask.sum().clamp(min=1.0)


def train_epoch(net, buffer, optimiser, device, batches=32, batch_size=128,
                ease_weight=1.0, cliff_weight=1.0, stab_weight=1.0):
    """
    Run gradient steps. Returns a dict of mean losses:
      {total, policy, value, ease, cliff, stab}
    """
    net.train()
    acc = dict(total=0.0, policy=0.0, value=0.0, ease=0.0, cliff=0.0, stab=0.0)
    actual = 0

    for _ in range(batches):
        if len(buffer) == 0:
            break

        batch = buffer.sample(batch_size)
        (planes, policy_t, value_t,
         ease_t, emask, cliff_t, cmask, stab_t, smask) = _collate(batch, device)

        policy_logits, value_p, ease_p, cliff_p, stab_p = net(planes)

        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_t * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_p, value_t)
        ease_loss = _masked_mse(ease_p, ease_t, emask)
        cliff_loss = _masked_mse(cliff_p, cliff_t, cmask)
        stab_loss = _masked_mse(stab_p, stab_t, smask)

        loss = (policy_loss + value_loss
                + ease_weight * ease_loss
                + cliff_weight * cliff_loss
                + stab_weight * stab_loss)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        acc["total"] += loss.item()
        acc["policy"] += policy_loss.item()
        acc["value"] += value_loss.item()
        acc["ease"] += ease_loss.item()
        acc["cliff"] += cliff_loss.item()
        acc["stab"] += stab_loss.item()
        actual += 1

    if actual == 0:
        return {k: 0.0 for k in acc}
    return {k: v / actual for k, v in acc.items()}
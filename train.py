
import random
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

class ReplayBuffer:
    """
    Buffer of examples, training happens on most recent game plus random selection of 
    past states, makes gradient more smooth, instead of training on lots of similar data
    one move apart.

    """

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
    shape batches + tensorise for GPU
    """
    planes = np.stack([b[0] for b in batch])             # (B, 18, 8, 8)
    policy = np.stack([b[1] for b in batch])             # (B, NUM_ACTIONS)
    value = np.array([b[2] for b in batch], np.float32)  # (B,)

    planes = torch.from_numpy(planes).to(device)
    policy = torch.from_numpy(policy).to(device)
    value = torch.from_numpy(value).to(device).unsqueeze(1)  # (B, 1)
    return planes, policy, value


def train_epoch(net, buffer, optimiser, device, batches=32, batch_size=128):
    """
    Run gradient steps, sampling from buffer/
    Returns:
        mean_total_loss:
        mean_policy_loss:
        mean_value_loss:
    """

    net.train()
    total, policy_l, value_l = 0.0, 0.0, 0.0
    actual = 0

    for _ in range(batches):
        if len(buffer) == 0:
            break

        batch = buffer.sample(batch_size)
        planes, policy_target, value_target = _collate(batch, device)

        policy_logits, value_pred = net(planes)

        # cross entropy between target dist and predicted.
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

        value_loss = F.mse_loss(value_pred, value_target)

        loss = policy_loss + value_loss

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total += loss.item()
        policy_l += policy_loss.item()
        value_l += value_loss.item()
        actual += 1

    if actual == 0:
        return 0.0, 0.0, 0.0
    
    # avg losses
    return total / actual, policy_l / actual, value_l / actual
import torch
import torch.nn as nn
import torch.nn.functional as F
from move_encoding import NUM_ACTIONS

NUM_PLANES = 18


class ResidualBlock(nn.Module):
    """Two conv layers with a skip connection."""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return F.relu(x)


class ScalarHead(nn.Module):
    """1x1 conv -> fc -> fc -> scalar, with a configurable output activation."""
    def __init__(self, channels, activation):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(1 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)
        self.activation = activation

    def forward(self, x):
        h = F.relu(self.bn(self.conv(x)))
        h = h.reshape(h.size(0), -1)
        h = F.relu(self.fc1(h))
        return self.activation(self.fc2(h))


class ChessNet(nn.Module):
    def __init__(self, channels=64, num_blocks=5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, NUM_ACTIONS)

        # value head -> game outcome in [-1, 1], mover's POV  (unchanged keys)
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # ease head -> forgiveness (frac_safe) in [0, 1]  (unchanged keys)
        # self.ease_conv = nn.Conv2d(channels, 1, 1, bias=False)
        # self.ease_bn = nn.BatchNorm2d(1)
        # self.ease_fc1 = nn.Linear(1 * 8 * 8, 64)
        # self.ease_fc2 = nn.Linear(64, 1)

        # # record-only heads, both in [0, 1]
        # self.cliff_head = ScalarHead(channels, torch.sigmoid)  # cliff-size discounted return
        # self.stab_head = ScalarHead(channels, torch.sigmoid)   # trajectory stability

    def forward(self, x):
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        # e = F.relu(self.ease_bn(self.ease_conv(x)))
        # e = e.reshape(e.size(0), -1)
        # e = F.relu(self.ease_fc1(e))
        # ease = torch.sigmoid(self.ease_fc2(e))

        # cliff = self.cliff_head(x)
        # stab = self.stab_head(x)

        return policy_logits, value
import torch
import torch.nn as nn
import torch.nn.functional as F
from move_encoding import NUM_ACTIONS

NUM_PLANES = 18


class ResidualBlock(nn.Module):
    """
    Two conv layers with a skip connection.
    """
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

        # value head -> game outcome in [-1, 1], mover's POV
        self.value_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.value_fc2 = nn.Linear(64, 1)

        # ease head -> forgiveness of the position in [0, 1]
        # absolute (not side-to-move-signed): a property of the position itself.
        self.ease_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.ease_bn = nn.BatchNorm2d(1)
        self.ease_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.ease_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # x shape: (batch, 18, 8, 8)
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        # policy head -> (batch, NUM_ACTIONS) logits
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy_logits = self.policy_fc(p)

        # value head -> (batch, 1) in [-1, 1]
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        # tanh to match reward, -1 loss 0 draw, 1 win
        value = torch.tanh(self.value_fc2(v))

        # ease head -> (batch, 1) in [0, 1]
        e = F.relu(self.ease_bn(self.ease_conv(x)))
        e = e.reshape(e.size(0), -1)
        e = F.relu(self.ease_fc1(e))
        # sigmoid to match the forgiveness target (a fraction in [0, 1])
        ease = torch.sigmoid(self.ease_fc2(e))

        return policy_logits, value, ease
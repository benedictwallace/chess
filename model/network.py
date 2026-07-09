import torch
import torch.nn as nn
import torch.nn.functional as F
from model.move_encoding import NUM_ACTIONS

# Must match model.encoding.NUM_PLANES (the conv stem's in-channels = encoded
# planes). 17 = 12 piece planes + 4 castling + 1 en passant; no side-to-move
# plane, since the board is canonicalised to the mover's POV.
NUM_PLANES = 17


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
    def __init__(self, channels=64, num_blocks=5, ease_detach=True):
        super().__init__()
        # ease_detach=True (default): the ease head reads DETACHED trunk
        # features, so its loss trains only ease_conv/bn/fc1/fc2 and
        # contributes zero gradient to the stem, residual blocks, policy head,
        # or value head -- they train exactly as if the ease head didn't
        # exist. This keeps ease a pure readout: any later behaviour change
        # must come from an explicit ease consumer, not from the aux loss
        # silently reshaping the shared representation. Set False to let ease
        # act as a KataGo-style auxiliary task that shapes the trunk.
        self.ease_detach = ease_detach

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

        # ease head -> forgiveness in [0, 1]  (unchanged keys)
        self.ease_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.ease_bn = nn.BatchNorm2d(1)
        self.ease_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.ease_fc2 = nn.Linear(64, 1)

    def forward(self, x, return_ease=False):
        """Returns (policy_logits, value) by default -- every existing search /
        arena / probe consumer unpacks a 2-tuple and stays untouched. Training
        (and anything else that wants the forgiveness estimate) passes
        return_ease=True for (policy_logits, value, ease) with ease already
        sigmoid-activated in [0, 1], matching the masked-MSE loss in
        training/train.py."""
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

        if not return_ease:
            return policy_logits, value

        feat = x.detach() if self.ease_detach else x
        e = F.relu(self.ease_bn(self.ease_conv(feat)))
        e = e.reshape(e.size(0), -1)
        e = F.relu(self.ease_fc1(e))
        ease = torch.sigmoid(self.ease_fc2(e))

        return policy_logits, value, ease
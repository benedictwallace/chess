import torch
import torch.nn as nn
import torch.nn.functional as F
from model.move_encoding import NUM_ACTIONS

NUM_PLANES = 19


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
        return F.relu(x + residual)


class ChessNet(nn.Module):
    """
    WHAT CHANGED vs the previous version, and why.

    POLICY HEAD -- was Conv2d(C, 2, 1) -> flatten(128) -> Linear(128, 4672).
    Every one of the 4672 action logits was a linear function of the SAME 128
    numbers, so the policy was a rank-<=128 map: it could not represent 4672
    independent action preferences, and it destroyed the spatial structure of
    the action space (the encoding is fromSq * 73 + moveType, i.e. a per-square
    73-way choice). The 2-plane bottleneck comes from AlphaGo Zero, where the
    policy had 362 outputs; it does not transfer to chess.

    Now: a 3x3 conv keeps full width, then a 1x1 conv emits 73 planes. Output
    (B, 73, 8, 8) permuted to (B, 8, 8, 73) and flattened gives exactly
    index = (rank*8 + file)*73 + moveType = fromSq*73 + moveType, matching
    model.move_encoding._encode_raw (verified). Each square's 73 move-type
    logits are now computed from that square's own feature column, and the head
    is translation-equivariant -- the same structure AlphaZero used for chess.

    VALUE HEAD -- was Conv2d(C, 1, 1) -> Linear(64, 64) -> Linear(64, 1). One
    plane and a 64-wide hidden layer is a very small readout for a quantity
    that has to integrate material, king safety and pawn structure over the
    whole board. Widened to 32 planes and a 256-wide hidden layer, plus a
    global-average-pooled summary concatenated in (KataGo-style): board-wide
    sums like material counts are then available directly instead of having to
    be reconstructed by the final linear layer.

    Cost: the trunk is unchanged, so self-play leaf latency moves very little
    (the heads are 1x1/3x3 convs on an 8x8 grid). Parameter count goes UP,
    but nearly all of it is in the policy conv, which is where it was missing.

    The forgiveness head is unchanged and still reads DETACHED trunk features
    by default, so training/train.py's decoupled two-optimiser path and the
    "forgiveness_" name split in main.py / main_multigpu.py both keep working
    untouched.

    NOT CHECKPOINT-COMPATIBLE with the old policy_fc / value_fc1 shapes. This
    is a fresh-run change.
    """

    def __init__(self, channels=128, num_blocks=8, forgiveness_detach=True,
                 policy_channels=None, value_channels=32, value_hidden=256):
        super().__init__()
        self.forgiveness_detach = forgiveness_detach
        pc = channels if policy_channels is None else policy_channels

        self.stem = nn.Sequential(
            nn.Conv2d(NUM_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # ---- policy head: (B, 73, 8, 8) -> fromSq*73 + moveType ----
        self.policy_conv1 = nn.Conv2d(channels, pc, 3, padding=1, bias=False)
        self.policy_bn1 = nn.BatchNorm2d(pc)
        self.policy_conv2 = nn.Conv2d(pc, 73, 1, bias=True)

        # ---- value head ----
        self.value_conv = nn.Conv2d(channels, value_channels, 1, bias=False)
        self.value_bn = nn.BatchNorm2d(value_channels)
        # flattened spatial features + a global-average-pooled summary
        self.value_fc1 = nn.Linear(value_channels * 8 * 8 + value_channels,
                                   value_hidden)
        self.value_fc2 = nn.Linear(value_hidden, 1)

        # ---- forgiveness head (unchanged keys / shapes) ----
        self.forgiveness_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.forgiveness_bn = nn.BatchNorm2d(1)
        self.forgiveness_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.forgiveness_fc2 = nn.Linear(64, 1)

    def forward(self, x, return_forgiveness=False):
        """Same 2-tuple contract as before; 3-tuple with return_forgiveness."""
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)

        p = F.relu(self.policy_bn1(self.policy_conv1(x)))
        p = self.policy_conv2(p)                       # (B, 73, 8, 8)
        # (B,73,H,W) -> (B,H,W,73) -> (B, 64*73); index = (rank*8+file)*73 + type
        policy_logits = p.permute(0, 2, 3, 1).reshape(p.size(0), NUM_ACTIONS)

        v = F.relu(self.value_bn(self.value_conv(x)))   # (B, vc, 8, 8)
        v_pool = v.mean(dim=(2, 3))                     # (B, vc) board-wide sums
        v = torch.cat([v.reshape(v.size(0), -1), v_pool], dim=1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))

        if not return_forgiveness:
            return policy_logits, value

        feat = x.detach() if self.forgiveness_detach else x
        e = F.relu(self.forgiveness_bn(self.forgiveness_conv(feat)))
        e = e.reshape(e.size(0), -1)
        e = F.relu(self.forgiveness_fc1(e))
        forgiveness = torch.sigmoid(self.forgiveness_fc2(e))

        return policy_logits, value, forgiveness

    
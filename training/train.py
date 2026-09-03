import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

from model.move_encoding import NUM_ACTIONS


class _Ring:
    """Fixed-capacity ring buffer backed by a LIST, not a deque.

    random.sample() indexes its population, and deque.__getitem__ is O(n) --
    so sampling k rows from a d-element deque costs O(k*d). Measured on a
    300k-row buffer: 64 batches of 256 took 59.7 ms from a deque vs 12.2 ms
    from a list. Same eviction semantics (oldest row overwritten first).
    """
    __slots__ = ("data", "capacity", "pos")

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self.data = []
        self.pos = 0

    def append(self, x):
        if len(self.data) < self.capacity:
            self.data.append(x)
        else:
            self.data[self.pos] = x
            self.pos += 1
            if self.pos == self.capacity:
                self.pos = 0

    def __len__(self):
        return len(self.data)

    def sample(self, k):
        return random.sample(self.data, min(k, len(self.data)))


def _has_policy(example):
    """True if this row carries a policy target (vs a value-only fast row)."""
    p = example[1]
    if isinstance(p, tuple):          # sparse (indices, probs)
        return len(p[0]) > 0
    return bool(p.sum() > 0)          # legacy dense row


class ReplayBuffer:
    """Replay buffer that keeps POLICY rows and VALUE-ONLY rows in separate
    pools and composes every batch from both.

    WHY THE SPLIT. With record_fast_rows=True most rows are value-only
    (playout-capped plies with an empty policy target). Sampling uniformly
    from one pool means a batch of 256 contains only ~256*full_search_prob
    policy rows -- at full_search_prob=0.25 that is 64. train_epoch already
    normalises the policy loss by the mask sum, so the policy gradient is not
    SCALED down, but it is still estimated from a quarter of the batch, i.e.
    it is ~2x noisier than the batch size suggests. Drawing a fixed fraction
    of each batch from the policy pool fixes that at no extra cost: same 256
    rows forward, twice the policy rows in them.

    policy_frac is the share of each batch drawn from the policy pool (0.5 =
    half). If either pool is short the other backfills, so early iterations
    and full_search_prob=1.0 both behave sensibly.

    CAPACITY SPLIT AND STALENESS. Each pool holds its own most-recent rows, so
    a pool of size M spans M / (arrival rate) of history. Policy rows arrive at
    rate full_search_prob and value-only rows at (1 - full_search_prob), so
    equal-sized pools give both the same time-horizon only when
    full_search_prob = 0.5. Set policy_capacity_frac = full_search_prob to keep
    the two windows aligned at other settings; otherwise the smaller stream is
    held longer and its rows are staler.
    """

    def __init__(self, capacity=50_000, policy_frac=0.5,
                 policy_capacity_frac=0.5):
        pc = max(1, int(capacity * policy_capacity_frac))
        vc = max(1, capacity - pc)
        self.policy_rows = _Ring(pc)
        self.value_rows = _Ring(vc)
        self.policy_frac = float(policy_frac)

    def add_examples(self, examples):
        pr, vr = self.policy_rows, self.value_rows
        for e in examples:
            (pr if _has_policy(e) else vr).append(e)

    def __len__(self):
        return len(self.policy_rows) + len(self.value_rows)

    def counts(self):
        """(policy_rows, value_only_rows) -- for logging the real mix."""
        return len(self.policy_rows), len(self.value_rows)

    def sample(self, batch_size):
        want_p = int(round(batch_size * self.policy_frac))
        n_p = min(want_p, len(self.policy_rows))
        n_v = min(batch_size - n_p, len(self.value_rows))
        # backfill from whichever pool still has rows
        if n_p + n_v < batch_size:
            n_p = min(len(self.policy_rows), batch_size - n_v)
        out = self.policy_rows.sample(n_p) + self.value_rows.sample(n_v)
        random.shuffle(out)
        return out


def _collate(batch, device, aux_forgiveness=False):
    """
    Examples are (planes, policy, value) or, with aux_forgiveness,
    (planes, policy, value, forgiveness_target, forgiveness_mask).

    `policy` per row is either a dense (NUM_ACTIONS,) float32 array (legacy
    self_play.py rows) or a SPARSE (action_indices, probs) pair (batched
    self-play). Sparse rows are densified here, per batch. An EMPTY sparse
    pair (or an all-zero dense row) is a VALUE-ONLY row: it gets policy-mask 0
    so it contributes nothing to the policy loss but trains value/forgiveness as
    normal.

    Returns (planes, policy, pmask, value[, forgiveness, emask]) -- pmask is (B,1).
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

    if not aux_forgiveness:
        return planes, policy, pmask, value

    forgiveness  = np.array([b[3] for b in batch], np.float32)
    emask = np.array([b[4] for b in batch], np.float32)
    forgiveness  = t(forgiveness).unsqueeze(1)
    emask = t(emask).unsqueeze(1)
    return planes, policy, pmask, value, forgiveness, emask


def _masked_mse(pred, target, mask):
    return (mask * (pred - target) ** 2).sum() / mask.sum().clamp(min=1.0)


def train_epoch(net, buffer, optimiser, device, batches=32, batch_size=128,
                aux_forgiveness=False, forgiveness_weight=1.0, forgiveness_optimiser=None,
                scaler=None):
    """
    Run gradient steps. Returns mean losses {total, policy, value, forgiveness}.
    With aux_forgiveness=False this is the original policy+value training (forgiveness 0).

    DECOUPLED FORGIVENESS TRAINING: when aux_forgiveness is on and the net's forgiveness head reads
    DETACHED trunk features (ChessNet's forgiveness_detach=True default), the forgiveness
    loss is never summed with the policy+value loss. Each batch takes TWO
    independent gradient steps from ONE shared trunk forward:

        step 1:  policy_loss + value_loss      -> `optimiser`
                 (trunk + policy head + value head)
        step 2:  forgiveness_weight * forgiveness_loss       -> `forgiveness_optimiser`
                 (the four forgiveness-head layers, nothing else)

    The detach makes the two computation graphs disjoint, so the second
    backward needs no retain_graph, costs no second forward, and physically
    cannot deposit gradient into the trunk or the other heads. Build the two
    optimisers over DISJOINT parameter sets (the mains split on "forgiveness_" in the
    parameter name); if forgiveness_optimiser is omitted, `optimiser` is stepped a
    second time -- only valid when it actually contains the forgiveness parameters.

    If the net was built with forgiveness_detach=False, the forgiveness subgraph shares the
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
    decoupled = aux_forgiveness and getattr(inner, "forgiveness_detach", True)

    net.train()
    acc = dict(total=0.0, policy=0.0, value=0.0, forgiveness=0.0,
               policy_kl=0.0, target_entropy=0.0)
    actual = 0
    # masked forgiveness-target statistics, accumulated over the whole epoch. The raw
    # forgiveness MSE is uninterpretable on its own (its floor is the label noise);
    # what matters is the fit relative to the target spread:
    #     forgiveness_R2 = 1 - MSE / Var(target).
    # R2 ~ 0  -> the head predicts no better than the batch mean: either the
    #            labels are noise at this tau/floor or the head can't read the
    #            needed feature from the (detached) trunk.
    # R2 -> 1 -> the head explains the target variance.
    e_w = e_t = e_t2 = e_se = 0.0

    for _ in range(batches):
        if len(buffer) == 0:
            break

        batch = buffer.sample(batch_size)
        collated = _collate(batch, device, aux_forgiveness=aux_forgiveness)
        if aux_forgiveness:
            planes, policy_t, pmask, value_t, forgiveness_t, emask = collated
        else:
            planes, policy_t, pmask, value_t = collated

        optimiser.zero_grad()
        if forgiveness_optimiser is not None:
            forgiveness_optimiser.zero_grad()

        with autocast(device_type=device.type, enabled=use_amp):
            # single trunk forward serves both steps; return_forgiveness only when
            # the forgiveness path consumes it -- ChessNet.forward keeps its 2-tuple
            # contract for every search/arena/probe caller
            out = net(planes, return_forgiveness=aux_forgiveness)
            policy_logits, value_p = out[0], out[1]

            log_probs = F.log_softmax(policy_logits, dim=1)
            # Per-row cross-entropy, averaged over POLICY rows only (pmask).
            # A plain .mean() would dilute the policy gradient by the fraction
            # of value-only rows in the batch (~75% with record_fast_rows), so
            # normalize by the mask sum -- the policy signal per policy row is
            # then identical to a run without value-only rows.
            ce = -(policy_t * log_probs).sum(dim=1, keepdim=True)     # (B,1)
            policy_loss = (ce * pmask).sum() / pmask.sum().clamp(min=1.0)

            # ---- TARGET ENTROPY AND KL --------------------------------------
            # Cross-entropy decomposes exactly as
            #     CE = H(target) + KL(target || prediction)
            # and ONLY the KL term is learnable. H(target) is a property of the
            # SEARCH, not the net: it is set by the simulation count, the root
            # Dirichlet noise, and how genuinely ambiguous the position is. It
            # does not shrink as the net improves -- measured on this codebase
            # it sits around 1.7 nats (perplexity ~5.4) at 400 sims, and rises
            # slightly at 800 because deeper search explores more.
            #
            # So a policy CE of ~2.05 is ~1.7 floor plus ~0.35 of real error,
            # and a run that moves CE by 0.05 has actually cut its error by
            # ~14%. Logging CE alone makes genuine progress look like a plateau
            # -- which is exactly what it did here, while the same checkpoints
            # gained 236 Elo head-to-head.
            #
            # These are diagnostics only: no gradient flows from them, and the
            # optimisation target is unchanged.
            with torch.no_grad():
                tent = -(policy_t * torch.log(policy_t.clamp(min=1e-12))
                         ).sum(dim=1, keepdim=True)                   # (B,1)
                denom = pmask.sum().clamp(min=1.0)
                target_entropy = (tent * pmask).sum() / denom
                policy_kl = policy_loss - target_entropy

            value_loss = F.mse_loss(value_p, value_t)
            loss_pv = policy_loss + value_loss

            forgiveness_loss = None
            if aux_forgiveness:
                forgiveness_loss = _masked_mse(out[2], forgiveness_t, emask)
                with torch.no_grad():
                    w = emask.float()
                    t = forgiveness_t.float()
                    p = out[2].detach().float()
                    e_w += w.sum().item()
                    e_t += (w * t).sum().item()
                    e_t2 += (w * t * t).sum().item()
                    e_se += (w * (p - t) ** 2).sum().item()

        if decoupled:
            # ---- ONE backward over a summed loss, then step both optimisers ----
            #
            # This used to be two separate .backward() calls, one per optimiser.
            # That is mathematically the same thing here but CRASHES under
            # torch.compile:
            #
            #     RuntimeError: Trying to backward through the graph a second
            #     time ... saved intermediate values ... have already been freed
            #
            # AOT autograd compiles the whole forward into ONE graph and frees
            # its saved tensors when the first backward completes, so the second
            # call finds nothing left. It does not matter that the two
            # subgraphs are logically disjoint -- the compiled region is a
            # single unit as far as the autograd engine is concerned. Eager
            # mode happened to tolerate the pattern, which is why this only
            # surfaced once torch.compile was switched on AND
            # forgiveness_targets became True (the second backward never ran
            # while aux_forgiveness was off).
            #
            # Summing is EXACT, not an approximation, because `decoupled` is
            # only true when the head reads DETACHED trunk features:
            #   d(forgiveness_loss)/d(trunk, policy, value params) == 0
            #       -- the detach severs it
            #   d(loss_pv)/d(forgiveness params)             == 0
            #       -- the forgiveness head is not on the policy/value path
            # so each parameter group receives precisely the gradient it
            # received before. The two optimisers still hold disjoint parameter
            # sets and still step with their own learning rates; only the
            # number of graph traversals changed. (retain_graph=True would also
            # work, but it keeps the entire forward's activations alive for a
            # second pass -- real memory for no benefit.)
            #
            # The empty-batch guard is preserved: when a batch carries no
            # forgiveness-labelled rows the masked loss is 0 with zero gradient,
            # but AdamW's weight_decay plus stale momentum would still shrink
            # the head on every such batch -- a slow drift toward the origin
            # driven by nothing. So the head's optimiser is only stepped when
            # there is something to learn from, and the loss term is only
            # summed in when it is live.
            have_forgiveness = emask.sum().item() > 0
            total = loss_pv
            if have_forgiveness:
                total = total + forgiveness_weight * forgiveness_loss

            scaler.scale(total).backward()
            scaler.step(optimiser)
            if have_forgiveness:
                eopt = (forgiveness_optimiser if forgiveness_optimiser is not None
                        else optimiser)
                if eopt is not optimiser:
                    scaler.step(eopt)
            scaler.update()

            loss = loss_pv + forgiveness_weight * forgiveness_loss    # logging only
        else:
            loss = loss_pv
            if aux_forgiveness:
                loss = loss + forgiveness_weight * forgiveness_loss
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()

        acc["total"] += loss.item()
        acc["policy"] += policy_loss.item()
        acc["policy_kl"] += policy_kl.item()
        acc["target_entropy"] += target_entropy.item()
        acc["value"] += value_loss.item()
        if aux_forgiveness:
            acc["forgiveness"] += forgiveness_loss.item()
        actual += 1

    forgiveness_stats = dict(forgiveness_R2=0.0, forgiveness_tvar=0.0, forgiveness_tmean=0.0)
    if e_w > 0:
        tmean = e_t / e_w
        tvar = max(e_t2 / e_w - tmean * tmean, 0.0)
        mse = e_se / e_w
        forgiveness_stats["forgiveness_tmean"] = tmean
        forgiveness_stats["forgiveness_tvar"] = tvar
        # guard: a (near-)constant target makes R2 meaningless; report 0
        forgiveness_stats["forgiveness_R2"] = (1.0 - mse / tvar) if tvar > 1e-6 else 0.0

    if actual == 0:
        return {**{k: 0.0 for k in acc}, **forgiveness_stats}
    return {**{k: v / actual for k, v in acc.items()}, **forgiveness_stats}


"""
Offline multi-mode forgiveness-head training.

Given a checkpoint, this script:

  1. loads the net and FREEZES it (trunk, policy head, value head -- nothing
     but new forgiveness heads is ever trained here);
  2. generates self-play games with the frozen net, computing the forgiveness target
     for EVERY mode ("gap", "entropy", "tree_gap", "tree_entropy", "flat_gap",
     "flat_entropy") from the SAME search tree at every recorded root -- so
     the modes are compared on identical positions and identical Q statistics,
     never on different games;
  3. trains ONE fresh forgiveness head PER MODE on those targets, all heads
     simultaneously off a single (frozen) trunk forward per batch, reporting
     held-out R^2 per mode per epoch;
  4. saves, for each mode: a full checkpoint (frozen backbone + that mode's
     BEST-val-R2 head, loadable by play_checkpoint / probe_forgiveness /
     arena exactly like a normal net_iterN.pt); a compact all-heads file
     (best + final state per mode); a metrics CSV; and
     forgiveness_heads_history.pt with EVERY head's state at every
     --snapshot-every epochs (rewritten incrementally, so an interrupted run
     keeps its trajectory). Load a specific snapshot with e.g.
         h = torch.load("forgiveness_heads_history.pt")
         head.load_state_dict(h["history"]["flat_entropy"][25]["state"])

Why this is sound: ChessNet's forgiveness head reads DETACHED trunk features
(forgiveness_detach=True), so in the online loop the forgiveness loss already trains only
the four forgiveness_* layers. Training them offline against a frozen checkpoint is
the same regression problem with a STATIONARY feature map and stationary
targets -- cleaner, and it lets a single expensive dataset serve many cheap
head-training experiments.

How the multi-mode targets are captured: training.self_play_batched imports
`forgiveness_target` from search.forgiveness at module level and calls it at exactly one
site (do_move), with the finished root tree in hand; the returned
(forgiveness_t, forgiveness_m) pair is carried through history/finalize untouched. We
monkey-patch that module-level name with a wrapper that calls the real
forgiveness_target once per mode and returns float32 VECTORS (K targets, K masks).
The self-play machinery never looks inside them, so nothing else changes.
The patch is restored afterwards.

Self-play settings differ from main.py's in two deliberate ways:
  * record_fast_rows=False -- fast rows carry no forgiveness target, useless here;
  * full_search_prob=1.0 (default) -- when only forgiveness rows count, a full move
    costs (sims + forgiveness_extra_sims) sims per row, while p=0.25 costs
    0.25*full + 0.75*fast sims per 0.25 rows (~1100 vs ~800 sims/row at the
    defaults). Full-searching every ply is the cheaper way to buy forgiveness rows,
    and subtree reuse compounds it (full->full moves hand over big subtrees).

Train/validation split is a CONTIGUOUS TAIL of the generation order, not a
random shuffle: rows of one game are appended as a block, so a random split
would put sister positions of the same game on both sides and leak
game-level correlation into the "held-out" R^2. With a tail split at most
one game straddles the boundary.

Run from the repo root (same import layout as main.py):

    python train_forgiveness_heads.py checkpoints/net_iter800.pt --games 512 \
        --dataset datasets/iter800_forgiveness.npz --out-dir forgiveness_heads_iter800

Re-running with an existing --dataset file skips self-play and goes straight
to head training (use --regen to force regeneration), so tau/mode/LR sweeps
cost seconds, not GPU-hours.
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.network import ChessNet
import training.self_play_batched as spb
from search.forgiveness import forgiveness_target as _forgiveness_target_single

ALL_MODES = ["gap", "entropy", "tree_gap", "tree_entropy",
             "flat_gap", "flat_entropy"]


# --------------------------------------------------------------------------- #
# checkpoint loading / freezing
# --------------------------------------------------------------------------- #
def load_frozen_net(path, device, channels=None, num_blocks=None):
    """Load a checkpoint into a frozen, eval-mode ChessNet.

    channels/num_blocks come from the checkpoint's saved config unless
    overridden (needed only for very old checkpoints without a config dict).
    Returns (net, ckpt) -- ckpt is kept around for its model_state (the
    backbone we merge trained heads back into) and its config (tau/gamma/
    search defaults, provenance).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})
    channels = channels or cfg.get("channels", 128)
    num_blocks = num_blocks or cfg.get("num_blocks", 8)

    net = ChessNet(channels=channels, num_blocks=num_blocks)
    missing, unexpected = net.load_state_dict(ckpt["model_state"], strict=False)
    if missing:
        # a pre-forgiveness-head checkpoint: backbone loads, forgiveness_* stay random --
        # irrelevant here, we never use the checkpoint's own head for training
        print(f"  note: {len(missing)} params not in checkpoint "
              f"(e.g. {missing[0]}); fine if they are forgiveness_* keys.")
    if unexpected:
        print(f"  note: {len(unexpected)} unexpected checkpoint keys ignored "
              f"(e.g. {unexpected[0]}).")

    net.to(device)
    net.eval()                          # trunk BN uses running stats, frozen
    net.requires_grad_(False)           # belt and braces: nothing here trains
    print(f"loaded {path}  ({channels}x{num_blocks}, "
          f"iteration {ckpt.get('iteration', '?')})")
    return net, ckpt


# --------------------------------------------------------------------------- #
# dataset generation: one self-play run, ALL forgiveness modes per root
# --------------------------------------------------------------------------- #
def _make_multi_mode_forgiveness_target(modes):
    """A drop-in replacement for search.forgiveness.forgiveness_target (same positional
    signature as the do_move call site) that evaluates EVERY requested mode on
    the root and returns (targets, masks) as float32 vectors of length K.
    Modes with an undefined statistic on this root get mask 0 for this row
    only -- the masks are per-mode because e.g. the root-local statistic
    (which respects the forced-visit floor) can be undefined where the
    tree/flat aggregates are not, and vice versa."""
    def multi(root, floor, tau, mode, gamma=0.85, stat=None):  # noqa: ARG001
        K = len(modes)
        ts = np.zeros(K, dtype=np.float32)
        ms = np.zeros(K, dtype=np.float32)
        for i, m in enumerate(modes):
            t, k = _forgiveness_target_single(root, floor, tau, m, gamma)
            ts[i], ms[i] = t, k
        return ts, ms
    return multi


def generate_dataset(net, modes, args):
    """Self-play with the frozen net; returns (planes, targets, masks) as
    (N,19,8,8) / (N,K) / (N,K) float32 arrays, keeping only rows where at
    least one mode's target was computable."""
    patched = _make_multi_mode_forgiveness_target(modes)
    original = spb.forgiveness_target
    spb.forgiveness_target = patched           # intercept the single call site
    try:
        t0 = time.time()
        examples = spb.generate_games_batched(
            net, args.games,
            iterations=args.sims,
            max_plies=args.max_plies,
            temp_moves=args.temp_moves,
            concurrency=args.concurrency,
            adj_margin=args.adj_margin,
            adj_plies=args.adj_plies,
            use_cache=True, cache_cap=args.cache_cap,
            reuse_tree=True,
            full_search_prob=args.full_search_prob,
            fast_iterations=args.fast_sims,
            root_force_m=args.root_force_m,
            root_force_visits=args.root_force_visits,
            forgiveness_targets=True,
            forgiveness_tau=args.tau,
            forgiveness_target_mode="gap",      # ignored by the patch; kept valid
            forgiveness_gamma=args.gamma,
            forgiveness_extra_sims=args.forgiveness_extra_sims,
            forgiveness_force_m=args.forgiveness_force_m,
            fpu_reduction=args.fpu_reduction,
            value_target_lambda=1.0,     # value targets unused here
            record_fast_rows=False,      # fast rows carry no forgiveness target
            verbose=not args.quiet,
        )
        gen_sec = time.time() - t0
    finally:
        spb.forgiveness_target = original       # always restore the real function

    # rows: (planes, sparse_policy, value, forgiveness_vec, mask_vec) -- keep rows
    # where any mode is defined; policy/value are discarded (frozen anyway)
    kept = [(p, t, m) for (p, _pol, _v, t, m) in examples
            if isinstance(m, np.ndarray) and m.any()]
    if not kept:
        raise RuntimeError(
            "self-play produced no rows with a computable forgiveness target -- "
            "check forgiveness_force_m/root_force_visits (the forced-visit floor "
            "must qualify >= 2 children) and that games are being recorded.")
    planes = np.stack([p for p, _, _ in kept]).astype(np.float32)
    targets = np.stack([t for _, t, _ in kept]).astype(np.float32)
    masks = np.stack([m for _, _, m in kept]).astype(np.float32)
    print(f"dataset: {len(kept)} usable rows from {args.games} games "
          f"({len(examples)} recorded) in {gen_sec/60:.1f} min")
    for i, m in enumerate(modes):
        n = int(masks[:, i].sum())
        tv = targets[masks[:, i] > 0, i]
        print(f"  {m:<13s} defined on {n:6d} rows   "
              f"mean {tv.mean():.3f}  var {tv.var():.4f}")
    return planes, targets, masks


def save_dataset(path, planes, targets, masks, modes, args, ckpt_path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(checkpoint=os.path.abspath(ckpt_path), tau=args.tau,
                gamma=args.gamma, games=args.games, sims=args.sims,
                forgiveness_extra_sims=args.forgiveness_extra_sims,
                forgiveness_force_m=args.forgiveness_force_m,
                root_force_visits=args.root_force_visits,
                full_search_prob=args.full_search_prob)
    np.savez_compressed(path, planes=planes, targets=targets, masks=masks,
                        modes=np.array(modes), meta=json.dumps(meta))
    print(f"dataset saved -> {path} "
          f"({os.path.getsize(path)/1e6:.0f} MB compressed)")


def load_dataset(path, modes):
    """Load a saved dataset and select the requested modes' columns (they
    must all be present in the stored file)."""
    d = np.load(path, allow_pickle=False)
    stored = [str(m) for m in d["modes"]]
    idx = []
    for m in modes:
        if m not in stored:
            raise ValueError(f"dataset {path} lacks mode {m!r} "
                             f"(has {stored}); regenerate with --regen")
    idx = [stored.index(m) for m in modes]
    meta = json.loads(str(d["meta"]))
    print(f"dataset loaded <- {path}: {d['planes'].shape[0]} rows, "
          f"modes {stored}, tau={meta.get('tau')} "
          f"(generated from {os.path.basename(str(meta.get('checkpoint')))})")
    return (d["planes"].astype(np.float32),
            d["targets"][:, idx].astype(np.float32),
            d["masks"][:, idx].astype(np.float32),
            meta)


# --------------------------------------------------------------------------- #
# heads
# --------------------------------------------------------------------------- #
class ForgivenessHead(nn.Module):
    """Exact replica of ChessNet's forgiveness head, reading (frozen) trunk features.
    Submodule names deliberately match ChessNet's attribute names, so
    head.state_dict() keys ("forgiveness_conv.weight", ..., "forgiveness_fc2.bias") merge
    straight into a ChessNet model_state with a dict.update()."""

    def __init__(self, channels):
        super().__init__()
        self.forgiveness_conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.forgiveness_bn = nn.BatchNorm2d(1)
        self.forgiveness_fc1 = nn.Linear(1 * 8 * 8, 64)
        self.forgiveness_fc2 = nn.Linear(64, 1)

    def forward(self, feat):
        e = F.relu(self.forgiveness_bn(self.forgiveness_conv(feat)))
        e = e.reshape(e.size(0), -1)
        e = F.relu(self.forgiveness_fc1(e))
        return torch.sigmoid(self.forgiveness_fc2(e))


def trunk_features(net, x, use_amp):
    """Frozen stem + residual blocks (eval-mode BN), no grad. This is the
    exact feature map ChessNet's own forgiveness head reads (post-detach), computed
    the way every search/probe consumer computes it (net.eval())."""
    with torch.no_grad():
        if use_amp:
            with torch.autocast("cuda"):
                h = net.stem(x)
                for b in net.blocks:
                    h = b(h)
        else:
            h = net.stem(x)
            for b in net.blocks:
                h = b(h)
    return h.float()


def _masked_mse(pred, target, mask):
    return (mask * (pred - target) ** 2).sum() / mask.sum().clamp(min=1.0)


def cosine_lr(e, base, floor, total):
    span = max(1, total - 1)
    t = min(max(e - 1, 0), span) / span
    return floor + 0.5 * (base - floor) * (1.0 + math.cos(math.pi * t))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def evaluate(heads, net, planes, targets, masks, device, batch_size, use_amp,
             feats=None):
    """Per-mode masked MSE and R^2 on a fixed row set. R^2 = 1 - MSE/Var(t)
    over that head's DEFINED rows -- 0 means 'no better than predicting the
    mean', the same yardstick train_epoch logs online."""
    modes = list(heads.keys())
    K = len(modes)
    se = np.zeros(K)
    n = np.zeros(K)
    tsum = np.zeros(K)
    tsq = np.zeros(K)
    for h in heads.values():
        h.eval()
    with torch.no_grad():
        for lo in range(0, len(planes), batch_size):
            hi = min(lo + batch_size, len(planes))
            if feats is not None:
                f = feats[lo:hi].to(device).float()
            else:
                x = torch.from_numpy(planes[lo:hi]).to(device)
                f = trunk_features(net, x, use_amp)
            t = torch.from_numpy(targets[lo:hi]).to(device)
            m = torch.from_numpy(masks[lo:hi]).to(device)
            for k, mode in enumerate(modes):
                p = heads[mode](f).squeeze(1)
                w = m[:, k]
                se[k] += (w * (p - t[:, k]) ** 2).sum().item()
                n[k] += w.sum().item()
                tsum[k] += (w * t[:, k]).sum().item()
                tsq[k] += (w * t[:, k] ** 2).sum().item()
    out = {}
    for k, mode in enumerate(modes):
        if n[k] == 0:
            out[mode] = dict(mse=0.0, r2=0.0, var=0.0, n=0)
            continue
        mse = se[k] / n[k]
        mean = tsum[k] / n[k]
        var = max(tsq[k] / n[k] - mean * mean, 0.0)
        r2 = (1.0 - mse / var) if var > 1e-6 else 0.0
        out[mode] = dict(mse=mse, r2=r2, var=var, n=int(n[k]))
    return out


def train_heads(net, planes, targets, masks, modes, args, device):
    """Train one fresh head per mode, simultaneously, on frozen features.
    One trunk forward (or cached-feature fetch) serves every head in the
    batch; head losses are summed into a single backward -- gradients cannot
    interact because the heads are parameter-disjoint and the features carry
    no graph.

    Every head is snapshotted every --snapshot-every epochs (default: every
    epoch) into forgiveness_heads_history.pt -- heads are ~4.4k params
    (~18 KB), so keeping the whole trajectory is essentially free. The file
    is rewritten at each snapshot, so an interrupted run keeps everything
    trained so far. The best-val-R2 snapshot per mode is still tracked
    separately (it is what gets merged into the per-mode full checkpoints).

    Returns (heads, best, history): the live end-of-training modules, the
    per-mode best-snapshot records, and history[mode][epoch] -> CPU
    state_dict."""
    N = len(planes)
    n_val = max(1, int(N * args.val_frac))
    tr = slice(0, N - n_val)             # contiguous tail split: see module
    va = slice(N - n_val, N)             # docstring (game-block leakage)
    print(f"train rows {N - n_val}, val rows {n_val} (contiguous tail split)")

    use_amp = (device.type == "cuda")
    channels = net.stem[0].out_channels

    # optional one-off feature cache: N x C x 8 x 8 fp16 on CPU. For 128
    # channels that is 16 KB/row (~1.6 GB per 100k rows); it removes every
    # trunk forward from the epoch loop.
    feats = None
    if args.cache_features:
        est = N * channels * 64 * 2 / 1e9
        print(f"caching trunk features ({est:.1f} GB fp16, CPU) ...")
        chunks = []
        for lo in range(0, N, args.batch_size):
            x = torch.from_numpy(planes[lo:lo + args.batch_size]).to(device)
            chunks.append(trunk_features(net, x, use_amp).half().cpu())
        feats = torch.cat(chunks)

    heads = {m: ForgivenessHead(channels).to(device) for m in modes}
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

    rng = np.random.default_rng(args.seed)
    best = {m: dict(r2=-np.inf, state=None, epoch=0) for m in modes}
    history = {m: {} for m in modes}     # history[mode][epoch] = state_dict
    hist_path = os.path.join(args.out_dir, "forgiveness_heads_history.pt")

    def _snapshot(head):
        return {k: t.detach().cpu().clone()
                for k, t in head.state_dict().items()}

    csv_path = os.path.join(args.out_dir, "forgiveness_heads_metrics.csv")
    csv_f = open(csv_path, "w")
    csv_f.write("epoch,mode,lr,train_mse,val_mse,val_R2,val_tvar,val_rows\n")

    for epoch in range(1, args.epochs + 1):
        lr = cosine_lr(epoch, args.lr, args.lr_min, args.epochs)
        for g in opt.param_groups:
            g["lr"] = lr

        for h in heads.values():
            h.train()                    # head BN uses batch stats + updates
        order = rng.permutation(np.arange(tr.start, tr.stop))
        tr_se = {m: 0.0 for m in modes}
        tr_n = {m: 0.0 for m in modes}

        for lo in range(0, len(order), args.batch_size):
            idx = order[lo:lo + args.batch_size]
            if feats is not None:
                f = feats[idx].to(device).float()
            else:
                x = torch.from_numpy(planes[idx]).to(device)
                f = trunk_features(net, x, use_amp)
            t = torch.from_numpy(targets[idx]).to(device)
            m = torch.from_numpy(masks[idx]).to(device)

            opt.zero_grad(set_to_none=True)
            loss = 0.0
            for k, mode in enumerate(modes):
                p = heads[mode](f).squeeze(1)
                l = _masked_mse(p, t[:, k], m[:, k])
                loss = loss + l
                tr_se[mode] += l.item() * max(m[:, k].sum().item(), 1.0)
                tr_n[mode] += m[:, k].sum().item()
            loss.backward()
            opt.step()

        val = evaluate(heads, net, planes[va], targets[va], masks[va],
                       device, args.batch_size, use_amp,
                       feats[va] if feats is not None else None)

        line = [f"epoch {epoch:3d}  lr {lr:.1e} "]
        for mode in modes:
            v = val[mode]
            tmse = tr_se[mode] / max(tr_n[mode], 1.0)
            csv_f.write(f"{epoch},{mode},{lr:.2e},{tmse:.6f},"
                        f"{v['mse']:.6f},{v['r2']:.4f},{v['var']:.5f},"
                        f"{v['n']}\n")
            line.append(f"{mode}:R2 {v['r2']:+.3f}")
            if v["r2"] > best[mode]["r2"]:
                best[mode] = dict(
                    r2=v["r2"], epoch=epoch, mse=v["mse"], var=v["var"],
                    state=_snapshot(heads[mode]))

        # ---- store EVERY head this epoch (BN running stats included), and
        # rewrite the history file so a crash loses nothing already trained.
        # val stats ride along so a snapshot can be picked by its R2 later
        # without re-reading the CSV. ----
        if epoch % args.snapshot_every == 0 or epoch == args.epochs:
            for mode in modes:
                history[mode][epoch] = dict(state=_snapshot(heads[mode]),
                                            val_R2=val[mode]["r2"],
                                            val_mse=val[mode]["mse"])
            torch.save({"history": history,
                        "_meta": dict(tau=args.tau, gamma=args.gamma,
                                      modes=modes, seed=args.seed,
                                      epochs=args.epochs,
                                      snapshot_every=args.snapshot_every)},
                       hist_path)

        csv_f.flush()
        print("  ".join(line))

    csv_f.close()
    print(f"metrics -> {csv_path}")
    print(f"per-epoch head snapshots -> {hist_path} "
          f"(history[mode][epoch]['state'])")
    return heads, best, history


# --------------------------------------------------------------------------- #
# reference: the checkpoint's own (online-trained) head, if it has one
# --------------------------------------------------------------------------- #
def eval_checkpoint_head(net, ckpt, planes, targets, masks, modes, args,
                         device):
    """Score the checkpoint's own forgiveness head against every mode's val targets.
    It was trained online for ONE mode (ckpt config forgiveness_target_mode), so its
    R^2 on that column is the like-for-like reference for the offline runs;
    the other columns show how transferable one definition's head is."""
    keys = ["forgiveness_conv.weight", "forgiveness_bn.weight", "forgiveness_bn.bias",
            "forgiveness_bn.running_mean", "forgiveness_bn.running_var",
            "forgiveness_fc1.weight", "forgiveness_fc1.bias",
            "forgiveness_fc2.weight", "forgiveness_fc2.bias"]
    ms = ckpt["model_state"]
    if not all(k in ms for k in keys):
        return None
    channels = net.stem[0].out_channels
    head = ForgivenessHead(channels).to(device)
    head.load_state_dict({k: ms[k] for k in ms
                          if k.startswith("forgiveness_")}, strict=False)
    N = len(planes)
    n_val = max(1, int(N * args.val_frac))
    use_amp = (device.type == "cuda")
    stats = evaluate({m: head for m in modes}, net, planes[N - n_val:],
                     targets[N - n_val:], masks[N - n_val:], device,
                     args.batch_size, use_amp)
    trained_for = ckpt.get("config", {}).get("forgiveness_target_mode", "?")
    print(f"\ncheckpoint's own head (online-trained for {trained_for!r}), "
          f"val R2 per target definition:")
    for m in modes:
        print(f"  vs {m:<13s} R2 {stats[m]['r2']:+.3f}")
    return stats


# --------------------------------------------------------------------------- #
# saving
# --------------------------------------------------------------------------- #
def save_heads(ckpt, ckpt_path, best, heads, modes, args):
    """Per mode: a FULL checkpoint (frozen backbone + the BEST-val-R2 head,
    config updated so downstream tools see the right mode/tau) -- plus one
    compact all-heads file holding both the best and the final-epoch state
    per mode. The full per-epoch trajectory lives in
    forgiveness_heads_history.pt, written incrementally during training."""
    base_state = {k: v.cpu() for k, v in ckpt["model_state"].items()}
    base_cfg = dict(ckpt.get("config", {}))
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]

    for mode in modes:
        b = best[mode]
        if b["state"] is None:
            print(f"  {mode}: no trained state to save (no defined rows?)")
            continue
        state = dict(base_state)
        state.update(b["state"])          # overwrite ONLY the forgiveness_* keys
        cfg = dict(base_cfg)
        cfg.update(forgiveness_target_mode=mode, forgiveness_tau=args.tau,
                   forgiveness_gamma=args.gamma,
                   forgiveness_head_offline=dict(
                       source_checkpoint=os.path.abspath(ckpt_path),
                       val_R2=b["r2"], best_epoch=b["epoch"],
                       games=args.games, seed=args.seed))
        path = os.path.join(args.out_dir, f"{stem}_forgiveness_{mode}.pt")
        torch.save({"iteration": ckpt.get("iteration", 0),
                    "model_state": state, "config": cfg}, path)
        print(f"  {mode:<13s} val R2 {b['r2']:+.3f} (epoch {b['epoch']:3d}) "
              f"-> {path}")

    compact = os.path.join(args.out_dir, "forgiveness_heads_all.pt")
    torch.save({m: dict(state=best[m]["state"], val_R2=best[m]["r2"],
                        best_epoch=best[m]["epoch"],
                        final_state={k: t.detach().cpu().clone()
                                     for k, t in heads[m].state_dict().items()})
                for m in modes if best[m]["state"] is not None}
               | {"_meta": dict(checkpoint=os.path.abspath(ckpt_path),
                                tau=args.tau, gamma=args.gamma,
                                modes=modes)},
               compact)
    print(f"  all heads (best + final)     -> {compact}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate self-play from a frozen checkpoint and train "
                    "one forgiveness head per target mode on the same search trees.")
    p.add_argument("checkpoint", help="path to a net_iterN.pt / latest.pt")
    p.add_argument("--out-dir", default="forgiveness_heads")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES)
    p.add_argument("--dataset", default=None,
                   help=".npz path: loaded if it exists (skipping self-play), "
                        "written after generation otherwise")
    p.add_argument("--regen", action="store_true",
                   help="regenerate even if --dataset exists")

    # target definition -- default tau/gamma come from the checkpoint config
    p.add_argument("--tau", type=float, default=None,
                   help="forgiveness temperature; default: checkpoint config "
                        "forgiveness_tau (probe_forgiveness calibration), else 0.044")
    p.add_argument("--gamma", type=float, default=None,
                   help="tree-mode decay; default: config forgiveness_gamma / 0.85")

    # self-play / search (defaults mirror main.CONFIG where sensible)
    p.add_argument("--games", type=int, default=512)
    p.add_argument("--sims", type=int, default=700)
    p.add_argument("--forgiveness-extra-sims", type=int, default=100)
    p.add_argument("--forgiveness-force-m", type=int, default=6)
    p.add_argument("--root-force-m", type=int, default=6)
    p.add_argument("--root-force-visits", type=int, default=80)
    p.add_argument("--full-search-prob", type=float, default=1.0,
                   help="1.0 default: cheapest sims-per-forgiveness-row (see "
                        "module docstring); set 0.25 to replicate the "
                        "training run's position distribution instead")
    p.add_argument("--fast-sims", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--temp-moves", type=int, default=20)
    p.add_argument("--adj-margin", type=float, default=5.0)
    p.add_argument("--adj-plies", type=int, default=20)
    p.add_argument("--fpu-reduction", type=float, default=0.25)
    p.add_argument("--cache-cap", type=int, default=200_000)

    # head training
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-min", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="store every head's state every N epochs in "
                        "forgiveness_heads_history.pt (last epoch always "
                        "stored); 1 = every epoch")
    p.add_argument("--cache-features", action="store_true",
                   help="precompute trunk features once (fp16, CPU RAM); "
                        "removes all trunk forwards from the epoch loop")
    p.add_argument("--seed", type=int, default=0)

    # misc
    p.add_argument("--channels", type=int, default=None)
    p.add_argument("--num-blocks", type=int, default=None)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    net, ckpt = load_frozen_net(args.checkpoint, device,
                                args.channels, args.num_blocks)
    cfg = ckpt.get("config", {})
    if args.tau is None:
        args.tau = cfg.get("forgiveness_tau", 0.044)
    if args.gamma is None:
        args.gamma = cfg.get("forgiveness_gamma", 0.85)
    print(f"modes: {args.modes}   tau={args.tau}   gamma={args.gamma}")

    # ---- dataset: load if present, otherwise self-play (and save) ----
    if args.dataset and os.path.exists(args.dataset) and not args.regen:
        planes, targets, masks, meta = load_dataset(args.dataset, args.modes)
        if abs(meta.get("tau", args.tau) - args.tau) > 1e-9:
            print(f"  WARNING: dataset was generated with tau="
                  f"{meta.get('tau')}, not {args.tau}; targets follow the "
                  f"dataset's tau. Use --regen to rebuild.")
            args.tau = meta["tau"]
    else:
        torch.backends.cudnn.benchmark = True
        planes, targets, masks = generate_dataset(net, args.modes, args)
        if args.dataset:
            save_dataset(args.dataset, planes, targets, masks,
                         args.modes, args, args.checkpoint)

    # ---- reference: how the checkpoint's own online head scores ----
    eval_checkpoint_head(net, ckpt, planes, targets, masks, args.modes,
                         args, device)

    # ---- train one fresh head per mode ----
    print(f"\ntraining {len(args.modes)} heads for {args.epochs} epochs")
    heads, best, _history = train_heads(net, planes, targets, masks,
                                        args.modes, args, device)

    # ---- save + final table ----
    print("\nbest held-out R2 per target definition:")
    save_heads(ckpt, args.checkpoint, best, heads, args.modes, args)


if __name__ == "__main__":
    main()
"""
Migrate pre-rename checkpoints: ease_* -> forgiveness_*.

The code rename changed ChessNet's head attribute names, so old checkpoints
carry model_state keys ("ease_conv.weight", "ease_bn.running_mean", ...) that
the renamed net no longer has. Because every load site uses strict=False,
loading an unmigrated checkpoint would NOT error -- it would silently leave
the forgiveness head randomly initialised (main.py would print its
"params freshly initialised" note) and drop the head's optimiser state
("ease_optim_state" no longer looked up). This script rewrites the keys in
place so nothing is lost:

  * model_state keys containing "ease_"      -> "forgiveness_"
    (handles a defensive "_orig_mod." prefix too, though checkpoints are
    saved from the unwrapped module);
  * the top-level "ease_optim_state" entry   -> "forgiveness_optim_state";
  * config keys starting with "ease_"        -> "forgiveness_" prefix
    (ease_tau, ease_target_mode, ease_gamma, ease_extra_sims, ease_force_m,
    ease_lr, ease_lr_min, ease_loss_weight, ease_targets, ...).

Adam's internal optimiser state is keyed by parameter INDEX, not name, so the
param-group split ("forgiveness_" in name) still lines up after migration --
the head's momentum/variance state survives intact.

Idempotent: a checkpoint with no ease_* keys is reported and left untouched,
so re-running over a mixed directory is safe.

Usage:
    python migrate_checkpoints.py checkpoints/*.pt          # in place, .bak kept
    python migrate_checkpoints.py old.pt --out new.pt       # write elsewhere
    python migrate_checkpoints.py checkpoints/*.pt --no-backup

Note the metrics CSV needs no migration: main.open_metrics detects the new
header (loss_forgiveness / forgiveness_R2 columns) and diverts new rows to a
fresh *_v2.csv automatically.
"""

import argparse
import os
import shutil


def _rename_key(k: str) -> str:
    return k.replace("ease_", "forgiveness_") if "ease_" in k else k


def migrate_ckpt_dict(ckpt):
    """Pure key rewrite on a loaded checkpoint dict. Returns (ckpt, changes),
    where changes is a list of 'old -> new' strings (empty = already
    migrated). Mutates and returns the same dict."""
    changes = []

    ms = ckpt.get("model_state")
    if isinstance(ms, dict):
        for k in list(ms.keys()):
            nk = _rename_key(k)
            if nk != k:
                ms[nk] = ms.pop(k)
                changes.append(f"model_state: {k} -> {nk}")

    if "ease_optim_state" in ckpt:
        ckpt["forgiveness_optim_state"] = ckpt.pop("ease_optim_state")
        changes.append("ease_optim_state -> forgiveness_optim_state")

    cfg = ckpt.get("config")
    if isinstance(cfg, dict):
        for k in list(cfg.keys()):
            if isinstance(k, str) and k.startswith("ease_"):
                nk = "forgiveness_" + k[len("ease_"):]
                cfg[nk] = cfg.pop(k)
                changes.append(f"config: {k} -> {nk}")

    return ckpt, changes


def main():
    ap = argparse.ArgumentParser(
        description="Rename ease_* keys to forgiveness_* inside checkpoints.")
    ap.add_argument("paths", nargs="+", help=".pt checkpoint files")
    ap.add_argument("--out", default=None,
                    help="output path (single input only); default: in place")
    ap.add_argument("--no-backup", action="store_true",
                    help="skip the .bak copy when migrating in place")
    args = ap.parse_args()

    if args.out and len(args.paths) != 1:
        ap.error("--out only makes sense with a single input path")

    import torch  # local import: the key logic above is torch-free/testable

    for path in args.paths:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict):
            print(f"{path}: not a checkpoint dict, skipped")
            continue
        ckpt, changes = migrate_ckpt_dict(ckpt)
        if not changes:
            print(f"{path}: already migrated (no ease_* keys), untouched")
            continue

        dest = args.out or path
        if dest == path and not args.no_backup:
            shutil.copy2(path, path + ".bak")
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        torch.save(ckpt, dest)
        head = [c for c in changes if c.startswith("model_state")]
        print(f"{path} -> {dest}: {len(changes)} keys renamed "
              f"({len(head)} model_state, "
              f"backup {'kept' if dest == path and not args.no_backup else 'n/a'})")


if __name__ == "__main__":
    main()
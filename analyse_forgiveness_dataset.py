"""
Controls for the forgiveness-head R2 numbers, saved to disk for the write-up.

WHY
---
A head that scores R2 0.78 has learned something -- but "something" has to be
measured against what a trivial predictor already gets, or the number means
nothing on its own. This script builds an escalating ladder of cheap baselines
from features that are read straight off the encoded planes, so every claim
about the head is stated as a MARGIN over a stated control rather than as a
bare number.

The ladder (each nests inside the next, so R2 is monotone along it):

  1. piece_count        total pieces on the board -- game phase, one number
  2. + ply proxy        halfmove clock (plane 17), a second phase signal
  3. + material         mover's material minus opponent's, standard values
  4. + mobility proxy   per-side piece counts by type (12 features)
  5. all of the above   the strongest cheap control

Every one of these is available WITHOUT a network, without search, and without
training -- so whatever margin the head holds over (5) is the part that
genuinely required learning a positional concept.

WHAT IT WRITES
--------------
  <out>/baselines.csv      one row per (mode, baseline): R2, adj R2, n
  <out>/mode_corr.csv      the mode-by-mode correlation matrix
  <out>/mode_stats.csv     per-mode n, mean, sd, and the piece-count R2
  <out>/summary.md         a table you can paste into the dissertation

Adjusted R2 is reported alongside raw R2 because baseline (4) uses 12
features: with 60k rows the inflation is negligible, but stating it forecloses
the obvious objection.

USAGE
    python analyse_forgiveness_dataset.py datasets/iter11275_tau0178.npz \\
        --heads-r2 gap=0.485,entropy=0.602,tree_gap=0.762,tree_entropy=0.767,\\
flat_gap=0.770,flat_entropy=0.778,tree_gap_me=0.747,tree_gap_opp=0.753,\\
flat_entropy_me=0.737,flat_entropy_opp=0.746 \\
        --out analysis/
"""

import argparse
import csv
import os

import numpy as np

# Encoded plane layout (model/encoding.py):
#   0-5   mover's pieces   P N B R Q K
#   6-11  opponent's       P N B R Q K
#   12-15 castling rights (uniform planes)
#   16    en-passant square
#   17    halfmove clock / 100   (uniform)
#   18    repetition count / 2   (uniform)
PIECE_VALUES = np.array([1, 3, 3, 5, 9, 0], dtype=np.float64)


def features(planes):
    """Cheap, search-free, network-free features straight off the planes."""
    per_type_us = planes[:, 0:6].sum(axis=(2, 3))          # (N, 6)
    per_type_them = planes[:, 6:12].sum(axis=(2, 3))       # (N, 6)
    piece_count = per_type_us.sum(1) + per_type_them.sum(1)
    halfmove = planes[:, 17, 0, 0]                          # uniform plane
    material = per_type_us @ PIECE_VALUES - per_type_them @ PIECE_VALUES
    return {
        "piece_count": piece_count[:, None],
        "halfmove": halfmove[:, None],
        "material": material[:, None],
        "per_type": np.concatenate([per_type_us, per_type_them], axis=1),
    }


def ols_r2(X, y):
    """R2 and adjusted R2 of an ordinary least-squares fit with an intercept.

    Uses lstsq rather than a normal-equation inverse: the per-type columns are
    near-collinear (they sum to piece_count), which makes X'X ill-conditioned.
    lstsq handles the rank deficiency instead of returning nonsense.
    """
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    ss_res = float((resid ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, p = len(y), X.shape[1] - 1
    adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else float("nan")
    return r2, adj, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--out", default="analysis")
    ap.add_argument("--heads-dir", default="forgiveness_heads",
                    help="directory written by train_forgiveness_heads.py. Its "
                         "forgiveness_heads_metrics.csv is read automatically "
                         "for each mode's BEST val_R2, so the margin column is "
                         "filled in without retyping ten numbers.")
    ap.add_argument("--heads-r2", default="",
                    help="override: comma-separated mode=R2. Only needed if the "
                         "metrics CSV is missing or you want a specific epoch.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    d = np.load(args.dataset, allow_pickle=True)
    planes, targets, masks = d["planes"], d["targets"], d["masks"]
    modes = [str(m) for m in d["modes"]]
    print(f"{args.dataset}: {len(planes):,} rows, {len(modes)} modes")

    # ---- head R2: read from the training run's own CSV --------------------
    # train_forgiveness_heads.py writes one row per (epoch, mode) with a
    # val_R2 column. Taking the max per mode reproduces exactly the "best
    # held-out R2 per target definition" summary it prints at the end, so the
    # margin column cannot drift from what the training run actually reported.
    heads, heads_src = {}, None
    csv_path = os.path.join(args.heads_dir, "forgiveness_heads_metrics.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                try:
                    r2 = float(row["val_R2"])
                except (KeyError, ValueError):
                    continue
                m = row["mode"]
                if m not in heads or r2 > heads[m]:
                    heads[m] = r2
        heads_src = csv_path
        print(f"head R2 read from {csv_path} ({len(heads)} modes)")
    elif not args.heads_r2:
        print(f"[warn] {csv_path} not found and --heads-r2 not given; "
              f"the margin column will be blank.")

    # explicit values win over the CSV
    for part in args.heads_r2.split(","):
        if "=" in part:
            k, v = part.split("=")
            heads[k.strip()] = float(v)
            heads_src = "--heads-r2"

    F = features(planes.astype(np.float64))

    LADDER = [
        ("piece_count", ["piece_count"]),
        ("+halfmove", ["piece_count", "halfmove"]),
        ("+material", ["piece_count", "halfmove", "material"]),
        ("+per_type", ["piece_count", "halfmove", "material", "per_type"]),
    ]

    rows = []
    for i, mode in enumerate(modes):
        k = masks[:, i] > 0
        y = targets[k, i].astype(np.float64)
        for name, keys in LADDER:
            X = np.concatenate([F[key][k] for key in keys], axis=1)
            r2, adj, p = ols_r2(X, y)
            rows.append(dict(mode=mode, baseline=name, n_features=p,
                             n_rows=int(k.sum()), r2=round(r2, 4),
                             adj_r2=round(adj, 4)))
            print(f"  {mode:>18} {name:>14} ({p:2d} feat)  R2={r2:6.3f}")

    with open(os.path.join(args.out, "baselines.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # ---- mode correlation matrix ----
    with open(os.path.join(args.out, "mode_corr.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + modes)
        for i, a in enumerate(modes):
            line = [a]
            for j in range(len(modes)):
                kk = (masks[:, i] > 0) & (masks[:, j] > 0)
                r = (np.corrcoef(targets[kk, i], targets[kk, j])[0, 1]
                     if kk.sum() > 10 else float("nan"))
                line.append(round(float(r), 4))
            w.writerow(line)

    # ---- per-mode stats ----
    with open(os.path.join(args.out, "mode_stats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "n", "mean", "sd", "piece_count_r2",
                    "best_baseline_r2", "head_r2", "margin"])
        for i, mode in enumerate(modes):
            k = masks[:, i] > 0
            y = targets[k, i]
            pc = next(r for r in rows
                      if r["mode"] == mode and r["baseline"] == "piece_count")
            best = max(r["r2"] for r in rows if r["mode"] == mode)
            h = heads.get(mode)
            w.writerow([mode, int(k.sum()), round(float(y.mean()), 4),
                        round(float(y.std()), 4), pc["r2"], round(best, 4),
                        "" if h is None else h,
                        "" if h is None else round(h - best, 4)])

    # ---- markdown summary ----
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(f"# Forgiveness target analysis\n\n")
        f.write(f"Dataset: `{args.dataset}` — {len(planes):,} positions.\n")
        if heads_src:
            f.write(f"Head R² source: `{heads_src}` (best epoch per mode).\n")
        f.write("\n")
        f.write("## Trivial-feature controls\n\n")
        f.write("R² of ordinary least squares on features read directly off the "
                "encoded planes — no network, no search, no training. The head's "
                "margin over the strongest control is the part that required "
                "learning a positional concept.\n\n")
        f.write("| mode | piece count | +halfmove | +material | +per-type | head | margin |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for mode in modes:
            r = {x["baseline"]: x["r2"] for x in rows if x["mode"] == mode}
            best = max(r.values())
            h = heads.get(mode)
            hs = f"{h:.3f}" if h is not None else "—"
            ms = f"**{h - best:+.3f}**" if h is not None else "—"
            f.write(f"| {mode} | {r['piece_count']:.3f} | {r['+halfmove']:.3f} | "
                    f"{r['+material']:.3f} | {r['+per_type']:.3f} | {hs} | {ms} |\n")
        f.write("\n## Mode redundancy\n\n")
        f.write("Pearson correlation between target definitions on rows where "
                "both are defined. Pairs above ~0.95 are effectively one target "
                "under two names.\n\n")
        f.write("| | " + " | ".join(m[:12] for m in modes) + " |\n")
        f.write("|---" * (len(modes) + 1) + "|\n")
        for i, a in enumerate(modes):
            line = [a]
            for j in range(len(modes)):
                kk = (masks[:, i] > 0) & (masks[:, j] > 0)
                r = np.corrcoef(targets[kk, i], targets[kk, j])[0, 1]
                line.append(f"{r:.3f}")
            f.write("| " + " | ".join(line) + " |\n")

    print(f"\nwrote {args.out}/baselines.csv, mode_corr.csv, mode_stats.csv, "
          f"summary.md")


if __name__ == "__main__":
    main()

    
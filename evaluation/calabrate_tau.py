"""
Recover / recompute the forgiveness temperature tau from a
forgiveness_probe.csv (written by probe_forgiveness.py).

The calibration rule (same as probe_forgiveness.py's auto mode):

    F_gap = exp(-gap / tau)  should map the MEDIAN gap to F = 0.5
    =>  tau = median(gap) / ln 2

One tau serves both local statistics: inside the entropy's softmax(Q / tau)
it plays the identical "how much Q-cost still counts as near-optimal" role.
As a cross-check this script also reports the tau at which the median
forgiveness_entropy would land exactly on 0.5 -- recomputed from (q1, q2) pairs, the
2-move entropy H(softmax([q1,q2]/tau)) / ln 2, which is a lower-bound proxy
since the CSV doesn't store the full Q vectors. If the two taus are within a
factor of ~1.5 of each other, keep the shared gap-derived tau.

It also verifies the CSV's own consistency (does exp(-gap/tau_rec) reproduce
the stored F_gap median?) and prints the implied F at the gap percentiles so
you can see the target contrast a given tau produces.

Usage:
    python calibrate_tau.py forgiveness_probe.csv
    python calibrate_tau.py forgiveness_probe.csv --tau 0.05  # audit a given tau
"""

import argparse
import csv
import math

import numpy as np

LN2 = math.log(2.0)


def load_rows(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{path}: empty CSV")
    return rows


def real_choice_rows(rows):
    """Rows with a genuine second move: q2 present and gap not pinned to the
    single-move sentinel (2.0)."""
    out = []
    for r in rows:
        if r.get("q2", "") == "":
            continue
        g = float(r["gap"])
        if g >= 2.0:
            continue
        out.append(r)
    return out


def two_move_entropy(q1, q2, tau):
    """Normalised entropy of softmax([q1, q2] / tau) -- H / ln 2 in [0, 1]."""
    d = (q1 - q2) / tau
    # p = [1/(1+e^-d), e^-d/(1+e^-d)]; stable form:
    p1 = 1.0 / (1.0 + math.exp(-abs(d)))
    p2 = 1.0 - p1
    h = 0.0
    for p in (p1, p2):
        if p > 1e-12:
            h -= p * math.log(p)
    return h / LN2


def median_entropy_at_tau(pairs, tau):
    return float(np.median([two_move_entropy(q1, q2, tau) for q1, q2 in pairs]))


def solve_entropy_tau(pairs, lo=1e-4, hi=10.0, iters=80):
    """Bisect for the tau at which the median 2-move entropy hits 0.5.
    Median entropy is monotonically increasing in tau."""
    if median_entropy_at_tau(pairs, hi) < 0.5:
        return None
    if median_entropy_at_tau(pairs, lo) > 0.5:
        return None
    for _ in range(iters):
        mid = math.sqrt(lo * hi)          # geometric: tau spans decades
        if median_entropy_at_tau(pairs, mid) < 0.5:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def pct(xs, ps=(10, 25, 50, 75, 90)):
    return {p: float(np.percentile(xs, p)) for p in ps}


def main():
    ap = argparse.ArgumentParser(description="tau calibration from a probe CSV")
    ap.add_argument("csv",
                    help="forgiveness_probe.csv written by probe_forgiveness.py")
    ap.add_argument("--tau", type=float, default=0.0,
                    help="also audit this specific tau (e.g. the 0.05 "
                         "placeholder your training run is using)")
    args = ap.parse_args()

    rows = load_rows(args.csv)
    usable = real_choice_rows(rows)
    print(f"{len(rows)} rows, {len(usable)} with a genuine second move\n")

    gaps = np.asarray([float(r["gap"]) for r in usable])
    med_gap = float(np.median(gaps))
    tau_gap = med_gap / LN2
    print("gap percentiles: " +
          "  ".join(f"p{p}={v:.4f}" for p, v in pct(gaps).items()))
    print(f"\ntau (median gap / ln 2)      = {tau_gap:.4f}"
          f"   <-- freeze this as forgiveness_tau for the next training run")

    # entropy cross-check from (q1, q2) pairs
    pairs = [(float(r["q1"]), float(r["q2"])) for r in usable]
    tau_ent = solve_entropy_tau(pairs)
    if tau_ent is not None:
        ratio = tau_ent / tau_gap
        note = ("close to the gap tau -> share one tau" if 1/1.5 <= ratio <= 1.5
                else "NB the 2-move proxy underestimates the true entropy tau "
                     "(real roots have n>2 qualified moves, which raises H at "
                     "fixed tau) -- trust the F/entropy medians of the probe "
                     "printout over this bound")
        print(f"tau (median 2-move entropy=.5) = {tau_ent:.4f}"
              f"   ({ratio:.2f}x the gap tau; {note})")
    else:
        print("entropy-tau bisection did not bracket 0.5 -- data degenerate?")

    # consistency: does the recovered tau reproduce the CSV's stored F_gap?
    if "F_gap" in rows[0]:
        f_stored = float(np.median([float(r["F_gap"]) for r in usable]))
        f_implied = math.exp(-med_gap / tau_gap)     # 0.5 by construction
        print(f"\nstored F_gap median {f_stored:.3f} vs implied {f_implied:.3f} "
              f"(match ~= this CSV was produced WITH the calibrated tau; "
              f"mismatch ~= it used --tau or a stale value)")

    # contrast audit: implied F at the gap percentiles, for candidate taus
    cands = [("calibrated", tau_gap)]
    if args.tau > 0:
        cands.append(("--tau given", args.tau))
    print("\nimplied F_gap at the gap percentiles:")
    header = "  ".join(f"{'p'+str(p):>7}" for p in (10, 25, 50, 75, 90))
    print(f"  {'tau':>12}  {header}")
    for name, t in cands:
        fs = [math.exp(-v / t) for v in pct(gaps).values()]
        print(f"  {name:>12}  " + "  ".join(f"{f:7.3f}" for f in fs))
        # note: percentile p of gap maps to percentile (100 - p) of F


if __name__ == "__main__":
    main()

    
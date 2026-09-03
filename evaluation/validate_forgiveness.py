"""Validity test for the forgiveness statistic.

The forgiveness heads are already known to be LEARNABLE (Section 7.1: R^2 well
above trivial plane features). That is a different question from whether F
measures what it claims to measure. This script tests the claim itself:

    when the intended action is not executed, is less lost in a high-F state
    than in a brittle one?

WHAT NOT TO TEST
----------------
Correlating F(s) against the Q-spread of the same root is near-tautological.
F is the normalised entropy of softmax(q/tau); the mean deviation of the
non-best actions from q1 is another summary of the same q vector. They must
correlate, and that correlation is evidence of nothing. This script therefore
uses the GAME OUTCOME following the decision as the label.

THE CONTROL THAT MAKES IT A TEST
--------------------------------
A positive slope of outcome on F among perturbed decisions is not sufficient.
F could simply be higher in positions that are winning anyway, in which case
the slope appears with or without a mistake. So the slope is fitted twice:

    beta_mistake : decisions where the injected error FIRED
    beta_clean   : decisions in the same games where it did not

The forgiveness hypothesis predicts beta_mistake > beta_clean. That difference,
not either slope alone, is the result. If they are equal, F is measuring
position quality rather than tolerance to error -- which is itself a finding,
and a cleaner explanation of the null Elo results than "the steering was too
weak".

Standard errors are bootstrapped with resampling CLUSTERED BY GAME, because
decisions within a game share one outcome and are not independent.

Input:
    --decisions  the .npz from robustness_arena.py --dataset-out
    --games      the .csv from robustness_arena.py --per-game-out
Joined on (pidx/game, player).

Usage:
    python validate_forgiveness.py --decisions dec.npz --games games.csv
    python validate_forgiveness.py --decisions dec.npz --games games.csv \
        --eps 0.10 --stat f_entropy --boot 5000
"""

import argparse
import json

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_decisions(path):
    z = np.load(path, allow_pickle=True)
    meta = json.loads(str(z["meta"])) if "meta" in z else {}
    cols = {k: z[k] for k in z.files if k != "meta"}
    df = pd.DataFrame(cols)
    # every column is stored as float32 or as str; the join keys must be str
    for k in ("pidx", "player"):
        if k in df:
            df[k] = df[k].astype(str)
    return df, meta


def load_games(path):
    g = pd.read_csv(path)
    g["game"] = g["game"].astype(str)
    g["player"] = g["player"].astype(str)
    return g


def join(dec, games, eps=None):
    """One row per decision, carrying that decision's state statistics and the
    eventual score of the game it occurred in (from the deciding player's
    point of view, which is what `score` already is in the per-game file)."""
    g = games.copy()
    if eps is not None:
        g = g[np.isclose(g["eps"], eps)]
        if g.empty:
            raise SystemExit(f"no per-game rows at eps={eps}; "
                             f"available: {sorted(games['eps'].unique())}")
    keep = ["game", "player", "score", "eps", "plies"]
    keep += [c for c in ("selector", "head_mode", "pairing") if c in g]
    g = g[keep].rename(columns={"game": "pidx"})
    df = dec.merge(g, on=["pidx", "player"], how="inner")
    if df.empty:
        raise SystemExit(
            "join produced no rows. The decision file's `pidx` and the "
            "per-game file's `game` must come from the SAME run -- rerun the "
            "arena with --dataset-out and --per-game-out together.")
    return df


# --------------------------------------------------------------------------- #
# estimation
# --------------------------------------------------------------------------- #
def ols_slope(x, y, controls=None):
    """Slope of y on x, optionally adjusting for `controls` (n, k). Returns nan
    for a degenerate design rather than raising, so a bootstrap resample that
    happens to be constant does not kill the run."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cols = [np.ones_like(x), x]
    if controls is not None and len(controls):
        C = np.asarray(controls, dtype=np.float64)
        C = C.reshape(len(x), -1)
        cols += [C[:, j] for j in range(C.shape[1])]
    X = np.column_stack(cols)
    try:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    return float(beta[1])


def cluster_bootstrap(df, stat_col, controls, n_boot, rng):
    """Resample GAMES with replacement; refit both slopes on each resample.

    Clustering matters here: a 400-game run at eps=0.10 yields tens of
    thousands of decisions but only 400 independent outcomes. Treating
    decisions as independent would shrink the interval by roughly an order of
    magnitude and manufacture significance."""
    games = df["pidx"].unique()
    idx_by_game = {gp: np.flatnonzero(df["pidx"].values == gp) for gp in games}
    out = []
    for _ in range(n_boot):
        pick = rng.choice(games, size=len(games), replace=True)
        rows = np.concatenate([idx_by_game[gp] for gp in pick])
        d = df.iloc[rows]
        out.append(_both_slopes(d, stat_col, controls))
    a = np.asarray(out, dtype=np.float64)
    return a


def _both_slopes(d, stat_col, controls):
    m = d[d["mistake"] == 1]
    c = d[d["mistake"] == 0]
    bm = bc = np.nan
    if len(m) > len(controls) + 3:
        bm = ols_slope(m[stat_col], m["score"], m[controls] if controls else None)
    if len(c) > len(controls) + 3:
        bc = ols_slope(c[stat_col], c["score"], c[controls] if controls else None)
    return bm, bc, bm - bc


def ci(a, lo=2.5, hi=97.5):
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def bin_table(df, stat_col, nbins=4):
    """Mean outcome by F quantile, split on whether the error fired. The
    regression is the test; this is the picture."""
    d = df.copy()
    try:
        d["bin"] = pd.qcut(d[stat_col], nbins, labels=False, duplicates="drop")
    except ValueError:
        return None
    rows = []
    for b, grp in d.groupby("bin"):
        m = grp[grp["mistake"] == 1]
        c = grp[grp["mistake"] == 0]
        rows.append(dict(
            bin=int(b),
            F_lo=round(float(grp[stat_col].min()), 4),
            F_hi=round(float(grp[stat_col].max()), 4),
            n_mistake=len(m),
            score_mistake=round(float(m["score"].mean()), 4) if len(m) else np.nan,
            n_clean=len(c),
            score_clean=round(float(c["score"].mean()), 4) if len(c) else np.nan,
        ))
    t = pd.DataFrame(rows)
    t["difference"] = (t["score_mistake"] - t["score_clean"]).round(4)
    return t


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decisions", required=True,
                   help=".npz from robustness_arena.py --dataset-out")
    p.add_argument("--games", required=True,
                   help=".csv from robustness_arena.py --per-game-out")
    p.add_argument("--eps", type=float, default=None,
                   help="restrict to one noise level (recommended: pick the "
                        "level with a decent mistake count, e.g. 0.10)")
    p.add_argument("--stat", default="f_entropy",
                   choices=["f_entropy", "f_gap", "eff_actions", "gap"],
                   help="which state statistic to test")
    p.add_argument("--player", default=None,
                   help="restrict to one arm (A / B), e.g. to test the "
                        "greedy arm only")
    p.add_argument("--controls", default="ply,q1",
                   help="comma-separated columns to adjust for, or '' for "
                        "none. q1 matters: it is the position's value, and "
                        "without it a slope on F may just be reading that "
                        "good positions are flat")
    p.add_argument("--bins", type=int, default=4)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the joined rows to CSV")
    args = p.parse_args()

    dec, meta = load_decisions(args.decisions)
    if meta:
        print("run meta: " + "  ".join(
            f"{k}={meta[k]}" for k in
            ("select_a", "select_b", "delta", "mode", "sims", "root",
             "stat", "agg", "parity") if k in meta))
    games = load_games(args.games)
    df = join(dec, games, args.eps)
    if args.player:
        df = df[df["player"] == args.player]

    controls = [c for c in args.controls.split(",") if c and c in df.columns]
    df = df.dropna(subset=[args.stat, "score", "mistake"] + controls)

    n_m = int((df["mistake"] == 1).sum())
    n_c = int((df["mistake"] == 0).sum())
    print(f"\n{len(df)} decisions over {df['pidx'].nunique()} games "
          f"({n_m} perturbed, {n_c} clean), statistic = {args.stat}, "
          f"controls = {controls or 'none'}")
    if n_m < 200:
        print("WARNING: few perturbed decisions. Raise --games or --eps; "
              "below a few hundred the interval will not exclude anything.")

    t = bin_table(df, args.stat, args.bins)
    if t is not None:
        print(f"\nMean game score by {args.stat} quantile:")
        print(t.to_string(index=False))

    bm, bc, diff = _both_slopes(df, args.stat, controls)
    boot = cluster_bootstrap(df, args.stat, controls, args.boot,
                             np.random.default_rng(args.seed))
    lo_m, hi_m = ci(boot[:, 0])
    lo_c, hi_c = ci(boot[:, 1])
    lo_d, hi_d = ci(boot[:, 2])

    print(f"\nSlope of game score on {args.stat}, "
          f"{args.boot} game-clustered bootstrap resamples:")
    print(f"  perturbed decisions   beta = {bm:+.4f}  [{lo_m:+.4f}, {hi_m:+.4f}]")
    print(f"  clean decisions       beta = {bc:+.4f}  [{lo_c:+.4f}, {hi_c:+.4f}]")
    print(f"  DIFFERENCE            beta = {diff:+.4f}  [{lo_d:+.4f}, {hi_d:+.4f}]")

    print("\nReading:")
    if not np.isfinite(diff):
        print("  undetermined -- not enough data in one of the two groups.")
    elif lo_d > 0:
        print("  The difference excludes zero and is positive: a forced error")
        print("  costs less in a high-F state than in a brittle one, over and")
        print("  above F's association with outcome in unperturbed play.")
        print("  This is the validity claim, and it holds.")
    elif hi_d < 0:
        print("  The difference excludes zero and is NEGATIVE: forced errors")
        print("  cost MORE in high-F states. The statistic is anti-predictive")
        print("  of tolerance, which would explain the Elo results directly.")
    else:
        print("  The difference does not exclude zero. F does not measurably")
        print("  predict tolerance to a forced error beyond its association")
        print("  with position quality. Report this: it is the cleanest")
        print("  available explanation of the null steering results, and it")
        print("  is a stronger claim than 'the steering was too weak'.")
        if np.isfinite(bc) and (lo_c > 0 or hi_c < 0):
            print("  Note the clean slope alone IS non-zero, i.e. F does track")
            print("  position quality -- which is exactly the confound the")
            print("  difference is designed to remove.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\njoined rows -> {args.out}")


if __name__ == "__main__":
    main()

    
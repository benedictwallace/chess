"""
Read metrics.csv and report the quantities that actually indicate progress.

    python watch_metrics.py checkpoints_sh/metrics.csv
    python watch_metrics.py checkpoints_sh/metrics.csv --window 50

Handles both schemas (main.py and main_multigpu.py). Nothing here is a
substitute for an Elo ladder -- these metrics tell you whether the run is
HEALTHY, not whether it is getting STRONGER. Only games against other
checkpoints tell you that.
"""
import argparse
import csv
import math
import sys


def load(path):
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)]
    if not rows:
        sys.exit("metrics file is empty")
    return rows


def col(rows, name):
    """Column as floats, None where absent/unparseable."""
    if name not in rows[0]:
        return None
    out = []
    for r in rows:
        try:
            out.append(float(r[name]))
        except (ValueError, TypeError):
            out.append(None)
    return out


def clean(xs):
    return [x for x in xs if x is not None and not math.isnan(x)]


def trend(xs, window):
    """(early mean, late mean, delta) over the last 2*window rows."""
    xs = clean(xs)
    if len(xs) < 4:
        return None
    w = min(window, len(xs) // 2)
    early = sum(xs[-2 * w:-w]) / w
    late = sum(xs[-w:]) / w
    return early, late, late - early


def arrow(delta, good_direction):
    if delta is None:
        return "?"
    if abs(delta) < 1e-9:
        return "flat"
    rising = delta > 0
    ok = (rising and good_direction > 0) or (not rising and good_direction < 0)
    return ("UP  " if rising else "DOWN") + ("  ok" if ok else "  <-- WATCH")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--window", type=int, default=25,
                    help="rows per half when comparing early vs late")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--train-batches", type=int, default=64)
    a = ap.parse_args()

    rows = load(a.path)
    n = len(rows)
    step_col = "train_step" if "train_step" in rows[0] else "iteration"
    runner = "main_multigpu" if "train_step" in rows[0] else "main"
    print(f"\n{a.path}  ({runner} schema, {n} rows, "
          f"last {step_col}={rows[-1].get(step_col)})")
    print("=" * 72)

    # ---------------------------------------------------------------- #
    # 1. THE FIRST THING TO CHECK ON A SEQUENTIAL-HALVING RUN
    # ---------------------------------------------------------------- #
    te = col(rows, "target_entropy")
    print("\n1. TARGET ENTROPY  -- is the pi' target informative?")
    if te is None or not clean(te):
        print("   column missing")
    else:
        t = clean(te)
        recent = sum(t[-min(10, len(t)):]) / min(10, len(t))
        print(f"   first={t[0]:.3f}  recent={recent:.3f} nats")
        if recent < 0.5:
            print("   *** TARGET HAS COLLAPSED TO NEAR ONE-HOT ***")
            print("   sh_c_scale is too high for this sim budget. The target")
            print("   carries no information about non-best moves. Halve")
            print("   sh_c_scale and restart -- this will not recover on its own.")
        elif recent > 3.2:
            print("   Target is very flat -- sh_c_scale may be too low, so pi'")
            print("   is barely improving on the prior. Try raising it.")
        else:
            print("   OK. Healthy range is roughly 1.2-2.6 nats; visit-count")
            print("   targets on this codebase sat near 1.7.")

    # ---------------------------------------------------------------- #
    # 2. THE ONLY LEARNABLE PART OF THE POLICY LOSS
    # ---------------------------------------------------------------- #
    kl = col(rows, "policy_kl")
    lp = col(rows, "loss_policy")
    print("\n2. POLICY  -- CE = target_entropy + policy_kl; only KL is learnable")
    if kl is None or not clean(kl):
        print("   policy_kl column missing")
    else:
        tr = trend(kl, a.window)
        k = clean(kl)
        print(f"   policy_kl   first={k[0]:.4f}  last={k[-1]:.4f}")
        if tr:
            print(f"   last {a.window}x2 rows: {tr[0]:.4f} -> {tr[1]:.4f}   "
                  f"{arrow(tr[2], -1)}")
        if lp and clean(lp) and te and clean(te):
            print(f"   (loss_policy={clean(lp)[-1]:.4f} is mostly the "
                  f"{clean(te)[-1]:.3f} entropy floor -- do not read it as progress)")

    # ---------------------------------------------------------------- #
    # 3. VALUE OVERFIT
    # ---------------------------------------------------------------- #
    lv = col(rows, "loss_value")
    print("\n3. VALUE  -- rising while policy_kl falls is the overfit signature")
    if lv is None or not clean(lv):
        print("   column missing")
    else:
        tr = trend(lv, a.window)
        v = clean(lv)
        print(f"   loss_value  first={v[0]:.4f}  last={v[-1]:.4f}")
        if tr:
            print(f"   last {a.window}x2 rows: {tr[0]:.4f} -> {tr[1]:.4f}   "
                  f"{arrow(tr[2], -1)}")
            ktr = trend(kl, a.window) if kl else None
            if ktr and tr[2] > 0.002 and ktr[2] < 0:
                print("   *** value rising while policy improving = OVERFIT ***")
                print("   Raise games_per_iter, or cut train_batches / "
                      "--target-ratio.")

    # ---------------------------------------------------------------- #
    # 4. DATA RATE AND REPLAY
    # ---------------------------------------------------------------- #
    print("\n4. DATA  -- is the learner outrunning self-play?")
    rr = col(rows, "replay_ratio")
    if rr and clean(rr):
        r = clean(rr)
        print(f"   replay_ratio  first={r[0]:.2f}  last={r[-1]:.2f}")
        if r[-1] > 8:
            print("   *** above 8: each position is being trained on many times."
                  "\n   This codebase's notes record value loss creeping at ~8."
                  "\n   Lower --target-ratio (default is 12).")
        else:
            print("   OK (aim for roughly 4-6).")
    else:
        buf = col(rows, "buffer_size") or col(rows, "buffer")
        if buf and clean(buf):
            b = clean(buf)
            consumed = a.batch_size * a.train_batches
            grew = [b[i] - b[i - 1] for i in range(1, len(b)) if b[i] >= b[i - 1]]
            recent_growth = (sum(grew[-a.window:]) / min(len(grew), a.window)
                             if grew else 0.0)
            print(f"   buffer  first={b[0]:.0f}  last={b[-1]:.0f}")
            print(f"   consumed/iter = {consumed} "
                  f"(batch_size {a.batch_size} x train_batches {a.train_batches})")
            if recent_growth > 1:
                print(f"   new rows/iter ~ {recent_growth:.0f} while the buffer "
                      f"fills -> replay ~ {consumed / recent_growth:.1f}x")
            else:
                print("   buffer is saturated; rows/iter not derivable from it.")
                print("   Compare selfplay_sec against your old run instead.")

    sp = col(rows, "selfplay_sec")
    tt = col(rows, "train_sec")
    if sp and clean(sp):
        s = clean(sp)
        recent = sum(s[-min(10, len(s)):]) / min(10, len(s))
        print(f"   selfplay_sec recent mean = {recent:.1f}s", end="")
        if tt and clean(tt):
            t2 = clean(tt)
            print(f"   train_sec = {sum(t2[-min(10,len(t2)):])/min(10,len(t2)):.1f}s")
        else:
            print()

    # ---------------------------------------------------------------- #
    # 5. FORGIVENESS COLUMNS
    # ---------------------------------------------------------------- #
    r2 = col(rows, "forgiveness_R2")
    if r2 and clean(r2):
        rv = clean(r2)
        tv = clean(col(rows, "forgiveness_tvar") or [])
        print("\n5. FORGIVENESS HEAD")
        if tv and max(tv) < 1e-8:
            print("   target variance is 0 -> forgiveness_targets is OFF. "
                  "R2 is meaningless; ignore these columns.")
        else:
            print(f"   forgiveness_R2 last={rv[-1]:+.3f}  "
                  f"(0 = no better than the batch mean)")

    print("\n" + "=" * 72)
    print("None of the above measures STRENGTH. For that:")
    print("  python -m evaluation.score_elo_batched --ckpt-dir <dir> \\")
    print("      --games 300 --iterations 200 --round-robin")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

    
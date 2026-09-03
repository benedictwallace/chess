"""
Pre-flight for the sequential-halving run.  Run from the repo root:

    python preflight.py

Checks that every patched file is actually DEPLOYED and that CONFIG is
internally coherent. The other two scripts do not cover this:
verify_movegen.py tests the engine, and test_sequential_halving.py passes its
own parameters explicitly and never reads CONFIG -- so both pass green on a
tree where you forgot to copy a file or forgot to flip a flag.

Exits non-zero if anything would waste a training run.
"""
import ast
import importlib
import inspect
import os
import sys

OK, WARN, FAIL = [], [], []


def ok(m):
    OK.append(m)
    print(f"  PASS  {m}")


def warn(m):
    WARN.append(m)
    print(f"  WARN  {m}")


def fail(m):
    FAIL.append(m)
    print(f"  FAIL  {m}")


def config_from_source(path="main.py"):
    """Read CONFIG without importing main (which needs torch + the .so)."""
    tree = ast.parse(open(path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and \
                getattr(node.targets[0], "id", "") == "CONFIG":
            out = {}
            for kw in node.value.keywords:
                try:
                    out[kw.arg] = ast.literal_eval(kw.value)
                except Exception:
                    out[kw.arg] = "<expr>"
            return out
    return None


print("\n" + "=" * 66)
print("1. FILES DEPLOYED")
print("=" * 66)

src = {}
for name, path in [
        ("sequential_halving", "search/sequential_halving.py"),
        ("self_play_batched", "training/self_play_batched.py"),
        ("fast_movegen", "engine/fast_movegen.py"),
        ("encoding", "model/encoding.py"),
        ("main", "main.py"),
        ("main_multigpu", "main_multigpu.py")]:
    if os.path.exists(path):
        src[name] = open(path).read()
        ok(f"{path} present")
    else:
        fail(f"{path} MISSING")

markers = [
    ("sequential_halving", "def improved_policy", "pi' target implemented"),
    ("sequential_halving", "_rescale", "Q rescaling present (target sharpness fix)"),
    ("self_play_batched", "from search.sequential_halving import",
     "self_play imports the SH module"),
    ("self_play_batched", "_policy_target_from_probs", "pi' target path wired"),
    ("self_play_batched", "net_value", "v_mix root value plumbed through"),
    ("fast_movegen", "_MOVE_TABLE", "Move interning deployed"),
    ("fast_movegen", "_build_perm", "derived bb permutation deployed"),
    ("encoding", "_piece_planes", "vectorised encode() deployed"),
    ("main_multigpu", "allow_tf32", "TF32 enabled in multigpu"),
    ("main_multigpu", "torch.compile", "learner compile enabled in multigpu"),
    ("main_multigpu", '_orig_mod', "compiled-state_dict fix present"),
    ("main_multigpu", 'cfg.get("resume"', "multigpu honours resume flag"),
]
for mod, needle, label in markers:
    if mod not in src:
        continue
    (ok if needle in src[mod] else fail)(label)


print("\n" + "=" * 66)
print("2. CONFIG")
print("=" * 66)

C = config_from_source()
if C is None:
    fail("could not parse CONFIG from main.py")
    C = {}

def show(k):
    return C.get(k, "<missing>")

for k in ["checkpoint_dir", "sequential_halving", "sh_m", "sh_c_visit",
          "sh_c_scale", "search_iterations", "root_force_m",
          "forgiveness_targets", "gumbel_select", "resume",
          "games_per_iter", "train_batches", "batch_size"]:
    print(f"        {k:<22} = {show(k)!r}")
print()

if C.get("sequential_halving") is True:
    ok("sequential_halving is ON")
else:
    fail("sequential_halving is OFF -- you would train the old baseline")

# the sharpness relation: sigma_total = (c_visit + max_N) * c_scale, and under
# halving max_N ~ budget/2 * (last phase share). Empirically max_N ~ budget/4.
sims = C.get("search_iterations")
cs = C.get("sh_c_scale")
if isinstance(sims, int) and isinstance(cs, float):
    max_n = sims * 0.24                      # measured: 240 at 1000, 97 at 400
    sigma = (C.get("sh_c_visit", 50.0) + max_n) * cs
    print(f"        predicted sigma_total = (50 + ~{max_n:.0f}) * {cs} = {sigma:.1f}")
    if sigma < 3.0:
        warn(f"sigma_total {sigma:.1f} is low -- target may be too flat "
             f"(pi' barely improves on the prior)")
    elif sigma > 9.0:
        fail(f"sigma_total {sigma:.1f} is HIGH -- expect a near-one-hot target. "
             f"Use sh_c_scale ~ {6.0/(50+max_n):.3f} for {sims} sims")
    else:
        ok(f"sigma_total {sigma:.1f} should give target_entropy ~1.5-2.2 nats")

if C.get("root_force_m", 0):
    warn("root_force_m > 0 -- harmless (SH disables it and says so at startup)")
else:
    ok("root_force_m = 0")

if C.get("gumbel_select"):
    warn("gumbel_select is True but unreachable under SH -- set False so the "
         "config does not misdescribe the run")

if C.get("forgiveness_targets"):
    warn("forgiveness_targets is ON at the same time as SH -- this confounds "
         "'halving helped' with 'forgiveness helped'. Intended?")
else:
    ok("forgiveness_targets OFF (clean SH arm)")


print("\n" + "=" * 66)
print("3. FRESH RUN")
print("=" * 66)

d = C.get("checkpoint_dir")
if isinstance(d, str):
    if not os.path.exists(d):
        ok(f"{d}/ does not exist yet -- will be created, run starts fresh")
    else:
        latest = os.path.join(d, "latest.pt")
        mcsv = os.path.join(d, C.get("metrics_file", "metrics.csv"))
        if os.path.exists(latest):
            if C.get("resume", True):
                warn(f"{latest} exists and resume=True -- this CONTINUES an "
                     f"existing run rather than starting fresh")
            else:
                ok(f"{latest} exists but resume=False -- fresh start")
        else:
            ok(f"{d}/ exists with no latest.pt -- fresh start")
        if os.path.exists(mcsv):
            n = sum(1 for _ in open(mcsv)) - 1
            warn(f"{mcsv} already has {n} rows -- new rows will append")


print("\n" + "=" * 66)
print("4. RUNTIME IMPORTS (needs the built .so and torch)")
print("=" * 66)
try:
    import engine.fast_movegen as fm
    ok(f"engine.movegen imports; bb permutation {fm._PERM}")
    if fm._IDENTITY_ORDER:
        warn("bb permutation is the identity -- unexpected for this codebase; "
             "confirm Board.bb ordering")
except Exception as e:
    fail(f"engine.fast_movegen: {e}")

try:
    from search.sequential_halving import SHState, improved_policy, plan_phases
    p = plan_phases(C.get("search_iterations", 400), C.get("sh_m", 16))
    spent = sum(a * b for a, b in p)
    ok("SH schedule for this budget: " +
       " -> ".join(f"{a}x{b}" for a, b in p) +
       f"  (spends {spent}/{C.get('search_iterations')})")
    if p and p[-1][0] != 2:
        warn(f"schedule does not halve to 2 finalists (ends at {p[-1][0]}) -- "
             f"sh_m may be too large for this budget")
except Exception as e:
    fail(f"search.sequential_halving: {e}")

try:
    from training.self_play_batched import run_selfplay
    sig = inspect.signature(run_selfplay).parameters
    missing = [k for k in ("sequential_halving", "sh_m", "sh_c_scale")
               if k not in sig]
    (ok if not missing else fail)(
        "run_selfplay accepts the SH options" if not missing
        else f"run_selfplay missing {missing} -- stale self_play_batched.py")
except Exception as e:
    fail(f"training.self_play_batched: {e}")


print("\n" + "=" * 66)
print(f"{len(OK)} passed, {len(WARN)} warnings, {len(FAIL)} failures")
if FAIL:
    print("\nDO NOT START. Fix the failures above.")
    sys.exit(1)
if WARN:
    print("\nSafe to start, but read the warnings first.")
else:
    print("\nAll clear -- start the run.")
print("=" * 66 + "\n")


"""Verification for sequential halving + Gumbel targets. No torch, no GPU."""
import math
import numpy as np

from model.move_encoding import NUM_ACTIONS
from search.sequential_halving import (SHState, plan_phases, improved_policy,
                                       root_v_mix)
from training.self_play_batched import run_selfplay, Node

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")
    if not cond:
        FAIL.append(name)


def fake_eval(seed=0):
    """Deterministic pseudo-network: stable priors/values per batch position."""
    rng = np.random.default_rng(seed)

    def eval_fn(planes_list):
        b = len(planes_list)
        logits = rng.normal(0, 1.5, size=(b, NUM_ACTIONS)).astype(np.float32)
        values = rng.uniform(-0.6, 0.6, size=b).astype(np.float32)
        return logits, values
    return eval_fn


print("\n1. plan_phases: budget is fully spent, never overspent")
for budget, m in [(1000, 16), (800, 16), (400, 8), (100, 16), (50, 32),
                  (13, 4), (7, 2), (3, 16), (1, 8)]:
    plan = plan_phases(budget, m)
    spent = sum(c * v for c, v in plan)
    shape = "->".join(f"{c}x{v}" for c, v in plan) or "(empty)"
    check(f"budget={budget:5d} m={m:3d}", spent <= budget,
          f"spends {spent:5d}/{budget:<5d} {shape}")

plan = plan_phases(1000, 16)
check("1000/16 uses >=95% of budget", sum(c * v for c, v in plan) >= 950,
      f"{sum(c*v for c,v in plan)}/1000")
check("1000/16 halves down to 2", plan[-1][0] == 2, f"final phase {plan[-1]}")


print("\n2. SHState on a synthetic root: schedule + elimination")
rng = np.random.default_rng(7)
root = Node()
root.net_value = 0.1
root.children = [Node(root, f"mv{i}", 0.0, 1) for i in range(30)]
pri = rng.dirichlet([0.6] * 30)
for ch, p in zip(root.children, pri):
    ch.prior = float(p)

sh = SHState(root, budget=1000, m=16, rng=rng)
check("samples exactly m candidates", len(sh.candidates) == 16,
      f"{len(sh.candidates)}")
check("candidates are distinct", len({id(c) for c in sh.candidates}) == 16)

sims = 0
qtrue = {id(c): rng.uniform(-1, 1) for c in root.children}
while sims < 1000:
    ch = sh.next_child()
    if ch is None:
        break
    ch.visits += 1
    ch.value += qtrue[id(ch)] + rng.normal(0, 0.3)
    root.visits += 1
    sims += 1
check("SH consumed most of the budget", sims >= 950, f"{sims}/1000 sims")
check("halved down to 2 finalists", len(sh.candidates) == 2,
      f"{len(sh.candidates)} left")
fin = sorted(c.visits for c in sh.candidates)
check("finalists have matched visit counts", abs(fin[0] - fin[1]) <= 2,
      f"visits {fin}")
check("final_floor() reports the survivor minimum",
      sh.final_floor() == fin[0], f"{sh.final_floor()}")

visited = [c for c in root.children if c.visits > 0]
check("eliminated actions exist (budget concentrated)",
      len(visited) <= 16, f"{len(visited)}/30 root actions ever visited")


print("\n3. improved_policy: a valid distribution over ALL legal moves")
pi = improved_policy(root)          # default c_scale
tot = sum(pi.values())
check("sums to 1", abs(tot - 1.0) < 1e-9, f"sum={tot:.12f}")
check("covers every child incl. never-visited", len(pi) == 30, f"{len(pi)} entries")
check("all strictly positive", all(v > 0 for v in pi.values()),
      f"min={min(pi.values()):.3e}")
vm = root_v_mix(root)
check("v_mix inside value range", -1.0 <= vm <= 1.0, f"v_mix={vm:+.4f}")

best_q = max(visited, key=lambda c: c.value / c.visits)
check("argmax pi' is a searched action", pi[best_q.move] > 0.01,
      f"best-Q action has pi'={pi[best_q.move]:.3f}")
_sharp = improved_policy(root, 50.0, 1.0)
_e = -sum(v*math.log(v) for v in _sharp.values() if v > 1e-12)
print(f"        (for contrast, c_scale=1.0 as in the paper/repo default: "
      f"{_e:.3f} nats -- a one-hot)")

# visit-count target on the same tree, for contrast
vc = {c.move: c.visits for c in root.children}
tv = sum(vc.values())
ent_pi = -sum(p * math.log(p) for p in pi.values() if p > 0)
ent_vc = -sum((n / tv) * math.log(n / tv) for n in vc.values() if n > 0)
print(f"        entropy: pi'={ent_pi:.3f} nats vs visit-count={ent_vc:.3f} nats "
      f"over {len(pi)} vs {sum(1 for n in vc.values() if n>0)} supported moves")


print("\n3b. target sharpness is sane at the default c_scale, and scale-invariant")
def _ent(d): return -sum(v*math.log(v) for v in d.values() if v > 1e-12)
ents = []
for spread in (0.03, 0.10, 0.30):
    r = Node(); r.net_value = 0.05
    rr = np.random.default_rng(4)
    r.children = [Node(r, f"m{i}", 0.0, 1) for i in range(32)]
    for ch, pp in zip(r.children, rr.dirichlet([0.5]*32)):
        ch.prior = float(pp)
    sh2 = SHState(r, budget=1000, m=16, rng=rr, c_scale=0.02)
    qt = {id(c): float(v) for c, v in
          zip(r.children, np.sort(rr.normal(0, spread, 32))[::-1])}
    k = 0
    while k < 1000:
        ch = sh2.next_child()
        if ch is None: break
        ch.visits += 1; ch.value += qt[id(ch)] + rr.normal(0, 0.25)
        r.visits += 1; k += 1
    ents.append(_ent(improved_policy(r, 50.0, 0.02)))
check("target entropy is informative, not one-hot",
      all(e > 0.8 for e in ents), f"nats: {[round(e,2) for e in ents]}")
check("entropy is stable across quiet vs tactical roots",
      max(ents) - min(ents) < 1.2, f"spread {max(ents)-min(ents):.2f} nats")


print("\n4. end-to-end self-play, SH on (real engine, fake net)")
ex = run_selfplay(fake_eval(1), num_games=4, iterations=200, concurrency=4,
                 max_plies=40, temp_moves=6, sequential_halving=True, sh_m=8,
                 forgiveness_targets=False, full_search_prob=0.5,
                 fast_iterations=40, record_fast_rows=True,
                 reuse_tree=True, seed=11, verbose=False)
check("produced training rows", len(ex) > 0, f"{len(ex)} rows")
pol = [e for e in ex if len(e[1][0]) > 0]
val = [e for e in ex if len(e[1][0]) == 0]
check("both policy and value-only rows present",
      len(pol) > 0 and len(val) > 0, f"{len(pol)} policy, {len(val)} value-only")
sums = [float(e[1][1].sum()) for e in pol]
check("every policy target normalised",
      all(abs(x - 1.0) < 1e-4 for x in sums),
      f"min={min(sums):.6f} max={max(sums):.6f}")
idxs = np.concatenate([e[1][0] for e in pol])
check("action indices in range", idxs.min() >= 0 and idxs.max() < NUM_ACTIONS,
      f"[{idxs.min()}, {idxs.max()}]")
check("value targets in [-1,1]",
      all(-1.0001 <= float(e[2]) <= 1.0001 for e in ex))
widths = [len(e[1][0]) for e in pol]
print(f"        policy target width: mean {np.mean(widths):.1f}, "
      f"max {max(widths)} (dense over legal moves, as pi' should be)")


print("\n5. baseline path untouched (sequential_halving=False)")
a = run_selfplay(fake_eval(3), num_games=3, iterations=120, concurrency=3,
                 max_plies=30, temp_moves=5, sequential_halving=False,
                 root_force_m=6, root_force_visits=20,
                 forgiveness_targets=False, full_search_prob=1.0,
                 reuse_tree=True, seed=5, verbose=False)
b = run_selfplay(fake_eval(3), num_games=3, iterations=120, concurrency=3,
                 max_plies=30, temp_moves=5, sequential_halving=False,
                 root_force_m=6, root_force_visits=20,
                 forgiveness_targets=False, full_search_prob=1.0,
                 reuse_tree=True, seed=5, verbose=False)
check("baseline is reproducible under a fixed seed", len(a) == len(b),
      f"{len(a)} vs {len(b)} rows")
same = all(np.array_equal(x[1][0], y[1][0]) and
           np.allclose(x[1][1], y[1][1]) and abs(float(x[2]) - float(y[2])) < 1e-9
           for x, y in zip(a, b))
check("baseline rows bit-identical across runs", same)


print("\n6. opening diversity: Gumbel resampling should decorrelate games")
def opening_variety(sh, seeds=14):
    """Distinct positions reached after White's first move.

    Measures the PLAYED move (via the ply-1 encoded planes), not the target
    argmax -- with a soft target those are different things, and it is the
    played move that determines whether self-play games actually diverge."""
    seen = set()
    for sd in range(seeds):
        e = run_selfplay(fake_eval(2), num_games=1, iterations=120,
                         concurrency=1, max_plies=3, temp_moves=0,
                         sequential_halving=sh, sh_m=8,
                         forgiveness_targets=False, full_search_prob=1.0,
                         reuse_tree=False, seed=sd, verbose=False)
        if len(e) > 1:
            seen.add(e[1][0].tobytes())
    return len(seen)

v_sh = opening_variety(True)
v_base = opening_variety(False)
check("SH openings diverge across seeds at temp=0", v_sh >= 4,
      f"{v_sh} distinct ply-1 positions over 14 games "
      f"(plain-PUCT temp=0 baseline: {v_base})")


print("\n" + "=" * 60)
print("ALL PASS" if not FAIL else f"FAILURES: {FAIL}")
print("=" * 60)


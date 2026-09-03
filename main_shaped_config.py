# Cython legal-move generation: patches Board.legalMoves in place. MUST be
# imported before anything that touches engine.board -- model.network and
# training.* both pull it in transitively, so this has to be the FIRST project
# import in the file. Build with
#     python setup_movegen.py build_ext --inplace
# and verify with verify_movegen.py + perft_check.py before trusting a run.
# Comment this line out to fall back to the pure-Python generator.
import engine.fast_movegen  # noqa: F401

import os
import csv
import time
import math
import numpy as np
import torch
from torch.amp import GradScaler

from model.network import ChessNet
from training.self_play_batched import generate_games_batched
from training.train import ReplayBuffer, train_epoch

torch._inductor.config.compile_threads = 1
torch.backends.cudnn.benchmark = True

CONFIG = dict(
    # network. WAS 192x10 (~7M params): at the measured self-play rate
    # (~16 recorded positions/sec -> ~2M positions per 33h run) that net is
    # heavily data-starved -- policy loss plateaued and value loss ROSE over
    # the final 200 iterations of the 599-iter run. Self-play is CPU-bound
    # (Python movegen), so a smaller net does not slow data generation; it
    # overfits the same stream far less and cuts learner + leaf latency.
    # Revert to 192/10 only once the data rate is several times higher.
    channels=128,
    num_blocks=8,

    # outer loop
    loop_iterations=4800,        # WAS 2400 (completed). Extending the horizon
                                 # doubles as a warm restart: cosine_lr is a
                                 # pure function of (it, total), so resuming at
                                 # it=2401 with total=4800 lands at lr ~5.5e-4
                                 # -- lifted off the 1e-4 floor it ground at
                                 # since ~iter 2300 -- and re-decays to lr_min
                                 # by 4800. The ~iter-870 resume showed exactly
                                 # this pattern preceding renewed Elo growth.
    # self-play
    games_per_iter=192,
    # STRENGTH PUSH: was 700. Self-play is CPU-bound on Python movegen, so cost
    # per ply is linear in this number while target quality improves ~log(N).
    # With full_search_prob=0.5 the budget per POLICY ROW goes 1100 -> 500
    # sims, i.e. ~2.2x more policy targets for the same wall-clock. A 400-sim
    # target from a net at this strength is only marginally softer than an
    # 800-sim one -- the limiting factor is the priors, not search depth.
    # Raise toward 600-800 once the net is past ~1600 and data is no longer
    # the binding constraint. Keep --sims in the gauntlet equal to this.
    search_iterations=800,     # PUCT iterations per move
    max_plies=300,             # hard ply cap. WAS 100 -- far too low: 100 plies
                               # is 50 moves, so every game not won by the
                               # adjudication margin within 50 moves was
                               # labelled a DRAW. That defines endgame
                               # conversion, mating technique, and sub-rook
                               # advantages as unwinnable -- the value head
                               # learns "small edge = draw" and strength caps.
                               # With early adjudication on, decided games
                               # still stop well before this, so 300 mainly
                               # lets balanced games resolve properly (and
                               # yields more recorded plies per game, which
                               # also helps the data-starved async learner).
    temp_moves=20,

    # Gumbel-AlphaZero move selection (see training/self_play_batched.py and
    # search.puct.select_move_gumbel): on full-search moves, play the argmax of
    #     log prior + (c_visit + max_visits) * c_scale * Q
    # over the forced-floor candidates instead of argmax pruned visits
    # (openings sample the same scores at temperature). Requires an active
    # forced floor (root_force_m > 0 or forgiveness_targets=True) to have >= 2
    # matched-variance candidates; otherwise it silently falls back to visit
    # selection. Policy targets are unchanged. Default off = legacy behavior.
    # STRENGTH PUSH: was True. select_move_gumbel scores are
    # log prior + (c_visit + max_N) * c_scale * Q, and max_N at a full budget
    # makes that sigma factor several hundred -- so a Q gap of 0.05 becomes
    # 20-30 logits and softmax(scores/temp) at temp=1 is numerically an ARGMAX.
    # The opening plies whose entire purpose is diversity were being played
    # deterministically, and every game in a batch opened alike. self_play_
    # batched now routes temp>0 plies through visit-count sampling regardless,
    # so this flag only affects post-opening moves -- but for a clean AlphaZero
    # baseline leave it off. (It also cannot fire at all while root_force_m=0.)
    gumbel_select=False,
    gumbel_c_visit=50.0,       # paper default; prior-vs-Q trust scale offset
    gumbel_c_scale=1.0,        # paper default; overall sigma(Q) gain

    # Delta-constrained FORGIVING move selection (see search/forgiveness.py,
    # select_move_forgiving): post-opening full-search moves play the most
    # forgiving member of { a : Q1 - Q(a) <= forgiving_delta } over the
    # floored candidates -- at most delta of Q sacrificed per move, subtree
    # forgiveness (forgiveness_tau / forgiveness_gamma; parity=1 = MY future
    # slack) breaking the near-tie. Needs the forced floor active. delta is
    # in Q units: at the 66-visit floor the Q1-Q2 difference carries a
    # standard error of roughly 0.05-0.10, so a delta much below that mostly
    # selects on noise -- calibrate against probe_forgiveness gap percentiles
    # and report sensitivity. Takes precedence over gumbel_select.
    # FORGIVENESS-SHAPED SEARCH (training-time mechanism; see
    # training/self_play_batched.py): every self-play leaf value backs up
    #     v' = clip(v + beta * (2*F_hat - 1), -1, 1)
    # with F_hat the net's own forgiveness head on the leaf position, so the
    # search -- and therefore the visit-count policy targets -- prefers
    # forgiving continuations. Both players are shaped symmetrically: seed
    # the run from a checkpoint whose head was trained on a parity-BLENDED
    # target (e.g. the offline flat_entropy head). beta is in value units;
    # keep it ~ the typical Q gap (0.02-0.05). 0.0 = off (legacy).
    # BASELINE / CONTROL ARM: was 0.003. Any nonzero beta means PUCT descends
    # on shaped values (Node.value_sh), so the search -- and the visit-count
    # policy targets it produces -- are no longer a plain-AlphaZero control.
    # 0.003 was in any case an order of magnitude below the typical Q gap it
    # was meant to break, so it cost a 3-output forward for no effect. When you
    # run the treatment arm, use 0.02-0.05 per the docstring.
    forgiveness_shaping_beta=0.03,
    # STAGING: shaping only activates once the run reaches this iteration --
    # head TRAINING can start from iteration 0 (fitting labels is harmless),
    # but the search should not CONSUME the head until value/policy/labels
    # are mature. Seeding from an offline-head checkpoint (recommended)
    # makes 0 fine; for a from-scratch run set this to a few hundred iters.
    forgiveness_shaping_start_iter=0,

    forgiving_select=False,
    forgiving_delta=0.05,
    forgiving_stat="gap",      # local statistic inside the aggregate
    forgiving_agg="tree",      # "tree" (gamma-decayed) or "flat"
    forgiving_parity=1,        # 1 = my future decision nodes (on a root child)

    # self-play throughput (see training/self_play_batched.py):
    #  * subtree reuse: after a move, carry the chosen child's already-searched
    #    tree over as the next root instead of rebuilding it. On by default;
    #    ~15-20% fewer simulations, no change to what gets recorded.
    #  * playout-cap randomization: run the full `search_iterations` budget on only
    #    `full_search_prob` of moves (ONLY these are recorded as training targets)
    #    and a cheap `fast_search_iterations` budget on the rest. ~2.5-3x less
    #    self-play compute per game. Tune the fraction/budget and watch strength;
    #    set full_search_prob=1.0 to disable (i.e. train on every move as before).
    reuse_tree=True,
    # STRENGTH PUSH: was 0.25. Halving search_iterations pays for doubling the
    # fraction of plies that produce a policy target, at roughly constant cost
    # per ply (0.5*400 + 0.5*100 = 250 vs 0.25*700 + 0.75*100 = 250).
    # Also makes policy and value-only rows arrive at equal rates, which is
    # what keeps the two replay pools' time-horizons aligned (see policy_frac).
    full_search_prob=0.5,
    fast_search_iterations=100,

    # batched self-play. Games are played in one process and their MCTS leaf
    # evaluations are batched together into a single GPU forward pass. `concurrency`
    # is how many games run simultaneously and is the main lever on GPU batch size:
    # the network sees batches of up to `concurrency` positions, versus <= workers
    # tiny batches under the old per-eval multiprocess path. Set it as high as the
    # forward pass fits in GPU memory; 64-256 is a good range for a 128x8 net.
    concurrency=128,

    # within-phase evaluation cache (transposition/repetition skip). Keyed by exact
    # position; rebuilt each self-play phase (the net changes between iterations).
    # Stores legal-move priors + value per entry (~a few KB); cache_cap bounds RAM
    # (200k ~= 0.7 GB). Set use_cache=False to disable.
    use_cache=True,
    cache_cap=200_000,

    # early adjudication ("resignation"): stop a game once one side has held a
    # material lead of >= adj_margin for adj_plies consecutive plies, and score it
    # for that side. Gives clearly-won games a clean +/-1 value target (instead of
    # a noisy ply-cap snapshot) and removes the biggest self-play cost: grinding
    # out already-won games. Set adj_plies=0 to disable.
    #
    # NOTE on the ply cap itself: games that still hit max_plies while materially
    # BALANCED are no longer labelled a hard 0.0 draw -- self_play_batched now
    # bootstraps their value target from the last full-search root value. See
    # finalize() in training/self_play_batched.py.
    adj_margin=5.0,            # 5 == 'up a rook'
    # consecutive plies the lead must hold
    adj_plies=20,
    no_adj_prob=0.1,           # fraction of games that IGNORE adjudication and
                               # play to a real terminal position. If every
                               # decided game stops the moment one side is up a
                               # rook, the value head never sees a conversion or
                               # a mate and learns "up a rook => +1" as a
                               # shortcut, while the policy never learns mating
                               # technique at all -- and in the gauntlet, where
                               # play_game adjudicates at ply 300 on the same
                               # >=5 rule, every won-a-knight game you cannot
                               # convert scores 0.5 instead of 1.0. AlphaGo Zero
                               # disabled resignation in 10% of games for
                               # exactly this reason. Costs ~10% self-play.              # consecutive plies the lead must hold
    # training
    buffer_capacity=600_000,   # WAS 200k. record_fast_rows grows rows/game ~4x
                               # (value-only rows from playout-capped moves), so
                               # the window must grow with it to keep the same
                               # count of POLICY rows in reach. Sparse policy
                               # storage keeps this at ~3 GB (dense would be
                               # ~14 GB): planes ~4.9 KB/row + ~40-entry
                               # (idx, prob) pairs instead of 4672 floats.
    train_batches=64,          # WAS 100. consumed/iter = batches*batch_size;
                               # at 100 the measured replay ratio pinned at
                               # ~8 consumptions per position and rolling
                               # value loss crept from ~0.16 (iter ~900) to
                               # ~0.216 (iter 2400) while policy loss kept
                               # falling -- the value-overfit signature. 64
                               # brings replay to ~5 at the same data rate;
                               # doubly important now that the LR restart
                               # lifts lr ~5.5x. If value loss still creeps,
                               # raise games_per_iter before cutting further.
    batch_size=256,
    policy_frac=0.5,           # share of every training batch drawn from the
                               # POLICY pool (see training/train.py). With
                               # record_fast_rows most rows are value-only, so a
                               # uniform buffer estimates the policy gradient
                               # from only full_search_prob * batch_size rows --
                               # not scaled down (train_epoch mask-normalises)
                               # but ~2x noisier than the batch size implies.
    policy_capacity_frac=0.5,  # share of buffer CAPACITY given to the policy
                               # pool. Each pool holds its own most-recent rows,
                               # so to give both the same time-horizon this
                               # should track full_search_prob.
    lr_schedule="constant",    # "constant" | "cosine". See schedule_lr().
                               # constant: lr is used verbatim, no horizon to
                               # guess, safe to stop/extend a run at any time.
                               # cosine: the old behaviour -- use it only for a
                               # short, deliberate final anneal.
    lr=4e-4,                   # was 1e-3 as a COSINE PEAK, which is not the
                               # same thing as a constant rate. 4e-4 is near the
                               # value the two successful warm restarts actually
                               # trained at (~5.5e-4 decaying), and is a sane
                               # constant. Raise toward 6e-4 if loss is stable
                               # and progress is slow; drop to 2e-4 if policy_kl
                               # becomes noisy or spikes.
    lr_min=1e-4,                # cosine-decay floor; LR goes lr -> lr_min over the run
    weight_decay=1e-4,
    forgiveness_lr=1e-3,               # forgiveness head's OWN optimiser, cosine-decayed
    forgiveness_lr_min=3e-4,           # ... to this floor over the run (a constant
                                # 1e-3 left the head jittering around its
                                # noise floor late in training)
    forgiveness_loss_weight=0.5,       # weight of the forgiveness-head loss in the total.

    # ---- forgiveness TARGET generation (now explicit; previously these silently
    # used the defaults inside self_play_batched, so a calibrated tau never
    # reached the actors in the async runner) ----
    # BASELINE / CONTROL ARM: was True. This is the SECOND shaping channel and
    # the easy one to miss: it extends full_cap by forgiveness_extra_sims AND
    # widens the forced floor via full_force_m = max(root_force_m,
    # forgiveness_force_m), so it changes how the search ALLOCATES visits even
    # though its stated job is label collection. Labels are generated offline
    # from any checkpoint by train_forgiveness_heads.py, which is also the
    # cleaner experimental design: the control's search was never touched.
    # With this False, train_epoch's aux_forgiveness path is off too.
    forgiveness_targets=True,
    forgiveness_tau=0.0178,             # probe_forgiveness calibration: median(gap)/ln 2.
                                # Recalibrated 2026-07 from a 300-position
                                # probe of the iter800 (128x8, FPU) net:
                                # median gap 0.0303 -> tau 0.0437 (the old
                                # 0.0313 squashed 37% of targets below F=0.1).
                                # The same tau serves the entropy statistic.
                                # FIX for the whole run -- changing it mid-run
                                # rescales the head's targets under its feet.
    forgiveness_target_mode="flat_entropy_me",
                                # visit-weighted normalised Q-entropy over the
                                # whole search subtree (flat_forgiveness with
                                # stat="entropy"). WAS "gap" (root-local action
                                # gap). Compound "agg_stat" strings are parsed
                                # by search.forgiveness.forgiveness_target; other options:
                                # gap | entropy | tree_gap | tree_entropy |
                                # flat_gap. NOTE: head targets change meaning
                                # AND scale vs gap-mode runs -- forgiveness_R2 /
                                # forgiveness_tvar are not comparable across the
                                # switch, and the head re-fits over a few
                                # iterations after a resume.
    forgiveness_gamma=0.85,            # only used by mode="tree"
    forgiveness_extra_sims=0,        # extra full-move sims for forgiveness (was 300; the
                                # deep-not-wide floor below needs less width,
                                # clawing back most of the +65% self-play cost)
    forgiveness_force_m=0,             # forced root children for forgiveness Qs (was 12)
    # ---- sample-efficiency / search levers (all new; see
    # training/self_play_batched.run_selfplay for the full rationale) ----
    fpu_reduction=0.25,         # first-play urgency: unvisited children assume
                                # parent-running-Q minus this, not a flat 0.
                                # Stops search over-exploring refuted moves
                                # whenever the mover is worse. 0 = legacy PUCT.
                                # Applies to self-play AND arena/gauntlet play.
    value_target_lambda=0.7,    # value target = lam*z + (1-lam)*Q_root (mover
                                # POV). Blending the position's own search root
                                # value into the outcome label is the cheapest
                                # variance cut available in a data-starved run.
                                # 1.0 = legacy pure-outcome labels.
    record_fast_rows=True,      # record playout-capped (fast) plies as
                                # VALUE-ONLY rows (empty policy, forgiveness mask 0):
                                # ~4x value-head data for one encode() per ply.
                                # The policy loss is mask-normalized in
                                # training/train.py so policy gradients are NOT
                                # diluted by these rows.
    # BASELINE: was 6. Restores plain PUCT roots. With forgiveness_targets on

    # and this at 6 the floor was min(80, 800//12) = 66 visits x 6 children =
    # 396 of 800 sims (49.5%) pinned to equalising the top-m children instead

    # of deepening the PV. Setting this to 0 also makes force_n 0, which is the
    # gate on BOTH gumbel_select and forgiving_select -- so neither can fire

    # even if left enabled. Restore 6 for the treatment arm.
    root_force_m=0,
    root_force_visits=80,       # per-child visit floor CEILING. Effective
                                # floor = min(this, cap // (2*m)) with
                                # cap = search_iterations + forgiveness_extra_sims:
                                # min(80, 800//12) = 66 visits/child. Same
                                # forced budget as before (6x66 ~ 12x40) but
                                # ~40% less Q-gap noise variance -- the gap
                                # statistic only needs the top-2 Qs solid.
                                # Watch the new forgiveness_R2 metrics column: ~0
                                # means the labels are still noise-dominated
                                # at this tau/floor; raise forgiveness_extra_sims
                                # (960 lifts the floor to the full 80) before
                                # blaming the head.
    # ---- GUMBEL SEQUENTIAL HALVING (see search/sequential_halving.py) ----
    # Replaces THREE root mechanisms at once on full-search moves: Dirichlet
    # noise -> Gumbel top-m sampling, forced root visits -> sequential halving,
    # visit-count policy target -> pi' = softmax(logit + sigma(completed-Q)).
    # They are a package; enabling this auto-disables root forcing and root
    # noise (announced at startup) because leaving either on would silently
    # corrupt the target rather than fail.
    sequential_halving=True,
    sh_m=16,                    # root actions sampled per move. The budget is
                                # spent over ceil(log2(m)) phases, halving the
                                # candidate set each time; with 1000 sims and
                                # m=16 the two finalists end on ~240 visits
                                # each with MATCHED standard errors -- the
                                # property root_force_m existed to buy, now
                                # obtained without pinning half the budget.
    sh_c_visit=50.0,            # paper default; prior-vs-Q trust offset
    sh_c_scale=0.03,            # TARGET SHARPNESS. Do not use the paper's 1.0:
                                # sharpness is set by (c_visit + max_N)*c_scale
                                # and halving drives max_N to ~240, so 1.0
                                # yields a ONE-HOT target (measured 0.000 nats)
                                # and mctx's 0.1 gives ~0.15 nats. 0.02 lands
                                # at ~1.8-2.2 nats, matching the visit-count
                                # target_entropy this policy head is already
                                # trained against. Falls as the sim budget
                                # rises -- see the table in the module docstring.

    # io
    checkpoint_dir="checkpoints_shaped",
    checkpoint_every=25,
    metrics_file="metrics_sh.csv",
    resume=True,                # auto-load latest.pt at startup if present
)


def open_metrics(path, header):
    """Append-open a metrics CSV, writing `header` if the file is new. If the
    file exists with a DIFFERENT header (schema change, e.g. the new forgiveness_R2
    columns), divert to <name>_v2.csv rather than appending misaligned rows.
    Returns (file, csv_writer, actual_path)."""
    if os.path.exists(path):
        with open(path, newline="") as f:
            first = f.readline().strip()
        if first and first.split(",") != header:
            base, ext = os.path.splitext(path)
            path = base + "_v2" + ext
            print(f"  metrics schema changed; logging to {path}")
    new_log = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.writer(f)
    if new_log:
        w.writerow(header)
        f.flush()
    return f, w, path


def cosine_lr(it, base_lr, lr_min, total):
    """Cosine-decayed learning rate: base_lr at it==1 down to lr_min at it==total.
    A pure function of the iteration number, so it is exactly correct across
    resumes -- there is no scheduler state to save or restore, and resuming from a
    checkpoint that predates the schedule still lands on the right LR for `it`.
    `total` should be the full planned run length (loop_iterations)."""
    span = max(1, total - 1)
    t = min(max(it - 1, 0), span) / span
    return lr_min + 0.5 * (base_lr - lr_min) * (1.0 + math.cos(math.pi * t))


def schedule_lr(it, base_lr, lr_min, total, cfg=None):
    """LR for step/iteration `it`. Dispatches on cfg["lr_schedule"].

    "constant"  -> base_lr, always. `total` is ignored.
    "cosine"    -> cosine_lr (the previous behaviour).

    WHY CONSTANT IS THE DEFAULT NOW
    -------------------------------
    cosine_lr is a pure function of it/total, which makes it exactly correct
    across resumes -- but it also means the LR you actually train at is decided
    by a number you have to guess in advance, and guessing it wrong is silent.
    Set `total` too low relative to where you resume and the whole run happens
    in the annealed tail.

    On this project that failure mode cost two multi-day runs. The two
    transitions that gained Elo (+236 and +255 head-to-head) both began at
    ~5.5e-4 after a horizon extension; the two that gained nothing (-19 and +4,
    both inside measurement noise) ran entirely at <=1.9e-4 because the resume
    point sat at 80-90% of the configured horizon.

    A constant LR removes the guess. Nothing depends on a planned run length,
    so a run can be stopped or extended freely and every step is spent at a
    productive LR. This is close to how KataGo and Leela actually train: a long
    body at essentially fixed LR, with decay only at the very end.

    You still want that final decay -- it is worth real Elo. Do it DELIBERATELY,
    as a short separate run once you have decided to stop: set
    lr_schedule="cosine" and total_train_steps a little beyond the current step,
    so the anneal is short and lands where you intended.
    """
    mode = (cfg or {}).get("lr_schedule", "constant")
    if mode == "constant":
        return base_lr
    if mode == "cosine":
        return cosine_lr(it, base_lr, lr_min, total)
    raise ValueError(f"unknown lr_schedule {mode!r}: use 'constant' or 'cosine'")


def main(cfg=CONFIG):
    """
    Run the self-play -> train -> checkpoint loop, logging losses per
    iteration. With CONFIG["forgiveness_targets"] on, self-play also computes
    forgiveness labels and train_epoch trains the (decoupled) forgiveness
    head; off (the strength-push configuration) it is the plain policy+value
    loop and the forgiveness metric columns log zeros.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"])
    net.to(device)

    # Default compile mode, NOT mode="reduce-overhead". reduce-overhead uses CUDA
    # graphs, which require a static input shape and a stable train/eval mode. The
    # batched self-play feeds a *variable* batch size (it shrinks from `concurrency`
    # as the game pool drains, and cache/terminal hits trim it round to round) and
    # we toggle net.train()/eval() every iteration, so a captured graph would be
    # recaptured constantly -- pure overhead for a net this small. Default mode
    # keeps the kernel fusion benefit without the cudagraph fragility.
    net = torch.compile(net)

    # DISJOINT optimisers: the main optimiser never contains the forgiveness-head
    # parameters and vice versa, so the two gradient steps in train_epoch are
    # fully decoupled -- neither optimiser can ever move the other's params.
    main_params = [p for n, p in net.named_parameters() if "forgiveness_" not in n]
    forgiveness_params = [p for n, p in net.named_parameters() if "forgiveness_" in n]
    # AdamW, not Adam. torch.optim.Adam implements weight_decay as L2 ADDED TO
    # THE GRADIENT, which Adam's per-parameter second-moment normalisation then
    # rescales -- so the effective decay on a parameter depends on its gradient
    # history, and large-gradient parameters (the BN-adjacent conv weights) are
    # decayed least. AdamW applies the decay directly to the weights, which is
    # what "weight_decay=1e-4" is meant to mean.
    optimiser = torch.optim.AdamW(
        main_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    forgiveness_optimiser = torch.optim.AdamW(
        forgiveness_params, lr=cfg.get("forgiveness_lr", 1e-3),
        weight_decay=cfg["weight_decay"]
    )
    # policy_frac: share of every batch drawn from the POLICY pool. See
    # training/train.py -- with record_fast_rows most rows are value-only, so a
    # uniform buffer estimates the policy gradient from only
    # full_search_prob * batch_size rows.
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"],
                          policy_frac=cfg.get("policy_frac", 0.5),
                          policy_capacity_frac=cfg.get(
                              "policy_capacity_frac", cfg["full_search_prob"]))

    # One persistent AMP loss scaler for the whole run. Recreating it each
    # iteration would reset the online loss-scale calibration and re-run its
    # warmup every time (occasionally skipping steps); see train_epoch.
    scaler = GradScaler("cuda", enabled=torch.cuda.is_available())

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ---- resume from the last checkpoint so re-launching never loses progress ----
    # Without this, every run starts from a fresh random net. We auto-load
    # latest.pt if it exists (disable with cfg["resume"]=False to force a fresh
    # run -- or just move/delete latest.pt). The buffer is NOT persisted (it would
    # be multi-GB); it refills from self-play within a few iterations.
    start_it = 1
    latest_path = os.path.join(cfg["checkpoint_dir"], "latest.pt")
    if cfg.get("resume", True) and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        # weights were saved from the UNWRAPPED module (no "_orig_mod." prefix),
        # so load them back into the unwrapped module under torch.compile.
        target = getattr(net, "_orig_mod", net)
        missing, unexpected = target.load_state_dict(ckpt["model_state"],
                                                     strict=False)
        if missing:
            print(f"  note: {len(missing)} params freshly initialised "
                  f"(e.g. {missing[0]}) -- expected when resuming a "
                  f"pre-forgiveness-head checkpoint; the backbone is loaded.")
        try:
            if "optim_state" in ckpt:
                optimiser.load_state_dict(ckpt["optim_state"])
            if "forgiveness_optim_state" in ckpt:
                forgiveness_optimiser.load_state_dict(ckpt["forgiveness_optim_state"])
        except Exception as e:
            print(f"  note: optimizer state incompatible with the split "
                  f"param groups ({e}); optimizers start fresh.")
        start_it = ckpt.get("iteration", 0) + 1
        print(f"resumed from {latest_path} at iteration {ckpt.get('iteration')}, "
              f"continuing at {start_it}")
        saved = ckpt.get("config", {})
        if saved.get("channels") != cfg["channels"] or saved.get("num_blocks") != cfg["num_blocks"]:
            print("  WARNING: checkpoint network size differs from current config; "
                  "a fresh run is needed if load failed above.")
    else:
        print("starting a fresh run (no checkpoint to resume)")

    if start_it > cfg["loop_iterations"]:
        print(f"already completed {start_it-1} iterations >= loop_iterations="
              f"{cfg['loop_iterations']}; nothing to do. Raise loop_iterations to train more.")
        return

    # ---- metrics log: write a header once, then append a row per iteration.
    # If an existing file has a different (older) header, divert to a fresh
    # _v2 file instead of appending misaligned rows. ----
    # policy_kl / target_entropy: CE = H(target) + KL(target||pred), and only
    # KL is learnable. Track policy_kl -- loss_policy alone conflates real
    # progress with a search-determined floor of ~1.7 nats and makes a learning
    # run look like a plateau. See training/train.py.
    header = ["iteration", "buffer_size",
              "loss_total", "loss_policy", "policy_kl", "target_entropy",
              "loss_value", "loss_forgiveness",
              "forgiveness_R2", "forgiveness_tvar",
              "selfplay_sec", "train_sec"]
    metrics_path = os.path.join(cfg["checkpoint_dir"], cfg["metrics_file"])
    metrics_f, writer, metrics_path = open_metrics(metrics_path, header)

    for it in range(start_it, cfg["loop_iterations"] + 1):
        n_iters = cfg["loop_iterations"]
        print(f"\n ===== Loop iteration {it}/{n_iters} =====")

        # cosine-decay the learning rate (lr -> lr_min over the planned run)
        lr = schedule_lr(it, cfg["lr"], cfg["lr_min"],
                         cfg["loop_iterations"], cfg)
        for grp in optimiser.param_groups:
            grp["lr"] = lr
        forgiveness_lr = schedule_lr(it, cfg["forgiveness_lr"],
                            cfg.get("forgiveness_lr_min", cfg["forgiveness_lr"]),
                            cfg["loop_iterations"], cfg)
        for grp in forgiveness_optimiser.param_groups:
            grp["lr"] = forgiveness_lr
        print(f"  lr {lr:.2e}  forgiveness_lr {forgiveness_lr:.2e}")

        print("self play")
        t0 = time.time()
        examples = generate_games_batched(
            net, cfg["games_per_iter"],
            iterations=cfg["search_iterations"],
            max_plies=cfg["max_plies"],
            temp_moves=cfg["temp_moves"],
            concurrency=cfg["concurrency"],
            adj_margin=cfg["adj_margin"],
            adj_plies=cfg["adj_plies"],
            no_adj_prob=cfg["no_adj_prob"],
            use_cache=cfg["use_cache"],
            cache_cap=cfg["cache_cap"],
            reuse_tree=cfg["reuse_tree"],
            full_search_prob=cfg["full_search_prob"],
            fast_iterations=cfg["fast_search_iterations"],
            root_force_m=cfg["root_force_m"],
            root_force_visits=cfg["root_force_visits"],
            forgiveness_targets=cfg["forgiveness_targets"], forgiveness_tau=cfg["forgiveness_tau"],
            forgiveness_target_mode=cfg["forgiveness_target_mode"],
            forgiveness_gamma=cfg["forgiveness_gamma"],
            forgiveness_extra_sims=cfg["forgiveness_extra_sims"],
            forgiveness_force_m=cfg["forgiveness_force_m"],
            fpu_reduction=cfg["fpu_reduction"],
            value_target_lambda=cfg["value_target_lambda"],
            record_fast_rows=cfg["record_fast_rows"],
            gumbel_select=cfg["gumbel_select"],
            gumbel_c_visit=cfg["gumbel_c_visit"],
            gumbel_c_scale=cfg["gumbel_c_scale"],
            sequential_halving=cfg.get("sequential_halving", False),
            sh_m=cfg.get("sh_m", 16),
            sh_c_visit=cfg.get("sh_c_visit", 50.0),
            sh_c_scale=cfg.get("sh_c_scale", 0.02),
            forgiving_select=cfg["forgiving_select"],
            forgiving_delta=cfg["forgiving_delta"],
            forgiving_stat=cfg["forgiving_stat"],
            forgiving_agg=cfg["forgiving_agg"],
            forgiving_parity=cfg["forgiving_parity"],
            forgiveness_shaping_beta=(
                cfg["forgiveness_shaping_beta"]
                if it >= cfg["forgiveness_shaping_start_iter"] else 0.0),
        )
        selfplay_sec = time.time() - t0
        buffer.add_examples(examples)
        _np, _nv = buffer.counts()
        print(f"  buffer size: {len(buffer)}  "
              f"(policy {_np}, value-only {_nv}; self-play {selfplay_sec:.1f}s)")

        print("training")
        t0 = time.time()
        losses = train_epoch(
            net, buffer, optimiser, device,
            batches=cfg["train_batches"],
            batch_size=cfg["batch_size"],
            aux_forgiveness=cfg["forgiveness_targets"],
            forgiveness_weight=cfg.get("forgiveness_loss_weight", 0.5),
            forgiveness_optimiser=forgiveness_optimiser,
            scaler=scaler,
        )
        train_sec = time.time() - t0

        print(f"  loss total={losses['total']:.4f}  policy={losses['policy']:.4f}  "
              f"value={losses['value']:.4f}  forgiveness={losses['forgiveness']:.4f}  "
              f"forgiveness_R2={losses['forgiveness_R2']:+.3f} (tvar {losses['forgiveness_tvar']:.4f})  "
              f"(train {train_sec:.1f}s)")

        # ---- log this iteration's metrics ----
        writer.writerow([
            it, len(buffer),
            f"{losses['total']:.6f}", f"{losses['policy']:.6f}",
            f"{losses['policy_kl']:.6f}", f"{losses['target_entropy']:.6f}",
            f"{losses['value']:.6f}",
            f"{losses['forgiveness']:.6f}",
            f"{losses['forgiveness_R2']:.4f}", f"{losses['forgiveness_tvar']:.5f}",
            f"{selfplay_sec:.2f}", f"{train_sec:.2f}",
        ])
        metrics_f.flush()

        # ---- checkpoint saving ----
        # net is torch.compile()'d, which (on most PyTorch versions) prefixes
        # state_dict keys with "_orig_mod.". arena.load_net loads with
        # strict=False, so a prefix mismatch would silently load ZERO weights
        # and evaluate a random network. Save the unwrapped module's keys.
        save_net = getattr(net, "_orig_mod", net)
        ckpt = {"iteration": it, "model_state": save_net.state_dict(),
                "optim_state": optimiser.state_dict(),
                "forgiveness_optim_state": forgiveness_optimiser.state_dict(),
                "config": cfg}

        # always overwrite "latest" so resuming is trivial
        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "latest.pt"))

        # keep a milestone every N iterations (and always the final one)
        if it % cfg["checkpoint_every"] == 0 or it == cfg["loop_iterations"]:
            path = os.path.join(cfg["checkpoint_dir"], f"net_iter{it}.pt")
            torch.save(ckpt, path)
            print(f"  saved milestone {path}")
        else:
            print("  saved latest.pt")

    metrics_f.close()
    print(f"\ndone. metrics -> {metrics_path}")


if __name__ == "__main__":
    main()


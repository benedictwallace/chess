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
    loop_iterations=800,        # self-play/train cycles

    # self-play
    games_per_iter=192,
    search_iterations=700,     # PUCT iterations per move
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
    full_search_prob=0.25,
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
    adj_plies=20,              # consecutive plies the lead must hold

    # training
    buffer_capacity=600_000,   # WAS 200k. record_fast_rows grows rows/game ~4x
                               # (value-only rows from playout-capped moves), so
                               # the window must grow with it to keep the same
                               # count of POLICY rows in reach. Sparse policy
                               # storage keeps this at ~3 GB (dense would be
                               # ~14 GB): planes ~4.9 KB/row + ~40-entry
                               # (idx, prob) pairs instead of 4672 floats.
    train_batches=100,
    batch_size=256,
    lr=1e-3,
    lr_min=1e-4,                # cosine-decay floor; LR goes lr -> lr_min over the run
    weight_decay=1e-4,
    ease_lr=1e-3,               # ease head's OWN optimiser, cosine-decayed
    ease_lr_min=3e-4,           # ... to this floor over the run (a constant
                                # 1e-3 left the head jittering around its
                                # noise floor late in training)
    ease_loss_weight=0.5,       # weight of the ease-head loss in the total.

    # ---- ease TARGET generation (now explicit; previously these silently
    # used the defaults inside self_play_batched, so a calibrated tau never
    # reached the actors in the async runner) ----
    ease_targets=True,
    ease_tau=0.0313,            # probe_ease calibration: median(gap)/ln 2.
                                # FIX for the whole run -- changing it mid-run
                                # rescales the head's targets under its feet.
    ease_target_mode="gap",
    ease_gamma=0.85,            # only used by mode="tree"
    ease_extra_sims=100,        # extra full-move sims for ease (was 300; the
                                # deep-not-wide floor below needs less width,
                                # clawing back most of the +65% self-play cost)
    ease_force_m=6,             # forced root children for ease Qs (was 12)
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
                                # VALUE-ONLY rows (empty policy, ease mask 0):
                                # ~4x value-head data for one encode() per ply.
                                # The policy loss is mask-normalized in
                                # training/train.py so policy gradients are NOT
                                # diluted by these rows.

    root_force_m=6,             # forced children on plain full moves
    root_force_visits=80,       # per-child visit floor CEILING. Effective
                                # floor = min(this, cap // (2*m)) with
                                # cap = search_iterations + ease_extra_sims:
                                # min(80, 800//12) = 66 visits/child. Same
                                # forced budget as before (6x66 ~ 12x40) but
                                # ~40% less Q-gap noise variance -- the gap
                                # statistic only needs the top-2 Qs solid.
                                # Watch the new ease_R2 metrics column: ~0
                                # means the labels are still noise-dominated
                                # at this tau/floor; raise ease_extra_sims
                                # (960 lifts the floor to the full 80) before
                                # blaming the head.

    # io
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    metrics_file="metrics.csv",
    resume=True,                # auto-load latest.pt at startup if present
)


def open_metrics(path, header):
    """Append-open a metrics CSV, writing `header` if the file is new. If the
    file exists with a DIFFERENT header (schema change, e.g. the new ease_R2
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


def main(cfg=CONFIG):
    """
    Run the baseline self-play -> train -> checkpoint loop (policy + value only),
    logging the policy and value losses.
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

    # DISJOINT optimisers: the main optimiser never contains the ease-head
    # parameters and vice versa, so the two gradient steps in train_epoch are
    # fully decoupled -- neither optimiser can ever move the other's params.
    main_params = [p for n, p in net.named_parameters() if "ease_" not in n]
    ease_params = [p for n, p in net.named_parameters() if "ease_" in n]
    optimiser = torch.optim.Adam(
        main_params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    ease_optimiser = torch.optim.Adam(
        ease_params, lr=cfg.get("ease_lr", 1e-3),
        weight_decay=cfg["weight_decay"]
    )
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"])

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
                  f"pre-ease-head checkpoint; the backbone is loaded.")
        try:
            if "optim_state" in ckpt:
                optimiser.load_state_dict(ckpt["optim_state"])
            if "ease_optim_state" in ckpt:
                ease_optimiser.load_state_dict(ckpt["ease_optim_state"])
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
    header = ["iteration", "buffer_size",
              "loss_total", "loss_policy", "loss_value", "loss_ease",
              "ease_R2", "ease_tvar",
              "selfplay_sec", "train_sec"]
    metrics_path = os.path.join(cfg["checkpoint_dir"], cfg["metrics_file"])
    metrics_f, writer, metrics_path = open_metrics(metrics_path, header)

    for it in range(start_it, cfg["loop_iterations"] + 1):
        n_iters = cfg["loop_iterations"]
        print(f"\n ===== Loop iteration {it}/{n_iters} =====")

        # cosine-decay the learning rate (lr -> lr_min over the planned run)
        lr = cosine_lr(it, cfg["lr"], cfg["lr_min"], cfg["loop_iterations"])
        for grp in optimiser.param_groups:
            grp["lr"] = lr
        ease_lr = cosine_lr(it, cfg["ease_lr"], cfg.get("ease_lr_min", cfg["ease_lr"]),
                            cfg["loop_iterations"])
        for grp in ease_optimiser.param_groups:
            grp["lr"] = ease_lr
        print(f"  lr {lr:.2e}  ease_lr {ease_lr:.2e}")

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
            use_cache=cfg["use_cache"],
            cache_cap=cfg["cache_cap"],
            reuse_tree=cfg["reuse_tree"],
            full_search_prob=cfg["full_search_prob"],
            fast_iterations=cfg["fast_search_iterations"],
            root_force_m=cfg["root_force_m"],
            root_force_visits=cfg["root_force_visits"],
            ease_targets=cfg["ease_targets"], ease_tau=cfg["ease_tau"],
            ease_target_mode=cfg["ease_target_mode"],
            ease_gamma=cfg["ease_gamma"],
            ease_extra_sims=cfg["ease_extra_sims"],
            ease_force_m=cfg["ease_force_m"],
            fpu_reduction=cfg["fpu_reduction"],
            value_target_lambda=cfg["value_target_lambda"],
            record_fast_rows=cfg["record_fast_rows"],
        )
        selfplay_sec = time.time() - t0
        buffer.add_examples(examples)
        print(f"  buffer size: {len(buffer)}  (self-play {selfplay_sec:.1f}s)")

        print("training")
        t0 = time.time()
        losses = train_epoch(
            net, buffer, optimiser, device,
            batches=cfg["train_batches"],
            batch_size=cfg["batch_size"],
            aux_ease=True,
            ease_weight=cfg.get("ease_loss_weight", 0.5),
            ease_optimiser=ease_optimiser,
            scaler=scaler,
        )
        train_sec = time.time() - t0

        print(f"  loss total={losses['total']:.4f}  policy={losses['policy']:.4f}  "
              f"value={losses['value']:.4f}  ease={losses['ease']:.4f}  "
              f"ease_R2={losses['ease_R2']:+.3f} (tvar {losses['ease_tvar']:.4f})  "
              f"(train {train_sec:.1f}s)")

        # ---- log this iteration's metrics ----
        writer.writerow([
            it, len(buffer),
            f"{losses['total']:.6f}", f"{losses['policy']:.6f}", f"{losses['value']:.6f}",
            f"{losses['ease']:.6f}",
            f"{losses['ease_R2']:.4f}", f"{losses['ease_tvar']:.5f}",
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
                "ease_optim_state": ease_optimiser.state_dict(),
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
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
    # network
    channels=192,
    num_blocks=10,

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
    buffer_capacity=200_000,
    train_batches=100,
    batch_size=256,
    lr=1e-3,
    lr_min=1e-4,                # cosine-decay floor; LR goes lr -> lr_min over the run
    weight_decay=1e-4,
    ease_lr=1e-3,               # ease head's OWN optimiser (constant LR; the
                                # cosine schedule applies to the main
                                # optimiser only)
    ease_loss_weight=0.5,       # weight of the ease-head loss in the total.
                                # Ease TARGET options (tau, gap/tree/flat mode,
                                # extra root sims, floor width) live as
                                # defaults on training/self_play_batched.py's
                                # generate_games_batched.

    # io
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    metrics_file="metrics.csv",
    resume=True,                # auto-load latest.pt at startup if present
)


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

    # ---- metrics log: write a header once, then append a row per iteration ----
    metrics_path = os.path.join(cfg["checkpoint_dir"], cfg["metrics_file"])
    new_log = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    writer = csv.writer(metrics_f)
    if new_log:
        writer.writerow([
            "iteration", "buffer_size",
            "loss_total", "loss_policy", "loss_value", "loss_ease",
            "selfplay_sec", "train_sec",
        ])
        metrics_f.flush()

    for it in range(start_it, cfg["loop_iterations"] + 1):
        n_iters = cfg["loop_iterations"]
        print(f"\n ===== Loop iteration {it}/{n_iters} =====")

        # cosine-decay the learning rate (lr -> lr_min over the planned run)
        lr = cosine_lr(it, cfg["lr"], cfg["lr_min"], cfg["loop_iterations"])
        for grp in optimiser.param_groups:
            grp["lr"] = lr
        print(f"  lr {lr:.2e}")

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
              f"(train {train_sec:.1f}s)")

        # ---- log this iteration's metrics ----
        writer.writerow([
            it, len(buffer),
            f"{losses['total']:.6f}", f"{losses['policy']:.6f}", f"{losses['value']:.6f}",
            f"{losses['ease']:.6f}",
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
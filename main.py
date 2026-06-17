import os
import csv
import time
import numpy as np
import torch

from network import ChessNet
from self_play_parallel import generate_games_parallel
from train import ReplayBuffer, train_epoch


CONFIG = dict(
    # network
    channels=256,
    num_blocks=10,

    # outer loop
    loop_iterations=500,        # self-play/train cycles

    # self-play
    games_per_iter=30,
    search_iterations=400,     # PUCT iterations per move
    max_plies=200,
    temp_moves=30,

    # training
    buffer_capacity=200_000,
    train_batches=32,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,
    ease_weight=1.0,           # weight on the forgiveness (ease) loss
    cliff_weight=1.0,          # weight on the cliff-return loss
    stab_weight=1.0,           # weight on the trajectory-stability loss

    # io
    workers=10,              # self-play processes; None = all cores
    checkpoint_dir="checkpoints",
    checkpoint_every=10,
    metrics_file="metrics.csv",
)


def main(cfg=CONFIG):
    """
    Run the full self-play -> train -> checkpoint loop, logging all losses.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"])
    net.to(device)

    optimiser = torch.optim.Adam(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"])

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # ---- metrics log: write a header once, then append a row per iteration ----
    metrics_path = os.path.join(cfg["checkpoint_dir"], cfg["metrics_file"])
    new_log = not os.path.exists(metrics_path)
    metrics_f = open(metrics_path, "a", newline="")
    writer = csv.writer(metrics_f)
    if new_log:
        writer.writerow([
            "iteration", "buffer_size",
            "loss_total", "loss_policy", "loss_value",
            "loss_ease", "loss_cliff", "loss_stab",
            "selfplay_sec", "train_sec",
        ])
        metrics_f.flush()

    for it in range(1, cfg["loop_iterations"] + 1):
        n_iters = cfg["loop_iterations"]
        print(f"\n ===== Loop iteration {it}/{n_iters} =====")

        print("self play")
        t0 = time.time()
        examples = generate_games_parallel(
            net, cfg["games_per_iter"],
            iterations=cfg["search_iterations"],
            max_plies=cfg["max_plies"],
            temp_moves=cfg["temp_moves"],
            workers=cfg["workers"],
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
            ease_weight=cfg["ease_weight"],
            cliff_weight=cfg["cliff_weight"],
            stab_weight=cfg["stab_weight"],
        )
        train_sec = time.time() - t0

        print(f"  loss total={losses['total']:.4f}  policy={losses['policy']:.4f}  "
              f"value={losses['value']:.4f}  ease={losses['ease']:.4f}  "
              f"cliff={losses['cliff']:.4f}  stab={losses['stab']:.4f}  (train {train_sec:.1f}s)")

        # ---- log this iteration's metrics ----
        writer.writerow([
            it, len(buffer),
            f"{losses['total']:.6f}", f"{losses['policy']:.6f}", f"{losses['value']:.6f}",
            f"{losses['ease']:.6f}", f"{losses['cliff']:.6f}", f"{losses['stab']:.6f}",
            f"{selfplay_sec:.2f}", f"{train_sec:.2f}",
        ])
        metrics_f.flush()

        # ---- checkpoint saving ----
        ckpt = {"iteration": it, "model_state": net.state_dict(),
                "optim_state": optimiser.state_dict(), "config": cfg}

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
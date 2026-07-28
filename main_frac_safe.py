"""
Standalone training run for the FRAC-SAFE ease signal.

Identical to main.py except:
  * the net has the auxiliary ease head (aux_ease=True),
  * self-play attaches a FracSafeEase signal so each example carries an
    (ease_target, ease_mask),
  * training adds the masked ease loss and logs it,
  * outputs go to their own checkpoint dir / metrics file so this never
    collides with the production run.

Run from the project root:  python main_fracsafe.py
"""

import os
import csv
import time
import torch

from model.network import ChessNet
from archive.self_play_parallel import generate_games_parallel
from training.train import ReplayBuffer, train_epoch
from ease.frac_safe import FracSafeEase

torch._inductor.config.compile_threads = 1
torch.backends.cudnn.benchmark = True

CONFIG = dict(
    # network
    channels=128,
    num_blocks=8,

    # outer loop
    loop_iterations=200,

    # self-play
    games_per_iter=24,
    search_iterations=400,
    max_plies=200,
    temp_moves=30,

    # frac-safe ease signal
    ease_weight=1.0,           # weight on the ease loss in the total
    ease_min_visits=5,         # ignore barely-explored root moves
    ease_delta=0.1,            # "safe" = within this of the best Q
    ease_gamma=0.9,            # discount for the future-state return

    # training
    buffer_capacity=200_000,
    train_batches=32,
    batch_size=256,
    lr=1e-3,
    weight_decay=1e-4,

    # io  (separate from production so runs don't clobber each other)
    workers=18,
    checkpoint_dir="checkpoints_fracsafe",
    checkpoint_every=10,
    metrics_file="metrics_fracsafe.csv",
)


def main(cfg=CONFIG):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  |  signal: frac-safe (ease)")

    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"], aux_ease=True)
    net.to(device)
    net = torch.compile(net, mode="reduce-overhead")

    optimiser = torch.optim.Adam(
        net.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"])

    ease_signal = FracSafeEase(
        min_visits=cfg["ease_min_visits"],
        delta=cfg["ease_delta"],
        gamma=cfg["ease_gamma"],
    )

    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

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

    for it in range(1, cfg["loop_iterations"] + 1):
        print(f"\n ===== Loop iteration {it}/{cfg['loop_iterations']} =====")

        print("self play")
        t0 = time.time()
        examples = generate_games_parallel(
            net, cfg["games_per_iter"],
            iterations=cfg["search_iterations"],
            max_plies=cfg["max_plies"],
            temp_moves=cfg["temp_moves"],
            workers=cfg["workers"],
            ease_signal=ease_signal,
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
            ease_weight=cfg["ease_weight"],
        )
        train_sec = time.time() - t0

        print(f"  loss total={losses['total']:.4f}  policy={losses['policy']:.4f}  "
              f"value={losses['value']:.4f}  ease={losses['ease']:.4f}  "
              f"(train {train_sec:.1f}s)")

        writer.writerow([
            it, len(buffer),
            f"{losses['total']:.6f}", f"{losses['policy']:.6f}",
            f"{losses['value']:.6f}", f"{losses['ease']:.6f}",
            f"{selfplay_sec:.2f}", f"{train_sec:.2f}",
        ])
        metrics_f.flush()

        # ---- checkpoint saving ----
        # unwrap the torch.compile module so checkpoint keys aren't "_orig_mod."-prefixed
        save_net = getattr(net, "_orig_mod", net)
        ckpt = {"iteration": it, "model_state": save_net.state_dict(),
                "optim_state": optimiser.state_dict(), "config": cfg}

        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "latest.pt"))

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
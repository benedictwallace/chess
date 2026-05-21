
import os
import numpy as np
import torch

from network import ChessNet
from self_play import generate_games
from train import ReplayBuffer, train_epoch



CONFIG = dict(
    # network
    channels=64,
    num_blocks=5,

    # outer loop
    loop_iterations=20,        # self-play/train cycles

    # self-play
    games_per_iter=10,
    search_iterations=100,     # PUCT iterations per move
    max_plies=200,
    temp_moves=30,

    # training
    buffer_capacity=50_000,
    train_batches=32,
    batch_size=128,
    lr=1e-3,
    weight_decay=1e-4,

    # io
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
)



def main(cfg=CONFIG):
    """
    Main function to run all training.
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

    for it in range(1, cfg["loop_iterations"]+1):
        print(f"\n ===== Loop iteration {it}/{cfg["loop_iterations"]} =====")


        print("self play")
        examples = generate_games(
            net, cfg["games_per_iter"],
            iterations=cfg["search_iterations"],
            max_plies=cfg["max_plies"],
            temp_moves=cfg["temp_moves"],
        )
        buffer.add_examples(examples)
        print(f"  buffer size: {len(buffer)}")

        print("training")
        total, policy_l, value_l = train_epoch(
            net, buffer, optimiser, device,
            batches=cfg["train_batches"],
            batch_size=cfg["batch_size"],
        )

        print(f"  loss total={total:.4f}  policy={policy_l:.4f}  value={value_l:.4f}")



        # checkpoint saving

        ckpt = {"iteration": it, "model_state": net.state_dict(),
                "optim_state": optimiser.state_dict(), "config": cfg}

        # always overwrite "latest" so resuming is trivial
        torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], "latest.pt"))

        # keep a milestone only every N iterations (and always the final one)
        if it % cfg["checkpoint_every"] == 0 or it == cfg["loop_iterations"]:
            path = os.path.join(cfg["checkpoint_dir"], f"net_iter{it}.pt")
            torch.save(ckpt, path)
            print(f"  saved milestone {path}")
        else:
            print("  saved latest.pt")

    print("\ndone.")



if __name__ == "__main__":
    main()


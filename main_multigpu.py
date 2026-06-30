"""
Multi-GPU self-play training: fan the self-play phase across GPUs, train on one.

Synchronous actor/learner loop (the standard scalable AlphaZero shape). Each
outer iteration:
  1. PUBLISH the current weights to a file.
  2. SELF-PLAY in parallel: `actors_per_gpu` processes per GPU, each loads the
     published weights onto its GPU and runs the batched self-play for a slice
     of the games. Every example is POOLED into one buffer.
  3. TRAIN one net on the pooled buffer (the learner GPU), checkpoint.

This is how multiple GPUs buy you strength: more/better self-play DATA for ONE
learner -- NOT averaged weights (independently trained nets live in different
loss basins, so averaging their parameters yields a broken net).

`actors_per_gpu` > 1 because each actor's MCTS is single-core: with ~18 cores
and a few GPUs you want several actors per GPU to keep the cores busy (the CPU
MCTS, not the GPU forward, is the bottleneck for this net).

Run:  python main_multigpu.py            # uses all visible GPUs
"""
import os
import time
import pickle
import tempfile
import multiprocessing as mp

import torch
from torch.amp import GradScaler

from model.network import ChessNet
from training.self_play_batched import generate_games_batched
from training.train import ReplayBuffer, train_epoch
from main import CONFIG, cosine_lr


def _selfplay_worker(gpu_id, threads, weights_path, out_path, n_games, cfg):
    torch.set_num_threads(max(1, threads))
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"]).to(device)
    ck = torch.load(weights_path, map_location=device, weights_only=False)
    net.load_state_dict(ck["model_state"])
    net.eval()
    examples = generate_games_batched(
        net, n_games,
        iterations=cfg["search_iterations"], max_plies=cfg["max_plies"],
        temp_moves=cfg["temp_moves"], concurrency=cfg["concurrency"],
        adj_margin=cfg["adj_margin"], adj_plies=cfg["adj_plies"],
        use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"], verbose=False)
    with open(out_path, "wb") as f:
        pickle.dump(examples, f, protocol=pickle.HIGHEST_PROTOCOL)


def selfplay_across_gpus(weights_path, n_games_total, gpus, actors_per_gpu, cfg):
    """Fan self-play over (gpus x actors_per_gpu) processes; pool all examples."""
    actor_gpus = [g for g in gpus for _ in range(actors_per_gpu)]
    n = len(actor_gpus)
    per = [n_games_total // n + (1 if i < n_games_total % n else 0) for i in range(n)]
    threads = max(1, (os.cpu_count() or n) // n)   # share cores across actors

    ctx = mp.get_context("spawn")
    procs, outs = [], []
    for i, (gpu_id, k) in enumerate(zip(actor_gpus, per)):
        if k == 0:
            continue
        out = os.path.join(tempfile.gettempdir(), f"sp_{os.getpid()}_{i}.pkl")
        p = ctx.Process(target=_selfplay_worker,
                        args=(gpu_id, threads, weights_path, out, k, cfg))
        p.start(); procs.append(p); outs.append(out)
    for p in procs:
        p.join()

    # robustness: if a worker died (e.g. CUDA OOM) its pickle won't exist -- use
    # what the survivors produced rather than crashing the whole run.
    failed = [p.exitcode for p in procs if p.exitcode not in (0, None)]
    if failed:
        print(f"  WARNING: {len(failed)} self-play worker(s) exited non-zero {failed}; "
              f"continuing with the rest (lower --actors-per-gpu if this is OOM)")
    examples = []
    for out in outs:
        if not os.path.exists(out):
            continue
        with open(out, "rb") as f:
            examples.extend(pickle.load(f))
        os.remove(out)
    if not examples:
        raise RuntimeError("all self-play workers failed -- check GPU memory and "
                           "lower --actors-per-gpu")
    return examples


def main(cfg=None, gpus=None, actors_per_gpu=2):
    cfg = dict(CONFIG if cfg is None else cfg)
    if gpus is None:
        gpus = tuple(range(torch.cuda.device_count())) or (0,)
    learner_device = torch.device(f"cuda:{gpus[0]}")
    print(f"learner on cuda:{gpus[0]}; self-play on GPUs {list(gpus)} "
          f"x {actors_per_gpu} actors each")

    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"]).to(learner_device)
    optimiser = torch.optim.Adam(net.parameters(), lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"])
    scaler = GradScaler("cuda")
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # resume if a checkpoint exists (otherwise every launch restarts from random)
    latest = os.path.join(cfg["checkpoint_dir"], "latest.pt")
    start_it = 1
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=learner_device, weights_only=False)
        net.load_state_dict(ck["model_state"])
        optimiser.load_state_dict(ck["optim_state"])
        start_it = ck["iteration"] + 1
        print(f"resumed from iter {ck['iteration']}")

    pub = os.path.join(cfg["checkpoint_dir"], "_actor_weights.pt")
    for it in range(start_it, cfg["loop_iterations"] + 1):
        print(f"\n===== iteration {it}/{cfg['loop_iterations']} =====")

        # cosine-decay the learning rate (lr -> lr_min over the planned run)
        lr = cosine_lr(it, cfg["lr"], cfg["lr_min"], cfg["loop_iterations"])
        for grp in optimiser.param_groups:
            grp["lr"] = lr
        print(f"  lr {lr:.2e}")

        # 1. publish weights the actors will load (no compile prefix -> clean keys)
        torch.save({"model_state": net.state_dict()}, pub)

        # 2. parallel self-play -> pooled examples
        t0 = time.time()
        examples = selfplay_across_gpus(pub, cfg["games_per_iter"], gpus,
                                        actors_per_gpu, cfg)
        buffer.add_examples(examples)
        print(f"  self-play: {len(examples)} positions in {time.time()-t0:.1f}s; "
              f"buffer {len(buffer)}")

        # 3. train on the learner GPU
        t0 = time.time()
        losses = train_epoch(net, buffer, optimiser, learner_device,
                             batches=cfg["train_batches"],
                             batch_size=cfg["batch_size"], scaler=scaler)
        print(f"  loss total={losses['total']:.4f} policy={losses['policy']:.4f} "
              f"value={losses['value']:.4f} (train {time.time()-t0:.1f}s)")

        # 4. checkpoint (same format arena/score_elo expect)
        ckpt = {"iteration": it, "model_state": net.state_dict(),
                "optim_state": optimiser.state_dict(), "config": cfg}
        torch.save(ckpt, latest)
        if it % cfg["checkpoint_every"] == 0 or it == cfg["loop_iterations"]:
            torch.save(ckpt, os.path.join(cfg["checkpoint_dir"], f"net_iter{it}.pt"))
            print(f"  saved milestone net_iter{it}.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Multi-GPU self-play training")
    ap.add_argument("--actors-per-gpu", type=int, default=2,
                    help="self-play processes per GPU (each uses one CPU core for MCTS)")
    ap.add_argument("--gpus", default=None,
                    help="comma-separated GPU ids, e.g. 0,1,2,3 (default: all visible)")
    args = ap.parse_args()
    gpus = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else None
    main(gpus=gpus, actors_per_gpu=args.actors_per_gpu)
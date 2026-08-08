"""
Multi-GPU ASYNC self-play training: self-play actors and the learner run
CONCURRENTLY instead of alternating.

The previous version alternated phases -- generate ALL self-play games, THEN
train -- so whichever GPU was training sat idle during self-play and the
self-play GPUs sat idle during training. Here the two overlap:

  * ACTORS (`actors_per_gpu` processes per GPU) run continuously. Each loads the
    latest published weights, plays a chunk of batched self-play games (with the
    subtree reuse + playout-cap randomization configured in CONFIG), drops the
    resulting examples into a spool directory as one file, and repeats.
  * The LEARNER (this process, on gpus[0]) runs continuously: it drains the spool
    into the replay buffer, takes gradient steps, and every `publish_every_steps`
    steps republishes its weights for the actors to pick up.

REPLAY-RATIO THROTTLE (the fix for the async pathology): in the old async loop
the ONLY things gating the learner were a 2k-example warmup and file-count
backpressure on the ACTORS. Once warm, the learner took gradient steps
non-stop, fully decoupled from how fast data arrived. With a slow (pure-Python
movegen) self-play engine and a fast GPU learner, the effective replay ratio
(samples trained / samples generated) explodes into the hundreds: the net
memorizes the buffer, publishes overfit weights, the actors generate the next
chunk with those weights, and progress stalls -- fast early Elo gains, then a
noisy plateau. Two changes fix this:

  1. `target_ratio` (default 8.0): the learner trains only while
         samples_trained_this_run <= target_ratio * samples_generated_this_run
     and otherwise sleeps + drains the spool. Accounting is PER RUN (deltas
     since this process started), so resuming from a large `train_step` with an
     empty spool does not deadlock the learner. 8x matches the implicit ratio
     of the synchronous main.py (100 batches x 256 consumed vs ~4-5k generated
     per iteration ~= 5-8x), which trained healthily.
  2. `min_buffer` warmup floor raised to buffer_capacity // 4 (150k at the
     current 600k capacity -- a long warmup at ~16 recorded positions/sec;
     pass --min-buffer to override) instead of buffer_capacity // 100 (2k).
     The old floor let the learner hammer the first ~2k positions at peak LR.

The throttle means the learner may spend long stretches sleeping while actors
catch up -- that is the intended behavior (it is the actors that are the
bottleneck, and more/faster actors now translate directly into more training).
With the default `games_per_chunk = concurrency` (128) data arrives in large
lumps; a smaller --games-per-chunk (e.g. 32) smooths the trickle and keeps the
buffer mixing fresher, at a small per-chunk overhead.

Communication is through the filesystem (atomic writes + an mtime check on the
weights file) -- the same idiom the old synchronous version used. There is no
blocking queue, so no actor/learner deadlock is possible: actors self-throttle on
the number of pending spool files and always exit when the STOP sentinel appears;
the learner always drains and cleans up on the way out (including on Ctrl-C).

As before, multiple GPUs buy strength through more/better self-play DATA for ONE
learner -- NOT averaged weights (independently trained nets live in different
loss basins, so averaging their parameters yields a broken net).

Runs on a single GPU too (actors share it with the learner -- useful when
self-play is CPU-bound, which is the common case for this net), and falls back to
CPU when no GPU is visible.

NOTE on metrics.csv: this version appends a `replay_ratio` column. If you are
appending to a metrics file written by the previous version, either rotate it
or expect the old rows to have one fewer column.

Run:  python main_multigpu.py                       # all visible GPUs, 2 actors each
      python main_multigpu.py --gpus 0,1,2,3 --actors-per-gpu 3
      python main_multigpu.py --dedicate-learner-gpu # reserve gpus[0] for training
      python main_multigpu.py --target-ratio 4      # stricter freshness
      python main_multigpu.py --target-ratio 0      # disable the throttle (old behavior)
"""
import os
import csv
import glob
import time
import pickle
import tempfile
import multiprocessing as mp

import torch
from torch.amp import GradScaler

from model.network import ChessNet
from training.self_play_batched import generate_games_batched
from training.train import ReplayBuffer, train_epoch
from main import CONFIG, cosine_lr, open_metrics


SPOOL_GLOB = "chunk_*.pkl"


# --------------------------------------------------------------------------- #
# small filesystem helpers
# --------------------------------------------------------------------------- #
def _pick_device(gpu_id):
    """cuda:<id> when a GPU is present, else cpu (so this runs/tests anywhere)."""
    return torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")


def _atomic_pickle(obj, path):
    """Write a pickle so a reader never sees a partial file (tmp + atomic rename)."""
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _atomic_torch_save(obj, path):
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# --------------------------------------------------------------------------- #
# actor process: continuously play chunks of self-play games
# --------------------------------------------------------------------------- #
def _actor_loop(gpu_id, threads, pub_path, spool_dir, stop_path,
                games_per_chunk, max_pending, cfg):
    torch.set_num_threads(max(1, threads))
    device = _pick_device(gpu_id)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"]).to(device)
    net.eval()
    last_mtime = -1.0
    counter = 0

    def reload_weights():
        nonlocal last_mtime
        try:
            m = os.path.getmtime(pub_path)
        except OSError:
            return
        if m > last_mtime:
            try:
                ck = torch.load(pub_path, map_location=device, weights_only=False)
                net.load_state_dict(ck["model_state"])
                net.eval()
                last_mtime = m
            except Exception as e:            # mid-rename race: retry next loop
                print(f"[actor {os.getpid()}] weight reload retry: {e}", flush=True)

    while not os.path.exists(pub_path) and not os.path.exists(stop_path):
        time.sleep(0.1)                       # wait for the learner's first publish

    while not os.path.exists(stop_path):
        reload_weights()
        # backpressure: don't run arbitrarily far ahead of the learner
        pending = len(glob.glob(os.path.join(spool_dir, SPOOL_GLOB)))
        if pending >= max_pending:
            time.sleep(0.3)
            continue
        try:
            examples = generate_games_batched(
                net, games_per_chunk,
                iterations=cfg["search_iterations"], max_plies=cfg["max_plies"],
                temp_moves=cfg["temp_moves"], concurrency=cfg["concurrency"],
                adj_margin=cfg["adj_margin"], adj_plies=cfg["adj_plies"],
                use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"],
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
                # --- forwarded so the async actors match main.py's search ---
                # WITHOUT these the actor uses generate_games_batched's
                # DEFAULTS (gumbel_select=False, forgiveness_shaping_beta=0.0,
                # forgiving_select=False): a plain run with no gumbel and no
                # forgiveness shaping, silently -- no argparse error to catch it.
                gumbel_select=cfg["gumbel_select"],
                gumbel_c_visit=cfg["gumbel_c_visit"],
                gumbel_c_scale=cfg["gumbel_c_scale"],
                forgiving_select=cfg["forgiving_select"],
                forgiving_delta=cfg["forgiving_delta"],
                forgiving_stat=cfg["forgiving_stat"],
                forgiving_agg=cfg["forgiving_agg"],
                forgiving_parity=cfg["forgiving_parity"],
                # start_iter gating cannot be honored here (actors carry no
                # global iteration counter), so beta is passed directly. Keep
                # forgiveness_shaping_start_iter=0 for async runs -- guarded in main().
                forgiveness_shaping_beta=cfg["forgiveness_shaping_beta"],
                verbose=False)
        except Exception as e:                # e.g. CUDA OOM: back off, stay alive
            print(f"[actor {os.getpid()}] self-play error, backing off: {e}",
                  flush=True)
            time.sleep(1.0)
            continue
        if not examples:
            continue
        counter += 1
        name = f"chunk_{gpu_id}_{os.getpid()}_{counter}.pkl"
        _atomic_pickle(examples, os.path.join(spool_dir, name))


# --------------------------------------------------------------------------- #
# learner (this process)
# --------------------------------------------------------------------------- #
def _drain_spool(spool_dir, buffer):
    """Load every completed chunk into the buffer and delete it; return #examples."""
    n = 0
    for path in sorted(glob.glob(os.path.join(spool_dir, SPOOL_GLOB))):
        try:
            with open(path, "rb") as f:
                ex = pickle.load(f)
        except Exception:
            continue                          # not fully written yet -> next drain
        buffer.add_examples(ex)
        n += len(ex)
        try:
            os.remove(path)
        except OSError:
            pass
    return n


def main(cfg=None, gpus=None, actors_per_gpu=2, dedicate_learner_gpu=False,
         total_train_steps=None, train_block=8, games_per_chunk=None,
         publish_every_steps=None, max_pending_per_actor=3, min_buffer=None,
         target_ratio=8.0):
    cfg = dict(CONFIG if cfg is None else cfg)

    if gpus is None:
        gpus = tuple(range(torch.cuda.device_count())) or (0,)
    learner_gpu = gpus[0]
    actor_pool = gpus[1:] if (dedicate_learner_gpu and len(gpus) > 1) else gpus
    actor_gpus = [g for g in actor_pool for _ in range(actors_per_gpu)]
    n_actors = len(actor_gpus)

    # derive async cadence from CONFIG so a run matches the synchronous one unless
    # explicitly overridden
    if total_train_steps is None:
        total_train_steps = cfg["loop_iterations"] * cfg["train_batches"]
    if publish_every_steps is None:
        publish_every_steps = cfg["train_batches"]          # ~once per old "iter"
    if games_per_chunk is None:
        games_per_chunk = max(1, cfg["concurrency"])        # one full batch-wave
    if min_buffer is None:
        # WARMUP FLOOR: a quarter of the buffer (150k at the current 600k
        # capacity), not one percent. The old buffer_capacity // 100 (= 2k
        # examples) let training start on a token dataset at the peak LR --
        # the net overfit it before real data existed.
        min_buffer = max(cfg["batch_size"], cfg["buffer_capacity"] // 4)
    checkpoint_every_steps = cfg["checkpoint_every"] * cfg["train_batches"]
    max_pending = max(1, max_pending_per_actor * max(1, n_actors))
    target_ratio = max(0.0, float(target_ratio))            # 0 disables the throttle

    learner_device = _pick_device(learner_gpu)
    print(f"learner on {learner_device}; {n_actors} actor(s) on GPUs "
          f"{sorted(set(actor_gpus))} x{actors_per_gpu}; {games_per_chunk} games/chunk; "
          f"{total_train_steps} train steps; publish every {publish_every_steps}; "
          f"min_buffer {min_buffer}; target replay ratio "
          f"{'off' if target_ratio == 0 else target_ratio}")

    net = ChessNet(channels=cfg["channels"], num_blocks=cfg["num_blocks"]).to(learner_device)
    # disjoint optimisers -> fully decoupled gradient steps (see train_epoch)
    main_params = [p for n, p in net.named_parameters() if "forgiveness_" not in n]
    forgiveness_params = [p for n, p in net.named_parameters() if "forgiveness_" in n]
    optimiser = torch.optim.Adam(main_params, lr=cfg["lr"],
                                 weight_decay=cfg["weight_decay"])
    forgiveness_optimiser = torch.optim.Adam(forgiveness_params,
                                      lr=cfg.get("forgiveness_lr", 1e-3),
                                      weight_decay=cfg["weight_decay"])
    buffer = ReplayBuffer(capacity=cfg["buffer_capacity"])
    scaler = GradScaler("cuda", enabled=(learner_device.type == "cuda"))
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

    # STAGED SHAPING IS NOT SUPPORTED IN THE ASYNC RUNNER: actors do not know
    # the learner's global iteration, so beta cannot be gated on it (see the
    # actor call). A nonzero start_iter would be silently ignored -- shaping
    # would be live from step 0 regardless. Fail loud instead of lying.
    if cfg.get("forgiveness_shaping_beta", 0.0) != 0.0 \
            and cfg.get("forgiveness_shaping_start_iter", 0) > 0:
        raise ValueError(
            "forgiveness_shaping_start_iter > 0 cannot be honored by "
            "main_multigpu (async actors have no global iteration counter). "
            "Set forgiveness_shaping_start_iter=0 and seed from a mature "
            "checkpoint, or use the synchronous main.py for staged shaping.")

    # spool + control files live in a private temp dir
    run_dir = tempfile.mkdtemp(prefix="azspool_")
    spool_dir = os.path.join(run_dir, "spool")
    os.makedirs(spool_dir, exist_ok=True)
    pub_path = os.path.join(run_dir, "actor_weights.pt")
    stop_path = os.path.join(run_dir, "STOP")

    # resume (checkpoints are saved with clean, uncompiled keys)
    latest = os.path.join(cfg["checkpoint_dir"], "latest.pt")
    step = 0
    if os.path.exists(latest):
        ck = torch.load(latest, map_location=learner_device, weights_only=False)
        missing, unexpected = net.load_state_dict(ck["model_state"],
                                                  strict=False)
        if missing:
            print(f"  note: {len(missing)} params freshly initialised "
                  f"(e.g. {missing[0]}) -- expected when resuming a "
                  f"pre-forgiveness-head checkpoint; the backbone is loaded.")
        try:
            if "optim_state" in ck:
                optimiser.load_state_dict(ck["optim_state"])
            if "forgiveness_optim_state" in ck:
                forgiveness_optimiser.load_state_dict(ck["forgiveness_optim_state"])
        except Exception as e:
            print(f"  note: optimizer state incompatible with the split "
                  f"param groups ({e}); optimizers start fresh.")
        step = ck.get("train_step", ck.get("iteration", 0) * cfg["train_batches"])
        print(f"resumed at train_step {step}")

    _atomic_torch_save({"model_state": net.state_dict()}, pub_path)  # before actors

    # metrics
    header = ["train_step", "iteration", "buffer", "consumed_total",
              "replay_ratio",
              "loss_total", "loss_policy", "loss_value", "loss_forgiveness",
              "forgiveness_R2", "forgiveness_tvar",
              "lr", "forgiveness_lr", "wall_sec"]
    metrics_path = os.path.join(cfg["checkpoint_dir"], cfg["metrics_file"])
    mf, writer, metrics_path = open_metrics(metrics_path, header)

    ctx = mp.get_context("spawn")
    threads = max(1, (os.cpu_count() or n_actors) // max(1, n_actors))
    procs = []
    for g in actor_gpus:
        p = ctx.Process(target=_actor_loop,
                        args=(g, threads, pub_path, spool_dir, stop_path,
                              games_per_chunk, max_pending, cfg))
        p.start()
        procs.append(p)

    def actors_alive():
        return any(p.is_alive() for p in procs)

    def spool_pending():
        return len(glob.glob(os.path.join(spool_dir, SPOOL_GLOB))) > 0

    consumed_total = 0                        # examples drained THIS RUN
    start_step = step                         # per-run accounting baseline: a
                                              # resume restores `step` but not
                                              # `consumed_total`, so the ratio
                                              # must be computed on deltas or the
                                              # throttle would deadlock on resume
    last_publish = step
    last_ckpt = step
    throttle_announced = False
    t_start = time.time()
    try:
        while step < total_train_steps:
            consumed_total += _drain_spool(spool_dir, buffer)

            # ---- warmup floor: never train on a token buffer ----
            if len(buffer) < min_buffer:
                if not actors_alive() and not spool_pending():
                    raise RuntimeError("all actors died before warmup -- check GPU "
                                       "memory / lower --actors-per-gpu")
                time.sleep(0.2)
                continue

            # ---- replay-ratio throttle: don't outrun the actors ----
            # Train only while samples trained (this run) stay within
            # target_ratio x samples generated (this run); otherwise sleep and
            # keep draining. This is what keeps the effective replay ratio at a
            # healthy ~target_ratio instead of letting the learner grind the
            # same stale buffer hundreds of times.
            if target_ratio > 0:
                trained_this_run = (step - start_step) * cfg["batch_size"]
                if trained_this_run >= target_ratio * max(consumed_total, 1):
                    if not actors_alive() and not spool_pending():
                        raise RuntimeError("all actors died and the spool is "
                                           "empty -- no more data will arrive")
                    if not throttle_announced:
                        print(f"  [throttle] waiting for self-play data "
                              f"(trained {trained_this_run}, generated "
                              f"{consumed_total}, target ratio {target_ratio})",
                              flush=True)
                        throttle_announced = True
                    time.sleep(0.5)
                    continue
                throttle_announced = False

            lr = cosine_lr(step + 1, cfg["lr"], cfg["lr_min"], total_train_steps)
            for grp in optimiser.param_groups:
                grp["lr"] = lr
            forgiveness_lr = cosine_lr(step + 1, cfg["forgiveness_lr"],
                                cfg.get("forgiveness_lr_min", cfg["forgiveness_lr"]),
                                total_train_steps)
            for grp in forgiveness_optimiser.param_groups:
                grp["lr"] = forgiveness_lr

            losses = train_epoch(net, buffer, optimiser, learner_device,
                                 batches=train_block, batch_size=cfg["batch_size"],
                                 aux_forgiveness=cfg["forgiveness_targets"],
                                 forgiveness_weight=cfg.get("forgiveness_loss_weight", 0.5),
                                 forgiveness_optimiser=forgiveness_optimiser,
                                 scaler=scaler)
            step += train_block

            if step - last_publish >= publish_every_steps:
                _atomic_torch_save({"model_state": net.state_dict()}, pub_path)
                last_publish = step
                it = step // cfg["train_batches"]
                ratio = ((step - start_step) * cfg["batch_size"]
                         / max(consumed_total, 1))
                writer.writerow([step, it, len(buffer), consumed_total,
                                 f"{ratio:.2f}",
                                 f"{losses['total']:.6f}", f"{losses['policy']:.6f}",
                                 f"{losses['value']:.6f}", f"{losses['forgiveness']:.6f}",
                                 f"{losses['forgiveness_R2']:.4f}", f"{losses['forgiveness_tvar']:.5f}",
                                 f"{lr:.3e}", f"{forgiveness_lr:.3e}",
                                 f"{time.time()-t_start:.1f}"])
                mf.flush()
                print(f"  step {step}/{total_train_steps}  buf {len(buffer)}  "
                      f"consumed {consumed_total}  ratio {ratio:.1f}  "
                      f"loss {losses['total']:.4f} "
                      f"(p {losses['policy']:.3f} v {losses['value']:.3f} "
                      f"e {losses['forgiveness']:.3f} R2 {losses['forgiveness_R2']:+.2f})  "
                      f"lr {lr:.2e}",
                      flush=True)

            if step - last_ckpt >= checkpoint_every_steps:
                it = step // cfg["train_batches"]
                ckpt = {"iteration": it, "train_step": step,
                        "model_state": net.state_dict(),
                        "optim_state": optimiser.state_dict(),
                        "forgiveness_optim_state": forgiveness_optimiser.state_dict(),
                        "config": cfg}
                _atomic_torch_save(ckpt, latest)
                _atomic_torch_save(ckpt, os.path.join(cfg["checkpoint_dir"],
                                                      f"net_iter{it}.pt"))
                last_ckpt = step
                print(f"  checkpoint net_iter{it}.pt (step {step})", flush=True)
    except KeyboardInterrupt:
        print("\ninterrupted -- stopping actors and saving", flush=True)
    finally:
        open(stop_path, "w").close()          # tell actors to exit
        for p in procs:
            p.join(timeout=30)
        for p in procs:
            if p.is_alive():
                p.terminate()
        it = step // cfg["train_batches"]
        ckpt = {"iteration": it, "train_step": step,
                "model_state": net.state_dict(),
                "optim_state": optimiser.state_dict(),
                "forgiveness_optim_state": forgiveness_optimiser.state_dict(),
                "config": cfg}
        _atomic_torch_save(ckpt, latest)
        _atomic_torch_save(ckpt, os.path.join(cfg["checkpoint_dir"], f"net_iter{it}.pt"))
        mf.close()
        for f in glob.glob(os.path.join(spool_dir, "*")):
            try: os.remove(f)
            except OSError: pass
        for pth in (pub_path, stop_path):
            try: os.remove(pth)
            except OSError: pass
        try:
            os.rmdir(spool_dir); os.rmdir(run_dir)
        except OSError:
            pass
        print(f"done at train_step {step}; final checkpoint net_iter{it}.pt "
              f"({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Async multi-GPU self-play training")
    ap.add_argument("--actors-per-gpu", type=int, default=2,
                    help="self-play processes per GPU (each uses ~one CPU core)")
    ap.add_argument("--gpus", default=None, help="comma-separated ids e.g. 0,1,2,3")
    ap.add_argument("--dedicate-learner-gpu", action="store_true",
                    help="reserve gpus[0] for the learner (default: share it)")
    ap.add_argument("--train-block", type=int, default=8,
                    help="gradient steps per learner round between drains/publishes")
    ap.add_argument("--games-per-chunk", type=int, default=None,
                    help="games an actor plays per emitted file (default: concurrency; "
                         "smaller values smooth data arrival under the throttle)")
    ap.add_argument("--total-train-steps", type=int, default=None,
                    help="default: loop_iterations * train_batches")
    ap.add_argument("--target-ratio", type=float, default=8.0,
                    help="max samples trained per sample generated (per run). "
                         "The learner sleeps when ahead of this. 0 disables "
                         "(old unthrottled behavior). Healthy range ~4-16.")
    ap.add_argument("--min-buffer", type=int, default=None,
                    help="examples required before the first gradient step "
                         "(default: buffer_capacity // 4)")
    args = ap.parse_args()
    gpus = tuple(int(x) for x in args.gpus.split(",")) if args.gpus else None
    main(gpus=gpus, actors_per_gpu=args.actors_per_gpu,
         dedicate_learner_gpu=args.dedicate_learner_gpu,
         train_block=args.train_block, games_per_chunk=args.games_per_chunk,
         total_train_steps=args.total_train_steps,
         target_ratio=args.target_ratio, min_buffer=args.min_buffer)

        
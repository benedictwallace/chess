"""
Single-process, leaf-batched checkpoint Elo scoring.

Same schedule, same Bradley-Terry Elo fit, and same resumable cache as
score_elo.py / score_elo_parallel.py -- only the way games are PLAYED changes.

score_elo.py plays one game at a time (batch-1 GPU forwards). score_elo_parallel
spreads games over CPU worker processes feeding a central GPU batcher, but each
worker blocks on its pipe after submitting ONE position, so at most `workers`
evaluations are ever in flight and the cores idle on the queue/GPU/pipe round
trip -- it is latency-bound, not compute-bound.

Here, as in training/self_play_batched.py, many games run CONCURRENTLY in one
process and their MCTS leaf evaluations are batched into single GPU forwards.
No IPC, nothing pickled. Two things differ from self-play:

  * Two nets per game. When an agent searches for its move it uses ITS OWN net
    for every leaf in that search (exactly as arena.NeuralAgent does). Different
    concurrent games -- and the two colours within one game -- may be searching
    with different nets, so each round we GROUP the pending leaves by the
    searching net and run one batched forward per net. Because the schedule
    emits games grouped by pairing, only ~2 nets are ever in flight, so the
    batches stay large (just like score_elo_parallel's per-net batching, minus
    the IPC).

  * Anchor moves (random / material) take no network evaluation, so a game
    alternates between "neural search" rounds (batched) and instant anchor
    moves (played inline). A game is only in the active batched set while it is
    waiting on a neural search.

Eval cache: checkpoints are FIXED for the whole run, so the cache is keyed by
(net_id, exact position) and persists across every game and pairing -- openings
recur constantly. It stores legal-move priors + value (a few KB/entry), bounded
by cache_cap.

The core run_elo_matches_batched() takes injected eval_fns and anchor agents and
imports no torch, so it is unit-testable on CPU with fakes; main() lazy-imports
the torch / arena / score_elo pieces.
"""

import math
import numpy as np

from engine.gameEnv import Chess
from model.encoding import encode
from model.move_encoding import encodeMovePOV, NUM_ACTIONS


# --------------------------------------------------------------------------- #
# MCTS primitives (torch-free; mirror search.puct with add_noise always off)
# --------------------------------------------------------------------------- #
class Node:
    __slots__ = ("parent", "move", "prior", "children",
                 "visits", "value", "moverSign", "terminal", "expanded")

    def __init__(self, parent=None, move=None, prior=0.0):
        self.parent = parent
        self.move = move
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.moverSign = 0
        self.terminal = False
        self.expanded = False


def _softmax(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def _puct_score(child, parent, c):
    q = 0.0 if child.visits == 0 else child.value / child.visits
    u = c * child.prior * math.sqrt(parent.visits) / (1 + child.visits)
    return q + u


def _expand(node, priors, mover):
    sign = 1 if mover == "white" else -1
    for m, p in priors.items():
        child = Node(parent=node, move=m, prior=p)
        child.moverSign = sign
        node.children.append(child)
    node.expanded = True


def _backprop(path, leaf_value_white):
    for n in path:
        n.visits += 1
        n.value += leaf_value_white * n.moverSign


def select_move(visit_counts, temp):
    moves = list(visit_counts.keys())
    counts = np.array([visit_counts[m] for m in moves], dtype=np.float64)
    if temp <= 1e-6 or counts.sum() == 0:
        return moves[int(counts.argmax())]
    logits = counts ** (1.0 / temp)
    probs = logits / logits.sum()
    rng = np.random.default_rng()
    return moves[rng.choice(len(moves), p=probs)]


# --------------------------------------------------------------------------- #
# per-game state
# --------------------------------------------------------------------------- #
class _AGame:
    """One arena game. `white`/`black` are each either a net_id (str -> neural,
    searched with that net) or an anchor agent object exposing .select(env, ply)."""
    __slots__ = ("pidx", "a_is_white", "white", "black",
                 "env", "ply", "done", "a_score",
                 "root", "sims_done", "search_net")

    def __init__(self, pidx, a_is_white, white, black):
        self.pidx = pidx
        self.a_is_white = a_is_white
        self.white = white
        self.black = black
        self.env = Chess()
        self.env.reset()
        self.ply = 0
        self.done = False
        self.a_score = None
        self.root = None
        self.sims_done = 0
        self.search_net = None


# --------------------------------------------------------------------------- #
# core batched runner
# --------------------------------------------------------------------------- #
def run_elo_matches_batched(tickets, eval_fns, *, iterations=400, c=1.5,
                            opening_plies=8, opening_temp=1.0, max_plies=160,
                            concurrency=128, use_cache=True, cache_cap=250_000,
                            on_game_done=None):
    """
    tickets : list of (pidx, a_is_white, white_mover, black_mover)
              white_mover/black_mover is a net_id str (neural) or an anchor agent.
    eval_fns: dict net_id -> callable(planes_list) -> (logits[B,A], values[B]),
              mover-POV policy logits and mover-POV value in [-1, 1].
    on_game_done(pidx, a_score) is called as each game finishes (for resumable
              caching / progress).

    Returns list of (pidx, a_score) for every ticket (a_score in {0.0,0.5,1.0}).
    """
    concurrency = max(1, min(concurrency, len(tickets)))
    cache = {} if use_cache else None
    ticket_iter = iter(tickets)
    results = []

    def finalize(g):
        r = g.env.result()
        if r is None:
            r = g.env.adjudicate()
        white_score = 0.5 if r == 0 else (1.0 if r > 0 else 0.0)
        g.a_score = white_score if g.a_is_white else (1.0 - white_score)
        g.done = True

    def advance_until_neural(g):
        """Play instant anchor moves / detect game end until it's a neural
        mover's turn (then set up its search root) or the game finishes."""
        while True:
            if g.env.isTerminal() or g.ply >= max_plies:
                finalize(g)
                return
            side = g.env.board.sideToMove
            mover = g.white if side == "white" else g.black
            if isinstance(mover, str):          # neural -> start a search
                g.root = Node()
                g.root.moverSign = 0
                g.sims_done = 0
                g.search_net = mover
                return
            move = mover.select(g.env, g.ply)   # anchor -> instant move
            if move is None:
                finalize(g)
                return
            g.env.step(move)
            g.ply += 1

    def start_next():
        """Pull the next ticket, fast-forward to its first neural move. Returns
        an active (awaiting-search) game, or None when tickets are exhausted.
        Games that finish before any neural move are recorded immediately."""
        while True:
            try:
                pidx, a_is_white, w, b = next(ticket_iter)
            except StopIteration:
                return None
            g = _AGame(pidx, a_is_white, w, b)
            advance_until_neural(g)
            if g.done:
                results.append((g.pidx, g.a_score))
                if on_game_done is not None:
                    on_game_done(g.pidx, g.a_score)
                continue
            return g

    active = []
    while len(active) < concurrency:
        g = start_next()
        if g is None:
            break
        active.append(g)

    while active:
        # ---- one simulation per active game; group leaves by searching net ----
        batches = {}
        for g in active:
            node = g.root
            env = g.env.clone()
            path = [node]
            while node.expanded and not node.terminal and node.children:
                sqrt_pv = math.sqrt(node.visits)
                best = None
                best_score = -1e30
                for ch in node.children:
                    v = ch.visits
                    q = ch.value / v if v else 0.0
                    s = q + c * ch.prior * sqrt_pv / (1 + v)
                    if s > best_score:
                        best_score = s
                        best = ch
                node = best
                env.step(node.move)
                path.append(node)

            if node.terminal:
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                g.sims_done += 1
                continue
            legal = env.legalMoves()
            if not legal:
                node.terminal = True
                r = env.result()
                _backprop(path, r if r is not None else 0.0)
                g.sims_done += 1
                continue
            if env.isRepetition() or env.isFiftyMove():
                node.terminal = True
                _backprop(path, 0.0)
                g.sims_done += 1
                continue

            mover = env.board.sideToMove
            if cache is not None:
                key = (g.search_net, env.board.stateKey())
                hit = cache.get(key)
                if hit is not None:
                    priors, value = hit
                    _expand(node, priors, mover)
                    _backprop(path, value if mover == "white" else -value)
                    g.sims_done += 1
                    continue

            batches.setdefault(g.search_net, []).append(
                (g, node, env, legal, path, mover))

        # ---- one batched forward per distinct searching net ----
        for net_id, items in batches.items():
            planes = [encode(it[2].board) for it in items]
            logits_b, values_b = eval_fns[net_id](planes)
            for (g, node, env, legal, path, mover), logits, value in zip(
                    items, logits_b, values_b):
                idxs = [encodeMovePOV(m, mover) for m in legal]
                probs = _softmax(np.asarray(logits)[idxs])
                priors = {m: float(p) for m, p in zip(legal, probs)}
                value = float(value)
                if cache is not None and len(cache) < cache_cap:
                    cache[(net_id, env.board.stateKey())] = (priors, value)
                _expand(node, priors, mover)
                _backprop(path, value if mover == "white" else -value)
                g.sims_done += 1

        # ---- games whose search is complete pick a move; refill the pool ----
        still = []
        for g in active:
            if g.sims_done < iterations:
                still.append(g)
                continue
            visit_counts = {ch.move: ch.visits for ch in g.root.children}
            if not visit_counts:
                finalize(g)
            else:
                temp = opening_temp if g.ply < opening_plies else 0.0
                move = select_move(visit_counts, temp)
                g.env.step(move)
                g.ply += 1
                if g.env.isTerminal() or g.ply >= max_plies:
                    finalize(g)
                else:
                    advance_until_neural(g)     # play anchor moves / set next search
            if g.done:
                results.append((g.pidx, g.a_score))
                if on_game_done is not None:
                    on_game_done(g.pidx, g.a_score)
                ng = start_next()
                if ng is not None:
                    still.append(ng)
            else:
                still.append(g)
        active = still

    return results


# --------------------------------------------------------------------------- #
# torch evaluator (lazy import)
# --------------------------------------------------------------------------- #
def _make_eval_fn(net):
    import torch
    device = next(net.parameters()).device
    use_amp = (device.type == "cuda")

    def eval_fn(planes_list):
        net.eval()
        x = torch.from_numpy(np.stack(planes_list)).to(device)
        with torch.no_grad():
            if use_amp:
                with torch.autocast("cuda"):
                    policy_logits, value = net(x)
            else:
                policy_logits, value = net(x)
        return (policy_logits.float().cpu().numpy(),
                value.float().cpu().numpy().reshape(-1))

    return eval_fn


# --------------------------------------------------------------------------- #
# pairing -> tickets, and a run-a-set-of-pairings helper (torch-free; the eval
# functions and anchor builder are injected, so this is unit-testable on CPU and
# is the single body shared by the serial path and each parallel worker)
# --------------------------------------------------------------------------- #
def _build_tickets(pairings, games, mover_for, seed_base):
    """pairings: list of (global_pidx, a_name, b_name). Returns ticket list with
    colours alternating per game; seeds derived from the GLOBAL pidx so they (and
    therefore the games, and the cache) are identical regardless of worker count."""
    tickets = []
    for (gpidx, a, b) in pairings:
        for gi in range(games):
            a_is_white = (gi % 2 == 0)
            seed = seed_base + gpidx * games + gi
            white_name, black_name = (a, b) if a_is_white else (b, a)
            tickets.append((gpidx, a_is_white,
                            mover_for(white_name, seed),
                            mover_for(black_name, seed)))
    return tickets


def _run_pairings(pairings, games, eval_fns, mover_for, on_pairing, *,
                  iterations, c, opening_plies, opening_temp, max_plies,
                  concurrency, use_cache, cache_cap, seed_base,
                  on_each_game=None):
    """Play every game of `pairings` with the batched runner, aggregate per
    pairing, and call on_pairing(a, b, w, d, l, n) as each pairing completes."""
    names = {gpidx: (a, b) for (gpidx, a, b) in pairings}
    agg = {gpidx: {"w": 0, "d": 0, "l": 0, "n": 0} for (gpidx, a, b) in pairings}
    received = {gpidx: 0 for (gpidx, a, b) in pairings}

    def on_game_done(gpidx, a_score):
        rec = agg[gpidx]
        if a_score == 1.0:
            rec["w"] += 1
        elif a_score == 0.0:
            rec["l"] += 1
        else:
            rec["d"] += 1
        rec["n"] += 1
        received[gpidx] += 1
        if on_each_game is not None:
            on_each_game()
        if received[gpidx] == games:
            a, b = names[gpidx]
            on_pairing(a, b, rec["w"], rec["d"], rec["l"], rec["n"])

    tickets = _build_tickets(pairings, games, mover_for, seed_base)
    run_elo_matches_batched(
        tickets, eval_fns,
        iterations=iterations, c=c,
        opening_plies=opening_plies, opening_temp=opening_temp,
        max_plies=max_plies, concurrency=concurrency,
        use_cache=use_cache, cache_cap=cache_cap,
        on_game_done=on_game_done,
    )


# --------------------------------------------------------------------------- #
# parallel worker (one process; loads only the nets its pairings reference)
# --------------------------------------------------------------------------- #
def _elo_worker(pairings, spec, games, cfg, seed_base, result_queue):
    import random
    import torch
    from evaluation.arena import RandomAgent, MaterialAgent, load_net
    from model.network import ChessNet

    device = torch.device(cfg["device"])
    # load just the nets this worker's pairings need (deduped)
    needed = {nm for (_p, a, b) in pairings for nm in (a, b) if nm in spec}
    eval_fns = {}
    for nm in needed:
        s = spec[nm]
        net = ChessNet().to(device).eval() if s == "untrained" else load_net(s, device)
        eval_fns[nm] = _make_eval_fn(net)

    def mover_for(name, seed):
        if name in spec:
            return name
        if name == "random":
            return RandomAgent(random.Random(seed))
        if name == "material":
            return MaterialAgent(random.Random(seed))
        raise ValueError(f"unknown player {name!r}")

    def on_pairing(a, b, w, d, l, n):
        result_queue.put(("pairing", a, b, w, d, l, n))

    def on_each_game():
        result_queue.put(("game",))

    try:
        _run_pairings(
            pairings, games, eval_fns, mover_for, on_pairing,
            iterations=cfg["iterations"], c=cfg["c"],
            opening_plies=cfg["opening_plies"], opening_temp=cfg["opening_temp"],
            max_plies=cfg["max_plies"], concurrency=cfg["concurrency"],
            use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"],
            seed_base=seed_base, on_each_game=on_each_game,
        )
    finally:
        result_queue.put(("worker_done",))


def _chunk(seq, n):
    """Split into n roughly-equal CONTIGUOUS chunks (keeps each worker's pairings
    on a nearby range of checkpoints, so it loads few distinct nets)."""
    n = max(1, min(n, len(seq)))
    k, r = divmod(len(seq), n)
    out, i = [], 0
    for w in range(n):
        size = k + (1 if w < r else 0)
        out.append(seq[i:i + size])
        i += size
    return [c for c in out if c]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    import argparse
    import csv
    import os
    import random
    import sys
    import time
    import multiprocessing as mp

    import torch

    from evaluation.arena import RandomAgent, MaterialAgent, load_net
    from evaluation.score_elo import (
        discover_checkpoints, fit_elo, load_cache, append_cache,
    )
    from model.network import ChessNet

    ap = argparse.ArgumentParser(description="Single/multi-process batched checkpoint Elo scoring")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--anchors", default="random,material")
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--iterations", type=int, default=400,
                    help="PUCT sims/move. For RANKING, 64-100 is plenty (training uses 400).")
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--opening-plies", type=int, default=8)
    ap.add_argument("--opening-temp", type=float, default=1.0)
    ap.add_argument("--concurrency", type=int, default=128,
                    help="games run/batched simultaneously PER PROCESS")
    ap.add_argument("--workers", type=int, default=1,
                    help="processes to split pairings over (1 = single process). "
                         "Each worker loads only the nets its pairings need.")
    ap.add_argument("--stride", type=int, default=1,
                    help="test every Nth checkpoint (1 = all); final always kept")
    ap.add_argument("--round-robin", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--cache-cap", type=int, default=250_000)
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(args.device if args.device
                          else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpts = discover_checkpoints(args.ckpt_dir)
    if not ckpts:
        print(f"no net_iter*.pt found in {args.ckpt_dir}")
        return
    if args.stride > 1:
        kept = ckpts[::args.stride]
        if ckpts[-1] not in kept:
            kept.append(ckpts[-1])
        ckpts = kept

    anchors = [a for a in args.anchors.split(",") if a]
    ckpt_names = [f"iter{it}" for it, _ in ckpts]
    spec = {f"iter{it}": os.path.join(args.ckpt_dir, f"net_iter{it}.pt") for it, _ in ckpts}
    players = anchors + ckpt_names
    name_idx = {n: i for i, n in enumerate(players)}

    schedule = []
    for cn in ckpt_names:
        for a in anchors:
            schedule.append((cn, a))
    if args.round_robin:
        for x in range(len(ckpt_names)):
            for y in range(x + 1, len(ckpt_names)):
                schedule.append((ckpt_names[x], ckpt_names[y]))
    else:
        for x in range(len(ckpt_names) - 1):
            schedule.append((ckpt_names[x], ckpt_names[x + 1]))

    cache_path = os.path.join(args.ckpt_dir, "elo_matches.csv")
    match_cache = load_cache(cache_path)

    cached, pending = [], []
    for (a, b) in schedule:
        if (a, b) in match_cache:
            aw, dr, bw, g = match_cache[(a, b)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        elif (b, a) in match_cache:
            bw, dr, aw, g = match_cache[(b, a)]
            cached.append((name_idx[a], name_idx[b], aw + 0.5 * dr, g))
        else:
            pending.append((a, b))

    print(f"{len(ckpts)} checkpoints, anchors={anchors}, {args.games} games/match, "
          f"{args.iterations} sims/move, concurrency={args.concurrency}, "
          f"workers={args.workers}, device={device}")
    print(f"{len(cached)} pairings cached, {len(pending)} to play "
          f"({len(pending) * args.games} games)")

    results = list(cached)
    # global pidx per pending pairing (keeps seeds/cache identical across worker counts)
    pending_idx = [(i, a, b) for i, (a, b) in enumerate(pending)]
    total_games = len(pending) * args.games
    t0 = time.time()

    def progress(done):
        frac = done / total_games if total_games else 1.0
        rate = done / (time.time() - t0) if time.time() > t0 else 0.0
        eta = (total_games - done) / rate if rate > 0 else 0.0
        sys.stdout.write(f"\r  games {done}/{total_games} ({frac*100:3.0f}%)  "
                         f"{rate:.1f}/s  eta {int(eta)}s   ")
        sys.stdout.flush()

    def record_pairing(a, b, w, d, l, n):
        append_cache(cache_path, a, b, w, d, l, n)
        results.append((name_idx[a], name_idx[b], w + 0.5 * d, n))

    if pending:
        cfg = dict(iterations=args.iterations, c=args.c,
                   opening_plies=args.opening_plies, opening_temp=args.opening_temp,
                   max_plies=args.max_plies, concurrency=args.concurrency,
                   use_cache=not args.no_cache, cache_cap=args.cache_cap,
                   device=str(device))

        if args.workers <= 1:
            # ---- single process ----
            needed = {nm for (_p, a, b) in pending_idx for nm in (a, b) if nm in spec}
            eval_fns = {}
            for nm in needed:
                s = spec[nm]
                eval_fns[nm] = _make_eval_fn(
                    ChessNet().to(device).eval() if s == "untrained" else load_net(s, device))
            print(f"loaded {len(eval_fns)} distinct nets onto {device}")

            def mover_for(name, seed):
                if name in spec:
                    return name
                if name == "random":
                    return RandomAgent(random.Random(seed))
                if name == "material":
                    return MaterialAgent(random.Random(seed))
                raise ValueError(f"unknown player {name!r}")

            done = [0]
            def on_each():
                done[0] += 1; progress(done[0])
            progress(0)
            _run_pairings(pending_idx, args.games, eval_fns, mover_for, record_pairing,
                          iterations=cfg["iterations"], c=cfg["c"],
                          opening_plies=cfg["opening_plies"], opening_temp=cfg["opening_temp"],
                          max_plies=cfg["max_plies"], concurrency=cfg["concurrency"],
                          use_cache=cfg["use_cache"], cache_cap=cfg["cache_cap"],
                          seed_base=args.seed, on_each_game=on_each)
            print()
        else:
            # ---- multi process: contiguous pairing chunks, each worker self-contained ----
            chunks = _chunk(pending_idx, args.workers)
            print(f"splitting {len(pending)} pairings over {len(chunks)} workers")
            ctx = mp.get_context("spawn")
            result_queue = ctx.Queue()
            procs = []
            for ch in chunks:
                p = ctx.Process(target=_elo_worker,
                                args=(ch, spec, args.games, cfg, args.seed, result_queue))
                p.start()
                procs.append(p)

            done = 0
            finished_workers = 0
            progress(0)
            while finished_workers < len(chunks):
                msg = result_queue.get()
                if msg[0] == "game":
                    done += 1
                    progress(done)
                elif msg[0] == "pairing":
                    _, a, b, w, d, l, n = msg
                    record_pairing(a, b, w, d, l, n)
                elif msg[0] == "worker_done":
                    finished_workers += 1
            for p in procs:
                p.join()
            print()

    pin = "random" if "random" in players else players[0]
    elo = fit_elo(players, results, pin=pin)

    print(f"\n{'='*46}\n  Elo ratings ({pin} = 0)\n{'='*46}")
    for a in anchors:
        print(f"  {a:14} {elo[a]:+7.0f}")
    print("  " + "-" * 30)
    rows = []
    for it, _ in ckpts:
        name = f"iter{it}"
        print(f"  {name:14} {elo[name]:+7.0f}")
        rows.append((it, elo[name]))

    out_csv = os.path.join(args.ckpt_dir, "elo_ratings.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iteration", "elo"])
        for it, e in rows:
            w.writerow([it, f"{e:.1f}"])
    print(f"\n  wrote {out_csv}")

    if rows:
        es = [e for _, e in rows]
        lo, hi = min(es), max(es)
        span = (hi - lo) or 1.0
        print(f"\n  Elo vs iteration  ({lo:+.0f} .. {hi:+.0f})")
        for it, e in rows:
            bar = "#" * int(1 + 40 * (e - lo) / span)
            print(f"  {it:>4} {bar}")


if __name__ == "__main__":
    main()
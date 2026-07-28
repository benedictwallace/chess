"""
Score checkpoints against EXTERNAL fixed-strength anchors (Stockfish via UCI)
and fit everything -- including your existing internal matches -- onto the one
Elo scale you already use (random pinned at 0).

Why this exists: the internal anchors (random / material) are swept 60-0 by
iteration ~150, so beyond that point elo_ratings.csv is chained purely off
noisy adjacent-checkpoint matches and the scale drifts. Stockfish at a fixed
skill level / node budget never changes between runs, so it is an anchor that
does not saturate: any real strength change shows up as a score change against
it, with a proper confidence interval.

Design notes
------------
* A UCIAgent duck-types evaluation.arena's agent interface (.select(env, ply)),
  so it drops into the same match machinery as your neural players. It keeps a
  python-chess board synced move-by-move (via .new_game()/.observe() hooks in
  this file's play_game), so the engine sees the FULL move history -- it knows
  about repetitions instead of being fed an amnesiac FEN every ply.
* Game rules/terminality are decided by YOUR env (engine.gameEnv.Chess), same
  as training and the internal arena, so numbers are comparable. The
  python-chess board is only a mirror used to talk to the UCI engine.
* Results are cached to a CSV in the same (a,b,a_wins,draws,b_wins,games)
  format as elo_matches.csv, so a long gauntlet is resumable, and the fit can
  merge internal + external matches into one Bradley-Terry solve.
* torch / evaluation.arena are imported lazily: `--selftest` and `--fit-only`
  run on a torch-free machine (e.g. a CPU box that just has stockfish).

Prereqs:  pip install chess      and a stockfish binary
          (apt: /usr/games/stockfish; or point --engine at any UCI engine)

Usage examples
--------------
# 1) sanity-check the engine bridge (no torch, no checkpoints needed):
python score_elo_external.py --selftest --engine /usr/games/stockfish

# 2) gauntlet: every milestone checkpoint vs the default anchor ladder,
#    60 games each, resumable, then a joint fit with your internal matches:
python score_elo_external.py --ckpt-dir checkpoints --games 60 \
    --engine /usr/games/stockfish --merge-internal elo_matches.csv

# 3) just a few checkpoints, more games on one anchor:
python score_elo_external.py --ckpt-dir checkpoints --iters 307,458,600 \
    --anchors sf:skill=1,nodes=200 --games 200

# 4) re-fit from cached CSVs without playing anything:
python score_elo_external.py --fit-only --merge-internal elo_matches.csv
"""

import argparse
import csv
import glob
import math
import os
import random
import re
import sys
import time

import chess
import chess.engine

from engine.gameEnv import Chess
from engine.fen import square_to_alg

DEFAULT_ANCHORS = [
    # a ladder of fixed-strength opponents. Node caps (not movetime) make them
    # deterministic-ish and machine-independent; Skill Level adds blunders.
    # '+' separates options within a spec so player names stay comma-free
    # (they are written into a plain CSV); ',' also works on the CLI.
    # The nodes=1 rung exists so a weak net's rating is INTERPOLATED between
    # anchors it beats and anchors that beat it, instead of extrapolated off
    # a handful of wins (the -500 vs -430 jitter between adjacent checkpoints
    # in earlier runs was exactly that ill-conditioning).
    "sf:skill=0+nodes=1",
    "sf:skill=0+nodes=50",
    "sf:skill=1+nodes=200",
    "sf:skill=3+nodes=400",
    "sf:skill=5+nodes=800",
]

MATCHES_FILE = "checkpoints/elo_matches_external.csv"
RATINGS_FILE = "checkpoints/elo_ratings_external.csv"


def canon(spec: str) -> str:
    """Canonical, comma-free player name for an anchor spec (CSV-safe)."""
    return spec.replace(",", "+")


def _uci(move) -> str:
    """Project Move -> UCI string. Castling is already king-two-square (e1g1),
    promotions are 'Q'/'R'/'B'/'N' -> lowercase suffix, matching python-chess."""
    p = move.promotion.lower() if move.promotion else ""
    return f"{square_to_alg(move.fromSq)}{square_to_alg(move.toSq)}{p}"


# --------------------------------------------------------------------------- #
# UCI (Stockfish) agent
# --------------------------------------------------------------------------- #
class UCIAgent:
    """
    Arena-compatible player backed by a UCI engine.

    spec grammar:  sf:key=val,key=val  with keys
        skill     Stockfish "Skill Level" 0..20 (blunder-prone at low values)
        elo       UCI_LimitStrength target Elo (min ~1320 for stockfish 15+)
        nodes     fixed node budget per move   (preferred: hardware-independent)
        movetime  milliseconds per move        (alternative to nodes)
        depth     fixed depth per move
    """

    def __init__(self, spec: str, engine_path: str, name: str = None):
        opts, limit = self._parse(spec)
        self.name = name or spec
        self.engine_path = engine_path
        self._opts = opts
        self._limit = limit
        self._engine = None            # started lazily / restarted on crash
        self.mirror = chess.Board()    # kept in sync by play_game hooks

    @staticmethod
    def _parse(spec):
        body = spec.split(":", 1)[1] if ":" in spec else spec
        kv = {}
        for part in re.split(r"[,+]", body):
            if not part:
                continue
            k, _, v = part.partition("=")
            kv[k.strip()] = v.strip()
        opts, limit_kw = {}, {}
        if "skill" in kv:
            opts["Skill Level"] = int(kv["skill"])
        if "elo" in kv:
            opts["UCI_LimitStrength"] = True
            opts["UCI_Elo"] = int(kv["elo"])
        if "nodes" in kv:
            limit_kw["nodes"] = int(kv["nodes"])
        if "movetime" in kv:
            limit_kw["time"] = float(kv["movetime"]) / 1000.0
        if "depth" in kv:
            limit_kw["depth"] = int(kv["depth"])
        if not limit_kw:
            limit_kw["nodes"] = 200    # never let the engine think unbounded
        return opts, chess.engine.Limit(**limit_kw)

    # -- engine lifecycle ---------------------------------------------------
    def _start(self):
        self._engine = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        for k, v in self._opts.items():
            try:
                self._engine.configure({k: v})
            except chess.engine.EngineError as e:
                print(f"  [warn] {self.name}: engine rejected option {k}={v} ({e})")

    def close(self):
        if self._engine is not None:
            try:
                self._engine.quit()
            except chess.engine.EngineError:
                pass
            self._engine = None

    # -- game hooks (called by play_game) ------------------------------------
    def new_game(self):
        self.mirror = chess.Board()

    def observe(self, move):
        """Mirror EVERY move of the game (both sides) so the engine sees full
        history -- repetition-aware play instead of stateless FEN probing."""
        self.mirror.push(chess.Move.from_uci(_uci(move)))

    # -- arena interface ------------------------------------------------------
    def select(self, env, ply):
        legal = env.legalMoves()
        if not legal:
            return None
        if self._engine is None:
            self._start()
        try:
            result = self._engine.play(self.mirror, self._limit)
        except chess.engine.EngineTerminatedError:
            print(f"  [warn] {self.name}: engine died; restarting")
            self._start()
            result = self._engine.play(self.mirror, self._limit)
        if result.move is None:                      # engine resigned/stalled
            return None
        want = result.move.uci()
        for m in legal:
            if _uci(m) == want:
                return m
        raise RuntimeError(
            f"{self.name} chose {want}, not legal in project env.\n"
            f"  mirror FEN: {self.mirror.fen()}\n"
            f"  env legal : {sorted(_uci(m) for m in legal)}\n"
            f"  (engine/env desync -- please report this position)"
        )


class _Seat:
    """Uniform wrapper so neural/random/material agents get no-op hooks."""
    def __init__(self, agent):
        self.agent = agent
    def new_game(self):
        if hasattr(self.agent, "new_game"):
            self.agent.new_game()
    def observe(self, move):
        if hasattr(self.agent, "observe"):
            self.agent.observe(move)
    def select(self, env, ply):
        return self.agent.select(env, ply)


# --------------------------------------------------------------------------- #
# Game / match (arena.play_game + observation hooks + optional sync check)
# --------------------------------------------------------------------------- #
def play_game(white_agent, black_agent, max_plies=300, check_sync=False):
    """Return white's score (1 / 0.5 / 0). Terminality by the PROJECT env."""
    white, black = _Seat(white_agent), _Seat(black_agent)
    env = Chess()
    env.reset()
    white.new_game()
    black.new_game()

    ref = chess.Board() if check_sync else None
    ply = 0
    while ply < max_plies:
        if env.isTerminal():
            break
        if check_sync:
            ours = sorted(_uci(m) for m in env.legalMoves())
            theirs = sorted(m.uci() for m in ref.legal_moves)
            assert ours == theirs, (
                f"legal-move mismatch at ply {ply}\nFEN {ref.fen()}\n"
                f"env-only : {sorted(set(ours) - set(theirs))}\n"
                f"chess-only: {sorted(set(theirs) - set(ours))}"
            )
        seat = white if env.board.sideToMove == "white" else black
        move = seat.select(env, ply)
        if move is None:
            break
        white.observe(move)
        black.observe(move)
        if check_sync:
            ref.push(chess.Move.from_uci(_uci(move)))
        env.step(move)
        ply += 1

    r = env.result()
    if r is None:                       # ply cap -> material adjudication,
        r = env.adjudicate()            # same rule as the internal arena
    if r == 0:
        return 0.5
    return 1.0 if r > 0 else 0.0


def match(agent_a, agent_b, games, max_plies=300, check_sync=False,
          verbose=True, label=""):
    wins = draws = losses = 0
    for g in range(games):
        a_white = (g % 2 == 0)
        if a_white:
            s = play_game(agent_a, agent_b, max_plies, check_sync)
        else:
            s = 1.0 - play_game(agent_b, agent_a, max_plies, check_sync)
        if s == 1.0:
            wins += 1
        elif s == 0.0:
            losses += 1
        else:
            draws += 1
        if verbose:
            score = (wins + 0.5 * draws) / (g + 1)
            print(f"    {label} game {g+1:>3}/{games} "
                  f"[A={'W' if a_white else 'B'}]  +{wins} ={draws} -{losses}"
                  f"  score={score:.3f}", flush=True)
    return wins, draws, losses


# --------------------------------------------------------------------------- #
# Match cache (same schema as elo_matches.csv -> resumable + mergeable)
# --------------------------------------------------------------------------- #
def load_matches(path):
    out = {}
    if os.path.exists(path):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                key = (row["a"], row["b"])
                out[key] = (int(row["a_wins"]), int(row["draws"]),
                            int(row["b_wins"]), int(row["games"]))
    return out


def append_match(path, a, b, w, d, l, n):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["a", "b", "a_wins", "draws", "b_wins", "games"])
        wr.writerow([a, b, w, d, l, n])


# --------------------------------------------------------------------------- #
# Elo fit: Bradley-Terry minorize-maximize, ties as half-wins, light prior.
# Same model as score_elo.fit_elo, re-implemented here so --fit-only runs
# without importing torch.
# --------------------------------------------------------------------------- #
def fit_elo(matches, pin="random", prior_games=2.0, steps=800):
    names = sorted({n for (a, b) in matches for n in (a, b)})
    idx = {n: i for i, n in enumerate(names)}
    P = len(names)
    wins = [0.0] * P
    pair_n = {}                                  # (i,j) -> games (i<j)
    for (a, b), (w, d, l, n) in matches.items():
        i, j = idx[a], idx[b]
        wins[i] += w + 0.5 * d
        wins[j] += l + 0.5 * d
        key = (min(i, j), max(i, j))
        pair_n[key] = pair_n.get(key, 0) + n
    # light prior: everyone drew `prior_games` vs a virtual average player,
    # keeps swept pairings (60-0) from sending ratings to +/- infinity.
    gamma = [1.0] * P
    for _ in range(steps):
        new = []
        for i in range(P):
            num = wins[i] + 0.5 * prior_games
            den = prior_games / (gamma[i] + 1.0)
            for (a, b), n in pair_n.items():
                if a == i:
                    den += n / (gamma[i] + gamma[b])
                elif b == i:
                    den += n / (gamma[i] + gamma[a])
            new.append(num / max(den, 1e-12))
        s = sum(new) / P
        gamma = [g / s for g in new]
    base = gamma[idx[pin]] if pin in idx else 1.0
    return {n: 400.0 * math.log10(gamma[idx[n]] / base) for n in names}


def wilson_elo_ci(w, d, l):
    """Score +/-1.96 SE mapped through the logistic Elo curve."""
    n = w + d + l
    if n == 0:
        return None
    s = (w + 0.5 * d) / n
    var = sum(((x - s) ** 2) * c for x, c in ((1, w), (0.5, d), (0, l))) / max(n - 1, 1)
    se = math.sqrt(var / n)
    def to_elo(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return 400.0 * math.log10(p / (1 - p))
    return to_elo(s), to_elo(max(s - 1.96 * se, 0.0)), to_elo(min(s + 1.96 * se, 1.0))


# --------------------------------------------------------------------------- #
# Checkpoint discovery / player construction
# --------------------------------------------------------------------------- #
def discover_checkpoints(ckpt_dir, iters=None):
    out = []
    for p in glob.glob(os.path.join(ckpt_dir, "net_iter*.pt")):
        m = re.search(r"net_iter(\d+)\.pt$", os.path.basename(p))
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    if iters:
        want = set(iters)
        out = [(i, p) for i, p in out if i in want]
    return out


def make_ckpt_agent(path, sims, c, device_str):
    """Lazy torch import so anchor-only / fit-only runs need no torch."""
    import torch
    from evaluation.arena import load_net, NeuralAgent
    device = torch.device(device_str if device_str else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    net = load_net(path, device)
    # opening_plies>0 gives game diversity (deterministic play would repeat
    # one game per colour); matches your internal arena's convention.
    return NeuralAgent(net, iterations=sims, c=c, opening_plies=8)


# --------------------------------------------------------------------------- #
# Self-test: random mover vs weakest stockfish, cross-checking legal moves
# between the project engine and python-chess EVERY ply.
# --------------------------------------------------------------------------- #
class _RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
    def select(self, env, ply):
        moves = env.legalMoves()
        return self.rng.choice(moves) if moves else None


def selftest(engine_path, games=4):
    print(f"selftest: random vs sf:skill=0,nodes=1  ({games} games, "
          f"per-ply legal-move cross-check vs python-chess)")
    sf = UCIAgent("sf:skill=0,nodes=1", engine_path, name="sf-selftest")
    try:
        w, d, l = match(_RandomAgent(), sf, games, check_sync=True,
                        verbose=True, label="selftest")
    finally:
        sf.close()
    print(f"selftest OK -- no desync. random vs stockfish: +{w} ={d} -{l}")
    print("(stockfish should win nearly every game; if random is winning, "
          "something is wrong with the bridge)")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="External (UCI) Elo gauntlet")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--iters", default="",
                    help="comma-separated iteration numbers (default: all milestones)")
    ap.add_argument("--anchors", default=",".join(DEFAULT_ANCHORS),
                    help="comma-separated anchor specs, e.g. 'sf:skill=0,nodes=50;sf:skill=3,nodes=400' "
                         "(separate multiple anchors with ';')")
    ap.add_argument("--engine", default=os.environ.get("STOCKFISH", "/usr/games/stockfish"),
                    help="path to the UCI engine binary")
    ap.add_argument("--games", type=int, default=60, help="games per (ckpt, anchor) pair")
    ap.add_argument("--sims", type=int, default=700,
                    help="PUCT sims/move for checkpoints. Default matches the "
                         "self-play full-search budget: benchmarking a net "
                         "trained for 700-sim targets at 100 sims (the old "
                         "default) understates it badly and is not comparable "
                         "across configs.")
    ap.add_argument("--c", type=float, default=1.5)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--device", default="", help="cuda | cpu (default: auto)")
    ap.add_argument("--matches-file", default=MATCHES_FILE)
    ap.add_argument("--ratings-file", default=RATINGS_FILE)
    ap.add_argument("--merge-internal", default="",
                    help="path to the existing elo_matches.csv to fit jointly "
                         "(puts external anchors on your random=0 scale)")
    ap.add_argument("--fit-only", action="store_true",
                    help="skip playing; just refit from cached CSVs")
    ap.add_argument("--selftest", action="store_true",
                    help="validate the engine bridge (no torch needed)")
    ap.add_argument("--check-sync", action="store_true",
                    help="cross-check legal moves vs python-chess during real games (slow)")
    args = ap.parse_args()

    if args.selftest:
        selftest(args.engine)
        return

    matches = load_matches(args.matches_file)

    if not args.fit_only:
        if not os.path.exists(args.engine):
            sys.exit(f"engine not found at {args.engine!r} -- install stockfish "
                     f"(apt install stockfish) or pass --engine")
        iters = [int(x) for x in args.iters.split(",") if x.strip()]
        ckpts = discover_checkpoints(args.ckpt_dir, iters or None)
        if not ckpts:
            sys.exit(f"no net_iter*.pt found in {args.ckpt_dir!r}")
        # anchors are ';'-separated; within a spec use '+' (or ',') between
        # options. 'sf:' boundaries also split, so plain comma lists work too.
        raw = args.anchors.replace(";", " ")
        raw = re.sub(r"[, ]*(sf:)", r" \1", raw)
        anchor_specs = [canon(s.strip().rstrip(",")) for s in raw.split() if s.strip()]

        print(f"{len(ckpts)} checkpoints x {len(anchor_specs)} anchors, "
              f"{args.games} games each  (cached pairs are skipped)")
        for it, path in ckpts:
            name = f"iter{it}"
            agent = None
            for spec in anchor_specs:
                aname = canon(spec)
                key = (name, aname)
                if key in matches or (aname, name) in matches:
                    print(f"  [cached] {name} vs {aname}")
                    continue
                if agent is None:      # load the net once per checkpoint
                    print(f"  loading {path}")
                    agent = make_ckpt_agent(path, args.sims, args.c, args.device)
                anchor = UCIAgent(spec, args.engine, name=canon(spec))
                print(f"  {name} vs {spec}")
                t0 = time.time()
                try:
                    w, d, l = match(agent, anchor, args.games,
                                    max_plies=args.max_plies,
                                    check_sync=args.check_sync,
                                    verbose=True, label=name)
                finally:
                    anchor.close()
                matches[key] = (w, d, l, args.games)
                append_match(args.matches_file, name, aname, w, d, l, args.games)
                ci = wilson_elo_ci(w, d, l)
                print(f"    -> +{w} ={d} -{l}  in {time.time()-t0:.0f}s  "
                      f"Elo diff {ci[0]:+.0f} [{ci[1]:+.0f}, {ci[2]:+.0f}]")

    # ---- joint fit ---------------------------------------------------------
    all_matches = dict(matches)
    pin = None
    if args.merge_internal:
        internal = load_matches(args.merge_internal)
        all_matches.update(internal)
        if any("random" in k for k in internal):
            pin = "random"
    if pin is None:
        # no internal file: pin the weakest anchor at 0 instead
        anchors = sorted({n for k in all_matches for n in k if n.startswith("sf:")})
        pin = anchors[0] if anchors else next(iter(all_matches))[0]

    if not all_matches:
        sys.exit("no matches to fit (play some games first, or pass --merge-internal)")

    elo = fit_elo(all_matches, pin=pin)
    rows = sorted(elo.items(),
                  key=lambda kv: (0, int(kv[0][4:])) if kv[0].startswith("iter")
                  else (1, kv[1]))
    print(f"\nElo (pin: {pin} = 0){' -- joint fit with internal matches' if args.merge_internal else ''}")
    print(f"{'player':<28}{'elo':>8}")
    with open(args.ratings_file, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["player", "elo"])
        for name, e in rows:
            print(f"{name:<28}{e:>8.1f}")
            wr.writerow([name, f"{e:.1f}"])
    print(f"\nwrote {args.ratings_file}  (matches cached in {args.matches_file})")


if __name__ == "__main__":
    main()
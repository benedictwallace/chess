"""
Evaluate any ease signal on any position.

Run an ease signal (frac-safe today; cliff / stability once they exist) on one
or more positions and print what it reports, so you can eyeball whether the
number matches the property you're trying to capture.

For each (position, signal) it shows:
  * the signal's LOCAL value at the position,
  * a `.explain()` breakdown if the signal provides one (e.g. frac-safe shows
    the move Qs, the best Q, the safety threshold, and how many moves clear it),
  * the top moves by visits with their search Q,
  * optionally the FUTURE-STATE RETURN target, computed by greedily rolling the
    game forward `--rollout-plies` plies (this is the actual training target).

The signal is derived from the SEARCH TREE, so values are only meaningful with a
trained --checkpoint. --random just verifies the pipeline runs.

Examples
--------
  # one position from a checkpoint
  python -m evaluation.eval_ease --checkpoint checkpoints_fracsafe/latest.pt \
         --fen "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"

  # the built-in suite, all signals, with the future-state return
  python -m evaluation.eval_ease --checkpoint checkpoints_fracsafe/latest.pt \
         --signal all --rollout-plies 8

  # just check it runs (values not meaningful)
  python -m evaluation.eval_ease --random --search-iters 50
"""

import argparse
import inspect
import sys

from engine.fen import env_from_fen, board_to_fen, square_to_alg
from training.ease_fracsafe import FracSafeEase

# register ease signals here; cliff / stability slot in as they're written
SIGNAL_REGISTRY = {
    "fracsafe": FracSafeEase,
}

# illustrative positions spanning the forgiving <-> sharp axis. Supply your own
# with --fen / --positions-file for the cases you actually care about.
DEFAULT_POSITIONS = [
    ("opening (balanced)",   "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
    ("open middlegame",      "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"),
    ("K+P endgame",          "8/8/8/4k3/8/4K3/4P3/8 w - - 0 1"),
]


# --------------------------------------------------------------------------- #
# signal construction (signal-agnostic: pass only the knobs a signal accepts)
# --------------------------------------------------------------------------- #
def build_signal(name, **knobs):
    if name not in SIGNAL_REGISTRY:
        raise SystemExit(f"unknown signal {name!r}; available: {', '.join(SIGNAL_REGISTRY)}")
    cls = SIGNAL_REGISTRY[name]
    params = inspect.signature(cls.__init__).parameters
    accepted = {k: v for k, v in knobs.items() if k in params and v is not None}
    return cls(**accepted)


def fmt_move(m):
    if m is None:
        return "-"
    if getattr(m, "castle", False):
        return "O-O" if m.toSq in (6, 62) else "O-O-O"
    s = square_to_alg(m.fromSq) + square_to_alg(m.toSq)
    if m.promotion:
        s += "=" + m.promotion
    return s


# --------------------------------------------------------------------------- #
# core: run search + signal on one position
# --------------------------------------------------------------------------- #
def analyze(env, net, signal, search_iters=400, c=1.5, topk=5, rollout_plies=0):
    from search.puct import search, select_move

    root, visit_counts = search(env, net, iterations=search_iters, c=c, add_noise=False)
    local = signal.local(root)

    kids = sorted(root.children, key=lambda ch: ch.visits, reverse=True)
    top = [(fmt_move(k.move), k.visits, (k.value / k.visits if k.visits else 0.0))
           for k in kids[:topk]]

    info = {"local": local, "top": top, "n_children": len(root.children)}
    if hasattr(signal, "explain"):
        info["explain"] = signal.explain(root)

    if rollout_plies and rollout_plies > 0:
        info["return"] = _rollout_return(env, net, signal, search_iters, rollout_plies, c)
    return info


def _rollout_return(env, net, signal, search_iters, plies, c):
    """Greedy rollout: collect local ease at plies 0..N, then the discounted
    future-state return at the start position. Returns (target, mask)."""
    from search.puct import search, select_move

    locals_ = []
    e = env.clone()
    for i in range(plies + 1):
        if e.isTerminal():
            break
        root, vc = search(e, net, iterations=search_iters, c=c, add_noise=False)
        if not vc:
            break
        locals_.append(signal.local(root))
        if i < plies:
            e.step(select_move(vc, temp=0.0))
    pairs = signal.returns(locals_)
    return pairs[0] if pairs else (None, 0.0)


# --------------------------------------------------------------------------- #
# net loading (torch imported lazily so the rest works without it)
# --------------------------------------------------------------------------- #
def _device(name):
    import torch
    if name in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def load_net(path, device=None):
    import torch
    from model.network import ChessNet
    dev = _device(device)
    ckpt = torch.load(path, map_location=dev, weights_only=False)
    cfg = ckpt.get("config", {})
    # aux_ease=False: this tool reads the signal off the SEARCH TREE, not the
    # net's ease head, so we only need policy+value. strict=False then ignores
    # any ease_head.* keys a frac-safe checkpoint carries.
    net = ChessNet(channels=cfg.get("channels", 64), num_blocks=cfg.get("num_blocks", 5))
    state = ckpt["model_state"]
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    return net.to(dev).eval()


def make_random_net(channels, num_blocks, device=None):
    from model.network import ChessNet
    dev = _device(device)
    return ChessNet(channels=channels, num_blocks=num_blocks).to(dev).eval()


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _fmt_local(v):
    return "  n/a" if v is None else f"{v:.3f}"


def print_position(label, fen, env, results, topk):
    print("=" * 72)
    print(f"{label}")
    print(f"  fen: {fen}")
    print(f"  side to move: {env.board.sideToMove}")
    for sig_name, info in results.items():
        print(f"\n  [{sig_name}]  local = {_fmt_local(info['local'])}", end="")
        if "return" in info:
            tgt, mask = info["return"]
            print(f"   future-return = {_fmt_local(tgt) if mask else 'n/a'}", end="")
        print()
        ex = info.get("explain")
        if ex and ex.get("defined"):
            print(f"      best Q {ex['q_best']:+.3f}  threshold {ex['threshold']:+.3f}  "
                  f"safe {ex['n_safe']}/{ex['n_considered']}")
        elif ex is not None:
            print(f"      undefined (only {ex.get('n_considered', 0)} well-visited move(s))")
        print(f"      top {min(topk, len(info['top']))} by visits:", end="  ")
        print("  ".join(f"{mv}(n={n},q={q:+.2f})" for mv, n, q in info["top"]))


def print_summary(rows, signal_names):
    print("\n" + "=" * 72)
    print("SUMMARY  (local ease value per position x signal)")
    w = max(len(lbl) for lbl, _ in rows) if rows else 8
    header = "  " + "position".ljust(w) + "  " + "  ".join(s.rjust(8) for s in signal_names)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for lbl, res in rows:
        cells = "  ".join(_fmt_local(res[s]["local"]).rjust(8) for s in signal_names)
        print("  " + lbl.ljust(w) + "  " + cells)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _load_positions(args):
    positions = []
    if args.positions_file:
        with open(args.positions_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "|" in line:                       # "FEN | label"
                    fen, lbl = line.split("|", 1)
                    positions.append((lbl.strip(), fen.strip()))
                else:
                    positions.append((line, line))
    for fen in (args.fen or []):
        positions.append((fen, fen))
    if not positions:
        positions = list(DEFAULT_POSITIONS)
    return positions


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate ease signals on positions.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--checkpoint", help="net checkpoint (.pt) to evaluate with")
    src.add_argument("--random", action="store_true",
                     help="use an untrained net (pipeline check only; values not meaningful)")
    ap.add_argument("--fen", action="append", help="a position FEN (repeatable)")
    ap.add_argument("--positions-file", help="file of 'FEN | label' lines")
    ap.add_argument("--signal", action="append",
                    help=f"signal name (repeatable), or 'all'. available: {', '.join(SIGNAL_REGISTRY)}")
    ap.add_argument("--search-iters", type=int, default=400)
    ap.add_argument("--rollout-plies", type=int, default=0,
                    help="if >0, also report the future-state return over this many plies")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--c", type=float, default=1.5)
    # signal knobs (passed only to signals that accept them)
    ap.add_argument("--min-visits", type=int, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--gamma", type=float, default=None)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--num-blocks", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)

    # signals
    names = args.signal or ["fracsafe"]
    if "all" in names:
        names = list(SIGNAL_REGISTRY)
    signals = {n: build_signal(n, min_visits=args.min_visits,
                               delta=args.delta, gamma=args.gamma) for n in names}

    # net
    if args.checkpoint:
        net = load_net(args.checkpoint, args.device)
        print(f"loaded net from {args.checkpoint}")
    elif args.random:
        net = make_random_net(args.channels, args.num_blocks, args.device)
        print("WARNING: --random net is untrained; ease values are NOT meaningful "
              "(use this only to check the pipeline runs).")
    else:
        raise SystemExit("provide --checkpoint PATH (recommended) or --random")

    positions = _load_positions(args)
    rows = []
    for label, fen in positions:
        env = env_from_fen(fen)
        results = {n: analyze(env, net, sig, search_iters=args.search_iters,
                              c=args.c, topk=args.topk, rollout_plies=args.rollout_plies)
                   for n, sig in signals.items()}
        print_position(label, fen, env, results, args.topk)
        rows.append((label, results))

    if len(rows) > 1 or len(signals) > 1:
        print_summary(rows, list(signals))


if __name__ == "__main__":
    main()
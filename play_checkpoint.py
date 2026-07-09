"""
Play against a trained checkpoint on a POPUP BOARD, with the search's visit
distribution and the ease/forgiveness statistics shown next to the board.

Default is a tkinter window (ships with Python -- no extra installs): click a
piece, click its destination. The right-hand panel shows, for every search:

  * a headline readout: eval (mover POV), F_local, F_tree
  * the visit distribution over root moves (visits, %, Q chooser-POV, prior;
    '*' marks moves that met the forced-visit floor, i.e. trustworthy Qs)
  * the ease block: the two local forgiveness values -- the action gap
    (F_gap = exp(-gap/tau)) and the normalised Q-entropy (with its effective
    move count exp(H)) -- plus the recursive F_tree from the project
    formulation

Buttons: Analyse (search YOUR position -- suggestion drawn as an arrow),
Undo (takes back a full move pair), New game, Flip board.

Every search (engine moves and Analyse alike) is the SAME forced-visit,
no-Dirichlet-noise, fp32 search as probe_ease.py, so numbers here are directly
comparable to your probe CSVs. The engine runs on a background thread; the
board stays responsive while it thinks.

--terminal restores the old text-mode loop (same commands as before: moves as
e2e4 / e7e8q, analyse, moves, fen, undo, new, resign/quit).

tau: pass the value probe_ease.py calibrated for this checkpoint (--tau). The
default 0.05 is a placeholder -- rankings between positions don't depend on
it, but calibrated tau puts the median position at F ~ 0.5.

Usage (from the repo root):
    python play_checkpoint.py --checkpoint checkpoints/07_07/net_iter600.pt
    python play_checkpoint.py --checkpoint ... --color black --sims 400 --tau 0.061
    python play_checkpoint.py --checkpoint ... --terminal
"""

import argparse
import math
import queue
import random
import sys
import threading

import torch

from engine.gameEnv import Chess
from evaluation.arena import load_net
from evaluation.probe_ease import (
    _ProbeItem, search_positions, root_q_vector, _uci, _probe_eval_fn,
)
from search.ease import ease_from_qs, tree_forgiveness

try:
    from engine.fen import board_to_fen
except ImportError:
    from fen import board_to_fen


PIECES = {("white", "pawn"): "P", ("white", "knight"): "N",
          ("white", "bishop"): "B", ("white", "rook"): "R",
          ("white", "queen"): "Q", ("white", "king"): "K",
          ("black", "pawn"): "p", ("black", "knight"): "n",
          ("black", "bishop"): "b", ("black", "rook"): "r",
          ("black", "queen"): "q", ("black", "king"): "k"}

UNICODE = {("white", "king"): "\u2654", ("white", "queen"): "\u2655",
           ("white", "rook"): "\u2656", ("white", "bishop"): "\u2657",
           ("white", "knight"): "\u2658", ("white", "pawn"): "\u2659",
           ("black", "king"): "\u265A", ("black", "queen"): "\u265B",
           ("black", "rook"): "\u265C", ("black", "bishop"): "\u265D",
           ("black", "knight"): "\u265E", ("black", "pawn"): "\u265F"}

LIGHT, DARK = "#F0D9B5", "#B58863"
SEL_COLOR, LAST_COLOR, DOT_COLOR = "#F7EC74", "#CDD26A", "#4a7a3a"


def piece_at(board, sq):
    for key, bb in board.bb.items():
        if (bb >> sq) & 1:
            return key
    return None


# --------------------------------------------------------------------------- #
# search + formatting shared by GUI and terminal
# --------------------------------------------------------------------------- #
def run_search(env, eval_fn, args):
    item = _ProbeItem(env.clone(), -1)
    force_eff = search_positions(
        [item], eval_fn, sims=args.sims, c=args.c,
        force_m=args.force_m, force_n=args.force_n, verbose=False)
    return item.root, force_eff


def format_distribution(root, force_eff, top=10):
    kids = sorted(root.children, key=lambda c: -c.visits)
    tot = max(1, sum(c.visits for c in kids))
    lines = [f"  {'move':<7}{'visits':>7}{'%':>7}{'Q':>8}{'prior':>8}"]
    for c in kids[:top]:
        if c.visits == 0:
            continue
        q = c.value / c.visits
        mark = "*" if c.visits >= force_eff > 0 else " "
        lines.append(f"{mark} {_uci(c.move):<7}{c.visits:>7}"
                     f"{100.0 * c.visits / tot:>6.1f}%{q:>+8.3f}{c.prior:>8.3f}")
    hidden = sum(1 for c in kids if c.visits == 0)
    if hidden:
        lines.append(f"  ({hidden} legal moves unvisited; "
                     f"* = met the {force_eff}-visit floor)")
    return "\n".join(lines)


def ease_summary(root, force_eff, tau, gamma):
    """Returns (stats dict or None). stats: st fields + ease fields + f_tree."""
    st = root_q_vector(root, force_eff)
    if st is None:
        return None
    ease = ease_from_qs(st["qs"], tau)
    st.update(ease)
    st["f_tree"] = tree_forgiveness(root, gamma, tau)
    return st


def format_ease(st, gamma):
    ftree = f"{st['f_tree']:.3f}" if st["f_tree"] is not None else "n/a"
    return (f"  eval (mover POV) {st['v_root']:+.3f} | gap {st['gap']:.3f}"
            f" -> F_gap {st['F_gap']:.3f}\n"
            f"  H_ease {st['ease_entropy']:.3f} | effA {st['eff_actions']:.2f}"
            f" | F_tree (g={gamma}) {ftree}")


def parse_move(text, legal):
    text = text.strip().lower()
    for m in legal:
        if _uci(m).lower() == text:
            return m
    hits = [m for m in legal if _uci(m).lower().startswith(text)]
    if len(hits) == 1:
        return hits[0]
    return None


def game_over(env):
    return env.isTerminal() or env.isRepetition() or env.isFiftyMove()


def result_text(env, human_white):
    r = env.result()
    if r is None or r == 0:
        return "Draw."
    winner = "White" if r > 0 else "Black"
    you = (r > 0) == human_white
    return f"{winner} wins -- {'you win!' if you else 'the engine wins.'}"


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
class BoardGUI:
    SQ = 64          # square size, px
    MARGIN = 22      # coordinate margin

    def __init__(self, tkroot, tk, env, eval_fn, args, human_white):
        self.tk = tk
        self.root = tkroot
        self.env = env
        self.eval_fn = eval_fn
        self.args = args
        self.human_white = human_white
        self.flip = not human_white
        self.undo_stack = []
        self.selected = None          # selected from-square
        self.targets = {}             # toSq -> list of candidate legal Moves
        self.last_move = None         # (fromSq, toSq) of the last played move
        self.suggestion = None        # (fromSq, toSq) arrow from Analyse
        self.busy = False
        self.msgq = queue.Queue()

        tkroot.title("play vs checkpoint -- ease probe view")
        main = tk.Frame(tkroot)
        main.pack(fill="both", expand=True, padx=6, pady=6)

        side = self.MARGIN + 8 * self.SQ
        self.canvas = tk.Canvas(main, width=side, height=side,
                                highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=4, sticky="n")
        self.canvas.bind("<Button-1>", self.on_click)

        piece_font = "Segoe UI Symbol" if sys.platform == "win32" else "DejaVu Sans"
        self.piece_font = (piece_font, int(self.SQ * 0.58))
        mono = "Consolas" if sys.platform == "win32" else "DejaVu Sans Mono"

        self.status = tk.Label(main, text="", font=(piece_font, 12, "bold"),
                               anchor="w")
        self.status.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.headline = tk.Label(main, text="F_local --  F_tree --  eval --",
                                 font=(mono, 12, "bold"), anchor="w")
        self.headline.grid(row=1, column=1, sticky="ew", padx=(10, 0))

        self.text = tk.Text(main, width=52, height=26, font=(mono, 10),
                            state="disabled", wrap="none")
        self.text.grid(row=2, column=1, sticky="nsew", padx=(10, 0))
        scroll = tk.Scrollbar(main, command=self.text.yview)
        scroll.grid(row=2, column=2, sticky="ns")
        self.text.configure(yscrollcommand=scroll.set)

        btns = tk.Frame(main)
        btns.grid(row=3, column=1, sticky="ew", padx=(10, 0), pady=(6, 0))
        self.buttons = []
        for label, cmd in (("Analyse", self.analyse), ("Undo", self.undo),
                           ("New game", self.new_game), ("Flip board", self.do_flip)):
            b = tk.Button(btns, text=label, command=cmd)
            b.pack(side="left", padx=(0, 6))
            self.buttons.append(b)
        main.rowconfigure(2, weight=1)
        main.columnconfigure(1, weight=1)

        self.draw()
        self.set_status()
        self.root.after(80, self.poll)
        if self.engine_to_move():
            self.kick_engine()

    # ---------------- board drawing ----------------
    def sq_to_xy(self, sq):
        f, r = sq % 8, sq // 8
        col = 7 - f if self.flip else f
        row = r if self.flip else 7 - r
        return (self.MARGIN + col * self.SQ, row * self.SQ)

    def xy_to_sq(self, x, y):
        col = (x - self.MARGIN) // self.SQ
        row = y // self.SQ
        if not (0 <= col < 8 and 0 <= row < 8):
            return None
        f = 7 - col if self.flip else col
        r = row if self.flip else 7 - row
        return int(r * 8 + f)

    def draw(self):
        c = self.canvas
        c.delete("all")
        for sq in range(64):
            x, y = self.sq_to_xy(sq)
            base = LIGHT if (sq // 8 + sq % 8) % 2 else DARK
            if self.last_move and sq in self.last_move:
                base = LAST_COLOR
            if sq == self.selected:
                base = SEL_COLOR
            c.create_rectangle(x, y, x + self.SQ, y + self.SQ,
                               fill=base, outline="")
            if sq in self.targets:
                c.create_oval(x + self.SQ * 0.38, y + self.SQ * 0.38,
                              x + self.SQ * 0.62, y + self.SQ * 0.62,
                              fill=DOT_COLOR, outline="")
            key = piece_at(self.env.board, sq)
            if key:
                c.create_text(x + self.SQ / 2, y + self.SQ / 2 + 2,
                              text=UNICODE[key], font=self.piece_font,
                              fill="#111")
        # coordinates
        files = "hgfedcba" if self.flip else "abcdefgh"
        ranks = range(1, 9) if self.flip else range(8, 0, -1)
        for i, ch in enumerate(files):
            c.create_text(self.MARGIN + i * self.SQ + self.SQ / 2,
                          8 * self.SQ + self.MARGIN / 2, text=ch)
        for i, rk in enumerate(ranks):
            c.create_text(self.MARGIN / 2, i * self.SQ + self.SQ / 2,
                          text=str(rk))
        # analysis suggestion arrow
        if self.suggestion:
            (x1, y1), (x2, y2) = (self.sq_to_xy(s) for s in self.suggestion)
            h = self.SQ / 2
            c.create_line(x1 + h, y1 + h, x2 + h, y2 + h, width=5,
                          fill="#2a78d6", arrow="last", arrowshape=(16, 20, 7))

    # ---------------- helpers ----------------
    def engine_color(self):
        return "black" if self.human_white else "white"

    def engine_to_move(self):
        return (self.env.board.sideToMove == self.engine_color()
                and not game_over(self.env))

    def set_status(self, extra=None):
        if extra:
            self.status.config(text=extra)
        elif game_over(self.env):
            self.status.config(text=result_text(self.env, self.human_white))
        elif self.engine_to_move():
            self.status.config(text=f"engine thinking ({self.args.sims} sims)...")
        else:
            side = "White" if self.env.board.sideToMove == "white" else "Black"
            self.status.config(text=f"your move ({side})")

    def set_busy(self, busy):
        self.busy = busy
        for b in self.buttons:
            b.config(state="disabled" if busy else "normal")

    def log(self, block):
        self.text.config(state="normal")
        self.text.insert("end", block + "\n" + "-" * 50 + "\n")
        self.text.see("end")
        self.text.config(state="disabled")

    def show_search(self, header, root, force_eff):
        st = ease_summary(root, force_eff, self.args.tau, self.args.gamma)
        block = header + "\n" + format_distribution(root, force_eff,
                                                    self.args.top)
        if st is not None:
            block += "\n" + format_ease(st, self.args.gamma)
            ftree = f"{st['f_tree']:.3f}" if st["f_tree"] is not None else "n/a"
            self.headline.config(
                text=f"F_gap {st['F_gap']:.3f}  F_tree {ftree}  "
                     f"eval {st['v_root']:+.3f}")
        self.log(block)
        return st

    # ---------------- threaded searches ----------------
    def kick_engine(self):
        self.set_busy(True)
        self.set_status()
        env_snapshot = self.env.clone()

        def work():
            try:
                root, feff = run_search(env_snapshot, self.eval_fn, self.args)
                self.msgq.put(("engine", root, feff))
            except Exception as e:
                self.msgq.put(("error", str(e), None))

        threading.Thread(target=work, daemon=True).start()

    def analyse(self):
        if self.busy or game_over(self.env) or self.engine_to_move():
            return
        self.set_busy(True)
        self.set_status("analysing your position...")
        env_snapshot = self.env.clone()

        def work():
            try:
                root, feff = run_search(env_snapshot, self.eval_fn, self.args)
                self.msgq.put(("analysis", root, feff))
            except Exception as e:
                self.msgq.put(("error", str(e), None))

        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        try:
            while True:
                kind, a, b = self.msgq.get_nowait()
                if kind == "error":
                    self.set_busy(False)
                    self.log(f"search error: {a}")
                    self.set_status()
                elif kind == "engine":
                    root, feff = a, b
                    kids = [c for c in root.children if c.visits > 0]
                    self.set_busy(False)
                    if not kids:
                        self.set_status()
                        continue
                    best = max(kids, key=lambda c: c.visits)
                    self.show_search(f"engine plays {_uci(best.move)}",
                                     root, feff)
                    self.undo_stack.append(self.env.clone())
                    self.env.step(best.move)
                    self.last_move = (best.move.fromSq, best.move.toSq)
                    self.suggestion = None
                    self.draw()
                    self.set_status()
                elif kind == "analysis":
                    root, feff = a, b
                    self.set_busy(False)
                    st = self.show_search("analysis of your position",
                                          root, feff)
                    if st is not None:
                        kids = [c for c in root.children if c.visits > 0]
                        best = max(kids, key=lambda c: c.visits)
                        self.suggestion = (best.move.fromSq, best.move.toSq)
                        self.draw()
                    self.set_status()
        except queue.Empty:
            pass
        self.root.after(80, self.poll)

    # ---------------- input ----------------
    def on_click(self, event):
        if self.busy or game_over(self.env) or self.engine_to_move():
            return
        sq = self.xy_to_sq(event.x, event.y)
        if sq is None:
            return
        legal = self.env.legalMoves()
        own = piece_at(self.env.board, sq)
        human_color = "white" if self.human_white else "black"

        if self.selected is not None and sq in self.targets:
            self.play_human(self.targets[sq])
            return
        if own is not None and own[0] == human_color:
            self.selected = sq
            self.targets = {}
            for m in legal:
                if m.fromSq == sq:
                    self.targets.setdefault(m.toSq, []).append(m)
        else:
            self.selected = None
            self.targets = {}
        self.draw()

    def play_human(self, candidates):
        move = candidates[0]
        if len(candidates) > 1:              # promotion: several moves share to-sq
            letter = self.ask_promotion()
            hits = [m for m in candidates
                    if m.promotion and m.promotion.upper() == letter]
            if hits:
                move = hits[0]
        self.undo_stack.append(self.env.clone())
        self.env.step(move)
        self.last_move = (move.fromSq, move.toSq)
        self.selected = None
        self.targets = {}
        self.suggestion = None
        self.draw()
        if game_over(self.env):
            self.set_status()
            self.log(result_text(self.env, self.human_white))
            return
        self.kick_engine()

    def ask_promotion(self):
        tk = self.tk
        win = tk.Toplevel(self.root)
        win.title("Promote to")
        choice = {"p": "Q"}

        def pick(letter):
            choice["p"] = letter
            win.destroy()

        for letter, name in (("Q", "Queen"), ("R", "Rook"),
                             ("B", "Bishop"), ("N", "Knight")):
            tk.Button(win, text=name, width=10,
                      command=lambda l=letter: pick(l)).pack(padx=8, pady=3)
        win.grab_set()
        self.root.wait_window(win)
        return choice["p"]

    # ---------------- buttons ----------------
    def undo(self):
        if self.busy or len(self.undo_stack) < 2:
            return
        self.undo_stack.pop()                # engine's move
        self.env = self.undo_stack.pop()     # back to before your move
        self.selected = None
        self.targets = {}
        self.last_move = None
        self.suggestion = None
        self.draw()
        self.set_status()
        self.log("took back one full move")

    def new_game(self):
        if self.busy:
            return
        self.env = Chess()
        self.env.reset()
        self.undo_stack.clear()
        self.selected = None
        self.targets = {}
        self.last_move = None
        self.suggestion = None
        self.draw()
        self.set_status()
        self.log("new game")
        if self.engine_to_move():
            self.kick_engine()

    def do_flip(self):
        self.flip = not self.flip
        self.draw()


def gui_main(env, eval_fn, args, human_white):
    try:
        import tkinter as tk
    except ImportError:
        print("tkinter is not available in this Python build -- "
              "falling back to --terminal mode.")
        terminal_main(env, eval_fn, args, human_white)
        return
    root = tk.Tk()
    BoardGUI(root, tk, env, eval_fn, args, human_white)
    root.mainloop()


# --------------------------------------------------------------------------- #
# terminal fallback (the original loop)
# --------------------------------------------------------------------------- #
def render(board, flip=False):
    ranks = range(8) if flip else range(7, -1, -1)
    files = range(7, -1, -1) if flip else range(8)
    lines = []
    for r in ranks:
        cells = []
        for f in files:
            key = piece_at(board, r * 8 + f)
            cells.append(PIECES[key] if key else ".")
        lines.append(f" {r + 1} " + " ".join(cells))
    footer = "hgfedcba" if flip else "abcdefgh"
    lines.append("   " + " ".join(footer))
    return "\n".join(lines)


def terminal_main(env, eval_fn, args, human_white):
    undo_stack = []

    def engine_turn():
        print("engine thinking...")
        root, feff = run_search(env, eval_fn, args)
        print(format_distribution(root, feff, args.top))
        st = ease_summary(root, feff, args.tau, args.gamma)
        if st is not None:
            print(format_ease(st, args.gamma))
        kids = [c for c in root.children if c.visits > 0]
        if not kids:
            return False
        best = max(kids, key=lambda c: c.visits)
        print(f"    engine plays {_uci(best.move)}\n")
        undo_stack.append(env.clone())
        env.step(best.move)
        return True

    while True:
        engine_color = "black" if human_white else "white"
        if env.board.sideToMove == engine_color and not game_over(env):
            if not engine_turn():
                break
            if game_over(env):
                print(render(env.board, flip=not human_white))
                print(result_text(env, human_white))
                break
        print(render(env.board, flip=not human_white))
        if game_over(env):
            print(result_text(env, human_white))
            break
        cmd = input(f"\nyour move ({'white' if human_white else 'black'}) > ")
        cmd = cmd.strip().lower()
        if cmd in ("quit", "resign", "exit"):
            print("game over -- you resigned.")
            break
        if cmd == "help":
            print("  e2e4 / e7e8q  play a move      analyse  search your position")
            print("  moves         list legal       fen      print FEN")
            print("  undo          take back        new      restart    quit/resign")
            continue
        if cmd == "new":
            env.reset()
            undo_stack.clear()
            print("new game.\n")
            continue
        if cmd == "fen":
            print(board_to_fen(env.board))
            continue
        if cmd == "moves":
            print("  " + "  ".join(sorted(_uci(m) for m in env.legalMoves())))
            continue
        if cmd == "undo":
            if len(undo_stack) >= 2:
                undo_stack.pop()
                env = undo_stack.pop()
                print("took back one full move.\n")
            else:
                print("nothing to undo.")
            continue
        if cmd in ("analyse", "analyze", "hint", "ease"):
            root, feff = run_search(env, eval_fn, args)
            print(format_distribution(root, feff, args.top))
            st = ease_summary(root, feff, args.tau, args.gamma)
            if st is not None:
                print(format_ease(st, args.gamma))
                print(f"    suggested: {st['best_move']}")
            continue
        move = parse_move(cmd, env.legalMoves())
        if move is None:
            print("  not a legal move (or ambiguous) -- 'moves' lists them; "
                  "promotions need the letter, e.g. e7e8q.")
            continue
        undo_stack.append(env.clone())
        env.step(move)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Play a trained checkpoint on a "
                                             "popup board with ease statistics")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--color", choices=["white", "black", "random"],
                    default="white", help="your color (default white)")
    ap.add_argument("--sims", type=int, default=700,
                    help="engine search budget per move (lower = faster)")
    ap.add_argument("--c", type=float, default=1.5, help="PUCT exploration c")
    ap.add_argument("--force-m", type=int, default=8)
    ap.add_argument("--force-n", type=int, default=40)
    ap.add_argument("--tau", type=float, default=0.05,
                    help="ease temperature -- use the value probe_ease "
                         "calibrated for this checkpoint")
    ap.add_argument("--gamma", type=float, default=0.85,
                    help="decay in the recursive tree forgiveness")
    ap.add_argument("--top", type=int, default=10,
                    help="rows shown in the visit distribution")
    ap.add_argument("--terminal", action="store_true",
                    help="text-mode loop instead of the popup board")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    net = load_net(args.checkpoint, device).eval()
    eval_fn = _probe_eval_fn(net)

    human_white = (args.color == "white" if args.color != "random"
                   else random.random() < 0.5)
    print(f"you play {'White' if human_white else 'Black'}; engine at "
          f"{args.sims} sims/move (floor: top-{args.force_m} by prior).")

    env = Chess()
    env.reset()
    if args.terminal:
        terminal_main(env, eval_fn, args, human_white)
    else:
        gui_main(env, eval_fn, args, human_white)


if __name__ == "__main__":
    main()
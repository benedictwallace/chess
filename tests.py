from gameEnv import Chess
from moves import Move
from board import Board, perft
from encoding import encode
import numpy as np
from network import ChessNet
import torch
from move_encoding import encodeMove, decodeMove, NUM_ACTIONS
from puct import search
import os
from self_play import generate_games
from train import ReplayBuffer, train_epoch

def testGPU():
    import torch
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())

def testEnv():
    env = Chess()
    state = env.reset()
    print("Legal moves at start:", len(env.legalMoves())) # 20
    print("Terminal?", env.isTerminal()) # False
    print("Result?", env.result()) # None
    move = env.legalMoves()[0]
    state, reward, done = env.step(move)
    print("After one move, side to move:", env.board.sideToMove) # black
    print("Reward:", reward, "Done:", done) # 0.0 False

def testFoolsMate():
    env = Chess()
    # Fool's mate sequence: 1. f3 e5 2. g4 Qh4#


    env.step(Move(fromSq=13, toSq=21)) # f2-f3 white
    env.step(Move(fromSq=52, toSq=36)) # e7-e5 black
    env.step(Move(fromSq=14, toSq=30)) # g2-g4 white
    state, reward, done = env.step(Move(fromSq=59, toSq=31)) # Qd8-h4# black

    print("Done?", done) # True
    print("Reward to mover:", reward) # 1.0 (black delivered mate)
    print("Result (white view):", env.result()) # -1.0

def testEncodingWhite():

    b = Board()
    planes = encode(b)
    print("Shape:", planes.shape)              # (18, 8, 8)

    # White to move, starting position. Plane 0 = our pawns.
    # Mover is white, no flip, white pawns are on rank 2 = row index 1.
    print("Our pawns (plane 0):")
    print(planes[0].astype(int))
    # Expect row 1 all 1s, everything else 0.

    # Plane 5 = our king. Should be a single 1 at e1 = row 0, col 4.
    print("Our king (plane 5):")
    print(planes[5].astype(int))

def testEncodingBlack():
    print("\n--- Black perspective test ---")

    # Start position, but pretend it's black's turn.
    b = Board()
    white_view = encode(b)

    b2 = Board()
    b2.sideToMove = "black"
    black_view = encode(b2)

    # 1. Shape sanity
    print("Black view shape:", black_view.shape)   # (18, 8, 8)

    # 3. Side-to-move plane: 1s for white, 0s for black.
    print("White side-to-move plane all 1s (expect True):",
          np.array_equal(white_view[12], np.ones((8, 8))))
    print("Black side-to-move plane all 0s (expect True):",
          np.array_equal(black_view[12], np.zeros((8, 8))))

    # 4. Our king from black's POV. Black king starts on e8 = square 60.
    #    flip: rank 7-7=0, file 7-4=3  ->  plane[5][0][3] should be 1.
    print("Black's own king (plane 5), expect a 1 at row 0, col 3:")
    print(black_view[5].astype(int))

    # 5. Flip is NOT a no-op asymmetric check.
    #    Put a single white pawn on a3 (square 16) and nothing else odd.
    #    From white's POV (no flip): square 16 -> plane[2][0].
    #    From black's POV (flip):    rank 7-2=5, file 7-0=7 -> plane[5][7].
    b3 = Board()
    for k in b3.bb:
        b3.bb[k] = 0
    b3.bb[("white", "pawn")] = 1 << 16     # a3
    b3.bb[("white", "king")] = 1 << 4      # e1, kings must exist
    b3.bb[("black", "king")] = 1 << 60     # e8
    b3.updatePieces()

    b3.sideToMove = "white"
    wv = encode(b3)
    b3.sideToMove = "black"
    bv = encode(b3)

    # white sees its own pawn on plane 0; black sees the same pawn as an
    # OPPONENT pawn on plane 6, at the 180-rotated square.
    print("White's own-pawn plane has 1 at [2][0] (expect True):",
          wv[0][2][0] == 1.0)
    print("Black's opp-pawn plane has 1 at [5][7] (expect True):",
          bv[6][5][7] == 1.0)

def testNetwork():
    net = ChessNet()
    net.eval()

    planes = encode(Board())                      # (18, 8, 8)
    x = torch.from_numpy(planes).unsqueeze(0)      # (1, 18, 8, 8) add batch dim

    with torch.no_grad():
        policy_logits, value, ease = net(x)

    print("Policy logits shape:", policy_logits.shape)   # (1, 4672)
    print("Value shape:", value.shape)                   # (1, 1)
    print("Value:", value.item())                        # some float in [-1, 1]

    # also test a batch
    batch = torch.from_numpy(np.stack([planes, planes, planes]))  # (3, 18, 8, 8)
    with torch.no_grad():
        pl, v, e = net(batch)
    print("Batch policy:", pl.shape, "Batch value:", v.shape)     # (3,4672) (3,1)

def testEncodingMove():
    b = Board()
    seen = {}
    for m in b.legalMoves("white"):
        idx = encodeMove(m)
        assert 0 <= idx < NUM_ACTIONS, f"index {idx} out of range for {m}"
        assert idx not in seen, f"collision: {m} and {seen[idx]} both -> {idx}"
        seen[idx] = m
    print(f"{len(seen)} moves encoded, all unique indices, all in range")

    # round-trip a queen move: encode then decode, from/to should survive
    m = b.legalMoves("white")[0]
    d = decodeMove(encodeMove(m))
    assert d.fromSq == m.fromSq and d.toSq == m.toSq
    print("round-trip from/to OK")

def testPUCT():

    net = ChessNet()
    env = Chess()
    env.reset()

    move = search(env, net, iterations=200)
    print("PUCT chose:", move)

def testBoard():
    b = Board()
    print(perft(b, "white", 1))
    print(perft(b, "white", 2))
    print(perft(b, "white", 3))
    print(perft(b, "white", 4))

def testOneMoveMate():
    print("\n--- One-move-mate test (back-rank) ---")
    from board import Board
    from gameEnv import Chess
    from moves import Move

    b = Board()
    for k in b.bb:
        b.bb[k] = 0

    # White: rook a1 (sq 0), king h1 (sq 7)
    b.bb[("white", "rook")] = 1 << 0
    b.bb[("white", "king")] = 1 << 7
    # Black: king h8 (sq 63), pawns f7/g7/h7 (sq 53/54/55)
    b.bb[("black", "king")] = 1 << 63
    b.bb[("black", "pawn")] = (1 << 53) | (1 << 54) | (1 << 55)

    # no castling rights in this position
    b.whiteKCastle = b.whiteQCastle = False
    b.blackKCastle = b.blackQCastle = False
    b.sideToMove = "white"
    b.updatePieces()

    env = Chess()
    env.board = b

    # The mating move: rook a1 -> a8 (sq 0 -> sq 56)
    mate = Move(fromSq=0, toSq=56)

    # sanity: confirm it really is mate before testing the search
    legal = env.legalMoves()
    assert any(m.fromSq == 0 and m.toSq == 56 for m in legal), \
        "Ra8 should be legal"

    trial = env.clone()
    trial.step(mate)
    assert trial.isTerminal(), "after Ra8 the position should be terminal"
    assert trial.board.checkMate("black"), "after Ra8 black should be mated"
    print("Position verified: Ra8 is checkmate.")

    net = ChessNet()
    net.eval()

    chosen = search(env, net, iterations=200)
    print("Search chose:", chosen)
    if chosen.fromSq == 0 and chosen.toSq == 56:
        print("PASS search found the mate.")
    else:
        print("FAIL search missed the mate.")


def testTrainingLoop():
    """
    Scaled-down self-play -> train -> checkpoint loop. Verifies the
    pipeline runs end to end and the loss is a finite number.
    """

    print("\n--- Training loop smoke test ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    # small net + tiny config so this finishes in well under a minute
    net = ChessNet(channels=16, num_blocks=2).to(device)
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    buffer = ReplayBuffer(capacity=10_000)

    for it in range(1, 3):  # 2 loop iterations
        print(f"loop iteration {it}/2")

        examples = generate_games(
            net, num_games=3,
            iterations=15,
            max_plies=16,
            temp_moves=4,
        )
        buffer.add_examples(examples)
        print(f"  buffer size: {len(buffer)}")

        tot, pol, val, ez = train_epoch(
            net, buffer, optimiser, device,
            batches=4, batch_size=32,
        )
        print(f"  loss total={tot:.4f}  policy={pol:.4f}  value={val:.4f}")

        # sanity checks -- catch the failure modes a long run would waste time on
        assert len(buffer) > 0, "buffer is empty -- self-play produced nothing"
        assert np.isfinite(tot), f"loss is not finite: {tot}"
        assert pol >= 0, f"policy loss should be non-negative, got {pol}"

    print("PASS training loop runs end to end.")

if __name__=="__main__":

    testTrainingLoop()
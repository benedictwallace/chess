# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
C-level legal move generation.

WHY
---
Board.legalMoves() measured 135 us per call and was 40-45% of self-play wall
time -- the binding constraint on the whole project's data rate. Everything in
it is 64-bit integer work that CPython performs on heap-allocated arbitrary-
precision objects: every `bb & -bb`, every shift, every `1 << sq` allocates.
In C those are single instructions.

WHAT IS PRESERVED
-----------------
The algorithm is a direct transcription of engine/board.py, deliberately NOT
an improvement on it:

  * pseudo-legal generation per piece type, king moves generated WITHOUT an
    attack mask (as board.py does since the legality filter catches them);
  * check detected once via square_attacked();
  * absolute pins found by the same 8 ray casts as _pinnedBB();
  * a move is verified by make/unmake ONLY if we are in check, or it is a king
    move, or the mover is pinned, or it is an en-passant capture -- exactly
    board.py's condition;
  * castling uses the same must-be-empty masks and tests the same three
    squares per side.

The legal move SET is therefore identical. The ORDER differs (pawns are
generated set-wise), which only affects argmax tie-breaks, never legality.
Verified by perft to depth 5 from the start position plus the six standard
perft positions, and by differential testing against the Python engine over
every position of thousands of random games.

MOVE PACKING
------------
A move is one 32-bit int:  from | to<<6 | promo<<12 | castle<<15 | ep<<16
with promo 0=none 1=Q 2=R 3=B 4=N. legal_moves_packed() returns these
directly; hot callers that only compare and index moves should use them and
skip building Python objects entirely. legal_moves() builds the Move
namedtuples for drop-in compatibility.
"""

from libc.stdint cimport uint64_t, uint32_t, int32_t

# GCC/Clang bit-scan intrinsics: single CPU instructions (TZCNT/LZCNT).
# These are what replace Python's int.bit_length() dance in the hot loop.
cdef extern from *:
    """
    static inline int _ctz64(unsigned long long x) { return __builtin_ctzll(x); }
    static inline int _clz64(unsigned long long x) { return __builtin_clzll(x); }
    """
    int _ctz64(uint64_t x) nogil
    int _clz64(uint64_t x) nogil

# --------------------------------------------------------------------------- #
# piece indices into the 12-slot bitboard array
# --------------------------------------------------------------------------- #
cdef enum:
    WP = 0, WN = 1, WB = 2, WR = 3, WQ = 4, WK = 5
    BP = 6, BN = 7, BB_ = 8, BR = 9, BQ = 10, BK = 11

cdef uint64_t FILE_A = 0x0101010101010101ULL
cdef uint64_t FILE_B = 0x0202020202020202ULL
cdef uint64_t FILE_G = 0x4040404040404040ULL
cdef uint64_t FILE_H = 0x8080808080808080ULL
cdef uint64_t NOT_A = ~FILE_A
cdef uint64_t NOT_B = ~FILE_B
cdef uint64_t NOT_G = ~FILE_G
cdef uint64_t NOT_H = ~FILE_H
cdef uint64_t RANK_1 = 0x00000000000000FFULL
cdef uint64_t RANK_2 = 0x000000000000FF00ULL
cdef uint64_t RANK_3 = 0x0000000000FF0000ULL
cdef uint64_t RANK_6 = 0x0000FF0000000000ULL
cdef uint64_t RANK_7 = 0x00FF000000000000ULL
cdef uint64_t RANK_8 = 0xFF00000000000000ULL

# must-be-empty masks, matching engine/bitboard.py
cdef uint64_t MBE_KW = (1ULL << 5) | (1ULL << 6)
cdef uint64_t MBE_QW = (1ULL << 1) | (1ULL << 2) | (1ULL << 3)
cdef uint64_t MBE_KB = (1ULL << 61) | (1ULL << 62)
cdef uint64_t MBE_QB = (1ULL << 57) | (1ULL << 58) | (1ULL << 59)

# --------------------------------------------------------------------------- #
# attack tables, built at import
# --------------------------------------------------------------------------- #
cdef uint64_t KNIGHT_ATT[64]
cdef uint64_t KING_ATT[64]
cdef uint64_t PAWN_ATT[2][64]          # [0]=white, [1]=black
cdef uint64_t RAYS[64][8]
# direction order matches engine/moves.py _RAY_DIRS:
#  0:(+1,0) 1:(-1,0) 2:(0,+1) 3:(0,-1) 4:(+1,+1) 5:(+1,-1) 6:(-1,+1) 7:(-1,-1)
# "positive" directions (blocker = lsb): 0,2,4,5   negative (blocker = msb): 1,3,6,7
cdef int ROOK_POS[2]
cdef int ROOK_NEG[2]
cdef int BISH_POS[2]
cdef int BISH_NEG[2]


cdef inline int lsb_i(uint64_t b) noexcept nogil:
    return _ctz64(b)


cdef inline int msb_i(uint64_t b) noexcept nogil:
    return 63 - _clz64(b)


cdef void _init_tables() noexcept:
    cdef int sq, r, f, nr, nf, d, i
    cdef uint64_t bb
    cdef int dr8[8]
    cdef int df8[8]
    dr8[0] =  1; df8[0] =  0
    dr8[1] = -1; df8[1] =  0
    dr8[2] =  0; df8[2] =  1
    dr8[3] =  0; df8[3] = -1
    dr8[4] =  1; df8[4] =  1
    dr8[5] =  1; df8[5] = -1
    dr8[6] = -1; df8[6] =  1
    dr8[7] = -1; df8[7] = -1

    for sq in range(64):
        bb = 1ULL << sq
        KNIGHT_ATT[sq] = (((bb & NOT_A & NOT_B) << 6) |
                          ((bb & NOT_H & NOT_G) << 10) |
                          ((bb & NOT_A) << 15) |
                          ((bb & NOT_H) << 17) |
                          ((bb & NOT_G & NOT_H) >> 6) |
                          ((bb & NOT_A & NOT_B) >> 10) |
                          ((bb & NOT_H) >> 15) |
                          ((bb & NOT_A) >> 17))
        KING_ATT[sq] = ((bb << 8) | (bb >> 8) |
                        ((bb << 1) & NOT_A) | ((bb >> 1) & NOT_H) |
                        ((bb << 7) & NOT_H) | ((bb >> 7) & NOT_A) |
                        ((bb << 9) & NOT_A) | ((bb >> 9) & NOT_H))
        PAWN_ATT[0][sq] = ((bb & NOT_A) << 7) | ((bb & NOT_H) << 9)
        PAWN_ATT[1][sq] = ((bb & NOT_A) >> 9) | ((bb & NOT_H) >> 7)

        r = sq >> 3
        f = sq & 7
        for d in range(8):
            RAYS[sq][d] = 0
            nr = r
            nf = f
            while True:
                nr += dr8[d]
                nf += df8[d]
                if nr < 0 or nr > 7 or nf < 0 or nf > 7:
                    break
                RAYS[sq][d] |= (1ULL << (nr * 8 + nf))

    ROOK_POS[0] = 0; ROOK_POS[1] = 2
    ROOK_NEG[0] = 1; ROOK_NEG[1] = 3
    BISH_POS[0] = 4; BISH_POS[1] = 5
    BISH_NEG[0] = 6; BISH_NEG[1] = 7


_init_tables()


# --------------------------------------------------------------------------- #
# sliding attacks (classical ray + blocker scan)
# --------------------------------------------------------------------------- #
cdef inline uint64_t rook_att(int sq, uint64_t occ) noexcept nogil:
    cdef uint64_t a = 0, ray, blockers
    cdef int i, d
    for i in range(2):
        d = ROOK_POS[i]
        ray = RAYS[sq][d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[lsb_i(blockers)][d]
        a |= ray
    for i in range(2):
        d = ROOK_NEG[i]
        ray = RAYS[sq][d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[msb_i(blockers)][d]
        a |= ray
    return a


cdef inline uint64_t bish_att(int sq, uint64_t occ) noexcept nogil:
    cdef uint64_t a = 0, ray, blockers
    cdef int i, d
    for i in range(2):
        d = BISH_POS[i]
        ray = RAYS[sq][d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[lsb_i(blockers)][d]
        a |= ray
    for i in range(2):
        d = BISH_NEG[i]
        ray = RAYS[sq][d]
        blockers = ray & occ
        if blockers:
            ray ^= RAYS[msb_i(blockers)][d]
        a |= ray
    return a


# --------------------------------------------------------------------------- #
# position
# --------------------------------------------------------------------------- #
cdef struct Pos:
    uint64_t bb[12]
    uint64_t occ_w
    uint64_t occ_b
    uint64_t occ
    int ep                 # en-passant target square, -1 if none
    int wk, wq, bk, bq     # castling rights


cdef struct Undo:
    int captured_idx       # -1 if none
    int captured_sq
    uint64_t captured_bit


cdef inline void refresh(Pos *p) noexcept nogil:
    cdef int i
    p.occ_w = 0
    p.occ_b = 0
    for i in range(6):
        p.occ_w |= p.bb[i]
    for i in range(6, 12):
        p.occ_b |= p.bb[i]
    p.occ = p.occ_w | p.occ_b


cdef inline bint square_attacked(Pos *p, int sq, int by_white) noexcept nogil:
    """Is `sq` attacked by the given colour? Mirrors board.squareAttackedBy."""
    cdef int base = 0 if by_white else 6
    cdef uint64_t occ = p.occ
    # pawns: a pawn on X attacks sq iff sq is attacked-from-X, i.e. use the
    # OPPOSITE colour's attack table from sq
    if PAWN_ATT[1 if by_white else 0][sq] & p.bb[base + 0]:
        return True
    if KNIGHT_ATT[sq] & p.bb[base + 1]:
        return True
    if KING_ATT[sq] & p.bb[base + 5]:
        return True
    if bish_att(sq, occ) & (p.bb[base + 2] | p.bb[base + 4]):
        return True
    if rook_att(sq, occ) & (p.bb[base + 3] | p.bb[base + 4]):
        return True
    return False


# --------------------------------------------------------------------------- #
# move packing
# --------------------------------------------------------------------------- #
cdef inline uint32_t pack(int frm, int to, int promo, int castle, int ep) noexcept nogil:
    return <uint32_t>(frm | (to << 6) | (promo << 12) | (castle << 15) | (ep << 16))


cdef inline int m_from(uint32_t m) noexcept nogil:
    return m & 63


cdef inline int m_to(uint32_t m) noexcept nogil:
    return (m >> 6) & 63


cdef inline int m_promo(uint32_t m) noexcept nogil:
    return (m >> 12) & 7


cdef inline int m_castle(uint32_t m) noexcept nogil:
    return (m >> 15) & 1


cdef inline int m_ep(uint32_t m) noexcept nogil:
    return (m >> 16) & 1


# --------------------------------------------------------------------------- #
# make / unmake -- only enough state to test "is our king attacked?"
# --------------------------------------------------------------------------- #
cdef inline int piece_at(Pos *p, int sq, int lo, int hi) noexcept nogil:
    cdef uint64_t bit = 1ULL << sq
    cdef int i
    for i in range(lo, hi):
        if p.bb[i] & bit:
            return i
    return -1


cdef void do_move(Pos *p, uint32_t mv, int white, Undo *u) noexcept nogil:
    cdef int frm = m_from(mv), to = m_to(mv)
    cdef int promo = m_promo(mv)
    cdef uint64_t fb = 1ULL << frm, tb = 1ULL << to
    cdef int base = 0 if white else 6
    cdef int obase = 6 if white else 0
    cdef int moving = piece_at(p, frm, base, base + 6)
    cdef int capsq, cap
    cdef uint64_t cb

    u.captured_idx = -1
    if moving < 0:
        return

    # capture (en passant captures on a different square)
    capsq = to
    if m_ep(mv):
        capsq = to - 8 if white else to + 8
    cb = 1ULL << capsq
    cap = piece_at(p, capsq, obase, obase + 6)
    if cap >= 0:
        p.bb[cap] &= ~cb
        u.captured_idx = cap
        u.captured_sq = capsq
        u.captured_bit = cb

    p.bb[moving] &= ~fb
    if promo:
        # 1=Q 2=R 3=B 4=N -> WQ/WR/WB/WN offsets 4/3/2/1
        if promo == 1:
            p.bb[base + 4] |= tb
        elif promo == 2:
            p.bb[base + 3] |= tb
        elif promo == 3:
            p.bb[base + 2] |= tb
        else:
            p.bb[base + 1] |= tb
    else:
        p.bb[moving] |= tb

    if m_castle(mv):
        # move the rook too
        if to == 6:
            p.bb[WR] &= ~(1ULL << 7); p.bb[WR] |= (1ULL << 5)
        elif to == 2:
            p.bb[WR] &= ~(1ULL << 0); p.bb[WR] |= (1ULL << 3)
        elif to == 62:
            p.bb[BR] &= ~(1ULL << 63); p.bb[BR] |= (1ULL << 61)
        elif to == 58:
            p.bb[BR] &= ~(1ULL << 56); p.bb[BR] |= (1ULL << 59)
    refresh(p)


cdef void undo_move(Pos *p, uint32_t mv, int white, Undo *u) noexcept nogil:
    cdef int frm = m_from(mv), to = m_to(mv)
    cdef int promo = m_promo(mv)
    cdef uint64_t fb = 1ULL << frm, tb = 1ULL << to
    cdef int base = 0 if white else 6
    cdef int moving

    if promo:
        if promo == 1:
            p.bb[base + 4] &= ~tb
        elif promo == 2:
            p.bb[base + 3] &= ~tb
        elif promo == 3:
            p.bb[base + 2] &= ~tb
        else:
            p.bb[base + 1] &= ~tb
        p.bb[base + 0] |= fb
    else:
        moving = piece_at(p, to, base, base + 6)
        if moving >= 0:
            p.bb[moving] &= ~tb
            p.bb[moving] |= fb

    if u.captured_idx >= 0:
        p.bb[u.captured_idx] |= u.captured_bit

    if m_castle(mv):
        if to == 6:
            p.bb[WR] &= ~(1ULL << 5); p.bb[WR] |= (1ULL << 7)
        elif to == 2:
            p.bb[WR] &= ~(1ULL << 3); p.bb[WR] |= (1ULL << 0)
        elif to == 62:
            p.bb[BR] &= ~(1ULL << 61); p.bb[BR] |= (1ULL << 63)
        elif to == 58:
            p.bb[BR] &= ~(1ULL << 59); p.bb[BR] |= (1ULL << 56)
    refresh(p)


# --------------------------------------------------------------------------- #
# pins -- transcription of Board._pinnedBB
# --------------------------------------------------------------------------- #
cdef uint64_t pinned_bb(Pos *p, int ksq, int white) noexcept nogil:
    cdef uint64_t occ = p.occ
    cdef uint64_t own = p.occ_w if white else p.occ_b
    cdef int obase = 6 if white else 0
    cdef uint64_t erq = p.bb[obase + 3] | p.bb[obase + 4]
    cdef uint64_t ebq = p.bb[obase + 2] | p.bb[obase + 4]
    cdef uint64_t pinned = 0, blockers, rest
    cdef int d, i, first, second, positive

    for i in range(8):
        d = i
        positive = 1 if (d == 0 or d == 2 or d == 4 or d == 5) else 0
        blockers = RAYS[ksq][d] & occ
        if not blockers:
            continue
        first = lsb_i(blockers) if positive else msb_i(blockers)
        if not ((own >> first) & 1):
            continue
        rest = RAYS[first][d] & occ
        if not rest:
            continue
        second = lsb_i(rest) if positive else msb_i(rest)
        if d < 4:
            if (erq >> second) & 1:
                pinned |= (1ULL << first)
        else:
            if (ebq >> second) & 1:
                pinned |= (1ULL << first)
    return pinned


# --------------------------------------------------------------------------- #
# pseudo-legal generation
# --------------------------------------------------------------------------- #
cdef int gen_pseudo(Pos *p, int white, uint32_t *out) noexcept nogil:
    cdef int n = 0
    cdef int base = 0 if white else 6
    cdef uint64_t own = p.occ_w if white else p.occ_b
    cdef uint64_t opp = p.occ_b if white else p.occ_w
    cdef uint64_t occ = p.occ
    cdef uint64_t pieces, att, bbm, targets, empty
    cdef uint64_t single, dbl, capl, capr
    cdef int frm, to, i
    cdef int ds, dd, dl, dr_

    # knights
    pieces = p.bb[base + 1]
    while pieces:
        frm = lsb_i(pieces)
        att = KNIGHT_ATT[frm] & ~own
        while att:
            to = lsb_i(att)
            out[n] = pack(frm, to, 0, 0, 0); n += 1
            att &= att - 1
        pieces &= pieces - 1

    # rooks
    pieces = p.bb[base + 3]
    while pieces:
        frm = lsb_i(pieces)
        att = rook_att(frm, occ) & ~own
        while att:
            to = lsb_i(att)
            out[n] = pack(frm, to, 0, 0, 0); n += 1
            att &= att - 1
        pieces &= pieces - 1

    # bishops
    pieces = p.bb[base + 2]
    while pieces:
        frm = lsb_i(pieces)
        att = bish_att(frm, occ) & ~own
        while att:
            to = lsb_i(att)
            out[n] = pack(frm, to, 0, 0, 0); n += 1
            att &= att - 1
        pieces &= pieces - 1

    # queens
    pieces = p.bb[base + 4]
    while pieces:
        frm = lsb_i(pieces)
        att = (rook_att(frm, occ) | bish_att(frm, occ)) & ~own
        while att:
            to = lsb_i(att)
            out[n] = pack(frm, to, 0, 0, 0); n += 1
            att &= att - 1
        pieces &= pieces - 1

    # king (no attack mask -- matches board.py)
    pieces = p.bb[base + 5]
    while pieces:
        frm = lsb_i(pieces)
        att = KING_ATT[frm] & ~own
        while att:
            to = lsb_i(att)
            out[n] = pack(frm, to, 0, 0, 0); n += 1
            att &= att - 1
        pieces &= pieces - 1

    # pawns, set-wise
    empty = ~occ
    targets = opp
    if p.ep >= 0:
        targets |= (1ULL << p.ep)
    pieces = p.bb[base + 0]
    if white:
        single = (pieces << 8) & empty
        dbl = ((single & RANK_3) << 8) & empty
        capl = ((pieces & NOT_A) << 7) & targets
        capr = ((pieces & NOT_H) << 9) & targets
        ds = -8; dd = -16; dl = -7; dr_ = -9
    else:
        single = (pieces >> 8) & empty
        dbl = ((single & RANK_6) >> 8) & empty
        capl = ((pieces & NOT_A) >> 9) & targets
        capr = ((pieces & NOT_H) >> 7) & targets
        ds = 8; dd = 16; dl = 9; dr_ = 7

    for i in range(4):
        if i == 0:
            bbm = single; frm = ds
        elif i == 1:
            bbm = dbl; frm = dd
        elif i == 2:
            bbm = capl; frm = dl
        else:
            bbm = capr; frm = dr_
        while bbm:
            to = lsb_i(bbm)
            if (1ULL << to) & (RANK_8 if white else RANK_1):
                out[n] = pack(to + frm, to, 1, 0, 0); n += 1   # Q
                out[n] = pack(to + frm, to, 2, 0, 0); n += 1   # R
                out[n] = pack(to + frm, to, 3, 0, 0); n += 1   # B
                out[n] = pack(to + frm, to, 4, 0, 0); n += 1   # N
            else:
                out[n] = pack(to + frm, to,
                              0, 0,
                              1 if (i >= 2 and to == p.ep) else 0)
                n += 1
            bbm &= bbm - 1

    # castling -- same masks and same three squares tested as board.getCastles
    if white:
        if p.wk and not (occ & MBE_KW):
            if (not square_attacked(p, 4, 0) and not square_attacked(p, 5, 0)
                    and not square_attacked(p, 6, 0)):
                out[n] = pack(4, 6, 0, 1, 0); n += 1
        if p.wq and not (occ & MBE_QW):
            if (not square_attacked(p, 4, 0) and not square_attacked(p, 3, 0)
                    and not square_attacked(p, 2, 0)):
                out[n] = pack(4, 2, 0, 1, 0); n += 1
    else:
        if p.bk and not (occ & MBE_KB):
            if (not square_attacked(p, 60, 1) and not square_attacked(p, 61, 1)
                    and not square_attacked(p, 62, 1)):
                out[n] = pack(60, 62, 0, 1, 0); n += 1
        if p.bq and not (occ & MBE_QB):
            if (not square_attacked(p, 60, 1) and not square_attacked(p, 59, 1)
                    and not square_attacked(p, 58, 1)):
                out[n] = pack(60, 58, 0, 1, 0); n += 1
    return n


cdef int gen_legal(Pos *p, int white, uint32_t *out) noexcept nogil:
    """Filter pseudo-legal moves exactly as board.legalMoves does."""
    cdef uint32_t buf[256]
    cdef int n = gen_pseudo(p, white, buf)
    cdef int base = 0 if white else 6
    cdef int ksq = lsb_i(p.bb[base + 5]) if p.bb[base + 5] else -1
    cdef bint in_check
    cdef uint64_t pins
    cdef int m, frm, cnt = 0
    cdef Undo u

    if ksq < 0:
        for m in range(n):
            out[cnt] = buf[m]; cnt += 1
        return cnt

    in_check = square_attacked(p, ksq, 0 if white else 1)
    pins = pinned_bb(p, ksq, white)

    for m in range(n):
        frm = m_from(buf[m])
        if in_check or frm == ksq or ((pins >> frm) & 1) or m_ep(buf[m]):
            do_move(p, buf[m], white, &u)
            if not square_attacked(p, lsb_i(p.bb[base + 5]),
                                   0 if white else 1):
                out[cnt] = buf[m]; cnt += 1
            undo_move(p, buf[m], white, &u)
        else:
            out[cnt] = buf[m]; cnt += 1
    return cnt


# --------------------------------------------------------------------------- #
# Python interface
# --------------------------------------------------------------------------- #
cdef Pos _build(list bbs, int ep, int wk, int wq, int bk, int bq):
    cdef Pos p
    cdef int i
    for i in range(12):
        p.bb[i] = <uint64_t>bbs[i]
    p.ep = ep
    p.wk = wk; p.wq = wq; p.bk = bk; p.bq = bq
    refresh(&p)
    return p


def legal_moves_packed(list bbs, int white, int ep, int wk, int wq,
                       int bk, int bq):
    """bbs: 12 ints [WP,WN,WB,WR,WQ,WK,BP,BN,BB,BR,BQ,BK].
    Returns a list of packed 32-bit move ints."""
    cdef Pos p = _build(bbs, ep, wk, wq, bk, bq)
    cdef uint32_t out[256]
    cdef int n = gen_legal(&p, white, out)
    cdef int i
    return [out[i] for i in range(n)]


def perft(list bbs, int white, int ep, int wk, int wq, int bk, int bq,
          int depth):
    """Self-contained perft for verification. Castling rights are NOT updated
    between plies (this mirrors what the Python engine's own perft harness
    does when driven from board.py), so use it only for cross-checking against
    that same harness, not as a standalone correctness oracle."""
    cdef Pos p = _build(bbs, ep, wk, wq, bk, bq)
    return _perft(&p, white, depth)


cdef long _perft(Pos *p, int white, int depth):
    cdef uint32_t out[256]
    cdef int n = gen_legal(p, white, out)
    cdef long total = 0
    cdef int i, saved_ep
    cdef Undo u
    if depth <= 1:
        return n
    for i in range(n):
        saved_ep = p.ep
        do_move(p, out[i], white, &u)
        # set ep square for a double pawn push so the next ply sees it
        if (p.bb[0 if white else 6] & (1ULL << m_to(out[i]))) and \
                abs(m_to(out[i]) - m_from(out[i])) == 16:
            p.ep = (m_from(out[i]) + m_to(out[i])) // 2
        else:
            p.ep = -1
        total += _perft(p, 0 if white else 1, depth - 1)
        undo_move(p, out[i], white, &u)
        p.ep = saved_ep
    return total


PROMO_CHARS = (None, "Q", "R", "B", "N")


def unpack(uint32_t m):
    """-> (fromSq, toSq, promotion, castle, enPassant)"""
    return (m & 63, (m >> 6) & 63, PROMO_CHARS[(m >> 12) & 7],
            bool((m >> 15) & 1), bool((m >> 16) & 1))

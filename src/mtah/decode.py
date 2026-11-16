"""Segmental decoding: Viterbi over {H, A, D} with switch penalties (frames scored independently, then smoothed
by the cost of changing state). Run separately inside each graded half."""
import numpy as np

H, A, D, X = 0, 1, 2, 3


def viterbi(emis, pen):
    """emis [n,3] log-score per state (higher = better); pen [3,3] cost of moving i->j. Returns int path."""
    n = len(emis)
    back = np.zeros((n, 3), np.int8)
    sc = emis[0].astype(np.float64).copy()
    for t in range(1, n):
        cand = sc[:, None] - pen
        back[t] = cand.argmax(0)
        sc = cand.max(0) + emis[t]
    path = np.zeros(n, np.int8)
    path[-1] = sc.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def penalties(p_switch, p_dead):
    pen = np.zeros((3, 3))
    pen[H, A] = pen[A, H] = p_switch
    for s in (H, A):
        pen[s, D] = pen[D, s] = p_dead
    return pen


def runs(path, state):
    m = np.concatenate(([0], (path == state).astype(np.int8), [0]))
    e = np.diff(m)
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def decode(emis, blocks, p_switch, p_dead, n_total, min_dead=0, protect=None):
    """Viterbi per block; then any dead run shorter than `min_dead` frames is forbidden and the block re-decoded
    (a stoppage shorter than a few seconds is slow live play, not a dead ball). `protect` [n_total] bool: frames
    where dead may not be forbidden (e.g. pre-kickoff)."""
    path = np.full(n_total, X, np.int8)
    pen = penalties(p_switch, p_dead)
    for a, b in blocks:
        e = emis[a:b].astype(np.float64).copy()
        pth = viterbi(e, pen)
        for _ in range(4 if min_dead else 0):
            short = [(i, j) for i, j in runs(pth, D) if j - i < min_dead
                     and not (protect is not None and protect[a + i:a + j].any())]
            if not short:
                break
            for i, j in short:
                e[i:j, D] = -1e6
            pth = viterbi(e, pen)
        path[a:b] = pth
    return path

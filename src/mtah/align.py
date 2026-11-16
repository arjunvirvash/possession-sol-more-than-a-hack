"""Broadcast -> tactical temporal alignment (label-free). See docs/notes/broadcast_alignment.md.

NOTE: io.load('broadcast') is wrong (parse_atd hard-codes 46 slots / 142 columns); use load_broadcast() here.
"""
import os
import numpy as np
from . import games, io

CACHE = io.CACHE


def parse_atd_n(raw_path, nslots):
    """Parse an ATD txt with `nslots` slots. Returns frame, xy [n,nslots,2] (0-1), speed, ball [n,2]."""
    with open(raw_path, "rb") as f:
        txt = f.read()
    txt = txt.replace(b":", b",").replace(b";", b",").replace(b"\n", b",")
    a = np.array(np.fromstring(txt.decode(), sep=","), dtype=np.float32)
    w = 1 + nslots * 3 + 3
    n = a.size // w
    assert a.size == n * w, (a.size, n, w)
    a = a.reshape(n, w)
    pl = a[:, 1:1 + nslots * 3].reshape(n, nslots, 3)
    return a[:, 0].astype(np.int64), pl[:, :, :2], pl[:, :, 2], a[:, -3:-1]


def load_broadcast(force=False):
    path = os.path.join(CACHE, "atd_broadcast_fixed.npz")
    if os.path.exists(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    meta, raw = games.require(["atd_meta", "atd_raw"], source="broadcast")
    team = io.slot_teams(meta)
    frame, xy, sp, ball = parse_atd_n(raw, len(team))
    L, W = io.PITCH
    d = dict(frame=frame, px=xy[:, :, 0] * L, py=xy[:, :, 1] * W, ps=sp,
             bx=ball[:, 0] * L, by=ball[:, 1] * W, team=team)
    hp = os.path.join(os.path.dirname(raw), "BAR-ATH_broadcast_homography.csv")
    import pandas as pd
    h = pd.read_csv(hp)
    d["hvalid"] = h["valid"].values.astype(bool)
    d["hrep"] = h["reprojection_error"].values.astype(np.float32)
    np.savez(path, **d)
    return d


# ---------------------------------------------------------------------------------------------
# Offset model: tactical_frame = broadcast_frame0 + off(half, b); two halves, constant up to a tiny drift.
HALF_SPLIT = 85500          # broadcast frame separating the two halves (halftime graphic gap 84692..86808)


def interp_ball(t, tf):
    """Tactical ball at fractional frames tf (NaN unless both neighbours observed)."""
    f0 = np.floor(tf).astype(int); w = tf - f0
    f0 = np.clip(f0, 0, len(t["bx"]) - 2)
    x = (1 - w) * t["bx"][f0] + w * t["bx"][f0 + 1]; y = (1 - w) * t["by"][f0] + w * t["by"][f0 + 1]
    return x, y


def ball_resid(t, b, fr, off):
    """Residual (m) between broadcast ball at frames fr and tactical ball at fr+off (fractional, interpolated)."""
    tx, ty = interp_ball(t, fr + off)
    d = np.hypot(b["bx"][fr] - tx, b["by"][fr] - ty)
    return d[np.isfinite(d)]


def fit_offsets(t, b, block=6000, lo_hi=((7200, 84700, 4014.0), (86800, 164400, 26920.0))):
    """Fractional per-block offset from ball residuals (robust mean of min(d,3)); returns per-half linear fit
    (a, s) with off(b) = a + s*(b-b0) plus the block table."""
    ball = np.isfinite(b["bx"]) & b["hvalid"]
    out = []
    for h, (lo, hi, base) in enumerate(lo_hi):
        rows = []
        for s0 in range(lo, hi, block):
            fr = np.flatnonzero(ball[s0:min(s0 + block, hi)]) + s0
            if len(fr) < 400: continue
            cand = base + np.arange(-3, 3.01, 0.25)
            sc = [np.minimum(ball_resid(t, b, fr, o), 3).mean() for o in cand]
            k = int(np.argmin(sc))
            # parabolic refine around best
            o = cand[k]
            if 0 < k < len(cand) - 1:
                y0, y1, y2 = sc[k - 1], sc[k], sc[k + 1]
                o = cand[k] + 0.25 * 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
            rows.append((h, s0 + block / 2, o, len(fr), sc[k]))
        rows = np.array(rows)
        out.append(rows)
    return out


def offset_model(t, b):
    """Linear per-half fractional offset: tactical = b + (a_h + s_h*b). Returns function off(b_array)."""
    fits = fit_offsets(t, b)
    pol = [np.polyfit(r[:, 1], r[:, 2], 1, w=r[:, 3]) for r in fits]
    def off(bf):
        bf = np.asarray(bf, float)
        return np.where(bf < HALF_SPLIT, np.polyval(pol[0], bf), np.polyval(pol[1], bf))
    return off, pol


def player_cost(t, b, bfr, tfr, clip=15.0):
    """Mean clipped nearest-neighbour distance between visible broadcast players at frames bfr and same-team
    tactical players at frames tfr (vectorised over frames; NaN if <min players). Also returns player counts."""
    n = len(bfr); cost = np.full(n, np.nan, np.float32); npl = np.zeros(n, np.int16)
    BIG = 1e4
    th, ta = t["team"] == 0, t["team"] == 1
    bh, ba = b["team"] == 0, b["team"] == 1
    def pk(d, m, fr):
        return np.nan_to_num(np.stack([d["px"][fr][:, m], d["py"][fr][:, m]], -1), nan=BIG).astype(np.float32)
    CH = 4000
    for s in range(0, n, CH):
        sl = slice(s, min(s + CH, n))
        tot = 0; cnt = 0
        for tm, bm in ((th, bh), (ta, ba)):
            P = pk(b, bm, bfr[sl]); T = pk(t, tm, tfr[sl])
            vis = P[..., 0] < BIG / 2
            D = np.sqrt(((P[:, :, None, :] - T[:, None, :, :]) ** 2).sum(-1)).min(2)
            D = np.minimum(D, clip)
            tot = tot + np.where(vis, D, 0).sum(1); cnt = cnt + vis.sum(1)
        cost[sl] = tot / np.maximum(cnt, 1); npl[sl] = cnt
    return cost, npl

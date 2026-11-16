"""Build cache/broadcast_ball.npy (+_conf) : broadcast ball on the tactical timeline, verified live (non-replay)."""
import sys, json
import numpy as np
sys.path.insert(0, "src")
from mtah import io, align
t = io.load("tactical"); b = align.load_broadcast()
N = len(t["bx"]); NB = len(b["bx"])
off, pol = align.offset_model(t, b)
print("poly", pol)
# --- per broadcast frame verification at the fitted offset (label-free: uses only players)
fr = np.arange(NB)
tf = np.clip(np.round(fr + off(fr)).astype(int), 0, N - 1)
inrange = (fr + off(fr) >= 0) & (fr + off(fr) < N - 1)
cost = np.full(NB, np.nan, np.float32); npl = np.zeros(NB, np.int16)
idx = np.flatnonzero(b["hvalid"] & inrange)
c, n = align.player_cost(t, b, idx, tf[idx])
cost[idx] = c; npl[idx] = n
np.savez("cache/align_cost.npz", cost=cost, npl=npl, tf=tf, inrange=inrange)

import pandas as pd
W = 11
med = pd.Series(cost).rolling(W, center=True, min_periods=3).median().values
verified = b["hvalid"] & inrange & np.isfinite(med) & (med < 3.0) & ~(np.isfinite(cost) & (cost >= 6.0))
ballx, bally = b["bx"].copy(), b["by"].copy()
inpitch = (ballx > -3) & (ballx < 108.3) & (bally > -3) & (bally < 71)
ball_ok = verified & np.isfinite(ballx) & inpitch
print("valid", b["hvalid"].sum(), "verified", verified.sum(), "verified&ball", ball_ok.sum())
# --- place onto the tactical grid with sub-frame interpolation
bo = np.full((N, 2), np.nan, np.float32); conf = np.zeros(N, np.float32); src = np.full(N, -1, np.int64)
tfrac = fr + off(fr)                                     # fractional tactical frame of each broadcast frame
for h, (lo, hi) in enumerate(((0, align.HALF_SPLIT), (align.HALF_SPLIT, NB))):
    p = pol[h]
    tt = np.arange(max(0, int(np.ceil(lo + np.polyval(p, lo)))), min(N, int(np.floor(hi + np.polyval(p, hi)))))
    # invert t = b*(1+s)+a  ->  b = (t-a)/(1+s)
    bb = (tt - p[1]) / (1 + p[0])
    b0 = np.floor(bb).astype(int); w = bb - b0; b1 = np.minimum(b0 + 1, NB - 1)
    ok0, ok1 = ball_ok[b0], ball_ok[b1]
    both = ok0 & ok1
    x = np.where(both, (1 - w) * ballx[b0] + w * ballx[b1], np.where(w < 0.5, ballx[b0], ballx[b1]))
    y = np.where(both, (1 - w) * bally[b0] + w * bally[b1], np.where(w < 0.5, bally[b0], bally[b1]))
    okn = np.where(w < 0.5, ok0, ok1) | both
    # jump guard between the two samples (ball teleport)
    jump = both & (np.hypot(ballx[b0] - ballx[b1], bally[b0] - bally[b1]) > 3.0)
    okn &= ~jump
    bo[tt[okn], 0] = x[okn]; bo[tt[okn], 1] = y[okn]
    near = np.where(w < 0.5, b0, b1)
    q = np.clip(1 - np.nan_to_num(med[near], nan=3.0) / 3.0, 0, 1)
    conf[tt[okn]] = q[okn]; src[tt[okn]] = near[okn]
np.save("cache/broadcast_ball.npy", bo); np.save("cache/broadcast_ball_conf.npy", conf); np.save("cache/broadcast_ball_src.npy", src)

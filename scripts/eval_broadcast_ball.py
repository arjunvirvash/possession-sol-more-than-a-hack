import sys, numpy as np
sys.path.insert(0, "src")
from mtah import io, align, games
t = io.load("tactical"); N = len(t["bx"])
bo = np.load("cache/broadcast_ball.npy"); conf = np.load("cache/broadcast_ball_conf.npy"); src = np.load("cache/broadcast_ball_src.npy")
z = np.load("cache/align_cost.npz"); cost = z["cost"]
b = align.load_broadcast()
BLK = [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]
g = np.zeros(N, bool)
for a, c in BLK: g[a:c] = True
tb = np.isfinite(t["bx"]); bb = np.isfinite(bo[:, 0])
has_src = src >= 0
print("graded", g.sum(), "tact ball present", (g & tb).sum(), "missing", (g & ~tb).sum())
# verified frame coverage (any broadcast frame verified, even without ball): reconstruct from src not possible -> use cost
print("tactical graded frames with broadcast ball:", (g & bb).sum(), "%.1f%%" % (100 * (g & bb).sum() / g.sum()))
print("  H1 %.1f%%  H2 %.1f%%" % tuple(100 * bb[a:c].mean() for a, c in BLK))
miss = g & ~tb
print("tactical-missing graded frames filled by broadcast: %d / %d = %.1f%%" % ((miss & bb).sum(), miss.sum(), 100 * (miss & bb).sum() / miss.sum()))
print("union ball coverage graded: %.1f%% (tactical alone %.1f%%)" % (100 * (g & (tb | bb)).sum() / g.sum(), 100 * (g & tb).sum() / g.sum()))
for nm, (a, c) in zip(("H1", "H2"), BLK):
    m = np.zeros(N, bool); m[a:c] = True
    print(nm, "missing", (m & ~tb).sum(), "filled", (m & ~tb & bb).sum(), "%.1f%%" % (100 * (m & ~tb & bb).sum() / (m & ~tb).sum()))
# gap-length stratification of tactical missing runs
def runs(mask):
    d = np.diff(np.r_[0, mask.astype(int), 0]); s = np.flatnonzero(d == 1); e = np.flatnonzero(d == -1); return s, e
s, e = runs(g & ~tb)
L = e - s
for lo, hi in ((1, 5), (5, 25), (25, 100), (100, 500), (500, 10**7)):
    sel = (L >= lo) & (L < hi)
    fr = sum((e_ - s_) for s_, e_ in zip(s[sel], e[sel]))
    filled = sum(bb[s_:e_].sum() for s_, e_ in zip(s[sel], e[sel]))
    # fully covered gaps
    full = sum(bb[s_:e_].all() for s_, e_ in zip(s[sel], e[sel]))
    print("gap len [%d,%d): %d gaps, %d frames, %.1f%% filled, %d gaps fully covered" % (lo, hi, sel.sum(), fr, 100 * filled / max(fr, 1), full))
# accuracy where both
both = bb & tb
d = np.hypot(bo[:, 0] - t["bx"], bo[:, 1] - t["by"])
dd = d[both & g]
print("both observed (graded): n=%d  median %.2f m  p75 %.2f p90 %.2f p95 %.2f p99 %.2f; <0.5: %.1f%% <1: %.1f%% <2: %.1f%% <5: %.1f%% >10: %.2f%%" % (
    len(dd), *np.percentile(dd, [50, 75, 90, 95, 99]), *(100 * (dd < x).mean() for x in (0.5, 1, 2, 5)), 100 * (dd > 10).mean()))
# by confidence
for lo, hi in ((0, .5), (.5, .8), (.8, 1.01)):
    m = both & g & (conf >= lo) & (conf < hi)
    print("conf [%.1f,%.1f): n=%d median %.2f p90 %.2f >5m %.1f%%" % (lo, hi, m.sum(), np.median(d[m]), np.percentile(d[m], 90), 100 * (d[m] > 5).mean()))
# negative control: shift tactical ball by +/-N frames
for sh in (25, 100, 500, 5000):
    tx = np.roll(t["bx"], sh); ty = np.roll(t["by"], sh)
    dc = np.hypot(bo[:, 0] - tx, bo[:, 1] - ty); m = bb & np.isfinite(tx) & g
    print("control shift %d: median %.2f m, <2m %.1f%%" % (sh, np.median(dc[m]), 100 * (dc[m] < 2).mean()))
# ball speed (px) compared: by tactical ball speed
sp = np.r_[0, np.hypot(np.diff(t["bx"]), np.diff(t["by"]))] * 25
for lo, hi in ((0, 3), (3, 10), (10, 20), (20, 100)):
    m = both & g & (sp >= lo) & (sp < hi)
    print("tact ball speed [%d,%d) m/s: n=%d median err %.2f" % (lo, hi, m.sum(), np.median(d[m])))
# player-config evidence over verified frames
v = np.isfinite(cost)
print("player NN cost on valid frames: median %.2f m, <1.5m: %.1f%%" % (np.nanmedian(cost), 100 * (cost[v] < 1.5).mean()))
# coverage report: uncovered stretches of graded frames with no broadcast frame at all (by src / hvalid)
hv = np.zeros(N, bool)
fr = np.arange(len(b["bx"])); off_poly = None

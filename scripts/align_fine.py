"""Fine matching: for every valid-homography broadcast frame with >=6 players, cost curve over delta in [-60,60]
around the coarse dominant offsets (4015 for first half, 26920 second). Output cache/align_fine.npz."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from mtah import io, align
torch.set_num_threads(12)
BIG, D = 1e4, 60
t = io.load("tactical"); b = align.load_broadcast()
N = len(t["bx"])
def tp(team):
    m = t["team"] == team
    return torch.from_numpy(np.nan_to_num(np.stack([t["px"][:, m], t["py"][:, m]], -1), nan=BIG).astype(np.float32))
Th, Ta = tp(0), tp(1)
bh, ba = b["team"] == 0, b["team"] == 1
bf = np.flatnonzero(b["hvalid"] & (np.isfinite(b["px"]).sum(1) >= 6))
print(len(bf), flush=True)
o0 = np.where(bf < 85500, 4015, 26920)
deltas = np.arange(-D, D + 1)
C = np.full((len(bf), len(deltas)), np.nan, np.float32); NP = np.zeros(len(bf), np.int16)
def cost(P, T):
    if len(P) == 0: return 0
    d = torch.cdist(P[None].expand(T.shape[0], -1, -1), T)
    return d.min(2).values.clamp(max=15.0).sum(1)
t0 = time.time()
for i, f in enumerate(bf):
    tt = f + o0[i] + deltas
    ok = (tt >= 0) & (tt < N)
    ph = np.stack([b["px"][f][bh], b["py"][f][bh]], -1); ph = ph[np.isfinite(ph).all(1)]
    pa = np.stack([b["px"][f][ba], b["py"][f][ba]], -1); pa = pa[np.isfinite(pa).all(1)]
    ti = torch.from_numpy(tt[ok])
    c = (cost(torch.from_numpy(ph), Th[ti]) + cost(torch.from_numpy(pa), Ta[ti])) / (len(ph) + len(pa))
    C[i, ok] = c.numpy(); NP[i] = len(ph) + len(pa)
    if i % 10000 == 0: print(i, time.time() - t0, flush=True)
np.savez("cache/align_fine.npz", bf=bf, o0=o0, deltas=deltas, C=C, npl=NP)

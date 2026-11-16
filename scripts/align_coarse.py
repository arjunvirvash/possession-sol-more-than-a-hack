"""Coarse broadcast->tactical match: for sampled broadcast frames (valid homography, >=8 players) find tactical
frames (stride 5) whose same-team player configuration best matches (mean nearest-neighbour distance).
Outputs cache/align_coarse.npz. Label-free."""
import sys, time
import numpy as np, torch
sys.path.insert(0, "src")
from mtah import io, align
torch.set_num_threads(12)
SAMPLE, STRIDE, BIG = 25, 5, 1e4
t = io.load("tactical"); b = align.load_broadcast()
tt = np.arange(0, len(t["bx"]), STRIDE)
def prep(d, team, idx):
    m = d["team"] == team
    x = np.stack([d["px"][idx][:, m], d["py"][idx][:, m]], -1)
    return np.nan_to_num(x, nan=BIG).astype(np.float32)
Th, Ta = torch.from_numpy(prep(t, 0, tt)), torch.from_numpy(prep(t, 1, tt))
bf = np.flatnonzero(b["hvalid"])
bf = bf[::SAMPLE]
npl = np.isfinite(b["px"][bf]).sum(1)
bf = bf[npl >= 8]
print(len(bf), "samples", flush=True)
bh, ba = b["team"] == 0, b["team"] == 1
def cost(P, T, clip=15.0):
    # P (m,2) ; T (n,k,2) -> (n,) sum of clipped min dists, count
    if len(P) == 0: return 0, 0
    d = torch.cdist(P[None].expand(T.shape[0], -1, -1), T)  # (n,m,k)
    return d.min(2).values.clamp(max=clip).sum(1), len(P)
K = 5
out_t = np.zeros((len(bf), K), np.int64); out_c = np.zeros((len(bf), K), np.float32); med = np.zeros(len(bf), np.float32)
t0 = time.time()
for i, f in enumerate(bf):
    ph = np.stack([b["px"][f][bh], b["py"][f][bh]], -1); ph = ph[np.isfinite(ph).all(1)]
    pa = np.stack([b["px"][f][ba], b["py"][f][ba]], -1); pa = pa[np.isfinite(pa).all(1)]
    ch, nh = cost(torch.from_numpy(ph), Th); ca, na = cost(torch.from_numpy(pa), Ta)
    c = ((ch if nh else 0) + (ca if na else 0)) / (nh + na)
    v, ix = torch.topk(c, 60, largest=False)
    # keep top K separated by >=50 tactical frames
    sel = []
    for vv, ii in zip(v.numpy(), ix.numpy()):
        if all(abs(ii - s) * STRIDE >= 250 for s, _ in sel): sel.append((ii, vv))
        if len(sel) == K: break
    for k, (ii, vv) in enumerate(sel): out_t[i, k] = tt[ii]; out_c[i, k] = vv
    med[i] = c.median()
    if i % 200 == 0: print(i, time.time() - t0, f, out_t[i, 0], out_c[i, 0], med[i], flush=True)
np.savez("cache/align_coarse.npz", bf=bf, top_t=out_t, top_c=out_c, med=med)

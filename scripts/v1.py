import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from mtah import io, features as F

d = io.load(); N = F.N; g = F.graded_mask()
dH, dA, dist = F.team_dists(d)
ok = ~np.isnan(d["bx"])
SIG=float(os.environ.get('SIG',2.5)); V0=float(os.environ.get('V0',0))
lr = F.kernel_logratio(dist, d["team"], sigma=SIG)
if V0>0: lr = lr*F.control_weight(F.ball_speed(d), V0)
ms = F.smooth_mean(np.nanmean(np.where(np.isnan(d["ps"]), np.nan, d["ps"]), 1), 25)

def viterbi(emis, pen):
    """emis [n,3] scores (higher better), pen [3,3] switch cost. returns state path."""
    n = len(emis); back = np.zeros((n, 3), np.int8); sc = emis[0].copy()
    for t in range(1, n):
        cand = sc[:, None] - pen          # from i to j
        back[t] = cand.argmax(0); sc = cand.max(0) + emis[t]
    path = np.zeros(n, np.int8); path[-1] = sc.argmax()
    for t in range(n - 1, 0, -1): path[t - 1] = back[t, path[t]]
    return path

def run(P_alive, P_dead):
    dead_logit = np.where(ok, -2.0, 0.0) + 3.0 * (1.5 - ms) * np.where(ok, 0.3, 1.0) + np.where(ok, 0.0, 1.0)
    emis = np.zeros((N, 3), np.float32)
    emis[:, 0] = lr / 2; emis[:, 1] = -lr / 2
    emis[:, 2] = dead_logit * 0.5
    pen = np.zeros((3, 3)); pen[0, 1] = pen[1, 0] = P_alive
    pen[0, 2] = pen[1, 2] = pen[2, 0] = pen[2, 1] = P_dead
    path = np.full(N, 3, np.int8)
    for a, b in ((11185, 90652), (110808, 190198)):
        path[a:b] = viterbi(emis[a:b], pen)
    return path

if __name__ == "__main__":
    import pandas as pd
    L = np.array(list("HADX"))
    for Pa, Pd in [(float(sys.argv[1]), float(sys.argv[2]))]:
        p = run(Pa, Pd)
        lab = L[p]
        # second-half pre-kickoff dead
        out = os.path.join(os.path.dirname(__file__), "..", "evaluation", "results", os.environ.get("OUT","v1")+f"_{int(Pa)}_{int(Pd)}.csv")
        pd.DataFrame({"frame": np.arange(N), "label": lab}).to_csv(out, index=False)
        print(out)

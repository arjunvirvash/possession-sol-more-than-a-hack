"""Label-free dead-ball probability detector  (Task 3).      Output: cache/dead_signal.npy

What it is
----------
A per-frame probability P(ball dead / out of play) in [0, 1], float32, one value per VIDEO frame
(row i = video frame i, 199,319 values).  Values are meaningful on the graded frames
(11185-90651, 110808-190197); outside them they are computed the same way but should be ignored,
except that pre-kickoff frames (derived from the Smart Tagging kickoff tag) are set to 1.

No ground truth, no hand labels and no Smart Tagging are used.  It is a 2-state, unsupervised
(EM-fitted) hidden Markov model over 0.2 s blocks; the 'dead' state is the one with the lower collective
player speed.  Emissions (each state learns its own parameters from the data itself):
  1. log of the mean tracker speed of visible players, 3 s centred window       (Gaussian)
  2. log of the ball speed (median-smoothed ball, central difference over 0.5 s,
     1 s rolling median); only when the ball is observed                         (Gaussian, marginalised if unseen)
  3. ball outside the pitch rectangle by more than 1 m (observed frames only)    (Bernoulli)
  (ball visibility itself is NOT an emission: unsupervised EM turns it into a 'ball unseen' cluster)
The transition matrix is learned too (initialised sticky: ~10 s dead / ~20 s live), so isolated blips are
smoothed away and a dead stretch needs sustained evidence.  The posterior (forward-backward) is the output.

Characterisation of the individual signals (shares of graded frames, run lengths) and validation against
automatic, label-free references (set-piece tag restart windows; ball-at-feet proxy frames) is printed by
`python3 scripts/dead_signals.py`.  Optional: `--save-only` to just write the array.
"""
import sys
from _common import *
from mtah import tags as mtags

BLOCK = 5            # frames per HMM step (0.2 s)
EPS = 1e-3
# Ball *visibility* as an emission is deliberately OFF: EM then clusters on "ball unseen" (14% vs 99.9% seen),
# which is not dead ball (the ball is also unseen for long stretches of live play, ~40% of graded frames).
USE_OBS = False


def player_speed_signals(d, win=75):
    vis = ~np.isnan(d["px"])
    ps = np.where(vis, d["ps"], np.nan)
    with np.errstate(all="ignore"):
        mps = pd.Series(np.nanmean(ps, 1)).rolling(win, center=True, min_periods=15).mean().values
    return mps


def ball_signals(d):
    sx, sy = smooth_ball(d)
    v = ball_speed(sx, sy, lag=6)
    v1 = pd.Series(v).rolling(25, center=True, min_periods=8).median().values
    out_dist = np.maximum.reduce([-sx, sx - PL, -sy, sy - PW, np.zeros(len(sx))])
    out_dist[np.isnan(sx)] = np.nan
    return sx, sy, v, v1, out_dist


def blockify(a, n):
    nb = n // BLOCK
    with np.errstate(all="ignore"):
        return np.nanmean(a[:nb * BLOCK].reshape(nb, BLOCK), axis=1)


def features(d):
    n = len(d["bx"])
    mps = player_speed_signals(d)
    sx, sy, v, v1, out_dist = ball_signals(d)
    obs = (~np.isnan(sx)).astype(float)
    X = dict(
        lmps=np.log(np.maximum(blockify(mps, n), 0.05)),
        lbv=np.log(np.maximum(blockify(v1, n), 0.05)),
        out=blockify((out_dist > 1.0).astype(float) + 0 * out_dist, n),   # NaN where ball unseen
        obs=blockify(obs, n) > 0.5,
    )
    return X


def _loglik(X, P):
    """[T,K] emission log-likelihood; NaN features are marginalised out."""
    T = len(X["lmps"]); K = len(P["mu_p"])
    ll = np.zeros((T, K))
    for k in range(K):
        for key, mu, sd in (("lmps", P["mu_p"][k], P["sd_p"][k]), ("lbv", P["mu_b"][k], P["sd_b"][k])):
            x = X[key]; m = ~np.isnan(x)
            ll[m, k] += -0.5 * ((x[m] - mu) / sd) ** 2 - np.log(sd)
        o = X["out"]; m = ~np.isnan(o)
        ll[m, k] += o[m] * np.log(P["p_out"][k]) + (1 - o[m]) * np.log(1 - P["p_out"][k])
        if USE_OBS:
            ll[:, k] += np.where(X["obs"], np.log(P["p_obs"][k]), np.log(1 - P["p_obs"][k]))
    return ll


def forward_backward(ll, A, pi):
    T, K = ll.shape
    mx = ll.max(1, keepdims=True)
    B = np.exp(ll - mx)
    al = np.zeros((T, K)); c = np.zeros(T)
    a = pi * B[0]; c[0] = a.sum(); al[0] = a / c[0]
    for t in range(1, T):
        a = (al[t - 1] @ A) * B[t]; c[t] = a.sum(); al[t] = a / c[t]
    be = np.ones((T, K))
    for t in range(T - 2, -1, -1):
        be[t] = (A @ (B[t + 1] * be[t + 1])) / c[t + 1]
    g = al * be; g /= g.sum(1, keepdims=True)
    xi = np.zeros((K, K))
    for t in range(T - 1):
        m = al[t][:, None] * A * (B[t + 1] * be[t + 1])[None]
        xi += m / m.sum()
    return g, xi, (np.log(c) + mx[:, 0]).sum()


def fit(X_seqs, n_iter=40, verbose=True):
    """EM over several sequences (halves).  Returns params P with state 0 = dead (lower player speed)."""
    allp = np.concatenate([x["lmps"] for x in X_seqs]); allp = allp[~np.isnan(allp)]
    thr = np.quantile(allp, 0.35)
    P = dict(mu_p=np.array([np.quantile(allp, .15), np.quantile(allp, .7)]), sd_p=np.array([.4, .4]),
             mu_b=np.array([np.log(1.5), np.log(6.)]), sd_b=np.array([.8, .8]),
             p_out=np.array([.05, .01]), p_obs=np.array([.5, .6]))
    # sticky init: 10 s dead (50 blocks), 20 s live (100 blocks)
    A = np.array([[1 - 1 / 50., 1 / 50.], [1 / 100., 1 - 1 / 100.]]); pi = np.array([.3, .7])
    for it in range(n_iter):
        G = []; XI = np.zeros((2, 2)); LL = 0
        for X in X_seqs:
            g, xi, ll = forward_backward(_loglik(X, P), A, pi); G.append(g); XI += xi; LL += ll
        A = XI / XI.sum(1, keepdims=True)
        for k in range(2):
            for key, mk, sk in (("lmps", "mu_p", "sd_p"), ("lbv", "mu_b", "sd_b")):
                num = den = 0.; xs = []; ws = []
                for X, g in zip(X_seqs, G):
                    m = ~np.isnan(X[key]); xs.append(X[key][m]); ws.append(g[m, k])
                xs = np.concatenate(xs); ws = np.concatenate(ws)
                mu = (xs * ws).sum() / ws.sum(); P[mk][k] = mu
                P[sk][k] = max(np.sqrt((ws * (xs - mu) ** 2).sum() / ws.sum()), 0.15)
            xo = np.concatenate([X["out"][~np.isnan(X["out"])] for X in X_seqs])
            wo = np.concatenate([g[~np.isnan(X["out"]), k] for X, g in zip(X_seqs, G)])
            P["p_out"][k] = np.clip((xo * wo).sum() / wo.sum(), EPS, 1 - EPS)
            if not USE_OBS:
                continue
            xb = np.concatenate([X["obs"].astype(float) for X in X_seqs]); wb = np.concatenate([g[:, k] for g in G])
            P["p_obs"][k] = np.clip((xb * wb).sum() / wb.sum(), EPS, 1 - EPS)
        if verbose and it % 10 == 9:
            print("  EM it %2d  loglik %.1f  dead-share %.3f" % (it + 1, LL, np.mean(np.concatenate([g[:, 0] for g in G]))))
    if P["mu_p"][0] > P["mu_p"][1]:                       # make state 0 the slow (dead) one
        for key in P: P[key] = P[key][::-1].copy()
        A = A[::-1, ::-1]; pi = pi[::-1]
    return P, A, pi


def dead_probability(d, params=None, verbose=False):
    n = len(d["bx"])
    X = features(d)
    nb = len(X["lmps"])
    seqs, spans = [], []
    for h, (a, b) in HALVES.items():
        a_, b_ = a // BLOCK, (b + 1) // BLOCK
        spans.append((a_, b_)); seqs.append({k: v[a_:b_] for k, v in X.items()})
    P, A, pi = fit(seqs, verbose=verbose) if params is None else params
    # inference over whole video in one pass (non-graded stretches get the same treatment)
    g, _, _ = forward_backward(_loglik(X, P), A, np.array([.5, .5]))
    pb = np.full(n, np.nan, np.float32)
    pb[:nb * BLOCK] = np.repeat(g[:, 0], BLOCK).astype(np.float32)
    pb[nb * BLOCK:] = pb[nb * BLOCK - 1]
    pb[mtags.pre_kickoff_mask(n)] = 1.0                    # frames before a half's kickoff restart are dead
    return pb, (P, A, pi)


def runs_of(mask):
    dd = np.diff(np.r_[0, mask.astype(int), 0])
    return np.where(dd == 1)[0], np.where(dd == -1)[0]


if __name__ == "__main__":
    d = io.load()
    pb, (P, A, pi) = dead_probability(d, verbose=True)
    os.makedirs(CACHE, exist_ok=True)
    np.save(os.path.join(CACHE, "dead_signal.npy"), pb)
    print("saved cache/dead_signal.npy", pb.shape, pb.dtype)
    if "--save-only" in sys.argv:
        sys.exit()
    gm = graded_mask(len(pb))
    print("\nlearned params (state0 = dead):")
    for k in ("mu_p", "sd_p", "mu_b", "sd_b", "p_out", "p_obs"):
        print("  %-5s %s" % (k, np.round(P[k], 3)))
    print("  mean player speed (dead / live) = %.2f / %.2f m/s; ball speed = %.2f / %.2f m/s" % (
        *np.exp(P["mu_p"]), *np.exp(P["mu_b"])))
    print("  transitions\n", np.round(A, 4), "\n  implied mean stay: dead %.1f s, live %.1f s" % (
        BLOCK / FPS / (1 - A[0, 0]), BLOCK / FPS / (1 - A[1, 1])))
    for thr in (0.5, 0.7, 0.9):
        m = (pb > thr) & gm
        s, e = runs_of(m)
        L = (e - s) / FPS
        print("   [P>%.1f: %.1f%% of graded frames lie in stretches >= 5 s; %.1f%% in stretches >= 10 s]" % (
            thr, 100 * L[L >= 5].sum() * FPS / gm.sum(), 100 * L[L >= 10].sum() * FPS / gm.sum()))
        print("P>%.1f: share of graded frames %.1f%% (%.1f min of %.1f); %d stretches; length median %.1fs IQR [%.1f,%.1f] p10 %.1f p90 %.1f max %.0f; share <5s %.0f%%, 5-30s %.0f%%, >30s %.0f%% of stretches" % (
            thr, 100 * m[gm].mean(), m.sum() / FPS / 60, gm.sum() / FPS / 60, len(L), np.median(L), *np.percentile(L, [25, 75, 10, 90]), L.max(),
            100 * (L < 5).mean(), 100 * ((L >= 5) & (L <= 30)).mean(), 100 * (L > 30).mean()))

    # ------------------------------------------------------------------ signal characterisation
    from smart_tag_reliability import proxy_possession
    n = len(pb)
    sx, sy, v, v1, out_dist = ball_signals(d)
    mps = player_speed_signals(d)
    unseen = np.isnan(sx)
    vis = ~np.isnan(d["px"])
    flag = (pb > 0.5)
    print("\n== Raw signals on graded frames: share, run-length stats, overlap with detector flag (P>0.5) ==")
    print("   (detector flags %.1f%% of graded frames)" % (100 * flag[gm].mean()))
    cands = {
        "ball unseen": unseen,
        "ball unseen >= 5 s (run)": None,
        "ball stationary (v<0.5 m/s, seen)": (v1 < 0.5),
        "ball outside pitch >1 m": (out_dist > 1.0),
        "players slow: mean speed < 1.0 m/s (3 s)": (mps < 1.0),
        "players slow: mean speed < 1.3 m/s": (mps < 1.3),
        "ball unseen AND mean speed < 1.3": unseen & (mps < 1.3),
        "ball seen slow (<1.5 m/s) AND mean speed < 1.3": (v1 < 1.5) & (mps < 1.3),
    }
    # unseen runs >= 5 s
    s_, e_ = runs_of(unseen)
    m5 = np.zeros(n, bool)
    for a, b in zip(s_, e_):
        if b - a >= 5 * FPS: m5[a:b] = True
    cands["ball unseen >= 5 s (run)"] = m5
    for k, m in cands.items():
        m = np.nan_to_num(m.astype(float)).astype(bool) & gm
        a, b = runs_of(m)
        L = (b - a) / FPS
        print("  %-48s share %5.1f%%  runs %4d med %5.1fs p90 %6.1fs | flagged-given-signal %.0f%%  signal-given-flag %.0f%%" % (
            k, 100 * m[gm].mean(), len(L), np.median(L) if len(L) else 0, np.percentile(L, 90) if len(L) else 0,
            100 * flag[m].mean(), 100 * m[flag & gm].mean()))
    for N in (1, 2, 3, 5):
        stat = (v1 < 0.5) & ~unseen
        a, b = runs_of(np.nan_to_num(stat.astype(float)).astype(bool) & gm)
        ok = [(x, y) for x, y in zip(a, b) if (y - x) >= N * FPS]
        tot = sum(y - x for x, y in ok)
        fl = sum(flag[x:y].sum() for x, y in ok)
        print("  ball stationary >= %d s: %4d runs, %.1f%% of graded frames, %.0f%% of them flagged" % (N, len(ok), 100 * tot / gm.sum(), 100 * fl / max(tot, 1)))
    print("  ball-unseen & seen-border: ball seen outside pitch >1 m for >= 1 s: runs", sum(1 for x, y in zip(*runs_of((out_dist > 1).astype(bool) & gm)) if y - x >= FPS))

    # ------------------------------------------------------------------ validation vs automatic references
    prox = proxy_possession(d)
    near = np.zeros(n, bool)
    dead_win = np.zeros(n, bool)
    off = pd.read_csv(os.path.join(CACHE, "setpiece_offsets.csv")) if os.path.exists(os.path.join(CACHE, "setpiece_offsets.csv")) else None
    for e in tags():
        if e["code"] != "SET PIECES": continue
        sfr = e["start_frame"]
        near[sfr - 15 * FPS:e["end_frame"] + 15 * FPS] = True
        if e["tags"]["Type"].startswith("Kick"):
            dead_win[sfr:sfr + int(3 * FPS)] = True
        else:
            dead_win[sfr + int(1 * FPS):sfr + int(4 * FPS)] = True    # conservative: s+1 .. s+4 s (restart ~ s+6 s)
    print("\n== Validation against label-free references ==")
    print("  recall on tag-derived dead windows (s+1..s+4 s; kick-offs s..s+3 s): mean P(dead)=%.2f, flagged %.0f%% (n=%d frames)" % (
        pb[dead_win & gm].mean(), 100 * flag[dead_win & gm].mean(), (dead_win & gm).sum()))
    live = gm & (prox >= 0) & ~near
    print("  proxy-live frames (ball seen, slow, at one team's feet, >15 s from any set piece tag): flagged %.1f%% (n=%d)" % (100 * flag[live].mean(), live.sum()))
    fast = gm & (v1 > 8)
    print("  ball moving > 8 m/s (flight): flagged %.1f%% (n=%d)" % (100 * flag[fast].mean(), fast.sum()))
    if off is not None:
        rel = []; relstart = []
        S, E = runs_of(flag)
        for _, r in off.iterrows():
            if np.isnan(r.t_restart): continue
            sf = int(r.s) + 3 * FPS
            k = [(a, b) for a, b in zip(S, E) if a <= sf < b]
            if k:
                a, b = k[0]
                rel.append((b - r.t_restart) / FPS); relstart.append((a - r.s) / FPS)
        rel = np.array(rel); relstart = np.array(relstart)
        print("  flagged stretch containing s+3 s found for %d/%d tags with a ball-based restart" % (len(rel), off.t_restart.notna().sum()))
        print("    stretch END - ball restart (s): median %.1f IQR [%.1f, %.1f]" % (np.median(rel), *np.percentile(rel, [25, 75])))
        print("    stretch START - tag start (s): median %.1f IQR [%.1f, %.1f]" % (np.median(relstart), *np.percentile(relstart, [25, 75])))
    for h, (a, b) in HALVES.items():
        print("  half %d: flagged %.1f%% of graded frames" % (h, 100 * flag[a:b + 1].mean()))
    print("  excluding the pre-kickoff block: flagged %.1f%% of the remaining graded frames" % (
        100 * flag[gm & ~mtags.pre_kickoff_mask(n)].mean()))

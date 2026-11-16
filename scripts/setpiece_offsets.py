"""Task 1: where is each SET PIECES tag interval relative to the restart visible in the tracking?

No ground truth is used.  For every SET PIECES instance (start s, end e, frames) we measure, from the
ATD tracking only:
  t_out      frame the ball arrives at the dead-ball spot (enters the spot zone: touchline band <1 m for
             throw-ins, goal-area box +/-1.5 m for goal kicks, 3 m around the corner flag, centre circle
             for kick-offs).  Only for the zone-able types; NaN if the ball is not seen there.
  t_restart  frame the ball leaves the spot zone (end of the zone run that starts nearest s)  [ball-based]
             - for corners the ball is almost never observed at the flag, so we use the first observed
               ball frame after s ("ball reappears"), which is an upper bound (ball already in flight).
             - for free kicks we use the end of the stationary ball run nearest s.
  t_speed    frame the collective player speed (mean tracker speed, 1 s window) rises half-way from its
             dead-ball minimum to its post-restart maximum [players-based, independent of ball visibility].
Offsets are reported in seconds relative to the tag start s and tag end e: (t - s)/25, (t - e)/25.

Run:  python3 scripts/setpiece_offsets.py       (writes cache/setpiece_offsets.csv, prints tables)
"""
from _common import *


def zone_mask(kind, sx, sy):
    if kind == "Throw-in":
        return np.minimum(sy, PW - sy) < 1.0
    if kind == "Corner Kick":
        return np.hypot(np.minimum(sx, PL - sx), np.minimum(sy, PW - sy)) < 3.0
    if kind == "Goal Kick":
        return (np.minimum(sx, PL - sx) < 7.0) & (np.abs(sy - PW / 2) < 10)
    if kind.startswith("Kick Off"):
        return np.hypot(sx - PL / 2, sy - PW / 2) < 2.5
    return None


def bridge_runs(mask, unknown, gap=75, minlen=10):
    m = pd.Series(mask.astype(float))
    m[unknown] = np.nan
    m = m.interpolate(limit=gap, limit_area="inside").fillna(0).values > 0.5
    dd = np.diff(np.r_[0, m.astype(int), 0])
    return [(a, b) for a, b in zip(np.where(dd == 1)[0], np.where(dd == -1)[0]) if b - a >= minlen]


def stationary_runs(sx, sy, v, thr=1.0, gap=75, minlen=50):
    return bridge_runs(np.nan_to_num(v, nan=99) < thr, np.isnan(sx), gap, minlen)


def speed_rise(mps, s, lo=-10, hi=15, up=20):
    seg = mps[s + lo * FPS:s + hi * FPS]
    if np.isnan(seg).all():
        return np.nan
    i0 = int(np.nanargmin(seg)) + s + lo * FPS
    post = mps[i0:i0 + up * FPS]
    mx = np.nanmax(post)
    mn = mps[i0]
    thr = mn + 0.5 * (mx - mn)
    k = np.where(post >= thr)[0]
    return i0 + int(k[0]) if len(k) else np.nan


def measure(d):
    sx, sy = smooth_ball(d)
    v = ball_speed(sx, sy)
    mps = mean_player_speed(d, 25)
    unk = np.isnan(sx)
    rows = []
    cache = {}
    for e in tags():
        if e["code"] != "SET PIECES":
            continue
        t = e["tags"]["Type"]
        s, en = e["start_frame"], e["end_frame"]
        r = dict(type=t, team=e["tags"]["Team"], half=e["tags"]["Half"], s=s, e=en, t_out=np.nan,
                 t_restart=np.nan, t_speed=speed_rise(mps, s))
        zm = zone_mask(t, sx, sy)
        if zm is not None and t not in ("Corner Kick",) and not t.startswith("Kick Off"):
            if t not in cache:
                cache[t] = bridge_runs(zm & ~unk, unk)
            cand = [(a, b) for a, b in cache[t] if abs(a - s) < 12 * FPS or (a <= s < b)]
            if cand:
                a, b = min(cand, key=lambda ab: abs(ab[0] - s))
                r["t_out"], r["t_restart"] = a, b
        elif t.startswith("Kick Off"):
            # the ball is (almost) never observed at the centre spot before the kick; the first observed
            # raw ball sample within 4 m of the centre in [s, e+2 s] coincides with the strike.
            f = np.arange(s, en + 2 * FPS)
            ok = ~np.isnan(d["bx"][f]) & (np.hypot(d["bx"][f] - PL / 2, d["by"][f] - PW / 2) < 4.0)
            if ok.any():
                r["t_restart"] = f[np.argmax(ok)]
        elif t == "Corner Kick":
            obs = np.where(~unk[s:s + 30 * FPS])[0]
            if len(obs):
                r["t_restart"] = s + obs[0]
        else:  # free kick: stationary ball run nearest s
            if "st" not in cache:
                cache["st"] = stationary_runs(sx, sy, v)
            cand = [(a, b) for a, b in cache["st"] if abs(a - s) < 12 * FPS or (a <= s < b)]
            if cand:
                a, b = min(cand, key=lambda ab: abs(ab[0] - s))
                r["t_out"], r["t_restart"] = a, b
        rows.append(r)
    df = pd.DataFrame(rows)
    for c in ("t_out", "t_restart", "t_speed"):
        df[c + "_rel_s"] = (df[c] - df["s"]) / FPS
        df[c + "_rel_e"] = (df[c] - df["e"]) / FPS
    return df


def q(x):
    x = x.dropna()
    return "n=%2d med=%6.1f IQR=[%6.1f,%6.1f]" % (len(x), x.median(), x.quantile(.25), x.quantile(.75)) if len(x) else "n=0"


if __name__ == "__main__":
    d = io.load()
    df = measure(d)
    os.makedirs(CACHE, exist_ok=True)
    df.to_csv(os.path.join(CACHE, "setpiece_offsets.csv"), index=False)
    pd.set_option("display.width", 200)
    for t, g in df.groupby("type"):
        print("\n==", t, "(%d tags, tag length %.0f s)" % (len(g), ((g.e - g.s) / FPS).median()))
        for c, lab in (("t_out_rel_s", "ball arrives at spot  - s"), ("t_restart_rel_s", "ball leaves spot      - s"),
                       ("t_speed_rel_s", "player-speed rise     - s"), ("t_restart_rel_e", "ball leaves spot      - e"),
                       ("t_speed_rel_e", "player-speed rise     - e")):
            print("  %-26s %s" % (lab, q(g[c])))
        dur = ((g.t_restart - g.t_out) / FPS).dropna()
        if len(dur):
            print("  dwell at spot (leave-arrive): " + q(dur))
        lag = ((g.t_speed - g.t_restart) / FPS).dropna()
        if len(lag):
            print("  lag speed-rise after ball restart: " + q(lag))
    print("\nPooled lag of player-speed rise after ball restart (all types):", q(((df.t_speed - df.t_restart) / FPS)))
    print("Tag interval as share of the way to restart: restart is at (t_restart-s)/(e-s) =",
          (((df.t_restart - df.s) / (df.e - df.s)).groupby(df.type).median()).round(2).to_dict())

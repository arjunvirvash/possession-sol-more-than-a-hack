"""Task 2: which team does each Smart Tagging phase refer to, which codes imply that team has the ball,
and how reliable is that, measured against a LABEL-FREE proximity proxy (no ground truth is read).

Proxy: a frame has proxy-possession team T when the (median-smoothed) ball is observed, slow
(< SLOW m/s) and exactly one team has a player within RADIUS m of it (the other team's nearest player
is farther than RADIUS).  proxy frames ~ 'ball clearly at the feet of one team'.

Implied possession of a tag: team tag if the code is an attacking/possession code, the OPPONENT of the
team tag if the code is a defending code ('DEFENDING IN X THIRD', 'DEFENSIVE TRANSITION').

Run:  python3 scripts/smart_tag_reliability.py
"""
from _common import *

RADIUS, SLOW = 1.5, 3.0
ATT = ["BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL",
       "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "GOALS"]
DEF = ["DEFENDING IN DEFENSIVE THIRD", "DEFENDING IN MIDDLE THIRD", "DEFENDING IN ATTACKING THIRD",
       "DEFENSIVE TRANSITION"]


def proxy_possession(d, radius=RADIUS, slow=SLOW):
    """int8 array: 0/1 team with the ball by proximity, -1 where the proxy does not apply."""
    sx, sy = smooth_ball(d)
    v = ball_speed(sx, sy)
    dist = np.hypot(d["px"] - sx[:, None], d["py"] - sy[:, None])
    t = d["team"]
    d0 = np.nanmin(np.where(t[None] == 0, dist, np.nan), axis=1)
    d1 = np.nanmin(np.where(t[None] == 1, dist, np.nan), axis=1)
    ok = ~np.isnan(sx) & (v < slow)
    out = np.full(len(sx), -1, np.int8)
    out[ok & (d0 < radius) & (d1 >= radius)] = 0
    out[ok & (d1 < radius) & (d0 >= radius)] = 1
    return out


def implied_team(e, home_name="Barcelona"):
    """tracking team index that the tag says has the ball, or None."""
    nm = e["tags"].get("Team")
    if nm not in TEAM_NAME:
        return None
    tm = TEAM_NAME[nm]
    if home_name != "Barcelona":
        tm = 1 - tm
    if e["code"] in ATT:
        return tm
    if e["code"] in DEF:
        return 1 - tm
    return None


def agreement(prox, evs, home_name="Barcelona", trim=0):
    """per code: pooled proxy-frame agreement and per-instance-majority agreement."""
    rows = []
    for code in ATT + DEF:
        agree = tot = 0
        inst_ok = inst_n = 0
        ninst = 0
        for e in evs:
            if e["code"] != code:
                continue
            imp = implied_team(e, home_name)
            if imp is None:
                continue
            ninst += 1
            seg = prox[e["start_frame"] + trim:e["end_frame"] - trim]
            seg = seg[seg >= 0]
            if len(seg) == 0:
                continue
            a = int((seg == imp).sum())
            agree += a
            tot += len(seg)
            if len(seg) >= 25:
                inst_n += 1
                inst_ok += a > len(seg) / 2
        rows.append(dict(code=code, n_inst=ninst, n_proxy_frames=tot, pooled_agree=agree / tot if tot else np.nan,
                         inst_with_proxy=inst_n, inst_majority_agree=inst_ok / inst_n if inst_n else np.nan))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    d = io.load()
    n = len(d["bx"])
    gm = graded_mask(n)
    prox = proxy_possession(d)
    print("proxy frames: %d total, %.1f%% of graded frames  (home %.1f%%, away %.1f%%)" % (
        (prox >= 0).sum(), 100 * (prox[gm] >= 0).mean(), 100 * (prox[gm] == 0).mean(), 100 * (prox[gm] == 1).mean()))
    evs = [e for e in tags() if e["code"] != "SET PIECES"]
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    print("\n== Team-name mapping check (attacking codes pooled, instance-majority) ==")
    for nm in ("Barcelona", "Atletic Club"):
        pass
    for hn in ("Barcelona", "Atletic Club"):
        a = agreement(prox, evs, hn)
        a = a[a.code.isin(ATT)]
        print("  home = %-12s -> pooled agreement over attacking codes: %.3f" % (
            hn, (a.pooled_agree * a.n_proxy_frames).sum() / a.n_proxy_frames.sum()))

    A = agreement(prox, evs)
    print("\n== Per-code agreement with proxy (name->team 0 = Barcelona) ==")
    print(A.to_string(index=False))
    for lab, codes in (("ATTACKING codes", ATT), ("DEFENDING codes (opponent has ball)", DEF)):
        a = A[A.code.isin(codes)]
        print("  %-38s pooled %.3f  instance-majority %.3f" % (
            lab, (a.pooled_agree * a.n_proxy_frames).sum() / a.n_proxy_frames.sum(),
            (a.inst_majority_agree * a.inst_with_proxy).sum() / a.inst_with_proxy.sum()))

    print("\n== Trimmed intervals (drop first/last 2 s of each phase) ==")
    A2 = agreement(prox, evs, trim=50)
    a = A2[A2.code.isin(ATT)]; print("  attacking pooled %.3f" % ((a.pooled_agree * a.n_proxy_frames).sum() / a.n_proxy_frames.sum()))
    a = A2[A2.code.isin(DEF)]; print("  defending pooled %.3f" % ((a.pooled_agree * a.n_proxy_frames).sum() / a.n_proxy_frames.sum()))

    print("\n== Agreement vs position inside the phase (tenths of duration; all non-N/A phases) ==")
    num = np.zeros(10); den = np.zeros(10)
    for e in evs:
        imp = implied_team(e)
        if imp is None:
            continue
        s, en = e["start_frame"], e["end_frame"]
        seg = prox[s:en]
        pos = (np.arange(len(seg)) * 10 // max(len(seg), 1))
        for k in range(10):
            m = (pos == k) & (seg >= 0)
            num[k] += (seg[m] == imp).sum(); den[k] += m.sum()
    print("  " + " ".join("%.2f" % x for x in num / den))
    print("  agreement in the 2 s before start / after end (is possession already/still the tagged team?):")
    for lab, off in (("   2 s before start", -1), ("   2 s after end", 1)):
        nn = dd = 0
        for e in evs:
            imp = implied_team(e)
            if imp is None:
                continue
            seg = prox[e["start_frame"] - 50:e["start_frame"]] if off < 0 else prox[e["end_frame"]:e["end_frame"] + 50]
            seg = seg[seg >= 0]
            nn += (seg == imp).sum(); dd += len(seg)
        print("  %s: %.3f (n=%d)" % (lab, nn / dd, dd))

    print("\n== Mirror pairs: DEFENDING/DEFENSIVE codes duplicate an attacking code with the opposite team ==")
    pairs = {"PROGRESSION": "DEFENDING IN MIDDLE THIRD", "BUILD UP": "DEFENDING IN ATTACKING THIRD",
             "CREATING CHANCES": "DEFENDING IN DEFENSIVE THIRD", "ATTACKING TRANSITION": "DEFENSIVE TRANSITION"}
    for a_, b_ in pairs.items():
        A_ = [(e["start_frame"], e["end_frame"], e["tags"]["Team"]) for e in evs if e["code"] == a_]
        B_ = [(e["start_frame"], e["end_frame"], e["tags"]["Team"]) for e in evs if e["code"] == b_]
        m_ = sum(any(x[0] == y[0] and x[1] == y[1] and x[2] != y[2] for y in B_) for x in A_)
        print("  %-22s <-> %-30s n=%d/%d, identical interval & opposite team: %d" % (a_, b_, len(A_), len(B_), m_))

    print("\n== Agreement vs seconds from phase start / to phase end (2 s bins; 'after end' = bins past the end) ==")
    for att, name in ((["PROGRESSION", "BUILD UP", "CREATING CHANCES", "ATTACKING TRANSITION", "LONG BALL"], "phase codes"),
                      (["BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX"], "zone codes")):
        for ref in ("start", "end"):
            out = []
            for lo_ in range(-6, 30, 2):
                nn = dd = 0
                for e in evs:
                    if e["code"] not in att:
                        continue
                    imp = implied_team(e)
                    if imp is None:
                        continue
                    if ref == "start":
                        a_, b_ = e["start_frame"] + lo_ * FPS, e["start_frame"] + (lo_ + 2) * FPS
                    else:   # bin [lo_, lo_+2) measured as seconds BEFORE the end (negative = after the end)
                        a_, b_ = e["end_frame"] - (lo_ + 2) * FPS, e["end_frame"] - lo_ * FPS
                    seg = prox[a_:b_]
                    seg = seg[seg >= 0]
                    nn += (seg == imp).sum(); dd += len(seg)
                out.append("%d:%.2f" % (lo_, nn / dd if dd else np.nan))
            print("  %-12s %s" % (name, "secs from start" if ref == "start" else "secs BEFORE end (neg = after)"), " ".join(out))

    print("\n== Coverage of GRADED frames by tags (union of intervals) ==")
    codes_dur = {}
    for e in tags():
        codes_dur.setdefault(e["code"], []).append((e["end_frame"] - e["start_frame"]) / FPS)
    for c, v in sorted(codes_dur.items(), key=lambda kv: -len(kv[1])):
        v = np.array(v)
        print("  %-30s n=%3d dur median %5.1fs [IQR %5.1f-%5.1f] total %6.1f min" % (c, len(v), np.median(v), *np.percentile(v, [25, 75]), v.sum() / 60))

    def cover(codes_pred):
        cnt = np.zeros(n, np.int16)
        poss = np.full((n, 2), 0, np.int16)
        for e in tags():
            if not codes_pred(e):
                continue
            cnt[e["start_frame"]:e["end_frame"]] += 1
            imp = implied_team(e)
            if imp is not None:
                poss[e["start_frame"]:e["end_frame"], imp] += 1
        return cnt, poss
    ng = gm.sum()
    c_all, _ = cover(lambda e: True)
    c_np, poss = cover(lambda e: e["code"] != "SET PIECES")
    c_imp, poss = cover(lambda e: implied_team(e) is not None)
    print("  any tag covers                : %.1f%% of graded frames" % (100 * (c_all[gm] > 0).mean()))
    print("  any possession-implying tag   : %.1f%%" % (100 * (c_imp[gm] > 0).mean()))
    both = (poss[:, 0] > 0) & (poss[:, 1] > 0)
    one = ((poss[:, 0] > 0) ^ (poss[:, 1] > 0))
    print("    unambiguous (one team only) : %.1f%%   conflicting (both teams implied) : %.1f%%" % (100 * one[gm].mean(), 100 * both[gm].mean()))
    for h, (a, b) in HALVES.items():
        print("    half %d: implied-possession coverage %.1f%%" % (h, 100 * (c_imp[a:b + 1] > 0).mean()))
    imp_lab = np.where(one & (poss[:, 0] > 0), 0, np.where(one & (poss[:, 1] > 0), 1, -1))
    m = gm & (imp_lab >= 0) & (prox >= 0)
    print("  overall proxy agreement on unambiguous covered frames: %.3f (n=%d)" % ((imp_lab[m] == prox[m]).mean(), m.sum()))
    # proxy frames outside any tag
    nt = gm & (prox >= 0) & (c_imp == 0)
    print("  proxy frames outside any implying tag: %.1f%% of graded proxy frames" % (100 * nt.sum() / ((prox >= 0) & gm).sum()))
    print("\n== Time-share of tag teams (sum of duration, minutes): Barcelona / Atletic ==")
    for nm in TEAM_NAME:
        print("  ", nm, round(sum((e['end_frame'] - e['start_frame']) / FPS / 60 for e in tags() if e['tags'].get('Team') == nm and e['code'] != 'SET PIECES'), 1))

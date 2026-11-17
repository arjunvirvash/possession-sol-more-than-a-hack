"""Smart Tagging -> per-frame evidence arrays (dead-ball prior from set pieces, weak team prior from phases)."""
import numpy as np
import xml.etree.ElementTree as ET
from . import games

FPS = 25
TEAM = {"Barcelona": 0, "Atletic Club": 1}
# agreement with a label-free proximity proxy, measured in docs/notes/smart_tagging_dead_ball.md (not GT)
ATTACK_RELIABILITY = {"BUILD UP": .864, "PROGRESSION": .858, "GOALS": .851, "LONG BALL": .792,
                      "BALL IN FINAL THIRD": .714, "ATTACKING TRANSITION": .62, "CREATING CHANCES": .576,
                      "BALL IN THE BOX": .576, "PLAYERS IN THE BOX": .573}
# seconds from tag start until the restart (medians measured in tracking, same note)
RESTART_AFTER = {"Throw-in": 6.0, "Goal Kick": 6.1, "Corner Kick": 6.7, "Free Kick": 9.0,
                 "Kick Off (After Goal)": 3.3, "Kick Off (Start)": 3.4}


def load():
    path = games.require(["smart_tagging"])[0]
    out = []
    for ins in ET.parse(path).iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in ins.findall("label")}
        out.append(dict(code=ins.findtext("code").strip().upper(), s=float(ins.findtext("start")),
                        e=float(ins.findtext("end")), tags=tags))
    return out


def pre_kickoff_mask(n):
    """[n] bool: graded-half frames before that half's kickoff restart. The restart is the 'Kick Off (Start)' set-piece
    tag of the half plus the measured restart offset (RESTART_AFTER); nothing here is tied to specific frame numbers."""
    m = np.zeros(n, bool)
    starts = {t["tags"].get("Half"): t["s"] for t in load()
              if t["code"] == "SET PIECES" and t["tags"].get("Type") == "Kick Off (Start)"}
    for half, a, b in games.half_video_blocks(games.GAME):
        name = {1: "1st Half", 2: "2nd Half"}.get(half)
        if name in starts:
            restart = int(round(starts[name] * FPS)) + int(round(RESTART_AFTER["Kick Off (Start)"] * FPS))
            if restart > a:
                m[a:min(restart, b + 1)] = True
    return m


def evidence(n):
    """returns (dead_prior [n] in {0,1} (set-piece window s..restart), team_logit [n] (+ => home), restart_frames)."""
    dead = np.zeros(n, np.float32)
    team = np.zeros(n, np.float32)
    restarts = []
    for t in load():
        a, b = int(round(t["s"] * FPS)), int(round(t["e"] * FPS))
        typ = t["tags"].get("Type")
        if t["code"] == "SET PIECES" and typ in RESTART_AFTER:
            r = a + int(round(RESTART_AFTER[typ] * FPS))
            lo = a - (FPS if typ == "Goal Kick" else 0)
            dead[max(lo, 0):min(r, n)] = 1.0
            restarts.append(r)
            continue
        p = ATTACK_RELIABILITY.get(t["code"])
        tm = TEAM.get(t["tags"].get("Team"))
        if p is None or tm is None:
            continue   # defending codes are exact mirrors of attacking ones -> no new information
        sign = 1.0 if tm == 0 else -1.0
        L = np.log(p / (1 - p))
        # reliability profile: full from start-2s to start+10s, fading to ~0 by +18s, 0 in last 3 s
        idx = np.arange(max(a - 2 * FPS, 0), min(b, n))
        age = (idx - a) / FPS
        w = np.clip((18 - age) / 8, 0, 1)
        w[idx > b - 3 * FPS] = 0
        np.add.at(team, idx, (sign * L * w).astype(np.float32))
    return dead, np.clip(team, -3, 3), restarts

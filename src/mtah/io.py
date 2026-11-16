"""Fast ATD parsing + cache. Arrays are indexed by VIDEO frame (0..199318) for the tactical source."""
import os, sys, xml.etree.ElementTree as ET
import numpy as np
from . import games

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "cache")
PITCH = games.GAMES[games.GAME]["pitch"]


def slot_teams(meta_path):
    """0 = home (Barcelona), 1 = away, per track slot."""
    root = ET.parse(meta_path).getroot()
    ids = [t.get("id") for t in root.iter("Team")]
    return np.array([0 if p.get("teamId") == ids[0] else 1 for p in root.iter("Player")], np.int8)


def parse_atd(raw_path):
    with open(raw_path, "rb") as f:
        txt = f.read()
    txt = txt.replace(b":", b",").replace(b";", b",").replace(b"\n", b",")
    a = np.fromstring(txt, sep=",", dtype=np.float64) if False else np.array(
        np.fromstring(txt.decode(), sep=","), dtype=np.float32)
    n = a.size // 142
    a = a[: n * 142].reshape(n, 142)
    frame = a[:, 0].astype(np.int64)
    pl = a[:, 1:139].reshape(n, 46, 3)
    ball = a[:, 139:142]
    return frame, pl[:, :, :2], pl[:, :, 2], ball[:, :2], ball[:, 2]


def load(source="tactical", force=False):
    """dict: px,py [n,46] metres-normalised->metres, ps speed, bx,by,bs [n], team [46]. Row i = video frame i."""
    path = os.path.join(CACHE, f"atd_{source}.npz")
    if os.path.exists(path) and not force:
        d = np.load(path)
        return {k: d[k] for k in d.files}
    meta, raw = games.require(["atd_meta", "atd_raw"], source=source)
    frame, xy, sp, ball, bsp = parse_atd(raw)
    L, W = PITCH
    d = dict(frame=frame, px=xy[:, :, 0] * L, py=xy[:, :, 1] * W, ps=sp,
             bx=ball[:, 0] * L, by=ball[:, 1] * W, bs=bsp, team=slot_teams(meta))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **d)
    return d

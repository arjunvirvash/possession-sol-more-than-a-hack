"""Shared helpers for the smart-tagging / dead-ball analysis scripts (no ground truth is read anywhere).

Everything is indexed by VIDEO frame (row i = video frame i), 25 fps, pitch 105.3 x 68.0 m.
"""
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "..", "more-than-a-hack-2027-main", "starter"))
from mtah import io, games  # noqa: E402
try:
    import load_smart_tagging as LST  # noqa: E402
except Exception:  # only the analysis scripts need it
    LST = None

FPS = 25
PL, PW = 105.3, 68.0
HALVES = {h: (a, b) for h, a, b in games.half_video_blocks(games.GAME)}   # graded video-frame spans (inclusive)
CACHE = os.path.join(ROOT, "cache")

# Smart Tagging team strings -> tracking team index (0 = home = Barcelona per DATA.md; verified in
# smart_tag_reliability.py by agreement with a label-free proximity proxy)
TEAM_NAME = {"Barcelona": 0, "Atletic Club": 1}


def graded_mask(n):
    m = np.zeros(n, bool)
    for a, b in HALVES.values():
        m[a:b + 1] = True
    return m


def tags():
    return LST.load()


def smooth_ball(d, win=9):
    """NaN-aware running median of the (noisy, jittery) ball track; NaN where < 3 samples in window."""
    out = []
    for k in ("bx", "by"):
        s = pd.Series(d[k].astype(np.float64))
        out.append(s.rolling(win, center=True, min_periods=3).median().values)
    return out[0], out[1]


def ball_speed(sx, sy, lag=6):
    """Speed (m/s) from smoothed ball positions, central difference +-lag frames. NaN if an end is NaN."""
    v = np.full(len(sx), np.nan)
    dx = sx[2 * lag:] - sx[:-2 * lag]
    dy = sy[2 * lag:] - sy[:-2 * lag]
    v[lag:-lag] = np.hypot(dx, dy) / (2 * lag / FPS)
    return v


def mean_player_speed(d, win=25):
    """Mean tracker speed over visible players, centred rolling mean over `win` frames (1 s)."""
    ps = np.where(np.isnan(d["px"]), np.nan, d["ps"])
    with np.errstate(all="ignore"):
        m = np.nanmean(ps, axis=1)
    return pd.Series(m).rolling(win, center=True, min_periods=5).mean().values

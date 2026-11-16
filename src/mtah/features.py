"""Per-frame label-free evidence from the cleaned tracking."""
import numpy as np
from . import games

N = games.GAMES[games.GAME]["n_video_frames"]


def graded_mask():
    g = np.zeros(N, bool)
    for _, a, b in games.half_video_blocks(games.GAME):
        g[a:b + 1] = True
    return g


def team_dists(d):
    """Per frame: distance (m) from ball to nearest seen player of each team. inf where ball/players missing."""
    dx = d["px"] - d["bx"][:, None]
    dy = d["py"] - d["by"][:, None]
    dist = np.hypot(dx, dy)
    dist = np.where(np.isnan(dist), np.inf, dist)
    T = d["team"]
    return dist[:, T == 0].min(1), dist[:, T == 1].min(1), dist


def kernel_logratio(dist, team, sigma=2.5, clip=8.0):
    """log( sum_home exp(-d^2/2s^2) / sum_away exp(...) ), clipped; 0 where ball unseen."""
    w = np.exp(-(dist ** 2) / (2 * sigma ** 2))
    h = w[:, team == 0].sum(1) + 1e-9
    a = w[:, team == 1].sum(1) + 1e-9
    lr = np.log(h) - np.log(a)
    return np.clip(lr, -clip, clip)


def smooth_mean(x, win):
    k = np.ones(win) / win
    x = np.where(np.isnan(x), np.nanmean(x), x)
    return np.convolve(x, k, "same")


def ball_speed(d, win=7):
    """Ball speed (m/s) from positions via central difference over +-win/2 frames; NaN where unseen on either side."""
    bx, by = d["bx"].astype(np.float64), d["by"].astype(np.float64)
    h = win // 2
    vx = (np.roll(bx, -h) - np.roll(bx, h)) * 25.0 / (2 * h)
    vy = (np.roll(by, -h) - np.roll(by, h)) * 25.0 / (2 * h)
    return np.hypot(vx, vy)


def control_weight(speed, v0=7.0):
    """~1 when the ball moves slowly (carried/held), ->0 when in flight. Unknown speed -> 0.5."""
    c = np.exp(-(speed / v0) ** 2)
    return np.where(np.isnan(speed), 0.5, c)

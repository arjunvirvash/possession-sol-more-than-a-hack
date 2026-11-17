"""End-to-end: tracking + tags -> per-frame labels."""
import numpy as np
from . import io, features as F, tags, decode, games

LET = np.array(list("HADX"))


def blocks():
    return [(a, b + 1) for _, a, b in games.half_video_blocks(games.GAME)]


def run(params=None, dead_signal=None, return_parts=False):
    P = dict(sigma=2.0, v0=7.0, p_switch=30.0, p_dead=20.0, w_team=0.5, w_tag_dead=2.5, w_hmm=1.0,
             w_unseen=0.8, tag_team=0.3, min_dead=100)
    P.update(params or {})
    d = io.load(); N = F.N
    dist = F.team_dists(d)[2]
    lr = F.kernel_logratio(dist, d["team"], sigma=P["sigma"])
    lr = lr * F.control_weight(F.ball_speed(d), P["v0"])
    ok = ~np.isnan(d["bx"])
    dead_tag, team_tag, _ = tags.evidence(N)
    if dead_signal is None:
        import os
        dead_signal = np.load(os.path.join(io.CACHE, "dead_signal.npy"))
    p = np.clip(dead_signal.astype(np.float64), 1e-3, 1 - 1e-3)
    dead_logit = P["w_hmm"] * np.log(p / (1 - p)) + P["w_tag_dead"] * dead_tag - P["w_unseen"] * 0 \
        + P["w_unseen"] * (~ok)
    emis = np.zeros((N, 3), np.float32)
    emis[:, 0] = P["w_team"] * lr + P["tag_team"] * team_tag
    emis[:, 1] = -emis[:, 0]
    emis[:, 2] = dead_logit * 0.5
    emis[:, 0] += 0.0
    prot = tags.pre_kickoff_mask(N)
    path = decode.decode(emis, blocks(), P["p_switch"], P["p_dead"], N, P["min_dead"], prot)
    # frames before a half's kickoff restart are dead (derived from the Smart Tagging kickoff tag, see tags.py)
    path[prot] = decode.D
    labels = LET[path]
    return (labels, emis, path) if return_parts else labels

# Broadcast -> tactical alignment (label-free)

Code: `src/mtah/align.py`, `scripts/align_coarse.py` (global search), `scripts/align_fine.py` (local curves),
`scripts/build_broadcast_ball.py` (builds the arrays), `scripts/eval_broadcast_ball.py` (numbers below).
No GT used, nothing hand-labelled. Evidence uses only ATD (players, ball) and homography `valid`.

## Bug to know about
`io.load('broadcast')` is WRONG: `io.parse_atd` hard-codes 46 slots / 142 columns, broadcast has 51 slots (157 columns) ->
181,863 garbage rows. Use `align.load_broadcast()` (164,488 rows, cached in `cache/atd_broadcast_fixed.npz`, includes
`hvalid`, `hrep`). I deleted the bad `cache/atd_broadcast.npz` that the check created. Broadcast ATD is only meaningful
where homography is valid: players/frame 16.7 (valid) vs 1.5 (invalid), ball visible 76% (valid) vs 0.8% (invalid).

## 1. Coordinates / orientation / teams: identical
Same 0-1 normalisation (x along the 105.3 m axis), same orientation, same team order (team 0 = Barcelona in both).
At the aligned offset, mean same-team nearest-neighbour distance between visible broadcast players and tactical players is
0.70 m median. Controls on 6000 verified frames: teams swapped 6.20 m, x mirrored 11.8 m, y mirrored 6.7 m, both 12.5 m.
Sides: Barcelona defends the low-x end in half 1 and the high-x end in half 2 in BOTH sources
(home-minus-away mean x is negative in H1, positive in H2 for tactical and for broadcast segments).

## 2. Temporal mapping: one linear map per half (no piecewise cuts)
Coarse global search (4083 broadcast frames, every 25th, vs all tactical frames stride 5, cost = mean clipped NN distance,
true minima 0.3-1 m vs ~11 m median of random frames) gives 97% of samples at two constants:
* first half and pre-match: tactical = broadcast + ~4014 (broadcast frame 0 = tactical 4014; live-verified up to about tactical 88700)
* second half: tactical = broadcast + ~26920 (broadcast frames >= 86808)
The broadcast clock is continuous inside each half (cuts/replays do NOT shift the timeline; replays are simply wrong
content on the right clock). Replays appear as a few short runs with a different offset (3140-3640, 25100-26460, etc.,
earlier tactical moments) or as frames that fail verification.
Fractional refinement with the ball (robust residual vs interpolated tactical ball, 6000-frame blocks): there is a small
real clock-rate difference, slope -1.36e-5 (H1) and -1.41e-5 (H2) frame/frame (about 1 frame per 70k):
  H1: tactical = b + 4014.307 - 1.3567e-5*b      H2: tactical = b + 26921.29 - 1.4071e-5*b   (split at b = 85500)
Halftime gap: broadcast 84692..86808 invalid; tactical 88.7k..90.65k (end of H1) and 110808..113727 (start of H2, all D
anyway) have no broadcast counterpart.

## 3. Verification (replay / close-up rejection)
Per broadcast frame: homography valid AND rolling median (11 frames) of the player NN cost at the model offset < 3.0 m AND
own cost < 6 m. 98,385 of 108,248 valid frames pass (replays, close-ups, mismatches rejected); valid&ball = 76,029 pass.
Ball additionally must be inside the pitch (+-3 m margin) and no >3 m jump between the two interpolated frames.
Player evidence over valid frames: NN cost median 0.78 m, 79% < 1.5 m, 8% > 3 m (rejected).

## 4. Ball accuracy where both ball tracks exist (graded, n = 70,485)
median 0.75 m, p75 2.05, p90 5.5, p95 10.2; 58% < 1 m, 74% < 2 m, 89% < 5 m, 5.1% > 10 m.
Negative control (tactical ball rolled by 25/100/500 frames): median 6.1 / 20 / 36 m -> alignment is frame-accurate.
Errors are speed dependent (ball < 10 m/s: 0.7 m; 10-20: 1.0; > 20 m/s: 3.5 m): aerial balls are displaced by the
ground-plane homography (parallax) and big residuals come in sustained runs (median 25 frames; 83% of the bad frames are in
runs >= 25 frames), not isolated noise. Neither track is self-inconsistent there, so one of the two projects an aerial ball
differently; conf (0.7 m median for conf>0.8, 0.93 m for conf<0.5) only mildly separates them.

## 5. Gain (graded frames 11185-90651, 110808-190197 = 158,857)
* Tactical ball present 94,214 (59.3%); missing 64,643.
* Aligned verified broadcast ball available on 75,944 graded frames (47.8%).
* **Tactical-missing frames filled: 5,459 / 64,643 = 8.4%** (H1 8.1%, H2 8.8%); total ball coverage 59.3% -> 62.7%.
* Why so low: a verified live broadcast frame exists on only 33% of tactical-missing frames, and even there the broadcast
  ball is seen only 25.5% of the time (92.6% where tactical sees the ball): both trackers lose the ball at the same moments
  (occlusion/air/crowd; the long tactical gaps are mostly dead/out-of-play stretches where the broadcast is not on the pitch).
* By gap length: gaps >= 500 frames (44 gaps, 43k frames) 3.7% filled, none fully; 100-499 frames (89 gaps, 21k frames)
  18% filled, 14 gaps fully covered; gaps < 25 frames are only 75 frames in total.
* Filled positions pass the broadcast own-track consistency check (99.9% within 2 m of the 11-frame median).

## Verdict
The alignment is reliable (frame-accurate, verified by 0.7 m player agreement and 0.75 m median ball agreement) and the
arrays are saved, but the ball gain is small: +5.5k frames (+3.4% of graded frames, 8.4% of the missing ones), mostly
inside long gaps, and the broadcast ball inherits ~11% >5 m disagreements in aerial phases. Worth using only as an optional
low-priority fill with conf (never to override a tactical ball). The broadcast player positions could additionally add
players to tactical frames where tactical tracking misses people (not explored).

Outputs (cache/): `broadcast_ball.npy` [199319,2] metres tactical coords NaN if none; `broadcast_ball_conf.npy` (0-1,
1 - median player NN cost/3; 0 where no ball); `broadcast_ball_src.npy` (source broadcast frame, -1 none);
`align_cost.npz` (per-broadcast-frame cost, #players, mapped tactical frame); `align_coarse.npz`, `align_fine.npz`.

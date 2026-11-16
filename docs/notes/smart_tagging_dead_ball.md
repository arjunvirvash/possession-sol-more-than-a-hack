# Smart Tagging timing / team semantics / dead-ball signals

All numbers are label-free (no ground-truth CSV read). Scripts: `scripts/setpiece_offsets.py`,
`scripts/smart_tag_reliability.py`, `scripts/dead_signals.py` (helpers in `scripts/_common.py`);
their printed reports are saved in `cache/*_report.txt`.

## 1. SET PIECES tag timing (setpiece_offsets.py)
Every set-piece tag is exactly 35.0 s long (kick-offs 5.0 s). The interval is NOT centred on the restart: it
starts about when the ball goes out / is placed, and the restart is early in it (~17% of the way in).
Measured from tracking (s = tag start, e = tag end; ball restart = ball leaves the spot zone, corner = ball first
re-observed, kick-off = first observed ball sample at the centre spot; player-speed rise = 1 s mean speed
half-way from its dead-ball minimum to its post-restart maximum):

| Type (n) | ball restart - s, median [IQR] s | ball at spot - s | player-speed rise - s | restart - e |
|---|---|---|---|---|
| Throw-in (26; n=19 with ball) | 6.0 [4.2, 7.7] | +1.1 [-1.6, 5.4] | 9.5 [6.8, 11.9] | -29.0 [-30.8, -27.3] |
| Goal kick (10; n=8) | 6.1 [5.0, 6.7] | -1.3 [-5.6, -0.2] | 8.1 [6.6, 12.6] | -28.9 [-30.0, -28.3] |
| Corner (12; ball re-seen, upper bound) | 6.7 [5.8, 8.0] | ball unseen at flag | 8.2 [6.3, 9.6] | -28.3 [-29.2, -27.0] |
| Free kick (5; n=1 with ball) | ~4 (n=1) | -0.3 (n=1) | 11.7 [7.0, 12.4] | -23.3 |
| Kick-off after goal (4; n=3) | 3.3 [3.1, 3.3] | - | noisy | -1.7 |
| Kick-off start (2) | 3.3 [3.2, 3.5] | - | 4.3 | -1.7 |

Consistency checks: published kick-offs (11185, 113742) are s+3.5 s and s+3.3 s; the ball is first observed at the
centre spot on exactly that frame in 5/6 kick-offs (it is basically only tracked once it moves). Pooled lag of the
player-speed rise after the ball restart is 2.5 s [0.1, 4.6], so speed rise is a late and noisy restart proxy.

Rule (tag -> estimate):
- Throw-in / goal kick / corner: dead ball from about s (goal kick s-1 s; throw-in s to s+1 s) until
  restart_frame = s + 6 s (= s + 150 frames), uncertainty about +/-1.5 s (IQR). After s+6 s the tag is just a 29 s
  window of live play.
- Kick-off (after goal or start): restart_frame = s + 3.3..3.5 s (= e - 1.7 s); dead until then.
- Free kick: low confidence (n=5, ball almost never seen); restart ~ s + 9 s (player-speed rise 11.7 s minus
  2.5 s lag), range 4-12 s.
- Coverage is partial: tags are only ~30% of the touchline/goal-area ball visits seen in tracking
  (90 touchline ball runs vs 26 throw-in tags; 45 goal-area runs vs 10 goal-kick tags; runs include some
  non-restart touches). Use the tags as a prior, not as the full list of stoppages.
- Team: set-piece Team is N/A for all throw-ins and kick-offs (36 N/A in total).

## 2. Team and code semantics (smart_tag_reliability.py)
- Mapping confirmed: Barcelona = team 0 (home). Agreement of attacking codes with the proximity proxy is 0.698 under
  Barcelona=0 and 0.302 under the swap.
- Proxy: ball observed (median-smoothed), slow (<3 m/s), exactly one team has a player within 1.5 m. It exists on
  10.1% of graded frames (home 6.1%, away 4.0%).
- A defending code is an exact mirror of an attacking code with the opposite team (same start/end, e.g. PROGRESSION
  (66) <-> DEFENDING IN MIDDLE THIRD (66); BUILD UP <-> DEFENDING IN ATTACKING THIRD; CREATING CHANCES <-> DEFENDING IN
  DEFENSIVE THIRD (28/30 mirror; 2 are N/A team); ATTACKING <-> DEFENSIVE TRANSITION). So the defending codes add no
  independent information, only a duplicate, and their agreement numbers equal the mirrored attacking ones.
- Per-code agreement (named team has the ball; defending codes: opponent has the ball), pooled over proxy frames:

| code | n | agreement |
|---|---|---|
| BUILD UP / DEF. IN ATTACKING THIRD | 22 | 0.864 |
| PROGRESSION / DEF. IN MIDDLE THIRD | 66 | 0.858 |
| GOALS | 4 | 0.851 (tag median 169 s long: includes celebration/kick-off) |
| LONG BALL | 29 | 0.792 (8 instances with proxy) |
| BALL IN FINAL THIRD | 93 | 0.714 |
| ATTACKING / DEFENSIVE TRANSITION | 24 | 0.620 (instance-majority 0.46: weak) |
| CREATING CHANCES / DEF. IN DEF. THIRD | 28 | 0.576 |
| BALL IN THE BOX | 72 | 0.576 |
| PLAYERS IN THE BOX | 97 | 0.573 |

  Attacking codes pooled 0.698; defending codes pooled 0.783 (different mix of codes). Box codes are weak partly
  because the proxy rarely applies in crowded boxes; they are the least trustworthy possession cues.
- Reliability varies inside a phase (agreement per 2 s bin): high (0.84-0.92) from s-2 to ~s+8, falls to ~0.55 after
  s+14 s for the shorter phase codes; and in the last ~6 s before the end it is only ~0.45-0.55, and 4-6 s after
  the end it is 0.31 (the opponent has taken over). So a phase ends at/after the possession loss, and the tagged team
  is already in possession ~2 s before the start (0.82). Trust the first ~10 s and the middle, not the last 6 s.
- Coverage of graded frames: any tag 82.3%; any possession-implying tag 75.8% (half 1 79.4%, half 2 72.3%);
  unambiguous (one team only) 66.7%; conflicting (both teams implied by overlapping tags) 9.2%. Agreement with the
  proxy on the unambiguous frames is 0.749. 26.6% of graded proxy frames lie outside any implying tag.

## 3. Dead-ball signals and detector (dead_signals.py -> cache/dead_signal.npy)
Raw signals on graded frames (106 min):
- ball unseen 40.3% (median unseen run 11.4 s, p90 38 s): too common in live play to be a dead indicator alone.
- ball stationary (<0.5 m/s, seen) only 2.4% (runs median 1.2 s; >=2 s: 22 runs, >=5 s: 6 runs): the tracker only
  sees the ball when it moves, and the track is smooth/interpolated, so a long still ball is almost never observed.
- ball seen outside the pitch >1 m: 1.6% (57 runs, 28 of >=1 s).
- mean player speed (3 s) < 1.0 m/s: 22.8% of frames; < 1.3 m/s: 37.6% (median speed over graded frames 1.54 m/s;
  runs median 6.5 s, p90 33 s). This is the main usable signal. Dead mean player speed is ~0.95 m/s vs 2.3 m/s live.
- Team spread: no useful discrimination (AUC 0.45-0.62 against tag-derived dead windows; y-spread slightly smaller
  in dead windows) - not used. Player count visible AUC 0.67 (not used).

Detector: unsupervised 2-state HMM (EM) on 0.2 s blocks with emissions log player speed, log ball speed (when seen),
ball-outside-pitch flag; learned transitions (mean stay dead 17 s, live 19 s). Ball visibility as an emission was
rejected: EM then simply clusters "ball unseen" (14% vs 99.9% seen). 2nd-half frames 110808-113741 are forced to 1.
- Flags (P>0.5): 48.1% of graded frames = 50.9 min dead, 55 min live (football prior: ~55-60 effective of ~95-106).
  Half 1 47.0%, half 2 49.2% (47.1% without the pre-kickoff block).
- 169 stretches; length median 7.6 s, IQR [3.0, 21.6] s, p90 40 s, max 216 s; by stretch count 38% are <5 s,
  46% 5-30 s, 16% >30 s; by frames 45.6% of the graded frames lie in stretches >=5 s and 42.2% in >=10 s.
  The <5 s stretches are likely slow live play (they could be dropped with a min-duration of 5 s).
- Checks against automatic references: P(dead) on tag-derived dead windows (s+1..s+4 s) mean 0.86, flagged 87%;
  flagged stretches containing s+3 s exist for 40 of 45 tags with a ball-based restart and END at the ball restart
  +0.4 s [-0.2, +1.9] (so the offset is well localised at the end). Start is poorly localised: the stretch starts 15.6 s
  [-29.4, -2.7] before the tag start, because player speed decays long before the ball leaves play (calm build-up
  is included). Frames with the ball moving >8 m/s (flight) are flagged 9.7%.
- Caveat: "proxy-live" frames (ball at feet, slow) are flagged 49.4%: that proxy also fires on taker-standing-over-
  ball moments (untagged throw-ins, free kicks, keeper holds) so it is not a clean specificity measure.
- Limits: the detector is a calm-state detector. For better onset timing combine it with the ball cues (ball
  leaving the pitch, last observed ball position at the border) and the tag rule s .. s+6 s; restart ends are good.

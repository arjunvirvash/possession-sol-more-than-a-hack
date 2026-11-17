# GUIDE: how this solution works, how we got here, and why

This is the "teach me everything" document. `README.md` is the quick start, `TECHNICAL.md` is the short
write-up for the organisers, `docs/notes/` holds the two research notes.
Read this one if you want to understand, defend or extend the solution.

---

## 1. The problem in one paragraph

One match (Barcelona v Athletic). For each of 199,319 video frames (25 fps) output `H` (Barcelona has the ball),
`A` (Athletic has the ball), `D` (ball dead) or `X` (abstain). Score = **spellF1**: predicted possessions are
matched one-to-one to true possessions by temporal overlap (IoU 0.3 to 0.7). A possession is a maximal run of
`H` or of `A`, and `D` runs are stoppages, not possessions. So the goal is **recovering whole possessions with
the right start and end**, not getting most frames right. Tie-break is macroF1 (per-frame).
The published floor is 0.2433 (nearest player to the ball, majority vote over a window).

Why the baseline is bad: nearest-player flickers. A pass flies past an opponent and the label flips for a
moment, so it invents about 600 possessions where the match has a few hundred. spellF1 punishes every extra
fragment (each is a false positive and breaks the real possession it cuts).

---

## 2. The data (what we have, what each piece is)

| Asset | What it is | Used? |
|---|---|---|
| Tactical ATD (`*_tactical_atd.txt/.xml`) | Automatic tracking, one record per video frame: 46 anonymous player "slots" (x, y, speed) plus the ball (x, y, speed). Coordinates 0 to 1, we convert to metres (105.3 x 68.0). NaN = not seen. | **Yes, the core input** |
| Smart Tagging XML | 638 automatic phases (Build Up, Progression, Set Pieces ...) with team, half and set-piece type. Seconds from video start. | **Yes, as a weak prior** |
| Tactical homography | 3x3 pixel mapping per frame | No (no video; tracking is already in pitch coordinates) |
| Broadcast ATD / homography | Second camera's tracking, own timeline | Studied, **dropped** (see 6.3) |
| Videos (.mp4) | Not downloaded | No |
| Public ground truth (5 min) | Real labels for frames 11185 to 18684 | **Sanity checks only. Never trained or tuned on** (a rule) |

Key facts we measured (all label-free):
- Ball is missing on 40.7% of graded frames, almost entirely in **long gaps** (median about 11 s, 149 gaps).
  So interpolation is useless.
- During those gaps mean player speed is about 1.2 m/s versus 2.2 m/s in play. So **a missing ball is mostly a
  stoppage** (the tracker basically only sees the ball when it moves).
- Frame alignment: ATD row i is video frame i (`frame_id = video_frame + 1`; we load it so row index = video frame).
- Graded frames: 11185 to 90651 and 110808 to 190197. Second-half frames before the kickoff restart are dead; the span is derived from the Smart Tagging kickoff tag, not hard-coded.

---

## 3. The model, in one picture

There is **no neural network and no supervised training.** It is a small probabilistic pipeline:

```
 tactical ATD (players, ball) ───────────────┐
                                             ▼
 [features.py]  ball-player distances → team evidence "who controls the ball" (+ for H, - for A)
 [dead_signals] unsupervised HMM on player/ball speed → P(dead) per frame
 [tags.py]      Smart Tagging → set-piece dead windows + weak team prior
                                             ▼
 [pipeline.py]  combine into per-frame scores for the 3 states {H, A, D}
                                             ▼
 [decode.py]    Viterbi: cheapest path through H/A/D with switch costs, then drop dead runs < 4 s
                                             ▼
                 labels → submission CSV (199,319 rows)
```

The "model" is a **3-state hidden Markov model decoded with Viterbi**, whose emission scores are hand-designed
log-evidence terms from football logic, and whose transition costs are the "how hard is it to change possession"
knobs. One sub-component (the dead-ball detector) is a separately **EM-fitted 2-state HMM**, fitted on this
match's own tracking with no labels.

---

## 4. Each stage in detail

### 4.1 Loading (`src/mtah/io.py`)
Reads the raw ATD text in one vectorised pass (about 2 s; each line is `frame:p1;...;p46:ball`, we replace
separators and parse as floats into a `[n, 142]` array). Team of each slot comes from the metadata XML (first team =
home = Barcelona). Cached to `cache/atd_tactical.npz`. Arrays: `px, py, ps` (players, [n,46]), `bx, by, bs` (ball).

### 4.2 Team-control evidence (`src/mtah/features.py`)
For every frame where the ball is seen:
1. Distance from the ball to every seen player.
2. **Kernel log-ratio**: `log(Σ_home exp(-d²/2σ²)) − log(Σ_away exp(-d²/2σ²))`, σ = 2.0 m, clipped to ±8.
   Positive means home players are closer. It is a soft "nearest player", and it sums over several players, so a
   crowd of one team near the ball counts more than a lone defender.
3. **Control weight** `exp(-(ball_speed / 7 m/s)²)`. A slow ball (dribble, hold, reception) gets weight about 1;
   a fast ball (pass, shot, clearance) is down-weighted to about 0. The ball speed comes from positions
   (central difference over 7 frames). *This is the single most important idea*: a pass flying past an opponent
   should not flip possession, but an opponent controlling the ball should. It mirrors the official definition
   ("a change of possession is when an opposing player gains **controlled** possession").
4. Where the ball is unseen the evidence is 0 (the decoder then carries the previous state).

### 4.3 Dead-ball detector (`scripts/dead_signals.py`, output `cache/dead_signal.npy`)
Built by a helper agent, and worth understanding:
- Unsupervised **2-state HMM** (state 0 = dead, state 1 = live) over 0.2 s blocks, fitted by **EM** (40 iterations)
  on the tracking of this one match. No labels.
- Emissions: log mean player speed (3 s window), log ball speed when seen, and a "ball outside pitch" flag.
  The dead state is whichever learns the lower player speed (about 0.95 m/s vs 2.3 m/s).
- We deliberately did **not** feed it "ball unseen": EM just clusters on that and learns nothing about dead
  balls (the ball is unseen about 40% of the time, much of it live play).
- Learned transitions give sticky states (mean stay about 17 s dead, about 19 s live), so blips are smoothed.
- Output: P(dead) per frame, used as a log-odds term. Pre-kickoff frames (from the Smart Tagging kickoff tag) are forced to 1.
- Caveat (important): it is a **calm-state** detector. It finds when everybody slows down, so it is good at the
  end of a stoppage (restart localised to about +0.4 s) and poor at the start.

### 4.4 Smart Tagging (`src/mtah/tags.py`)
Two uses, both with measured, label-free calibration (`docs/notes/smart_tagging_dead_ball.md`):
- **Set-piece windows → dead ball.** Every set-piece tag is exactly 35 s long and starts about when the ball goes
  out; the restart happens at a measured median offset after the start: throw-in 6.0 s, goal kick 6.1 s,
  corner 6.7 s, kick-offs about 3.3 s, free kick about 9 s. We mark dead from tag start to the estimated restart.
  Tags cover only about 30% of real stoppages, so it is a prior, not the full list.
- **Phase codes → weak team prior.** Codes like Build Up / Progression imply the tagged team has the ball.
  The defending codes are exact mirrors of attacking ones (no new information), so we ignore them.
  Reliability per code was measured as agreement with a proxy ("ball slow and exactly one team within 1.5 m"),
  from 0.86 (Build Up) down to 0.57 (Players in the Box). The prior fades 10 to 18 s after a phase starts and
  is off in its last 3 s, because agreement drops there (measured). It is a small push (max about 0.3 per
  frame, clipped), mostly relevant when the ball is unseen.

### 4.5 Combining into 3 scores (`src/mtah/pipeline.py`)
Per frame:
```
score_H =  0.5 * team_evidence + 0.3 * tag_team_prior
score_A = -score_H
score_D =  0.5 * ( 1.0*logit(P_dead_HMM) + 2.5*set_piece_window + 0.8*ball_unseen )
```
Everything is a log-evidence ("how much more likely"). Dead gets a bonus when the ball is unseen because we found
that missing ball means stoppage in most cases.

### 4.6 Decoding (`src/mtah/decode.py`)
**Viterbi** finds the single best state sequence over each half given the scores and switch costs:
- H↔A switch costs **30** (you need several seconds' worth of clear opposing control to take a possession),
- alive↔D switch costs **20**.

Then the **minimum-stoppage rule**: any D run shorter than **100 frames (4 s)** is forbidden and the half is
re-decoded (up to 4 passes). Reason: a real stoppage lasts longer, and short "calm" dips are slow live play. This
fixed a visible failure (spurious short D inside possessions) and raised macroF1 0.70 → 0.78.

Finally, pre-kickoff frames are set to D, and everything outside the graded halves is `X`.

Why this beats the baseline structurally: the cost of switching makes the output a **small number of long, stable
segments**, exactly what spellF1 rewards, while the control weight stops passes from creating false switches.

---

## 5. How we arrived at it (the path)

1. **Understood the task** by reading all docs and the scorer's code. Key takeaways: spellF1 rewards
   whole possessions; `X` is forward-filled by the scorer (free inside a possession, costly at hand-overs); ground
   truth derives from corrected tracking; the public 5 min must not be tuned on.
2. **Looked at the data**: ball missing in long gaps, and missing-ball ≈ stoppage. That reframed the problem:
   dead-ball detection matters as much as team assignment.
3. **v1 (nearest-team evidence + simple dead rule + Viterbi)**: public spellF1 0.32, about equal to the baseline.
   Showed that the decoder alone is not enough.
4. **v2 (+ control weight, σ = 2 m)**: public spellF1 about 0.52. The "slow ball only" evidence was the big win.
   Switch penalties 25, 40, 60 all gave 0.516 to 0.525, a plateau, which is evidence we are not overfitting one knob.
5. **Two helper agents in parallel** (ran while I built v2):
   - *Smart Tagging and dead-ball study*: measured tag timing, tag reliability, built the dead-ball HMM.
   - *Broadcast alignment study*: showed the broadcast clock maps to tactical by one linear map per half
     (0.7 m agreement, frame-accurate) but only fills 8.4% of missing ball frames.
6. **v3 (+ dead-ball HMM + set-piece windows + tag team prior)**: macroF1 0.70 → 0.77.
7. **v3b (+ minimum stoppage 4 s)**: macroF1 0.78, spellF1 0.53. Frozen as final.
8. **Robustness check** (not selection): varied switch cost (15, 60, 100), dead cost (10, 40), σ (3.0) and the speed
   constant (5). Public spellF1 stayed 0.50 to 0.53 except at extremes (switch 15: 0.45, σ 3.0: 0.44).
   The final constants had been fixed before this sweep, from priors.
9. **Reproducibility**: cleared the cache, regenerated twice, and the SHA-256 of the CSV was identical.

### What we did NOT do, deliberately
- Tune on the public 5 minutes. It has 17 possessions; one possession moves spellF1 about 6 points and the
  bootstrap noise band is about ±0.4. The repo's own baselines show tuning on it picks the wrong parameter.
- Hand-label or hand-correct anything. Frames were inspected only to debug, and no rule is tied to specific frames.
- Train a neural net. No labels exist for this match, and a pseudo-label network would add risk without a way
  to validate it. CPU-only also made heavy training unattractive.

---

## 6. Things worth knowing

### 6.1 What the numbers mean
Public window (5 min): baseline spellF1 0.326 / macroF1 0.621, ours 0.529 / 0.782. This is a sanity check that
the pipeline works and is aligned, **not** a prediction of the full-match score (the official floor was 0.2433
there). Expect the same direction and a similar-sized gain, but nobody can verify it until the private GT is used.
Full-match structural indicators (label-free): 250 possessions, median about 10 s, D share 42%, versus the
baseline's 594 possessions with median 4.6 s.

### 6.2 Known weaknesses (the honest list)
- **Ball unseen but live** (about 7.6% of live frames): the previous team is carried over; no ball means no
  evidence. A players-only classifier could help (see 7).
- **Dead-ball onset** is imprecise (calm-state detector); we localise the *end* well but the *start* poorly.
- **Slow live play flagged as D**: partly fixed by the 4 s minimum, not eliminated.
- **Contested ownership**: when an opposing player is closest to the ball for a couple of seconds but the original
  team is still credited (observed in the public window around frames 11420 to 11470), we flip and the truth does not.
- Constants come from priors, not validation on this match.

### 6.3 Why the broadcast tracking was dropped
It aligns perfectly (tactical = broadcast + 4014 in half 1, + 26920 in half 2, with a tiny clock drift), but the
broadcast tracker loses the ball at the same moments (long stoppages), so it fills only 5,459 of 64,643 missing
frames (8.4%). The gain was small and risky (about 11% of its ball positions disagree by more than 5 m in
aerial phases), so it is documented but not in the final pipeline. The code is still in `src/mtah/align.py` and
`scripts/build_broadcast_ball.py` if you want to try it.

### 6.4 Rules compliance
Algorithmic labels only; no use of private GT or any human data about the match; the only learned component is
an unsupervised EM model on the match's own tracking; the repo regenerates the identical file; data is git-ignored.
(The competition terms also forbid sharing the data with hosted LLMs; you chose to allow that for this work.)

---

## 7. Where to improve next (ordered by expected value)

1. **Players-only possession classifier** for ball-unseen live frames: self-training from frames where the ball is
   slow and exactly one team is within 1.5 m; features such as team-centroid shift and pressing; validate on masked
   *observed* frames (label-free). Targets the 7.6% blind spot.
2. **Better dead-ball onset**: combine "ball left the pitch" with the last observed ball position and the set-piece
   tag, rather than relying on player calm alone.
3. **Touch/event detection** (ball direction change next to a player) to score control more precisely than distance.
4. **Optional broadcast fill** for long gaps (gaps of 100 to 499 frames get 18% filled).
5. **Tune hyper-parameters label-free**, for example choose the switch cost at the centre of the plateau of
   "possession count vs penalty" curves.

---

## 8. Repo map and how to run

```
src/mtah/io.py         parse + cache ATD                 src/mtah/decode.py    Viterbi + min-stoppage rule
src/mtah/features.py   ball speed, kernel evidence       src/mtah/pipeline.py  combine scores + decode  (all constants here)
src/mtah/tags.py       Smart Tagging evidence            src/mtah/align.py     broadcast alignment (unused in final)
scripts/run_all.py     one-command rebuild → submission/<team>.csv
scripts/dead_signals.py   dead-ball HMM (also run by run_all if cache missing)
evaluation/            vendored official scorer + public GT   baselines/   the four published baselines
submission/possession_final.csv   the current submission     docs/notes/   research notes from the two agents
```
Run: `export MTAH_DATA_DIR=<data folder>; python scripts/run_all.py --team <name>; python evaluation/validate_submission.py submission/<name>.csv`.
All tunable constants are in the `P = dict(...)` at the top of `pipeline.run`:
`sigma, v0, p_switch, p_dead, w_team, w_tag_dead, w_hmm, w_unseen, tag_team, min_dead`.

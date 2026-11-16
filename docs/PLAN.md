# Plan: frame-level ball-possession solution (More than a Hack 2027, Phase 1)

## Context
Label each of 199,319 tactical-video frames H / A / D / X. Ranked on spellF1 (possessions recovered whole,
IoU 0.3-0.7), macroF1 breaks ties. Floor to beat: **spellF1 0.2433** (nearest player to ball + 50-frame vote).
Inputs: tactical ATD (46 anonymous slots, normalised x,y,speed; ball missing on ~41% of graded frames),
tactical homography, Smart Tagging (638 auto phases), broadcast ATD/homography (unsynced). No videos on disk.
Only ground truth: public 5 min (frames 11185-18684, 21 segments) = sanity check only, never train/tune on it.
Graded frames: 11185-90651 and 110808-190197 (2nd half frames 110808-113741 are all D).
CPU only (12 cores, 15 GB, numpy/pandas/scipy/sklearn/torch present). Nothing in this plan needs a GPU;
if that changes I stop and ask.

What the public GT shows about the target: possessions are *controlled* spells (0.8 s to 42 s), brief opposing
touches do not flip it, dead stretches are 5-20 s, D->possession restarts are abrupt. Scorer: X is forward-filled
for spellF1, so abstaining inside a possession is free, abstaining at a hand-over is costly.

## Data handling
Per your instruction I (and subagents) may read the data freely, including raw frame dumps, to understand it.
Hard line: **no hand-labelling or hand-correcting anything** - I may inspect frames to design features and debug,
but every label and every constant must come from code, not from rules tied to specific frames. (Reminder, your
call: the competition terms also bar sharing data with hosted LLMs; you've accepted that.) data/ and derived
caches stay git-ignored and are never committed.

## Approach (no hand labels; every label comes out of an algorithm)
New repo `/home/imart/Downloads/Barca-hackathon-2027/possession-solution/` (git init, data via `MTAH_DATA_DIR`
pointing at the existing `more-than-a-hack-2027-main/data`, vendored copy of the MIT scorer + public GT).

Pipeline (`src/mtah/`):
1. **io.py** - fast vectorised ATD parser -> cached `.npz` (players [n,46,2], speeds, ball [n,2]), slot->team from
   metadata, video-frame alignment via `games.py` per-half offsets (reuse, do not rewrite), half/kickoff detection
   from tracking, attack direction per half from data (keeper / centroid), halves grading mask.
2. **ball.py** - clean + fill the ball: outlier rejection (>~35 m/s jumps, off-pitch), short-gap interpolation,
   physically constrained smoothing (Kalman/RTS), and a self-supervised *ball-from-players* regressor
   (permutation-invariant features, HistGradientBoosting on CPU) trained on frames where the ball IS observed,
   validated by masking observed frames (no GT involved). Gives ball estimate + uncertainty on long gaps.
   Stretch: align broadcast ATD to tactical (piecewise offsets via ball/centroid cross-correlation) to fill gaps
   from an independent camera; keep only if alignment is verifiably reliable.
3. **dead.py** - dead-ball (D) detector: ball out of bounds / stationary at a restart spot (corners, goal-kick
   area, touchline), team-wide speed collapse (players walking/standing), tracker-lost ball with static players,
   Smart Tagging SET PIECES windows as a prior once their alignment to the restart is measured. Pre-kickoff /
   half-time rules from detected kickoffs.
4. **control.py** - per-frame "who controls the ball" evidence: soft proximity of nearest player per team
   (distance- and speed-aware), ball-kinematics touch detection (velocity/direction changes near a player),
   possession-team classifier on players-only features (self-training: pseudo-labels from ball<~1.5 m from a
   player & slow; validated on held-out pseudo-labels) for ball-missing stretches; Smart Tagging team phases
   (Build Up / Progression / Creating Chances / Attacking Transition / Defending ... ) as weak, reliability-
   weighted evidence (reliability measured by agreement with high-confidence frames, not GT).
5. **decode.py** - segmental decoding, the core of the gain: semi-Markov / Viterbi over {H, A, D} with
   emission = combined evidence, min-duration for a "controlled" spell, switch penalty = evidence needed to
   hand possession over (opponent must keep control ~>=1-2 s / >=2 touches; deflections and aerial duels do not
   flip), restarts forced from D, and X emitted only deep inside well-supported spells (free under spellF1).
   Hyper-parameters (penalties, min durations) fixed from football priors and from unsupervised stability
   criteria (plateau of possession-count vs penalty), **not** from the public 5-min score.
6. **submit.py** - writes `frame,label` (199,319 rows), runs the vendored `validate_submission.py`.
7. Docs: README + technical document (approach, assets used/dropped, rules, limitations, repro command).

## Execution (parallel where independent)
- Step 0 (first action after approval): create the repo and save this plan into it as `docs/PLAN.md`
  (and keep it updated with decisions/results as we go).
- Phase A (parallel agents, free to inspect raw data): (a) parser/cache + QC + orientation + kickoff detection;
  (b) ball gap structure, cleaning, ball-from-players model, broadcast-alignment feasibility;
  (c) Smart Tagging time alignment vs. restarts + dead-ball signal study.
- Phase B: control evidence + dead-ball detector + decoder, built on the cache; ablations scored with
  self-consistency diagnostics (possession count/duration plausibility, flip stability), public window used once
  per major version as a pipeline-correctness check with `--bootstrap 1000`.
- Phase C: pick final config a priori, generate submission, validate, write repo docs, re-run end-to-end from
  scratch to confirm byte-identical reproduction (fixed seeds, pinned requirements).

## Critical existing code to reuse
`games.py` (asset_paths, half_video_blocks, video_to_id), `starter/baseline.py` (ATD line format, team mapping
`team_of_each_track`, `smooth`), `evaluation/possession_metrics.py`, `run_evaluation.py`, `validate_submission.py`.

## Verification
- `python evaluation/run_evaluation.py` with our CSV in `evaluation/results/`: must beat baseline_smooth50 on the
  public window (sanity, not a ranking claim; 17 possessions ~ +/-0.06 noise) and look sane on counts/durations.
- `validate_submission.py` passes (199,319 rows, frames 0..199318, labels in HADX).
- Ball model and possession classifier: held-out accuracy on masked *observed-ball* frames (label-free).
- Independent check that our full-match possession count/duration is in a realistic football range and far
  below the raw baseline's ~7x over-segmentation.
- Clean-checkout rerun reproduces the identical CSV (hash match).
- Compute stays CPU-only; stop and report if any step truly needs a GPU.

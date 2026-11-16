# Technical document

## Approach and why
Possession is a segmentation problem, not a per-frame one (spellF1 rewards whole possessions). Per-frame evidence
is therefore turned into segments by a Viterbi decode whose switch penalties encode the rule "possession changes
only when the opponent gains *controlled* possession".

## Assets used / dropped
- Used: tactical ATD (players + ball), tactical Smart Tagging.
- Tried, not used: broadcast ATD. It aligns to the tactical timeline exactly (one linear map per half, 0.7 m player
  agreement, `docs/notes/broadcast_alignment.md`) but fills only 8.4% of missing-ball frames (both trackers lose the
  ball in the same long stoppages), so it was left out of the final pipeline (arrays kept in cache/ only if rebuilt).
- Not used: homography (no video; tracking is already in pitch coordinates), videos.

## Training data
None. The only fitted component is an unsupervised EM-trained 2-state HMM on the tracking of this match itself
(no labels of any kind). Smart Tagging is Metrica's automatic output, not human annotation. Tag reliability and
set-piece timing offsets were measured against label-free proxies from the tracking.

## Possession rules
- Ball in flight (fast) carries no team evidence; possession persists through passes/clearances.
- A change needs sustained proximity of the opposing team to a slow ball (switch cost 30 nats ~ several seconds
  of clear evidence); deflections and fleeting contests do not flip.
- Dead ball: calm-state HMM probability + set-piece windows + ball-unseen; D runs < 4 s are re-decoded as live.
- Frames 110808-113741 (pre-kickoff, published as dead) are D. Halftime/outside = X (ungraded).

## Limitations
Ball missing in long gaps: the team is carried over from the previous possession (7.6% of alive frames). The
calm-state dead detector marks slow live play as D and is imprecise at stoppage onset; dead balls with the ball
visible are only caught via player calm and set-piece tags. Contested duels where the opponent is closest but the
original team keeps ownership are mislabelled. Constants come from priors, not from validation on this match, because
only 17 public possessions exist; sensitivity of the public score to the main constants was flat (0.50-0.53) except
at extremes (switch cost 15, kernel sigma 3.0).

## Reproduction
See README.md; `scripts/run_all.py` rebuilds everything from the tracking and the tags.

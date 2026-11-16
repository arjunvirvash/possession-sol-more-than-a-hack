# More than a Hack 2027 - Phase 1 (ball possession): solution

Label every tactical-video frame H / A / D / X. Algorithmic, CPU-only, no hand labels, no training on the public
5 minutes. Deterministic: the same data gives byte-identical output (verified from a cleared cache).

## Reproduce
```bash
pip install -r requirements.txt
export MTAH_DATA_DIR=/path/to/data      # folder with tactical/ (and optionally broadcast/); videos are not needed
python scripts/run_all.py --team <team_name>     # -> submission/<team_name>.csv  (~1 min)
python evaluation/validate_submission.py submission/<team_name>.csv
cp submission/<team_name>.csv evaluation/results/ && python evaluation/run_evaluation.py
```
`data/`, `cache/` and result CSVs are git-ignored (the match data may not be redistributed).

## Method (see TECHNICAL.md)
1. Parse the tactical ATD, ball cleaned by speed; ball-to-player distances per team.
2. Team evidence = Gaussian-kernel log-ratio of home vs away players around the ball, weighted by how
   "controlled" the ball is (slow ball = carried/held; fast ball = pass in flight, ignored). Weak prior from
   Smart Tagging phases (reliability measured against a label-free proxy).
3. Dead-ball evidence = unsupervised 2-state HMM on player/ball speed, plus Smart Tagging set-piece windows
   (dead from tag start until restart = start + measured offset), ball-unseen term, pre-kickoff rule.
4. Viterbi over {H, A, D} with switch penalties (opponent must keep control to take a possession), stoppages
   shorter than 4 s disallowed. Constants fixed from priors, not tuned on the public window.

## Results (public 5 min, sanity check only: 17 possessions, noise band ~ +/-0.4 on spellF1)
| | macroF1 | spellF1 | possessions |
|---|---|---|---|
| baseline_smooth50 (published floor, 0.2433 on the full match) | 0.621 | 0.326 | 26 |
| this solution | 0.782 | 0.529 | 14 (truth 17) |

Full-match output: 250 possessions (median ~10 s) vs 594 for the baseline; D share 42%.
The full-match score is unknown (private GT); the public window cannot rank anything.

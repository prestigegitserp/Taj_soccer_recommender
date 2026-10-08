# Model card and methodology

**Taj v0.2 — Experimental baseline / October 2026.**

## Purpose
Screen match tracking for spatial access and candidate tactical exposures. Assist a human football analyst in reviewing episodes. Not a betting engine, not an autonomous coach, not a validated match predictor.

## Sources
IDSSE/Sportec open tracking and events (2022/23 Bundesliga); SkillCorner broadcast-derived A-League tracking (2024/25). Source matches are historical; no 2026 top-team continuous tracking is included.

## Reference frame
Canonical pitch centre: x ∈ [-52.5,52.5], y ∈ [-34,34], metres.
Kloppy is explicitly transformed to 105x68 metric dimensions and static home/away orientation before centering; SkillCorner raw exports use centred metres. Both physical direction and who attacks which goal must be checked against period metadata. Analysis takes defended-goal sign as a **user-supplied** parameter, never silently guesses at half-time.

## Access heuristic
At each grid point compute approximate time of arrival for each side (min across players). Approximate adjusted position after reaction time using capped velocity, with a shared speed ceiling. Logistic mapping of arrival-time difference creates an uncalibrated *relative access index*, not a measured probability.

## Exposure hypothesis
Compute mean opponent access across lanes of a defending team's final third. Select intervals above a configurable threshold, count at most one sample per disjoint 12-second window for each team/period/match. Require >=4 windows before emitting a hypothesis. Note that time windows may still be tactically dependent, and this cannot prove conceded chances.

## Recommendation
Map supported exposed zones to a **small explicit candidate playbook**. Rank by access intensity, nonadjacent support, and an arbitrary risk penalty. Scores are relative only, not probabilities, goal lift or causal estimates. Link every suggestion to frame IDs and match periods. If no support, return an empty list.

## Limitations
- Mixed providers may have different visibility, observation flags and raw event definitions.
- SkillCorner's new-format tracking can contain extrapolated positions; avoid interpreting them as physically observed.
- First derivative velocities can be noisy and become NaN over large timestamp gaps.
- The current code has no opponent trajectory prediction, robust pass interception, goalkeeper-specific physics, ball flight, uncertainty intervals or calibrated reception labels.
- Expected goals and future match outcome predictions are absent by design.
- Frames from the same match are correlated; evaluation should use held-out matches and analysts' labels, not random frame-level splitting.
- Small historical sample cannot justify universal generalization to live professional teams.

## Planned scientific evaluation
1. Check geometric invariance under controlled transformations and regression fixtures.
2. Annotate independent possession episodes with pass targets, receivers and access events.
3. Optimize arrival-time params on training matches only; test calibration on held-out matches.
4. Compare against simple baselines (nearest player / inverse distance).
5. Have blinded football analysts judge recommendations and log disagreements.
6. Release model cards with precision, coverage, sample count, error distributions and dated snapshots.

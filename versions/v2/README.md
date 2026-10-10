# TAJ Football Intelligence — V2.0.0

**[V2 Live](https://prestigegitserp.github.io/Taj_soccer_recommender/v2/)** ·
**[V2 Research](https://prestigegitserp.github.io/Taj_soccer_recommender/v2/research.html)** ·
**[Archived V1](https://prestigegitserp.github.io/Taj_soccer_recommender/v1/)**

## Version layout
| Version | Source | Public website | Model and data |
| --- | --- | --- | --- |
| V1.0.0 | `versions/v1/` plus immutable branch `v1.0.0` | `docs/v1/` | Unchanged 20/42-feature neural, Poisson, scores |
| V2.0.0 | `versions/v2/` | `docs/v2/` | 56-feature LightGBM, Dixon-Coles, V2 OOS report |

## How it works
1. Use 3,836 genuine completed ESPN match results.
2. Generate features from matches strictly before each kickoff: V1 20 form
   features, 22 causal opponent-graph features, 14 weighted recent-form and
   venue variables.
3. Compare independent Poisson, Dixon-Coles, LightGBM-42, LightGBM-56 and
   validation-selected ensembles.
4. Refit each ML candidate from scratch at 3 independently held-out
   calendar boundaries. Temperature, rho, and blend weights are selected on
   validation only. Save individual probability vectors and actual outcomes.
5. Refit production forest on all completed games only after fixing
   hyperparameters from earlier validation. Export the forest as static JSON.
6. The visitor's browser loads the archive/model and performs inference
   in a Web Worker, without a prediction backend or API key.

## Reproducibility
Training command (optional for developers, NOT required for site visitors):
```bash
python -m pip install "numpy>=1.26,<2.2" "tensorflow-cpu==2.19.1" "lightgbm==4.6.0" "scikit-learn==1.6.1"
python versions/v2/train_v2.py
```
Training runs automatically under `.github/workflows/v2-training.yml`.
Browser proofs run under `.github/workflows/v2-e2e.yml`.
Cross-language parity tests: `tests/test_v2_parity.py`.

## Falsifiable success criteria
A model is not promoted based solely on larger layers, high top-1 accuracy,
or one lucky test fold. Our automatic gate needs a 3-fold out-of-sample
paired *day-block 95% CI* below zero for the Brier difference and non-worse
Log Loss. Report every failure rather than selectively reporting wins.

The first V2 run raised 1X2 top-choice accuracy from 47.63% to 48.60%, but
failed the strict Brier promotion gate. The main V2 dashboard therefore
keeps Poisson as the official conservative probability and explicitly shows
LightGBM/Dixon-Coles challenge outputs.

**Next research needs**: historical pre-match xG, lineups, team/player
availability and tracking from sources with clear usage rights; larger
cross-season holdouts; truly prospective timestamped predictions.

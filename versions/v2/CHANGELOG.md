# TAJ V2 Changelog

## 2.1.0 — 2026-10-10

### Genuine prospective validation, before kick-off
- New append-only research ledger in `docs/v2/data/prospective.json`.
- Every prediction is captured at least 15 minutes before a real ESPN fixture's
  scheduled kickoff, with source data timestamp, probabilities and SHA256 of
  the exact production model; those probabilities are never replaced.
- Verified final scores are attached when observed, while original predictions
  remain frozen. Accuracy/Brier/Log Loss appear only after real finals exist.
- A 3-hour GitHub Actions schedule and an independent immutable-timestamp
  audit maintain the ledger; the browser still runs all interactive inference.
- First executed prospective log: 97 genuine pre-kickoff fixture records,
  with zero completed outcomes at time of the first log.
- Expose real prospective status in the V2 research dashboard.
- `v2.0.0` continues as an immutable release branch with the original
  first-trained 56-feature forest. The model artifact's training version
  remains `2.0.0` (the UI and research infrastructure are `2.1.0`).

## 2.0.0 — 2026-10-10

### Versioning and rollback
- V1 is archived byte-for-byte at `versions/v1/`, served from `docs/v1/`,
  and pinned to GitHub branch `v1.0.0`.
- V2 source lives at `versions/v2/`, browser app at `docs/v2/`, with
  independently generated model/evaluation/history artifacts.
- No previous model, visual, CI test or knowledge graph was deleted.

### Scientific challenger architecture
- Keep the reproducible Poisson baseline from V1.
- Add validated Dixon–Coles low-scoring goal correction with time-safe rho.
- Add two multiclass gradient-boosting challengers using 42 and 56 features.
- V2 features add 60-day-half-life weighted attack/defence/points, BTTS,
  clean sheets, goal-difference form trends and venue-rest adjustments.
- Use strict historical-only features. Current squad membership never
  retroactively influences older matches.
- Export LightGBM model as ~73 KB JSON forest, execute locally in plain
  JavaScript browser Worker with no AI server. Confirm Python/export parity.
- Train production forest using all completed historical games only *after*
  selecting temperature, rho and blend on earlier validation data.
- Preserve the reference forecast unless paired historical evidence meets
  predeclared promotion conditions.

### V2 evaluated on REAL football results
- 3 expanding-window training/validation/test partitions, 1,753 disjoint
  unseen completed ESPN match results from six competitions.
- V1 Poisson accuracy 47.633%, Brier 0.62433, 835 / 1,753 correct.
- V2 validation-selected challenger accuracy 48.602%, Brier 0.62457,
  852 / 1,753 correct. 17 more correct top-1 1X2 guesses.
- Standalone 56-feature LightGBM accuracy 48.431%, Brier 0.62420.
- Day-block bootstrap 95% CI for challenger minus Poisson Brier:
  [-0.003294, +0.004224]. Not statistically resolved.
- Therefore V2 is EXPERIMENTAL and does not automatically displace V1
  Poisson for the official forecast.
- Research process was iterated after inspecting V1 outcomes; not
  preregistered, unbiased blind external evaluation.

### Real product and test evidence
- V2 dashboard at `/v2/`; V2 method/results at `/v2/research.html`.
- 3,836 verified completed historical games downloaded after first load.
- Five nearest real fixtures, rival graph, sourced ESPN squad/boxscore panel.
- Chrome desktop/mobile and four additional *offline* fixture forecasts after
  first download tested by GitHub Actions.
- Independent Python-vs-JavaScript feature/forest parity tests.

### Limitations
- No historical licensed xG, confirmed lineups, injuries, video, or tracking
  in the learned match-level feature matrix.
- Backtest is chronological retrospective reconstruction, NOT a prospective
  ledger of actual bets or pre-match logged live forecasts.
- Football remains probabilistic; improvement is not guaranteed.

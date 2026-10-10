# TAJ Research Methodology and Model Card v0.3
**Updated 2026-10-08** · Research software / unvalidated tactical recommendations.

## The intended task

Assist a qualified analyst by visualizing movements in open historical tracking, detecting repeated geometric access to a defending third, and ranking hypotheses for manual review. This is NOT an autonomous coach, a causal tactical simulator, or a betting model.

## Data

- **IDSSE/Sportec:** 7 released Bundesliga / 2. Bundesliga games with detailed event + tracking. Kloppy \`sportec.load_open_tracking_data\` used for normalized tracking. Network integration smoke has verified 60 real frames, 1320 player observations and 60 ball coordinates for match \`J03WMX\`.
- **SkillCorner:** A-League 2024/25 10 broadcast-tracked games, each match with extrapolated player positions (\`is_detected\`), player metadata, dynamic events and phases. Git LFS files are separate from Git pointers. Dynamic events provide EPV/pressure signals but **are not** an exhaustive Opta-style event feed.
- **Synthetic demonstration:** procedurally generated geometry only; should never be used to claim team weaknesses or match results.

## Coordinate frames: critical

Canonical units: metres, a 105×68 centred grid. However, two *orientations* must not be conflated:
1. **Kloppy / IDSSE**: \`STATIC_HOME_AWAY\` transforms source positions so home attacks +x in *both* periods. This can rotate the physical second-half field; Home therefore *analytically* defends -x at all times.
2. **SkillCorner raw**: x/y centred on original field coordinates. Team defensive direction can switch after halftime. User MUST supply the defended goal sign for every period, confirmed against meta/video.

The algorithm ranks regions by **defended goal**, not one arbitrary left-to-right assumption. No automatic direction inference is claimed for raw SkillCorner.

## Provider ingestion and observation quality

- Schema includes match, provider, period, relative seconds, frame ID, side, player ID, x/y metre coordinates, is_detected.
- A value of \`False\` under \`is_detected\` means broadcast extrapolation rather than actual vision detection. Missing values are not treated as observed.
- Warm-up frames with \`period=None\` are dropped rather than misassigned.
- Positions outside field bounds with 1 m tolerance are excluded from spatial model rather than clamped to touchline.
- First differences are used for velocity and are ignored at long temporal gaps; later smoothing/calibration is a research milestone.
- Models can refuse a frame with fewer than 7 positioned players per team.

## Spatial access approximation

For each of the grid locations, compute minimum arrival time of each team, using:
- current position and measured velocity component projected along route;
- fixed reaction time; bounded acceleration up to a speed ceiling;
- nearest-team arrival time gap transformed with a sigmoid.

The result is a **relative access index**, not an empirically calibrated probability. Overlapping uncertainty and incomplete broadcast coverage can invalidate the ranking. A strict observed-only option is available in plots.

## Passing lanes

Two diagnostic algorithms:
1. Static geometric interception corridors (fast baseline).
2. Ball trajectory at a fixed hypothesized speed versus defender time of arrival to sampled pass segments.

These are **not** pass-completion predictions; they omit keeper positioning, tackle reach, acceleration uncertainty, ball height and player decision-making.

## Evidence and tactical recommendations

- Frames from the same possession are highly correlated; therefore evidence is sampled at separated time windows (default 12 seconds) and thresholded by geometric risk proxy.
- At least 4 separated windows are needed before emitting a hypothesis; 4 windows **do not establish statistical significance**.
- A small playbook maps spatial exposure zones to conditional candidate actions. Rankings are relative: intensity, support and risk aversion, not probability of scoring or success.
- Recommendations are attached to frames, periods and timestamps. Empty evidence = empty recommendations; refusal is a valid result.
- Advanced dashboard can shift one player to compare access maps. This is a sensitivity study, **not a simulated opponent response**.

## Optional match-result forecasting

Completely separate from spatial analysis. Inputs: a **user supplied** historical match results CSV. The model:
- fits average home/away goal rates strictly before match \`as_of\`;
- smooths each team's attack/defence samples toward league base rates;
- produces Poisson score distribution, normalized 1X2 probabilities;
- reports temporal walk-forward multiclass Brier, not random frame-level cross validation.

Missing recent injury, lineups, transfers, xG and tracking mean predictions can be badly misspecified. No live results feed or forecast calibration is asserted.

## Validation gates and future work

1. **Software:** Pytest regression tests for pitch bounds, conservation of access indices, symmetry, cross-period event alignment, no future leakage, provenance/CSV guards and no unsupported recommendations.
2. **Real provider smoke:** source dataset LFS/HTTP integration checks in GitHub Actions; do not claim full-game performance from a 60-frame smoke.
3. **Scientific:** manually labelled independent possessions from multiple games; calibrate arrival times and risk; compare held-out match performance against inverse-distance baseline; publish error bars.
4. **Tactical:** review selected video episodes blindly with two analysts, publish inter-rater agreement and false positives, inspect confounding.
5. **Data:** only compare teams with sufficient current *licensed* tracking; historical small public data is not a substitute for every future fixture.

## Security and reproducibility

- Git ignores downloaded match data and outputs.
- API only reads local prepared datasets by sanitized name.
- Browser view loads JSON locally (no upload to server); no private match data committed by default.
- User remains responsible for provider license and attribution. Code MIT license does not relicense external data.

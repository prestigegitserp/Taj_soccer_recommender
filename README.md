# ⚽ TAJ Soccer Recommender
### Explainable Spatial Football Intelligence · Python + Open Tracking · Colab + Interactive UI

**Status: experimental research MVP, not a validated match predictor.**

[![Python checks](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/ci.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prestigegitserp/Taj_soccer_recommender/blob/main/notebooks/Taj_Colab.ipynb)

[🔬 Google Colab notebook](https://colab.research.google.com/github/prestigegitserp/Taj_soccer_recommender/blob/main/notebooks/Taj_Colab.ipynb) · [🎛️ Static visual lab](https://prestigegitserp.github.io/Taj_soccer_recommender/) (enable Pages) · [📄 Methodology](docs/METHODOLOGY.md)

## What is implemented

- **IDSSE/Sportec**: Kloppy tracking + optional events, coordinate normalization to physical pitch metres.
- **SkillCorner**: local new-format Git LFS JSONL adapter; tracks \`is_detected\` separately.
- **Spatial**: access-map baseline with velocity-aware arrival estimates, static pass lanes, team width/length, attack-zone exposure.
- **Evidence**: report windows de-correlated in time; suppress conclusions when evidence is insufficient.
- **Explainable recommender**: small candidate playbook, relative rank, uncertainty and risk, referenced match periods / frames.
- **UI**: Persian-friendly Streamlit match explorer with timeline, heatmap, players/ball, intelligent recommendation cards and downloadable report.
- **Colab**: installation, demo, real-data acquisition (explicit opt-in), interactive pitch, report export and Drive option.
- **Static site**: standalone visual sandbox, explicitly synthetic. This does not execute Python remotely.
- **API + CLI**: simple FastAPI route and repeatable command-line pipeline.
- **Tests + CI**: numerical guards and evidence-gating.

**What is NOT implemented:** verified pre-match prediction, tracking for current top-league fixtures, end-to-end GNN training, causal effect of tactical interventions, or calibrated probabilistic pitch control. We do not invent metrics for unavailable tracking.

## 1. Run locally

\`\`\`bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all]"
pytest -q

taj demo --out data/demo
streamlit run dashboard.py
\`\`\`

The demo creates **SYNTHETIC** positions strictly for offline UI and smoke testing. No real match inference.

## 2. Analyze IDSSE (network needed)
\`\`\`bash
taj ingest --provider idsse --match J03WMX --limit 1500 --sample-rate 2 --no-events
taj analyze --data data/idsse_J03WMX --defending Home --goal 1
streamlit run dashboard.py
\`\`\`
Remove \`--no-events\` when you need event data and have sufficient Colab resources. The \`limit\` is for early smoke-testing, not full-match tactical scouting.

## 3. Analyze SkillCorner (LFS needed)
\`\`\`bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/SkillCorner/opendata.git external/opendata
(cd external/opendata && git lfs pull -I "data/matches/1925299/1925299_tracking_extrapolated.jsonl")
taj ingest --provider skillcorner --match 1925299 --source-dir external/opendata --limit 2000 --sample-rate 2
taj analyze --data data/skillcorner_1925299 --defending Away --goal -1
\`\`\`

Tracking is sampled at 10Hz in source; the CLI translates sample rate to a frame stride. *SkillCorner dynamic events are not a comprehensive event feed.*

## 4. Run API

\`\`\`bash
python -m pip install -e ".[api]"
uvicorn taj.api:app --reload
# GET /health ; GET /datasets ; POST /analyze  {"dataset":"demo","defending":"Home","goal":1}
\`\`\`

## Architecture

\`\`\`text
Open-data providers
  ├─ IDSSE (Kloppy) → normalized tracking / events
  └─ SkillCorner (Git LFS JSONL) → normalized tracking + observed flag
               ↓
       Parquet match bundles
               ↓
     Pitch Access + Geometry
               ↓
   Evidence windows + hypotheses
               ↓
     Rule-based tactical ranking
               ↓
  Streamlit / FastAPI / JSON export
\`\`\`

\`\`\`text
src/taj/{schema,ingest,spatial,weakness,recommender,analysis,plotting,cli,api}.py
dashboard.py
notebooks/Taj_Colab.ipynb
docs/index.html
docs/METHODOLOGY.md
tests/
.github/workflows/ci.yml
\`\`\`

## Deploy static GitHub Pages

Open **Settings → Pages → Build and deployment → Deploy from a branch**, select **main** and **/docs**, then save. The static lab will become available at [the project Pages URL](https://prestigegitserp.github.io/Taj_soccer_recommender/). Pages cannot compute Python tracking features or publish Colab data until you export and upload outputs. Static demonstration is labelled synthetic.

## Data provenance and licenses

- [IDSSE dataset](https://github.com/spoho-datascience/idsse-data) — scientific open dataset, attribution required (CC BY 4.0).
- [SkillCorner open data](https://github.com/SkillCorner/opendata) — respect separate dataset terms and attribution. Tracking is broadcast-inferred and some player positions are extrapolated.
- [Kloppy](https://kloppy.pysport.org/) — provider-normalization library.

**Code license:** MIT (this repo's source only). Source dataset rights and licenses are NOT transferred by this repository.

## Engineering guardrails

1. Physical coordinate system fixed across half-times, never silently infer attacking direction from just team name.
2. No cross-match synchronization; align timestamps only within each provider's match.
3. No leakage of future state into pre-match evaluation.
4. Count independent temporal evidence windows, not adjacent tracking frames as separate proof.
5. No exact-goal-lift or causal success percentages without labels and validation.
6. Keep Colab optional downloads gated; real match data must remain external to Git.

Contributions welcome: validation on independently labelled possessions, passing corridor reachability model, and pitch-access calibration are key next milestones.

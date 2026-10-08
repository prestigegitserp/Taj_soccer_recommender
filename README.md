# TAJ Soccer Recommender ⚽
## Football Tactical Intelligence · Tracking + Events + Spatial ML · Google Colab

[![CI](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/ci.yml/badge.svg)](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/ci.yml)
[![Real-data integration](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/integration.yml/badge.svg)](https://github.com/prestigegitserp/Taj_soccer_recommender/actions/workflows/integration.yml)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prestigegitserp/Taj_soccer_recommender/blob/main/notebooks/Taj_Colab.ipynb)

**فارسی:** این پروژه یک محیط تحقیقاتی برای تحلیل فضایی فوتبال، بررسی الگوهای دفاعی و ارائه پیشنهادهای تاکتیکی مستند به Tracking و Event Data است. از Google Colab برای پردازش Python و از Streamlit یا یک نمایشگر مستقل مرورگری برای Visual Analytics استفاده می‌کند.

**Crucial honesty:** This is an explainable **research system**, not a validated universal match oracle. It does not invent tracking from scores, prove tactics cause goals, or give calibrated winning chances without historical match-results input.

### 🚀 Quick links
- [Interactive Google Colab](https://colab.research.google.com/github/prestigegitserp/Taj_soccer_recommender/blob/main/notebooks/Taj_Colab.ipynb)
- [Static Visual Lab (synthetic)](https://prestigegitserp.github.io/Taj_soccer_recommender/)
- [Real-data browser viewer](https://prestigegitserp.github.io/Taj_soccer_recommender/viewer.html) — requires exporting a local JSON bundle and enabling GitHub Pages
- [Methodology / model limitations](docs/METHODOLOGY.md)
- [Provider licenses](https://github.com/SkillCorner/opendata) and [IDSSE documentation](https://github.com/spoho-datascience/idsse-data)

## Features and maturity

| Capability | Status | Interpretation |
|---|---|---|
| IDSSE Kloppy tracking | Network smoke tested | Reads real frames, normalizes to metres and frame IDs |
| SkillCorner Git LFS tracking | Integrated + integration tests | 10Hz broadcast observations with \`is_detected\` |
| Dynamic event feed | Preserved from SkillCorner CSV | Not a complete pass/shot event feed |
| Full IDSSE events | Ingestion API available | Event-to-frame clock sync requires verification per match |
| Advanced pitch access | Implemented; unit tested | Vectorized kinematic surrogate, NOT calibrated probability |
| Passing interception | Implemented; unit tested | Ball-flight vs defender arrival margin |
| Team compactness/spacing | Implemented; unit tested | Geometry diagnostics, not a tactical ground truth |
| Space exposure hypothesis | Implemented; evidence-gated | Requires independent time windows |
| Tactical recommendation | Rule-ranked candidates with source timestamps | Exploration only, not validated causal gain |
| What-if player movement | Implemented in dashboard and notebook | Geometry sensitivity, not match simulation |
| Pre-match 1X2 | Optional historical Poisson baseline | Requires user-provided CSV; no live injury/lineup data |
| Intelligent UI | 5-tab Streamlit and offline real-data viewer | Scouting workspace, not an autonomous LLM |
| Deep GNN / video-to-tracking | Not implemented | Needs much larger labelled data and compute |

## 1. Start in Colab (recommended)

Click [Open in Colab](https://colab.research.google.com/github/prestigegitserp/Taj_soccer_recommender/blob/main/notebooks/Taj_Colab.ipynb).

1. Run **Install**. It clones the repository and installs Python dependencies.
2. Run **Synthetic smoke test** and the interactive pitch — it verifies installation and clearly labels all demo data **SYNTHETIC**.
3. Change \`RUN_IDSSE = True\` in the IDSSE cell to fetch actual Bundesliga tracking.
4. Change \`RUN_SKILLCORNER = True\` in the SkillCorner cell for actual A-League data using Git LFS.
5. Choose a processed dataset and compute spatial analysis and evidence windows.
6. Export \`analysis_report.json\`, standalone HTML and \`viewer.json\`.
7. On the [real-data viewer](https://prestigegitserp.github.io/Taj_soccer_recommender/viewer.html), select \`viewer.json\` to replay **actual exported tracking frames and model heatmaps**.

No GPU or external LLM API key is needed for the current models. Full match processing may require reducing frame sampling to fit Colab RAM/time. Colab storage is ephemeral; save generated files to Drive if needed.

## 2. Local setup

Python 3.11 recommended:

\`\`\`bash
git clone https://github.com/prestigegitserp/Taj_soccer_recommender.git
cd Taj_soccer_recommender
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[all]"
pytest -q
taj demo --out data/demo
streamlit run dashboard.py
\`\`\`

## 3. Real-data ingestion and analysis

### IDSSE

\`\`\`bash
taj ingest --provider idsse --match J03WMX --sample-rate 2 --limit 1500 --no-events
taj analyze --data data/idsse_J03WMX --defending Home
taj viewer-export --data data/idsse_J03WMX --max-frames 150
taj html-export --data data/idsse_J03WMX
\`\`\`

\`--no-events\` makes the first test faster; remove it to download IDSSE events. For a full match, add \`--sample-all\`, then choose sampling rate according to resource constraints. Kloppy uses a normalized \`STATIC_HOME_AWAY\` orientation, so analytical Home defends -x in all periods; this is **not** necessarily the physical camera direction.

### SkillCorner

SkillCorner's tracking file is **Git LFS**; a 133-byte pointer file is *not* real tracking. You must fetch the actual content:

\`\`\`bash
git lfs install
GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/SkillCorner/opendata.git external/opendata
(cd external/opendata && git lfs pull -I "data/matches/1925299/1925299_tracking_extrapolated.jsonl")
taj ingest --provider skillcorner --match 1925299 --source-dir external/opendata --limit 3000 --sample-rate 2
\`\`\`

**IMPORTANT:** raw SkillCorner period-specific defended-goal directions must be verified. The signs below are *illustrative*, not verified for the specific match:

\`\`\`bash
taj analyze --data data/skillcorner_1925299 --defending Home --directions "1:-1"
# For full two-period data, specify a VERIFIED sign for EACH period, e.g. 1:-1,2:1
taj viewer-export --data data/skillcorner_1925299 --directions "1:-1,2:1" --max-frames 160
\`\`\`

If your sample covers both halves, include both periods. The API intentionally refuses to guess raw SkillCorner direction without an explicit sign.

### Event provenance

IDSSE contains an Event feed; SkillCorner's \`dynamic_events.csv\` is a different taxonomy with off-ball runs, pressure and EPV fields — NOT a universal full pass/shot feed. They are persisted separately by provider, and only same-match, same-period timestamps may be joined.

## 4. Full dashboard

\`\`\`bash
streamlit run dashboard.py
\`\`\`

The main interface has:
- **Match Explorer** — frame selector, player/ball tracks, pitch-access heatmap, team shapes and passing corridors;
- **Spatial Intelligence** — risk zones, evidence windows, replay individual frame references;
- **Tactical Recommender** — evidence-backed cards, relative rank, risks and explicit abstention;
- **What-if Lab** — move a player, recompute differential map (geometry-only counterfactual);
- **Data Quality & Export** — coverage, data provenance, JSON, standalone HTML and a web-viewer bundle.

## 5. Real interactive site on GitHub Pages

Under **Settings → Pages**, configure deployment from **main /docs**. Pages requires this one-time action from a repository administrator. The site contains two experiences:

- \`docs/index.html\`: polished **synthetic** visual learning sandbox and summary report importer.
- \`docs/viewer.html\`: actual **data-bound** browser replay from the exported \`viewer.json\`. Upload file in the browser; no server upload.

GitHub Pages cannot run Python or private Colab kernels. Compute the data with Colab / Streamlit and export it.

## 6. Optional historical match prediction (separate model)

Prepare your own trustworthy results CSV with:

\`\`\`csv
date,home_team,away_team,home_goals,away_goals
2024-01-01,Team A,Team B,2,1
\`\`\`

Supply **many actual matches**; a single row is not enough.

\`\`\`bash
taj predict --results results.csv --home "Team A" --away "Team B" --as-of 2026-10-15
taj evaluate --results results.csv --min-history 50
\`\`\`

The predictor uses home/away rate shrinkage and independent Poisson. Temporal evaluation emits multiclass Brier score. It does **not** have player availability, tactical causal effects or access to live fixture data.

## 7. Optional API

\`\`\`bash
uvicorn taj.api:app --reload
# GET /health, GET /datasets, POST /analyze
\`\`\`

## Repository layout

\`\`\`text
src/taj/
  schema.py        canonical tracking schema / velocity / synthetic fixtures
  ingest.py        IDSSE and SkillCorner adapters
  events.py        temporal event-to-frame join
  features.py      shape geometry and static pass corridors
  spatial.py       transparent first baseline
  advanced.py      vectorized kinematic access and pass interception
  episodes.py      separated evidence-window extraction
  weakness.py      original threshold-based exposure detector (legacy)
  recommender.py   candidate playbook + relative ranking
  prediction.py    independent Poisson historical results baseline
  analysis.py      end-to-end report orchestration
  viewer.py        exported replay frames, positions and heatmaps
  plotting.py      Plotly pitch and what-if differential figures
  export.py        standalone HTML
  cli.py           reproducible commands
  api.py           local FastAPI
dashboard.py       five-tab Streamlit UI
notebooks/Taj_Colab.ipynb
docs/{index.html,viewer.html,METHODOLOGY.md}
tests/               Pytest numerical/contract tests
scripts/             real-provider network smoke checks
.github/workflows/   automated unit + real-provider integration
\`\`\`

## Research and licensing safeguards

- **Never** treat adjacent frames as independent tactical proof.
- Missing or extrapolated positions reduce evidence quality; do not silently fill them as observed.
- Home/Away normalized orientation **does not equal physical camera position** across periods.
- Do not use a statistical match-result baseline as a scouting report.
- Match outcome probability and spatial access scores have different meanings.
- A hypothetical player move demonstrates geometry sensitivity, **not** an observed tactical result.
- Code MIT license covers code only; dataset licenses and attribution are separate. Respect [SkillCorner](https://github.com/SkillCorner/opendata) and [IDSSE](https://github.com/spoho-datascience/idsse-data) source terms.

For detailed assumptions and evaluation limits see [METHODOLOGY.md](docs/METHODOLOGY.md).

# TAJ Website — zero-install product

**Live site:** https://prestigegitserp.github.io/Taj_soccer_recommender/

The **home page is NOT the old synthetic one-match demo**. It is a standalone Persian-language browser application that:

- Finds the **closest five future fixtures** across Premier League, La Liga, Bundesliga, Serie A, Ligue 1 and Champions League (or five within a selected league).
- Shows official-looking team names and logo URLs as returned by the source, kickoff in the visitor's time zone, and any actual live scores if available.
- Calculates last-six recorded results / mean goals, independent-Poisson 1X2 **experimental** estimates when enough match results exist, and conditional scouting hypotheses with sources and caveats.
- Allows 10% what-if goal-rate changes *inside the browser*; does not pretend to simulate tactics or players.
- Does not require Colab, Python, an API key, server access, installation or sign-up.

## How updates work

Every 30 minutes, GitHub Actions runs \`scripts/build_public_feed.py\`, collecting public scoreboard data from six ESPN league endpoints for the prior two months, current and next month (with caching of the oldest month). This writes \`docs/data/feed.json\`, and publishes GitHub Pages **within the same workflow**.

On-site browser refresh attempts more up-to-date cross-origin public API requests. Those may be rejected by ESPN / browser CORS; the site then retains the **honestly dated GitHub snapshot**. Schedules and scores are never fabricated. GitHub Actions scheduled triggers are best effort and may run late.

**Source risk:** ESPN endpoints are public but unofficial, without availability commitments; parsing is tested and cached, but source outages are possible.

## All computation is client-side

The small dependency-free \`docs/app.mjs\` does:
- sorting and filtering to exactly five upcoming fixtures,
- chronological leakage-safe recent-team feature creation,
- conservative goal-rate shrinkage to league baselines,
- Poisson outcome probability estimates and form badges,
- explainable conditional suggestions and sensitivity changes.

The GitHub job **only downloads normalized historical results and schedules**, not model inference. The site has no user uploads, no API key and no database backend.

## Files

- \`index.html\`: complete application skeleton and semantic layout
- \`ui.css\`: responsive Persian high-contrast styling with native CSS radar art
- \`app.mjs\`: pure browser models + data handling and interaction
- \`data/feed.json\`: scheduled public sports records, with per-league freshness metadata
- \`viewer.html\`: separate offline Tracking replay for historical downloaded match data

## What it cannot honestly do

The free public match schedule does **not** contain 22-player continuous tracking for upcoming games. This means no authentic pitch-control heatmaps, pressing traps or proof of half-space vulnerability for those future fixtures. Those analyses require licensed or legally extracted tracking or video. The separate Tracking Viewer remains optional for historical examples.

Code QA: \`node --test tests/test_browser.mjs\` (GitHub Actions handles this, not the website visitor).

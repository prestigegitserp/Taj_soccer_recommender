# Static Visual Lab

The directory \`docs/\` is a no-build GitHub Pages site. It has an interactive synthetic tactical pitch and a **local-only** parser for the report JSON exported by \`taj analyze\`.

How to publish: **Repository Settings → Pages → Build and deployment → Deploy from a branch → main /docs → Save**.

Then open https://prestigegitserp.github.io/Taj_soccer_recommender/

The page is a viewer and **not a server**. It never computes the Python pipeline for live matches, nor can it remotely fetch local Colab output. The canvas is always labeled synthetic. JSON reports only update the Evidence Studio summary, not the canvas coordinates.

For live Python computation, run \`streamlit run dashboard.py\` locally or load the Colab notebook.

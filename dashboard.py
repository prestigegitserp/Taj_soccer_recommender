"""High-contrast local dashboard. Colab users can run in notebook via Plotly cells."""
from __future__ import annotations
from pathlib import Path
import json
import streamlit as st
import pandas as pd
from taj.analysis import create_demo,build_report
from taj.ingest import read_bundle
from taj.schema import at_frame,validate_tracking
from taj.plotting import pitch_figure

st.set_page_config(page_title="TAJ | Tactical Intelligence",layout="wide",page_icon="⚽",
                   initial_sidebar_state="expanded")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:'Vazirmatn',sans-serif!important}
.stApp{background:radial-gradient(circle at 85% -15%,#12334a,transparent 42%),#07111f;color:#e5f2ff}
[data-testid="stSidebar"]{background:#0d1d2c}
div[data-testid="stMetric"]{border:1px solid #254457;border-radius:16px;background:#112437;padding:13px}
.stButton>button{border-radius:12px;background:#19cba8;color:#02151c;font-weight:800}
h1,h2,h3{letter-spacing:-.03em}
</style>
""",unsafe_allow_html=True)
st.markdown("<h1>⚽ TAJ <span style='color:#22e4bd'>TACTICAL INTELLIGENCE</span></h1>",unsafe_allow_html=True)
st.caption("Evidence-led spatial football analytics | Python × Open Data × Explainable Recommendations")
DATA=Path("data")
paths=sorted(x for x in DATA.iterdir() if x.is_dir() and (x/"tracking.parquet").exists()) if DATA.exists() else []
if not paths:
    st.info("No match data found; generating an explicitly SYNTHETIC demonstration.")
    create_demo(DATA/"demo")
    paths=[DATA/"demo"]
with st.sidebar:
    st.subheader("⚙️ Analysis workspace")
    folder=st.selectbox("Dataset",paths,format_func=lambda p:p.name)
    st.caption("Demo is synthetic. IDSSE / SkillCorner require running ingestion first.")
    team=st.selectbox("Defending team",["Home","Away"])
    goal=st.selectbox("Defended goal on physical pitch",[-1,1],
                      format_func=lambda g:"Left (-x)" if g<0 else "Right (+x)")
    show_heat=st.toggle("Show spatial access map",True)
    st.markdown("---")
    st.caption("Experimental indices: NOT calibrated scoring probabilities.")
@st.cache_data(show_spinner=False)
def load(path):
    return read_bundle(path)
tracking,ball,events=load(str(folder))
quality=validate_tracking(tracking)
frames=tracking[["period","frame_id","t_s"]].drop_duplicates().sort_values(["period","t_s"]).reset_index(drop=True)
i=st.slider("⏯️ Match timeline (sampled frames)",0,len(frames)-1,len(frames)//2)
pick=frames.iloc[i]
f=at_frame(tracking,int(pick.period),str(pick.frame_id))
b=ball[(ball.period==pick.period)&(ball.frame_id.astype(str)==str(pick.frame_id))]
label="SYNTHETIC / NOT REAL MATCH" if str(tracking.provider.iloc[0])=="synthetic" else f"{tracking.provider.iloc[0].upper()} / OPEN DATA"
st.warning(label) if label.startswith("SYNTHETIC") else st.info(label)
c1,c2,c3,c4=st.columns(4)
c1.metric("Tracking rows",f"{quality['rows']:,}")
c2.metric("Frames",quality["frames"])
c3.metric("Visible players",len(f))
c4.metric("Observed ratio","Unknown" if quality["observed_rate"] is None else f"{quality['observed_rate']:.0%}")
fig=pitch_figure(f,b,heatmap=show_heat,
    title=f"Period {pick.period} · {float(pick.t_s):.1f} seconds · Frame {pick.frame_id}")
st.plotly_chart(fig,use_container_width=True)
st.markdown("### 🧠 Evidence-based tactical insights")
with st.spinner("Computing evidence windows and recommendations ..."):
    report=build_report(tracking,ball,events,defending=team,goal=goal,
                        stride=max(1,len(frames)//150),min_windows=4)
left,right=st.columns([1.1,1])
with left:
    st.subheader("Spatial hypotheses")
    if not report["findings"]:
        st.info("Insufficient independent evidence windows. No weakness claim was generated.")
    for n,h in enumerate(report["findings"],1):
        with st.expander(f"{n}. {h['statement_fa']} · {h['n_independent_windows']} windows"):
            st.write(h["disclaimer_fa"])
            st.dataframe(pd.DataFrame(h["evidence"]),use_container_width=True)
with right:
    st.subheader("Tactical recommendations")
    if not report["recommendations"]:
        st.info("No supported recommendations yet; increase sample duration, not confidence artificially.")
    for n,r in enumerate(report["recommendations"],1):
        with st.container(border=True):
            st.markdown(f"**#{n} · {r['title_fa']}**")
            st.progress(min(int(r["priority"]),100),text=f"Relative priority {r['priority']:.1f}/100")
            st.write("**Evidence:** "+r["reason_fa"])
            st.caption("Risk: "+r["risk_fa"])
            st.caption(r["caveat_fa"])
st.download_button("⬇️ Export auditable JSON report",
                   json.dumps(report,ensure_ascii=False,indent=2),file_name="taj_report.json",
                   mime="application/json")
with st.expander("Methodology & limitations"):
    st.write(report["disclaimer_fa"])
    st.write("Source data availability, sample size and visible players determine conclusions. "
             "The dashboard does not predict outcomes or claim causal tactical effectiveness.")

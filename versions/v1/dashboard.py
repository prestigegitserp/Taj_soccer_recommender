"""TAJ Tactical Intelligence — explainable multi-workspace Streamlit research UI."""
from __future__ import annotations
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from taj.ingest import read_bundle
from taj.analysis import create_demo,build_report
from taj.schema import at_frame,validate_tracking
from taj.plotting import pitch_figure,control_difference
from taj.advanced import SpatialConfig,quick_report_frame,pass_interception
from taj.features import spatial_summary
from taj.export import export_html
from taj.viewer import build_viewer_payload

st.set_page_config(page_title="TAJ — Football Intelligence",page_icon="⚽",
                   initial_sidebar_state="expanded",layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');
html,body,[class*="css"],.stApp{font-family:Vazirmatn,system-ui!important}
.stApp{background:radial-gradient(circle at 82% -18%,#18394e,transparent 43%),#06111e;color:#ebf4fb}
[data-testid="stSidebar"]{background:#0b1d2c}
[data-testid="stMetric"]{background:#0f2435;border:1px solid #26495b;border-radius:17px;padding:15px}
h1,h2,h3{letter-spacing:-.028em}
.block-container{padding-top:1.35rem;max-width:1660px}
.stButton>button{border-radius:12px;border:1px solid #30605a;background:#103a3a;color:#d5fff5}
[data-testid="stTabs"] button{font-weight:800}
.taj-title{font-size:33px;font-weight:800;color:#f1fffb}
.taj-title span{color:#25e1b5}
.taj-sub{color:#9ab8c9;font-size:13px;line-height:1.9;margin-bottom:16px}
.taj-card{padding:19px;border:1px solid #2b5266;border-radius:16px;background:linear-gradient(140deg,#112839,#0c1b2c);margin:10px 0}
.taj-tag{border-radius:20px;background:#17483c;padding:3px 10px;color:#4aeac3;display:inline-block;font-size:11px}
.taj-note{color:#a5bcca;font-size:12px;line-height:2}
.taj-risk{color:#f2c882;font-size:12px}
</style>""",unsafe_allow_html=True)

st.markdown("<div class='taj-title'>⚽ TAJ <span>TACTICAL INTELLIGENCE</span></div>",unsafe_allow_html=True)
st.markdown("<div class='taj-sub'>Python spatial engine · Evidence-gated recommendations · Interactive tactical lab · Open Tracking</div>",unsafe_allow_html=True)

DATA=Path("data")
if not DATA.exists() or not any(x.is_dir() and (x/"tracking.parquet").exists() for x in DATA.iterdir()):
    with st.spinner("Preparing labelled synthetic demonstration..."):
        create_demo(DATA/"demo")

bundles=sorted(x for x in DATA.iterdir() if x.is_dir() and (x/"tracking.parquet").exists())
if not bundles:
    st.error("No playable datasets. Run taj demo or taj ingest.");st.stop()

@st.cache_data(show_spinner=False,max_entries=4)
def load(path:str,mtime:int):
    return read_bundle(path)

with st.sidebar:
    st.subheader("⚙️ WORKSPACE")
    bundle=st.selectbox("Match dataset",bundles,format_func=lambda x:x.name)
    tracking,ball,events=load(str(bundle),int((bundle/"tracking.parquet").stat().st_mtime))
    st.caption(f"Source: {tracking.provider.iloc[0]}")
    defending=st.selectbox("Defending team",["Home","Away"])
    periods=sorted(int(p) for p in tracking.period.unique())
    st.markdown("**Defended goal by period**")
    st.caption("x = −1 (left goal), +1 (right goal). Verify against metadata/video.")
    def_guess=-1 if defending=="Home" else 1
    goal_signs={}
    for p in periods:
        if str(tracking.provider.iloc[0])=="skillcorner" and p%2==0:
            guess=-def_guess
        else:
            guess=def_guess
        goal_signs[p]=st.selectbox(f"Period {p}",[-1,1],index=0 if guess==-1 else 1,
             format_func=lambda v:"Left goal (−x)" if v<0 else "Right goal (+x)",key=f"goal_{defending}_{p}")
    st.markdown("---")
    heatmap=st.toggle("Pitch access heatmap",value=True)
    observed_only=st.toggle("Observed positions only",value=False)
    cell_m=st.slider("Spatial resolution (m)",2.0,8.0,4.0,step=.5)
    sample_limit=st.slider("Analysis samples target",40,400,180,step=20)
    risk_threshold=st.slider("Risk proxy evidence threshold",.15,.65,.36,step=.01)
    risk_aversion=st.slider("Recommendation risk aversion",0.,1.,.4,step=.1)
    st.caption("All control/risk metrics are uncalibrated heuristic scores.")

q=validate_tracking(tracking)
keys=tracking[["period","frame_id","t_s"]].drop_duplicates().sort_values(["period","t_s"]).reset_index(drop=True)
st.warning("SYNTHETIC SAMPLE — NOT A REAL FOOTBALL MATCH") if str(tracking.provider.iloc[0])=="synthetic" else st.info(f"HISTORICAL OPEN TRACKING — {tracking.provider.iloc[0].upper()}")
top=st.columns(5)
top[0].metric("Frames",f"{q['frames']:,}")
top[1].metric("Player observations",f"{q['rows']:,}")
top[2].metric("Periods",len(periods))
top[3].metric("Event records",len(events))
top[4].metric("Directly detected","unknown" if q["observed_rate"] is None else f"{q['observed_rate']:.0%}")

if "timeline" not in st.session_state:st.session_state.timeline=len(keys)//2
st.session_state.timeline=min(st.session_state.timeline,len(keys)-1)
idx=st.slider("⏯ Time travel · sampled frame",0,len(keys)-1,key="timeline")
selected=keys.iloc[idx]
frame=at_frame(tracking,int(selected.period),str(selected.frame_id))
ball_frame=ball[(ball.period==int(selected.period))&(ball.frame_id.astype(str)==str(selected.frame_id))]
config=SpatialConfig(grid_m=cell_m)

tabs=st.tabs(["🟢 Match Explorer","🧠 Spatial Intelligence","🎯 Tactical Recommender","⚡ What-if Lab","🧪 Data Quality & Export"])

@st.cache_data(show_spinner=False,max_entries=8)
def cached_report(path:str,mtime:int,defending:str,goals:tuple,limit:int,threshold:float):
    players,balls,evs=read_bundle(path)
    fcount=players[["period","frame_id"]].drop_duplicates().shape[0]
    stride=max(1,int(np.ceil(fcount/limit)))
    return build_report(players,balls,evs,defending=defending,
        direction_by_period=dict(goals),stride=stride,
        min_windows=4,high_risk=threshold)

with tabs[0]:
    st.plotly_chart(pitch_figure(frame,ball_frame,heatmap=heatmap,step=cell_m,
        observed_only=observed_only,title=f"Period {selected.period} · {selected.t_s:.1f}s · {selected.frame_id}"),
        use_container_width=True)
    if observed_only:
        st.caption("Filtered to genuinely observed players when that flag exists; heatmap may be unavailable if coverage is insufficient.")
    st.caption("Spatial colours indicate relative access, not the odds of a pass or goal.")
    metrics=spatial_summary(frame)
    a,b=st.columns(2)
    with a:
        st.subheader("Home team shape")
        st.json(metrics["team_shape"].get("Home",{}),expanded=False)
    with b:
        st.subheader("Away team shape")
        st.json(metrics["team_shape"].get("Away",{}),expanded=False)
    if len(ball_frame):
        b0=ball_frame.iloc[0]
        opts=spatial_summary(frame,ball_xy=(float(b0.x),float(b0.y)))["passing_corridors"]
        with st.expander("Geometric passing corridors (NOT completion odds)"):
            side=st.radio("Attacking team",["Home","Away"],horizontal=True)
            st.dataframe(pd.DataFrame(opts[side]),use_container_width=True,hide_index=True)

with st.spinner("Computing separated evidence windows and recommending only supported patterns..."):
    report=cached_report(str(bundle),int((bundle/"tracking.parquet").stat().st_mtime),
                         defending,tuple(sorted(goal_signs.items())),sample_limit,risk_threshold)

with tabs[1]:
    st.subheader("Spatial vulnerabilities — evidence first")
    if report["warnings"]:
        for warn in report["warnings"]:st.warning(warn)
    try:
        features=quick_report_frame(frame,defending,goal_signs[int(selected.period)],config)
        st.dataframe(pd.DataFrame(features["zones"]),use_container_width=True,hide_index=True)
        st.caption("Risk proxy weights opponent access near the defended goal; do not interpret as probability of a conceded goal.")
    except ValueError as e:
        st.info(f"Frame unavailable for advanced model: {e}")
    if not report["findings"]:
        st.info("No validated-by-threshold spatial exposure. Reduce the threshold only for exploratory inspection; do not change it to force a weakness claim.")
    for finding in report["findings"]:
        with st.expander(f"{finding['statement_fa']} · {finding['n_independent_windows']} separated windows"):
            st.markdown(f"**Geometric risk:** {finding['severity']:.3f}")
            st.write(finding["disclaimer_fa"])
            ev=pd.DataFrame(finding["evidence"])
            st.dataframe(ev,use_container_width=True,hide_index=True)
            if len(ev):
                sel=st.selectbox("Inspect evidence frame",list(range(len(ev))),format_func=lambda i:f"P{ev.iloc[i].period} · {ev.iloc[i].t_s:.1f}s · {ev.iloc[i].frame_id}",key="evidence_"+finding["zone"])
                er=ev.iloc[sel]
                ef=at_frame(tracking,int(er.period),str(er.frame_id))
                eb=ball[(ball.period==int(er.period))&(ball.frame_id.astype(str)==str(er.frame_id))]
                st.plotly_chart(pitch_figure(ef,eb,heatmap=True,step=cell_m,
                                              title="Evidence frame · "+str(er.frame_id)),use_container_width=True)

with tabs[2]:
    st.subheader("Candidate playbook · Intelligent explanation cards")
    st.caption("Ranked suggestions for review. They are not causal recommendations, success odds, or predicted xG improvements.")
    from taj.recommender import recommend
    plans=recommend(report["findings"],risk_aversion=risk_aversion)
    if not plans:
        st.info("Insufficient evidence to recommend a tactic. The system intentionally abstains.")
    for n,r in enumerate(plans,1):
        safe=lambda x:html.escape(str(x))
        st.markdown(f"""<div class="taj-card"><span class="taj-tag">#{n} · EVIDENCE-BASED CANDIDATE</span>
        <h3>{safe(r['title_fa'])}</h3>
        <p>{safe(r['reason_fa'])}</p>
        <p class="taj-risk">ریسک: {safe(r['risk_fa'])}</p>
        <div class="taj-note">رتبه نسبی: {safe(r['priority'])}/100 ·
           بازه‌های مستقل: {safe(r['sample_windows'])} · سطح اطمینان: اکتشافی</div>
        <div class="taj-note">{safe(r['caveat_fa'])}</div></div>""",unsafe_allow_html=True)
        with st.expander("🎞 Audit evidence / match timestamps",expanded=False):
            st.dataframe(pd.DataFrame(r["evidence"]),use_container_width=True)
    st.markdown("**How to improve validity:** human video labels → independent matches → compare against baseline → publish calibration.")

with tabs[3]:
    st.subheader("Counterfactual geometry sandbox")
    st.caption("Move ONE tracked player in a copied frame. This shows change in *geometric access* only, not what would actually happen in a game.")
    if len(frame):
        label=lambda i:f"{frame.iloc[i].team} · player {frame.iloc[i].player_id}"
        choice=st.selectbox("Select a player",list(range(len(frame))),format_func=label)
        dx,dy=st.columns(2)
        x_shift=dx.slider("Move horizontally (m)",-15.,15.,5.,step=.5)
        y_shift=dy.slider("Move vertically (m)",-15.,15.,0.,step=.5)
        moved=frame.copy()
        row=moved.index[choice]
        moved.loc[row,"x"]=float(np.clip(moved.loc[row,"x"]+x_shift,-52.5,52.5))
        moved.loc[row,"y"]=float(np.clip(moved.loc[row,"y"]+y_shift,-34,34))
        a,b=st.columns(2)
        with a:st.plotly_chart(pitch_figure(frame,ball_frame,heatmap=False,title="Observed/estimated baseline"),use_container_width=True)
        with b:st.plotly_chart(pitch_figure(moved,ball_frame,heatmap=False,title="Hypothetical moved player"),use_container_width=True)
        try:st.plotly_chart(control_difference(frame,moved,config),use_container_width=True)
        except ValueError as e:st.info(f"Not enough players to compare control maps: {e}")

with tabs[4]:
    st.subheader("Data quality & reproducibility")
    st.json(report["quality"],expanded=True)
    st.json({"model":report["model"],"focus":report["focus"],
             "capabilities":report["capabilities"],"processed_frames":report["processed_frames"],
             "events":report["events"],"warnings":report["warnings"]},expanded=False)
    st.download_button("⬇️ Report JSON",json.dumps(report,ensure_ascii=False,indent=2),
        file_name=f"taj_{report['match_id']}.json",mime="application/json")
    st.download_button("⬇️ Self-contained HTML",export_html(report,tracking,ball,
        Path("outputs")/f"taj_{report['match_id']}.html").read_bytes(),
        file_name=f"taj_{report['match_id']}.html",mime="text/html")
    bundle_json=json.dumps(build_viewer_payload(tracking,ball,report,max_frames=130,step_m=6),
                       ensure_ascii=False,allow_nan=False).encode("utf-8")
    st.download_button("⬇️ Interactive browser viewer.json",bundle_json,
                       file_name=f"taj_{report['match_id']}_viewer.json",mime="application/json")
    st.markdown("**Methodology:** [model card](https://github.com/prestigegitserp/Taj_soccer_recommender/blob/main/docs/METHODOLOGY.md)")
    st.info("For raw SkillCorner you must check the half-time defended-goal selector yourself. Tracking inference is never silently converted into verified tactical outcome.")

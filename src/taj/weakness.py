"""Evidence-gated *hypotheses*, not categorical scouting claims."""
from __future__ import annotations
from dataclasses import asdict
import pandas as pd
import numpy as np
from .spatial import pitch_access,zonal_access

def space_evidence(tracking:pd.DataFrame, *, defending="Home", defending_goal=1,
                   stride_frames=8, min_players=7, observed_threshold=.5):
    evidence=[]
    keys=tracking[["match_id","period","frame_id","t_s"]].drop_duplicates()
    for _,k in keys.iloc[::stride_frames].iterrows():
        frame=tracking[(tracking.match_id==k.match_id)&(tracking.period==k.period)&
                       (tracking.frame_id==k.frame_id)]
        counts=frame.groupby("team").size()
        if min(counts.get("Home",0),counts.get("Away",0))<min_players:continue
        if frame.is_detected.notna().any() and float(frame.is_detected.fillna(False).mean())<observed_threshold:continue
        h,a,X,Y=pitch_access(frame,step=5)
        for zone in zonal_access(h,X,Y,defending,defending_goal):
            evidence.append(dict(match_id=str(k.match_id),period=int(k.period),
                frame_id=str(k.frame_id),t_s=float(k.t_s),zone=zone["zone"],
                opponent_access=zone["opponent_access"],provider=str(frame.provider.iloc[0])))
    return pd.DataFrame(evidence)

def hypotheses(evidence:pd.DataFrame, *, threshold=.58, min_samples=4,
               window_seconds=12.0):
    """Temporal-dependence guard: count separated windows, not adjacent frames."""
    if evidence.empty:return []
    findings=[]
    for zone,sub in evidence.groupby("zone"):
        high=sub[sub.opponent_access>=threshold].sort_values(["match_id","period","t_s"])
        picked=[]
        for r in high.itertuples():
            if not picked or (r.match_id,r.period)!=(picked[-1]["match_id"],picked[-1]["period"]) or r.t_s-picked[-1]["t_s"]>=window_seconds:
                picked.append(dict(match_id=r.match_id,period=r.period,frame_id=r.frame_id,
                                   t_s=r.t_s,zone=zone,access=round(float(r.opponent_access),3),
                                   source=r.provider))
        if len(picked)<min_samples:continue
        findings.append(dict(id="opp-space-"+zone,kind="spatial_exposure",
            zone=zone,severity=round(float(np.mean([r["access"] for r in picked])),3),
            n_independent_windows=len(picked),evidence=picked[:20],
            statement_fa=f"دسترسی فضایی حریف در ناحیه «{zone}» در چند بازه مستقل بالا بوده است.",
            disclaimer_fa="این یک فرضیه فضایی است؛ به‌تنهایی اثبات ضعف تاکتیکی یا ایجاد موقعیت خطرناک نیست."))
    return sorted(findings,key=lambda d:(d["severity"],d["n_independent_windows"]),reverse=True)

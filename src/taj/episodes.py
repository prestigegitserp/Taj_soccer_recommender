"""Evidence selection from tracking without pretending to infer an exact possession ID."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .advanced import SpatialConfig,quick_report_frame

def _safe_bool_mean(series):
    d=series.dropna()
    return float(d.mean()) if len(d) else None

def analyse_frames(tracking:pd.DataFrame, *, defending="Home",
                   direction_by_period:dict[int,int]|None=None,
                   frame_stride=20,min_team_players=8,high_risk=.36,
                   window_s=12.,min_windows=4,cfg:SpatialConfig=SpatialConfig()):
    """Evidence windows for each zone. One selected frame per separated temporal interval.
    The threshold is exploratory and user-tunable; frames in a window are correlated.
    """
    if frame_stride<1:raise ValueError("frame_stride must be >=1")
    if direction_by_period is None:
        raise ValueError("Must provide direction_by_period; half-time direction must not be guessed")
    keys=tracking[["match_id","period","frame_id","t_s"]].drop_duplicates().sort_values(["period","t_s"])
    result=[]
    for (match,period),group in keys.groupby(["match_id","period"],sort=True):
        if int(period) not in direction_by_period:
            raise ValueError(f"Missing defended goal for period {period}")
        sign=int(direction_by_period[int(period)])
        if sign not in (-1,1):raise ValueError("Goal direction must be -1 or +1")
        for k in group.iloc[::frame_stride].itertuples():
            frame=tracking[(tracking.match_id==match)&(tracking.period==period)&
                           (tracking.frame_id==k.frame_id)]
            counts=frame.groupby("team").size()
            if min(counts.get("Home",0),counts.get("Away",0))<min_team_players:continue
            try:report=quick_report_frame(frame,defending,sign,cfg)
            except ValueError:continue
            for z in report["zones"]:
                result.append({"match_id":str(match),"period":int(period),
                    "frame_id":str(k.frame_id),"t_s":round(float(k.t_s),3),
                    "zone":z["zone"],"opponent_access":z["access"],
                    "risk_proxy":z["risk_proxy"],
                    "observed_ratio":report["coverage"]["observed_fraction"],
                    "direction":sign,
                    "provider":str(frame.provider.iloc[0])})
    snapshots=pd.DataFrame(result)
    if snapshots.empty:return snapshots,[]
    findings=[]
    for zone,g in snapshots.groupby("zone"):
        g=g[g.risk_proxy>=high_risk].sort_values(["match_id","period","t_s"])
        selected=[]
        last={}
        for r in g.itertuples():
            key=(r.match_id,r.period)
            if key not in last or r.t_s-last[key]>=window_s:
                ev={"match_id":r.match_id,"period":int(r.period),"frame_id":r.frame_id,
                    "t_s":float(r.t_s),"risk_proxy":float(r.risk_proxy),
                    "opponent_access":float(r.opponent_access),"source":r.provider,
                    "direction":r.direction}
                selected.append(ev);last[key]=r.t_s
        if len(selected)<min_windows:continue
        scores=np.array([e["risk_proxy"] for e in selected])
        findings.append({"kind":"spatial_exposure","id":"danger-"+zone,
            "zone":zone,"severity":round(float(scores.mean()),4),
            "n_independent_windows":len(selected),
            "evidence":selected[:60],
            "statement_fa":f"قرارگیری مکرر فضای ناحیه {zone} در دسترس نسبی حریف",
            "disclaimer_fa":"شاخص خطر هندسی و فاقد کالیبراسیون گل؛ برای شناسایی ضعف تاکتیکی باید با Event یا ویدیو راستی‌آزمایی شود."})
    return snapshots,sorted(findings,key=lambda x:x["severity"],reverse=True)

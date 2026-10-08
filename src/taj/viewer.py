"""Offline interactive match payload from actual sampled tracking + computed spatial maps."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import at_frame
from .advanced import pitch_control,SpatialConfig

def number(v):
    if v is None:return None
    try:
        f=float(v)
        return round(f,3) if np.isfinite(f) else None
    except (ValueError,TypeError):return None

def build_viewer_payload(tracking:pd.DataFrame,ball:pd.DataFrame|None,report:dict,
                         max_frames=110,step_m=6.):
    if max_frames<1:raise ValueError("max_frames must be >0")
    keys=tracking[["period","frame_id","t_s"]].drop_duplicates().sort_values(["period","t_s"])
    indices=np.linspace(0,len(keys)-1,min(max_frames,len(keys))).astype(int)
    wanted=set((int(r.period),str(r.frame_id)) for r in keys.iloc[indices].itertuples())
    for f in report.get("findings",[]):
        for ev in f.get("evidence",[])[:30]:wanted.add((int(ev["period"]),str(ev["frame_id"])))
    frames=[]
    for r in keys.itertuples():
        ident=(int(r.period),str(r.frame_id))
        if ident not in wanted:continue
        frame=at_frame(tracking,*ident)
        bp=ball[(ball.period==ident[0])&(ball.frame_id.astype(str)==ident[1])] if ball is not None and not ball.empty else None
        players=[{"id":str(p.player_id),"team":str(p.team),
                  "x":number(p.x),"y":number(p.y),
                  "observed":None if pd.isna(p.is_detected) else bool(p.is_detected)}
                 for p in frame.itertuples()]
        b={"x":number(bp.iloc[0].x),"y":number(bp.iloc[0].y)} if bp is not None and not bp.empty else None
        heatmap=None
        try:
            pc=pitch_control(frame,SpatialConfig(grid_m=step_m,min_players_per_team=7))
            heatmap={"x":[number(x) for x in pc["x"]],
                     "y":[number(y) for y in pc["y"]],
                     "access":[[number(x) for x in row] for row in pc["home_access"]]}
        except ValueError:pass
        frames.append({"period":ident[0],"frame_id":ident[1],
                       "t_s":number(r.t_s),"players":players,
                       "ball":b,"heatmap":heatmap})
    return {"format":"taj-viewer-v1","match_id":report.get("match_id"),
            "provider":report.get("provider"),"synthetic":bool(report.get("synthetic")),
            "model":report.get("model"),"quality":report.get("quality",{}),
            "warnings":report.get("warnings",[]),"focus":report.get("focus",{}),
            "findings":report.get("findings",[]),"recommendations":report.get("recommendations",[]),
            "frames":frames,"notice":"Spatial access is an uncalibrated kinematic heuristic, not causal tactical impact."}

def export_viewer_json(payload:dict,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(",",":")),encoding="utf-8")
    return path

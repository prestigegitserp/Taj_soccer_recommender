"""Match pipeline. No hidden inference; preserve every evidence time stamp."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .schema import synthetic_match,validate_tracking,at_frame
from .ingest import read_bundle,write_bundle
from .spatial import pitch_access,shape_metrics,zonal_access
from .weakness import space_evidence,hypotheses
from .recommender import recommend

def build_report(players:pd.DataFrame,balls:pd.DataFrame | None=None, events:pd.DataFrame | None=None,
                 defending="Home",goal=1,stride=8,min_windows=4):
    q=validate_tracking(players)
    sample=players[["period","frame_id"]].drop_duplicates().iloc[len(players[["period","frame_id"]].drop_duplicates())//2]
    f=at_frame(players,int(sample.period),str(sample.frame_id))
    h,a,X,Y=pitch_access(f,step=5)
    ev=space_evidence(players,defending=defending,defending_goal=goal,stride_frames=stride)
    findings=hypotheses(ev,min_samples=min_windows)
    plans=recommend(findings,min_windows=min_windows)
    report={
        "schema_version":"1.0",
        "match_id":str(players.match_id.iloc[0]),
        "provider":str(players.provider.iloc[0]),
        "synthetic":bool(str(players.provider.iloc[0])=="synthetic"),
        "capabilities":{"tracking":True,"events":bool(events is not None and not events.empty),
                        "tracking_observation_known":bool(players.is_detected.notna().any()),
                        "result_prediction":False},
        "quality":q,
        "focus":{"defending_team":defending,"defended_goal_sign":int(goal)},
        "sample_frame":{"period":int(sample.period),"frame_id":str(sample.frame_id),
                        "shape":shape_metrics(f),"zones":zonal_access(h,X,Y,defending,goal)},
        "findings":findings,"recommendations":plans,
        "evidence_windows":int(len(ev)),
        "disclaimer_fa":"تحلیل اکتشافی مبتنی بر هندسه Tracking است؛ احتمال گل، توصیه قطعی مربیگری یا تحلیل بازی آینده نیست.",
        "model":"Taj heuristic pitch-access v0.2"
    }
    return report

def export_report(report:dict, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return path

def create_demo(root="data/demo"):
    tracking,ball=synthetic_match()
    write_bundle(tracking,ball,pd.DataFrame(),root)
    report=build_report(tracking,ball,defending="Home",goal=1,stride=3,min_windows=2)
    export_report(report,Path(root)/"report.json")
    return tracking,ball,report

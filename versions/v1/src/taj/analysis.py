"""End-to-end report orchestrator: consistent sources, directions, evidence and diagnostics."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
from .schema import synthetic_match,validate_tracking,at_frame
from .ingest import read_bundle,write_bundle
from .advanced import SpatialConfig,quick_report_frame
from .episodes import analyse_frames
from .events import event_summary
from .recommender import recommend

def _directions(tracking, defending, goal, direction_by_period):
    periods=sorted(int(p) for p in tracking.period.unique())
    if direction_by_period is not None:
        directions={int(p):int(v) for p,v in direction_by_period.items()}
    elif goal is not None:
        directions={p:int(goal) for p in periods}
    elif str(tracking.provider.iloc[0]) in ("idsse","synthetic"):
        # Kloppy ingest uses STATIC_HOME_AWAY; synthetic demo also has fixed sides.
        directions={p:(-1 if defending=="Home" else 1) for p in periods}
    else:
        raise ValueError("Direction per period is required for raw SkillCorner tracking. "
                         "Pass direction_by_period={1:-1, 2:1} after verifying team orientation.")
    if set(directions)!=set(periods) or not all(v in (-1,1) for v in directions.values()):
        raise ValueError("Exactly one defended-goal direction (-1 or +1) required for every period")
    return directions

def build_report(players:pd.DataFrame,balls:pd.DataFrame|None=None,
                 events:pd.DataFrame|None=None, defending="Home", goal:int|None=None,
                 direction_by_period:dict[int,int]|None=None,
                 stride:int|None=None,min_windows=4,
                 high_risk=.36,config:SpatialConfig=SpatialConfig()):
    if defending not in ("Home","Away"):raise ValueError("defending team must be Home or Away")
    quality=validate_tracking(players)
    keys=players[["period","frame_id","t_s"]].drop_duplicates().sort_values(["period","t_s"])
    if keys.empty:raise ValueError("No available frames")
    signs=_directions(players,defending,goal,direction_by_period)
    sample=keys.iloc[len(keys)//2]
    frame=at_frame(players,int(sample.period),str(sample.frame_id))
    sample_features=None
    try:sample_features=quick_report_frame(frame,defending,signs[int(sample.period)],config)
    except ValueError:
        # Keep report usable even when a single frame has insufficient coverage.
        sample_features={"zones":[],"shape":{},"coverage":{"warning":True},
                         "model":"insufficient coverage"}
    # Maximum ~250 sampled frames, to bound work on Colab CPU for full matches.
    stride=int(stride) if stride is not None else max(1,(len(keys)+249)//250)
    evidence,findings=analyse_frames(players,defending=defending,
        direction_by_period=signs,frame_stride=stride,min_windows=min_windows,
        high_risk=high_risk,cfg=config)
    plans=recommend(findings,min_windows=min_windows)
    coverage=quality.get("observed_rate")
    warnings=[]
    if coverage is not None and coverage<config.min_observed_ratio:
        warnings.append("Low directly observed coverage; extrapolated player positions may dominate.")
    if str(players.provider.iloc[0])=="synthetic":
        warnings.append("SYNTHETIC data: not a real match, findings only test the code.")
    if events is None or events.empty:
        warnings.append("No event feed loaded; exposures are not independently confirmed by actions/chances.")
    if str(players.provider.iloc[0])=="skillcorner":
        warnings.append("SkillCorner dynamic events are not a full Opta/StatsBomb-style event feed.")
    if goal is not None and len(signs)>1 and direction_by_period is None and str(players.provider.iloc[0])=="skillcorner":
        warnings.append("A single goal sign was applied to all periods: verify half-time direction.")
    return {"schema_version":"1.1",
            "match_id":str(players.match_id.iloc[0]),"provider":str(players.provider.iloc[0]),
            "synthetic":str(players.provider.iloc[0])=="synthetic",
            "capabilities":{"tracking":True,"events":bool(events is not None and not events.empty),
               "tracking_observation_known":coverage is not None,"result_prediction":False},
            "quality":quality,"focus":{"defending_team":defending,
                                      "defended_goal_by_period":{str(k):v for k,v in signs.items()}},
            "sample_frame":{"period":int(sample.period),"frame_id":str(sample.frame_id),
                            "t_s":float(sample.t_s),**sample_features},
            "events":event_summary(events),
            "findings":findings,"recommendations":plans,
            "evidence_windows":int(len(evidence)),"processed_frames":int(evidence[["period","frame_id"]].drop_duplicates().shape[0]) if len(evidence) else 0,
            "warnings":warnings,
            "disclaimer_fa":"تحلیل فضایی اکتشافی و بدون کالیبراسیون؛ اثبات نقطه ضعف، میزان بهبود xG یا احتمال نتیجه مسابقه نیست.",
            "model":"Taj kinematic access + evidence gates v0.3"}

def export_report(report:dict,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
    return path

def create_demo(root="data/demo"):
    tracking,ball=synthetic_match()
    write_bundle(tracking,ball,pd.DataFrame(),root)
    report=build_report(tracking,ball,defending="Home",goal=-1,stride=3,min_windows=2)
    export_report(report,Path(root)/"report.json")
    return tracking,ball,report

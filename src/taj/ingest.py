"""Provider adapters, preserving physical units and provenance.
IDSSE is loaded via Kloppy. New-format SkillCorner is loaded from its Git LFS JSONL
locally so that is_detected survives normalization.
"""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np
from .schema import LENGTH, WIDTH, validate_tracking, add_velocity

IDSSE_IDS = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")

def _coerce_seconds(value) -> float:
    if hasattr(value, "total_seconds"):
        return float(value.total_seconds())
    if hasattr(value, "timestamp") and hasattr(value.timestamp, "total_seconds"):
        return float(value.timestamp.total_seconds())
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value)
    try:
        return float(pd.to_timedelta(s).total_seconds())
    except (ValueError, TypeError):
        raise ValueError("Cannot interpret match-relative timestamp: "+s)

def from_kloppy(dataset, match_id: str, provider: str):
    """Normalize transformed Kloppy frames to pitch-centered meters.
    Export transforms *before* reading; no silent interpretation of 0..1 as meters.
    """
    from kloppy.domain import MetricPitchDimensions, Dimension
    metric = MetricPitchDimensions(x_dim=Dimension(0, LENGTH), y_dim=Dimension(0, WIDTH),
                                    pitch_length=LENGTH,pitch_width=WIDTH,standardized=False)
    ds = dataset.transform(to_pitch_dimensions=metric, to_orientation="STATIC_HOME_AWAY")
    home, away = ds.metadata.teams
    rows, balls = [], []
    for idx, frame in enumerate(ds):
        period = getattr(getattr(frame,"period",None),"id", None)
        if period is None:
            period = getattr(getattr(getattr(frame,"time",None),"period",None),"id", 1)
        t_obj = getattr(frame, "timestamp", None)
        if t_obj is None:
            t_obj = getattr(frame, "time",None)
        t = _coerce_seconds(t_obj) if t_obj is not None else idx/10
        fid = str(getattr(frame,"frame_id",idx))
        for pl, p in frame.players_coordinates.items():
            if p is None or p.x is None or p.y is None:
                continue
            team = "Home" if pl.team == home else ("Away" if pl.team == away else "Unknown")
            if team == "Unknown":
                continue
            detected = None
            pdata = getattr(frame,"players_data",{}).get(pl) if hasattr(frame,"players_data") else None
            if pdata is not None and hasattr(pdata,"is_detected"):
                detected = bool(pdata.is_detected)
            rows.append(dict(match_id=match_id, provider=provider,period=int(period),
                  t_s=t,frame_id=fid,team=team,
                  player_id=str(getattr(pl,"player_id",str(pl))),
                  x=float(p.x)-LENGTH/2,y=float(p.y)-WIDTH/2,is_detected=detected))
        bp = frame.ball_coordinates
        if bp is not None and bp.x is not None and bp.y is not None:
            balls.append(dict(match_id=match_id,period=int(period),t_s=t,frame_id=fid,
                              x=float(bp.x)-LENGTH/2,y=float(bp.y)-WIDTH/2))
    return add_velocity(pd.DataFrame(rows)), pd.DataFrame(balls)

def load_idsse(match_id="J03WMX", limit=2000, sample_rate=2.0, with_events=True):
    if match_id not in IDSSE_IDS:
        raise ValueError(f"Unknown IDSSE id. Choose {IDSSE_IDS}")
    from kloppy import sportec
    ds = sportec.load_open_tracking_data(match_id=match_id, limit=limit, sample_rate=sample_rate)
    players, ball = from_kloppy(ds,match_id,"idsse")
    events = pd.DataFrame()
    if with_events:
        raw = sportec.load_open_event_data(match_id=match_id)
        from kloppy.domain import MetricPitchDimensions, Dimension
        metric = MetricPitchDimensions(x_dim=Dimension(0, LENGTH), y_dim=Dimension(0, WIDTH),
                                        pitch_length=LENGTH,pitch_width=WIDTH,standardized=False)
        ev = raw.transform(to_pitch_dimensions=metric,to_orientation="STATIC_HOME_AWAY")
        edf = ev.to_df()
        if not edf.empty:
            for dim, mid in [("coordinates_x",LENGTH/2),("end_coordinates_x",LENGTH/2),
                             ("coordinates_y",WIDTH/2),("end_coordinates_y",WIDTH/2)]:
                if dim in edf:
                    edf[dim] = pd.to_numeric(edf[dim],errors="coerce") - mid
            edf["match_id"] = match_id
            events = edf
    return players,ball,events

def load_skillcorner_local(match_id: str, root: str | Path, limit: int | None=10000, stride: int=5):
    """Read SkillCorner new-format {id}_tracking_extrapolated.jsonl via checked-out Git LFS.
    No fabricated player team IDs; group is resolved from match metadata, if present.
    """
    root=Path(root)
    candidates=[root/"data"/"matches"/match_id, root/match_id, root]
    folder=next((x for x in candidates if (x/(match_id+"_tracking_extrapolated.jsonl")).exists()), None)
    if folder is None:
        raise FileNotFoundError(f"Could not find tracking JSONL for {match_id} under {root}")
    meta=json.loads((folder/(match_id+"_match.json")).read_text())
    trfile=folder/(match_id+"_tracking_extrapolated.jsonl")
    if trfile.stat().st_size < 500:
        raise ValueError("Tracking file is a Git LFS pointer. Run git lfs pull on the SkillCorner repository.")
    teams={}
    for side in ("home_team","away_team"):
        t=meta.get(side, {})
        if isinstance(t,dict):
            teams[str(t.get("id"))]="Home" if side=="home_team" else "Away"
    players_map={}
    for p in meta.get("players",[]):
        if isinstance(p,dict):
            group=p.get("team_id") or p.get("team",{}).get("id") if isinstance(p.get("team"),dict) else p.get("team_id")
            if group is not None:
                players_map[str(p.get("id"))]=teams.get(str(group),"Unknown")
    rows,balls=[],[]
    with trfile.open(encoding="utf-8") as f:
        for ix,line in enumerate(f):
            if ix % max(1,stride): continue
            if limit is not None and ix//max(1,stride)>=limit: break
            frame=json.loads(line)
            fid=str(frame.get("frame",ix))
            period=int(frame.get("period",1))
            stamp=frame.get("timestamp",ix/10)
            if isinstance(stamp,(int,float)):
                sec=float(stamp)
            elif isinstance(stamp,str):
                sec=_coerce_seconds(stamp)
            else:
                sec=float(ix/10)
            for p in frame.get("player_data",[]) or []:
                if p.get("x") is None or p.get("y") is None:continue
                pid=str(p.get("player_id"))
                group=p.get("group")
                if isinstance(group,dict):group=group.get("name") or group.get("id")
                team={"home":"Home","away":"Away"}.get(str(group).lower(), players_map.get(pid,"Unknown"))
                if team not in ("Home","Away"):continue
                rows.append(dict(match_id=match_id,provider="skillcorner",period=period,t_s=sec,
                    frame_id=fid, team=team, player_id=pid,
                    x=float(p["x"]),y=float(p["y"]),
                    is_detected=p.get("is_detected")))
            ball=frame.get("ball_data") or {}
            if ball.get("x") is not None and ball.get("y") is not None:
                balls.append(dict(match_id=match_id,period=period,t_s=sec,frame_id=fid,
                                  x=float(ball["x"]),y=float(ball["y"])))
    if not rows:
        raise ValueError("No player positions resolved: check the match metadata/team mapping.")
    return add_velocity(pd.DataFrame(rows)), pd.DataFrame(balls),pd.DataFrame()

def write_bundle(players,ball,events,out_dir):
    path=Path(out_dir);path.mkdir(parents=True,exist_ok=True)
    quality=validate_tracking(players)
    players.to_parquet(path/"tracking.parquet",index=False)
    ball.to_parquet(path/"ball.parquet",index=False)
    events.to_parquet(path/"events.parquet",index=False)
    (path/"quality.json").write_text(json.dumps(quality,indent=2),encoding="utf-8")
    return quality

def read_bundle(path):
    path=Path(path)
    return (pd.read_parquet(path/"tracking.parquet"),
            pd.read_parquet(path/"ball.parquet"),
            pd.read_parquet(path/"events.parquet"))

"""Canonical tracking schema. All positions use *metres* in pitch-centred coordinates.
The x-axis follows fixed home-left/away-right orientation, never possession flips.
A physical position can be absent; estimated positions are distinguished from observed.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional
import math
import numpy as np
import pandas as pd

LENGTH = 105.0
WIDTH = 68.0
TRACK_COLS = ["match_id", "provider", "period", "t_s", "frame_id", "team",
              "player_id", "x", "y", "is_detected"]
BALL_COLS = ["match_id", "period", "t_s", "frame_id", "x", "y"]

@dataclass(frozen=True)
class Player:
    player_id: str
    team: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    is_detected: Optional[bool] = None

@dataclass(frozen=True)
class Evidence:
    match_id: str
    period: int
    start_s: float
    end_s: float
    frame_id: str
    note: str
    source: str

def validate_tracking(df: pd.DataFrame) -> dict:
    missing = set(TRACK_COLS).difference(df.columns)
    if missing:
        raise ValueError("Missing tracking columns: " + ", ".join(sorted(missing)))
    if df.empty:
        raise ValueError("Empty tracking data")
    oob = (~df["x"].between(-LENGTH/2 - 1, LENGTH/2 + 1) |
           ~df["y"].between(-WIDTH/2 - 1, WIDTH/2 + 1))
    if df[["x","y"]].isna().any().any() or oob.any():
        raise ValueError("Coordinates missing or outside metric pitch bounds. Check provider normalization.")
    duplicates = df.duplicated(["match_id","period","frame_id","team","player_id"]).sum()
    return {"rows": int(len(df)), "frames": int(df[["period","frame_id"]].drop_duplicates().shape[0]),
            "duplicate_players": int(duplicates),
            "observed_rate": float(df["is_detected"].dropna().mean()) if df["is_detected"].notna().any() else None,
            "teams": sorted(df["team"].astype(str).unique().tolist())}

def at_frame(df: pd.DataFrame, period: int, frame_id: str) -> pd.DataFrame:
    frame = df.loc[(df["period"] == period) & (df["frame_id"].astype(str) == str(frame_id))].copy()
    if frame.empty:
        raise ValueError("Frame not found")
    frame["vx"] = frame.get("vx", pd.Series(0., index=frame.index)).fillna(0.)
    frame["vy"] = frame.get("vy", pd.Series(0., index=frame.index)).fillna(0.)
    return frame

def add_velocity(df: pd.DataFrame, max_speed: float = 11.0) -> pd.DataFrame:
    """Central differences grouped by player AND period; reject large temporal gaps.
    One-sided finite differences are used at edges; velocities above max_speed become NaN.
    """
    df = df.sort_values(["match_id","period","team","player_id","t_s"]).copy()
    groups = ["match_id","period","team","player_id"]
    for axis in ["x","y"]:
        dt = df.groupby(groups)["t_s"].diff()
        dv = df.groupby(groups)[axis].diff()
        df["v"+axis] = (dv / dt.where((dt > 0.001) & (dt < 1.5))).astype(float)
    speed = np.hypot(df["vx"], df["vy"])
    df.loc[speed > max_speed, ["vx","vy"]] = np.nan
    return df

def synthetic_match(match_id: str = "demo", frames: int = 80, hz: float = 2.0, seed: int = 17):
    """Explicitly synthetic sample: meaningful for UI/algorithm smoke tests only."""
    rng = np.random.default_rng(seed)
    rows, balls = [], []
    for k in range(frames):
        t = k / hz
        for team, sign in (("Home", -1), ("Away", 1)):
            for i in range(11):
                x = sign * (8 + 3*(i//3)) + 2*np.sin(t/15+i)
                y = -28 + i*5.6 + np.sin(t/7+i)*1.2
                x += rng.normal(0,.12); y += rng.normal(0,.12)
                rows.append(dict(match_id=match_id, provider="synthetic", period=1,
                                 t_s=t, frame_id=str(k), team=team, player_id=str(i),
                                 x=float(x),y=float(y),is_detected=True))
        balls.append(dict(match_id=match_id,period=1,t_s=t,frame_id=str(k),
                          x=float(-15+ t*.65),y=float(10*np.sin(t/15))))
    return pd.DataFrame(rows), pd.DataFrame(balls)

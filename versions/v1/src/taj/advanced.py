"""Advanced, transparent tracking analytics, built to run on CPU/Colab.

Definitions:
- arrival: kinematic earliest-arrival surrogate, *not* calibrated interception probability.
- territorial control: sigmoid of minimum arrival time difference; uncertainty depends on coverage.
- space danger: context-free geometric opportunity proxy, *not* xG or learned EPV.
- lane feasibility: estimates a defender's arrival to ball's time along pass path.
All numeric scores MUST be labelled model-derived and uncalibrated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from .schema import LENGTH, WIDTH

@dataclass(frozen=True)
class SpatialConfig:
    grid_m: float = 3.0
    reaction_s: float = .65
    max_speed_mps: float = 7.0
    acceleration_mps2: float = 3.0
    temperature_s: float = .65
    min_players_per_team: int = 7
    min_observed_ratio: float = .35

    def __post_init__(self):
        for name in ("grid_m","max_speed_mps","acceleration_mps2","temperature_s"):
            if getattr(self,name)<=0: raise ValueError(name+" must be positive")
        if self.reaction_s<0: raise ValueError("reaction_s must be >=0")

def _prepared(frame:pd.DataFrame,team:str, observed_only:bool=False):
    subset=frame[frame.team==team].copy()
    if observed_only and "is_detected" in subset.columns:
        subset=subset.loc[subset.is_detected.fillna(False).astype(bool)]
    if len(subset)==0:return np.zeros((0,4),dtype=float)
    for col in ("vx","vy"):
        if col not in subset:subset[col]=0.
    v=subset[["x","y","vx","vy"]].apply(pd.to_numeric,errors="coerce").to_numpy(dtype=float)
    v[:,2:]=np.nan_to_num(v[:,2:],nan=0,posinf=0,neginf=0)
    return v[np.isfinite(v[:,:2]).all(axis=1)]

def reach_time(players:np.ndarray,targets:np.ndarray, cfg:SpatialConfig):
    """Vectorized ETA for N players × M pitch targets in (meters, seconds).
    Constant-acceleration model from present velocity projected toward target,
    constrained to max speed. Returned ETA includes decision reaction delay.
    """
    if len(players)==0:return np.full(targets.shape[0],np.inf)
    xy=players[:,:2]
    vel=players[:,2:]
    vectors=targets[None,:,:]-xy[:,None,:]
    distance=np.linalg.norm(vectors,axis=-1)
    unit=vectors/np.maximum(distance[...,None],1e-9)
    initial_along=np.clip((vel[:,None,:]*unit).sum(axis=-1),0,cfg.max_speed_mps)
    a=cfg.acceleration_mps2; top=cfg.max_speed_mps
    accel_time=(top-initial_along)/a
    accel_dist=(initial_along+top)*accel_time/2
    t_accel=(-initial_along+np.sqrt(initial_along**2+2*a*distance))/a
    t=np.where(distance<=accel_dist,t_accel,accel_time+(distance-accel_dist)/top)
    return np.min(cfg.reaction_s + t,axis=0)

def pitch_control(frame:pd.DataFrame,cfg:SpatialConfig=SpatialConfig(), observed_only=False):
    """Structured map plus coverage. Do not interpret 'home_access' as calibrated probability."""
    xs=np.arange(-LENGTH/2,LENGTH/2+1e-5,cfg.grid_m)
    ys=np.arange(-WIDTH/2,WIDTH/2+1e-5,cfg.grid_m)
    X,Y=np.meshgrid(xs,ys);tgt=np.stack([X.ravel(),Y.ravel()],axis=-1)
    home=_prepared(frame,"Home",observed_only);away=_prepared(frame,"Away",observed_only)
    if len(home)<cfg.min_players_per_team or len(away)<cfg.min_players_per_team:
        raise ValueError(f"Insufficient tracking coverage: home={len(home)}, away={len(away)}")
    eta_h=reach_time(home,tgt,cfg);eta_a=reach_time(away,tgt,cfg)
    delta=np.clip((eta_h-eta_a)/cfg.temperature_s,-40,40)
    access=(1/(1+np.exp(delta))).reshape(X.shape)
    obs=float(frame.is_detected.dropna().mean()) if "is_detected" in frame and frame.is_detected.notna().any() else None
    return {"x":xs,"y":ys,"home_access":access,
            "home_eta_s":eta_h.reshape(X.shape),"away_eta_s":eta_a.reshape(X.shape),
            "coverage":{"home":len(home),"away":len(away),"observed_fraction":obs,
                        "warning":obs is not None and obs<cfg.min_observed_ratio},
            "method":"kinematic arrival-time index (uncalibrated)"}

def zone_risk(control:dict, defending:str, defended_goal:int):
    """Opponent access in defended third weighted by closeness to goal.
    Higher score = higher open-space exposure, *not* likelihood of conceding.
    """
    if defending not in ("Home","Away") or defended_goal not in (-1,1):
        raise ValueError("Invalid defending team or goal")
    x=control["x"][None,:];y=control["y"][:,None]
    access=control["home_access"] if defending=="Away" else (1-control["home_access"])
    in_final_third=np.broadcast_to((x>17.5 if defended_goal==1 else x < -17.5),access.shape)
    distance_to_goal=(52.5 - defended_goal*x)
    weight=np.clip(1-distance_to_goal/58,0,1)
    zones=[]
    for name,a,b in (("left",-34,-11.33),("central",-11.33,11.33),("right",11.33,34.01)):
        region=in_final_third & np.broadcast_to((y>=a)&(y<b),access.shape)
        if not region.any():continue
        scores=access[region];w=np.broadcast_to(weight,access.shape)[region]
        zones.append({"zone":name,
            "access":round(float(scores.mean()),4),
            "risk_proxy":round(float((scores*w).mean()),4),
            "n_cells":int(region.sum())})
    return sorted(zones,key=lambda z:z["risk_proxy"],reverse=True)

def pass_interception(frame:pd.DataFrame,start:tuple[float,float],end:tuple[float,float],
                      defending_team:str,ball_speed_mps:float=15.,
                      max_pass_s:float=5.,cfg:SpatialConfig=SpatialConfig()):
    """Sample a prospective ball trajectory, compare defender ETA to arrival.
    Ball flight assumed constant speed on straight horizontal line, no bounces.
    Returns *margin*, not estimated completion probability.
    """
    if ball_speed_mps<=0:raise ValueError("ball_speed_mps must be positive")
    a,b=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    d=float(np.linalg.norm(b-a))
    if d<.5:return {"plausible":False,"distance_m":d,"interception_margin_s":None}
    n=max(8,int(np.ceil(d/1.5)))
    steps=np.linspace(.05,.95,n)
    points=a[None,:]+steps[:,None]*(b-a)[None,:]
    eta=reach_time(_prepared(frame,defending_team),points,cfg)
    ball_eta=(steps*d/ball_speed_mps)
    slack=eta-ball_eta
    if len(slack)==0 or not np.isfinite(slack).any():
        return {"plausible":False,"distance_m":d,"interception_margin_s":None}
    margin=float(np.min(slack))
    return {"plausible":bool(margin>0 and d/ball_speed_mps<=max_pass_s),
            "distance_m":round(d,2),"interception_margin_s":round(margin,3),
            "method":"constant-speed pass versus approximate defender ETA"}

def quick_report_frame(frame:pd.DataFrame,defending="Home",goal=-1,cfg:SpatialConfig=SpatialConfig()):
    from .spatial import shape_metrics
    z=pitch_control(frame,cfg)
    return {"zones":zone_risk(z,defending,goal),
            "shape":shape_metrics(frame),"coverage":z["coverage"],
            "model":z["method"]}

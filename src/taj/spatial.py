"""Transparent *heuristic* pitch access: NOT calibrated pitch-control probability."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .schema import LENGTH,WIDTH

def grid(step:float=3.0):
    if step<=0:raise ValueError("step must be > 0")
    x=np.linspace(-LENGTH/2,LENGTH/2,int(np.ceil(LENGTH/step))+1)
    y=np.linspace(-WIDTH/2,WIDTH/2,int(np.ceil(WIDTH/step))+1)
    return np.meshgrid(x,y)

def arrival_time(players:pd.DataFrame, X, Y, reaction:float=.7, vmax:float=6.0):
    """Simple time-to-arrival approximation with bounded movement speeds.
    Velocity shifts predicted position during a short reaction interval.
    """
    if players.empty:
        return np.full(X.shape,np.inf)
    best=np.full(X.shape,np.inf)
    for r in players.itertuples():
        vx_raw=getattr(r,"vx",0.)
        vy_raw=getattr(r,"vy",0.)
        vx=float(vx_raw) if pd.notna(vx_raw) else 0.
        vy=float(vy_raw) if pd.notna(vy_raw) else 0.
        rx=float(r.x)+np.clip(vx,-vmax,vmax)*reaction
        ry=float(r.y)+np.clip(vy,-vmax,vmax)*reaction
        t=reaction+np.hypot(X-rx,Y-ry)/vmax
        best=np.minimum(best,t)
    return best

def pitch_access(frame:pd.DataFrame, step=3., reaction=.7, vmax=6., temperature=.65,
                 observed_only=False):
    """Returns access(HOME), access(AWAY), X,Y arrays. Logistic transforms differential ETA.
    Not an empirical probability until calibrated on pass-reception labels.
    """
    if temperature<=0: raise ValueError("temperature must be >0")
    X,Y=grid(step)
    work=frame
    if observed_only and "is_detected" in frame:
        work=frame[frame.is_detected.fillna(False).astype(bool)]
    home=work[work.team=="Home"]; away=work[work.team=="Away"]
    th=arrival_time(home,X,Y,reaction,vmax)
    ta=arrival_time(away,X,Y,reaction,vmax)
    if home.empty and away.empty: raise ValueError("No players in frame")
    if home.empty:return np.zeros_like(X),np.ones_like(X),X,Y
    if away.empty:return np.ones_like(X),np.zeros_like(X),X,Y
    delta=np.clip((th-ta)/temperature,-50,50)
    access_home=1/(1+np.exp(delta))
    return access_home,1-access_home,X,Y

def shape_metrics(frame:pd.DataFrame):
    """Robust team geometry summaries (no claims about tactical intention)."""
    result={}
    for team in ("Home","Away"):
        sub=frame[frame.team==team]
        if len(sub)<3:continue
        xy=sub[["x","y"]].to_numpy(dtype=float)
        span=np.ptp(xy,axis=0)
        result[team]=dict(players=len(sub),centroid_x=round(float(xy[:,0].mean()),2),
                          centroid_y=round(float(xy[:,1].mean()),2),
                          length_m=round(float(span[0]),2),
                          width_m=round(float(span[1]),2))
    return result

def passing_lane(ball:tuple[float,float], target:tuple[float,float],
                 defenders:pd.DataFrame, corridor_m:float=2., max_pass_distance:float=55.):
    """Static interception corridor proxy; not a pass-success probability."""
    a=np.asarray(ball,dtype=float);b=np.asarray(target,dtype=float)
    ab=b-a; length=float(np.linalg.norm(ab))
    if length<.5 or length>max_pass_distance:return {"open":False,"blockers":0,"distance_m":length}
    if defenders.empty:return {"open":True,"blockers":0,"distance_m":length}
    pts=defenders[["x","y"]].to_numpy(dtype=float)
    u=np.clip(((pts-a)@ab)/(length**2),0,1)
    dist=np.linalg.norm(pts-(a+u[:,None]*ab),axis=1)
    blocks=(dist<corridor_m)&(u>.05)&(u<.98)
    return {"open":bool(not blocks.any()),"blockers":int(blocks.sum()),"distance_m":round(length,1)}

def zonal_access(access,X,Y,team="Home",direction=1):
    """Rank opponent-controlled zones in defending team's final third (illustrative).
    'direction' is defended goal: +1 right-hand goal or -1 left-hand goal.
    """
    opp=1-access if team=="Home" else access
    mask= (X>17.5) if direction>0 else (X < -17.5)
    zones=[]
    for name,lo,hi in [("left",-34,-11),("centre",-11,11),("right",11,34)]:
        m=mask&(Y>=lo)&(Y<hi)
        if m.any():
            zones.append({"zone":name,"opponent_access":round(float(opp[m].mean()),3)})
    return sorted(zones,key=lambda d:d["opponent_access"],reverse=True)

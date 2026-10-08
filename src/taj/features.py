"""Additional physically interpretable team shape and passing features."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .spatial import passing_lane

def line_spacing(frame:pd.DataFrame,team="Home",attack_sign=1):
    """Three equal-count x-axis bands: diagnostic geometry, not inferred player roles."""
    sub=frame[frame.team==team].sort_values("x",ascending=attack_sign<0)
    if len(sub)<9: return None
    bins=np.array_split(sub["x"].to_numpy(),3)
    centres=[float(np.mean(b)) for b in bins]
    return {
        "band_centres_m":[round(x,2) for x in centres],
        "adjacent_gaps_m":[round(abs(centres[i+1]-centres[i]),2) for i in range(2)],
        "max_gap_m":round(max(abs(centres[1]-centres[0]),abs(centres[2]-centres[1])),2),
        "method":"equal-size longitudinal bands; not actual tactical lines"
    }

def passing_options(frame:pd.DataFrame,ball_xy:tuple[float,float],team="Home",limit=20):
    """Geometric pass corridors, static interception only.
    If the ball-possessing team is unknown, team must be specified by the caller.
    """
    attackers=frame[frame.team==team]
    defenders=frame[frame.team!=team]
    out=[]
    for r in attackers.itertuples():
        info=passing_lane(ball_xy,(float(r.x),float(r.y)),defenders)
        out.append({"player_id":str(r.player_id),"x":float(r.x),"y":float(r.y),**info})
    return sorted(out,key=lambda x:(not x["open"],x["distance_m"]))[:limit]

def spatial_summary(frame:pd.DataFrame,ball_xy:tuple[float,float]|None=None):
    from .spatial import shape_metrics
    results={"team_shape":shape_metrics(frame),
             "line_spacing":{x:line_spacing(frame,x) for x in ("Home","Away")}}
    if ball_xy is not None:
        results["passing_corridors"]={x:passing_options(frame,ball_xy,x)
                                      for x in ("Home","Away")}
    return results

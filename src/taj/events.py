"""Explicit event-to-tracking temporal alignment within a *single match and period*.

Never merge separate providers' timestamps without a verified clock offset.
A nearest timestamp is *not* proof an event and a frame describe one action.
"""
from __future__ import annotations
import pandas as pd
import numpy as np

def _to_seconds(x):
    if pd.isna(x):return np.nan
    if hasattr(x,"total_seconds"):return float(x.total_seconds())
    if isinstance(x,(float,int,np.integer,np.floating)):return float(x)
    try:return float(pd.to_timedelta(x).total_seconds())
    except (TypeError,ValueError):return np.nan

def event_summary(events:pd.DataFrame) -> dict:
    if events is None or events.empty:return {"total":0,"types":{},"provider_event_coverage":False}
    key="event_type" if "event_type" in events else ("type" if "type" in events else None)
    counts=events[key].astype(str).value_counts().head(25).to_dict() if key else {}
    return {"total":int(len(events)),"types":{str(k):int(v) for k,v in counts.items()},
            "provider_event_coverage":True}

def link_events(events:pd.DataFrame,tracking:pd.DataFrame,tolerance_s=.5) -> pd.DataFrame:
    """As-of join to sampled frame; verifies matches, periods and relative seconds.
    Output link_lag_s is nonnegative; unmatched event frame IDs are null.
    """
    cols=["event_id","period_id","timestamp"]
    if events is None or events.empty:
        return pd.DataFrame(columns=["event_id","period_id","t_s","matched_frame_id","link_lag_s"])
    for c in cols[1:]:
        if c not in events:
            raise ValueError("Event data missing required field "+c)
    if tracking.empty:raise ValueError("Cannot join to empty tracking")
    if tracking.match_id.nunique()!=1:raise ValueError("Tracking must belong to one match")
    if "match_id" in events and events.match_id.dropna().nunique() and (events.match_id.dropna().astype(str)!=str(tracking.match_id.iloc[0])).any():
        raise ValueError("Event and tracking match IDs differ")
    e=events.copy()
    if "event_id" not in e:e["event_id"]=e.index.astype(str)
    e["period_id"]=pd.to_numeric(e["period_id"],errors="coerce")
    e["t_s"]=e["timestamp"].map(_to_seconds)
    e=e.dropna(subset=["period_id","t_s"]).copy()
    if e.empty:return pd.DataFrame(columns=["event_id","period_id","t_s","matched_frame_id","link_lag_s"])
    e["period_id"]=e["period_id"].astype(int)
    e=e.sort_values("t_s")
    frames=tracking[["period","frame_id","t_s"]].drop_duplicates().copy()
    frames=frames.rename(columns={"period":"period_id","frame_id":"matched_frame_id","t_s":"frame_t_s"})
    frames=frames.sort_values("frame_t_s")
    joined=pd.merge_asof(e,frames,left_on="t_s",right_on="frame_t_s",
                         by="period_id",direction="nearest",tolerance=float(tolerance_s))
    joined["link_lag_s"]=(joined["t_s"]-joined["frame_t_s"]).abs()
    return joined

import pandas as pd
import pytest
from taj.events import link_events,event_summary
from taj.features import line_spacing,passing_options
from taj.schema import synthetic_match,at_frame

def _tracking():
    return pd.DataFrame([
      dict(match_id="one",period=1,frame_id="f0",t_s=0.),
      dict(match_id="one",period=1,frame_id="f1",t_s=.2),
      dict(match_id="one",period=2,frame_id="f2",t_s=.2)
    ])

def test_nearest_event_never_crosses_periods():
    ev=pd.DataFrame([
      dict(event_id="A",match_id="one",period_id=1,timestamp=.19,event_type="PASS"),
      dict(event_id="B",match_id="one",period_id=2,timestamp=.19,event_type="SHOT")
    ])
    got=link_events(ev,_tracking(),tolerance_s=.1)
    assert got.set_index("event_id").loc["A","matched_frame_id"]=="f1"
    assert got.set_index("event_id").loc["B","matched_frame_id"]=="f2"
    assert event_summary(ev)["types"]=={"PASS":1,"SHOT":1}

def test_asof_rejects_wrong_match():
    ev=pd.DataFrame([dict(event_id="x",match_id="two",period_id=1,timestamp=0)])
    with pytest.raises(ValueError,match="differ"):link_events(ev,_tracking())

def test_no_match_when_clock_lag_exceeds_tolerance():
    ev=pd.DataFrame([dict(event_id="x",period_id=1,timestamp=15.)])
    assert pd.isna(link_events(ev,_tracking(),tolerance_s=.1).matched_frame_id.iloc[0])

def test_spacing_and_options_are_diagnostics():
    tr,_=synthetic_match(frames=1)
    frame=at_frame(tr,1,"0")
    assert line_spacing(frame,"Home")["max_gap_m"]>=0
    opts=passing_options(frame,(-6,0),"Home")
    assert len(opts)==11 and all("blockers" in x for x in opts)

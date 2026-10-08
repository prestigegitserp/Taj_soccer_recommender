import pandas as pd
from taj.weakness import hypotheses
from taj.recommender import recommend
from taj.analysis import build_report
from taj.schema import synthetic_match

def test_adjacent_frames_are_not_independent():
    evidence=pd.DataFrame([dict(match_id="m",period=1,frame_id=str(i),
        t_s=float(i),zone="left",opponent_access=.85,provider="fake") for i in range(20)])
    assert hypotheses(evidence,min_samples=3,window_seconds=12)==[]

def test_separated_windows_generate_traceable_hypothesis():
    evidence=pd.DataFrame([dict(match_id="m",period=1,frame_id=str(i),
       t_s=i*15.,zone="right",opponent_access=.8,provider="fake") for i in range(6)])
    found=hypotheses(evidence,min_samples=4)
    assert len(found)==1 and len(found[0]["evidence"])==6
    rec=recommend(found)
    assert len(rec)==1 and rec[0]["priority"]>0
    assert rec[0]["expected_goal_lift"] is None
    assert all("frame_id" in ev for ev in rec[0]["evidence"])

def test_empty_report_does_not_fabricate_findings():
    tr,b=synthetic_match(frames=2)
    report=build_report(tr,b,stride=2,min_windows=9)
    assert report["synthetic"] is True
    assert report["findings"]==[]
    assert report["recommendations"]==[]

def test_low_support_no_recommendation():
    assert recommend([dict(kind="spatial_exposure",zone="right",
        n_independent_windows=1,severity=.9,evidence=[],statement_fa="x")])==[]

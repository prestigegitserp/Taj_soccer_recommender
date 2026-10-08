import json
from taj.analysis import build_report,create_demo,export_report
from taj.schema import synthetic_match
from taj.recommender import recommend
from taj.export import export_html

def test_full_report_v3_with_sources(tmp_path):
    t,b,r=create_demo(tmp_path/"demo")
    assert r["schema_version"]=="1.1"
    assert r["focus"]["defended_goal_by_period"]=={"1":-1}
    assert r["events"]["total"]==0
    saved=export_report(r,tmp_path/"report.json")
    assert json.loads(saved.read_text())["quality"]["rows"]==len(t)
    f=export_html(r,t,b,tmp_path/"report.html")
    assert "<html" in f.read_text() and "TAJ" in f.read_text()

def test_skillcorner_requires_direction_by_period():
    t,b=synthetic_match(frames=2)
    t["provider"]="skillcorner"
    try:
        build_report(t,b)
    except ValueError as e:
        assert "Direction per period" in str(e)
    else:
        raise AssertionError("Should require directions")
    r=build_report(t,b,direction_by_period={1:-1},min_windows=12)
    assert r["provider"]=="skillcorner"

def test_central_zone_supported_and_causal_claim_absent():
    rec=recommend([{"kind":"spatial_exposure","zone":"central",
        "severity":.5,"n_independent_windows":6,"statement_fa":"centre",
        "evidence":[{"frame_id":"1"}]}])
    assert rec[0]["expected_goal_lift"] is None
    assert rec[0]["confidence_label"]=="exploratory"

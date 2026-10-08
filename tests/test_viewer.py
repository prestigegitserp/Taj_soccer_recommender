import json
from taj.schema import synthetic_match
from taj.analysis import build_report
from taj.viewer import build_viewer_payload,export_viewer_json

def test_actual_sampled_frames_export(tmp_path):
    t,b=synthetic_match(frames=7)
    r=build_report(t,b,min_windows=20)
    p=build_viewer_payload(t,b,r,max_frames=3,step_m=10)
    assert p["format"]=="taj-viewer-v1" and len(p["frames"])==3
    assert len(p["frames"][0]["players"])==22
    assert p["frames"][0]["heatmap"] is not None
    assert json.loads(export_viewer_json(p,tmp_path/"viewer.json").read_text())["frames"][0]["ball"] is not None

def test_reject_zero_frames():
    t,b=synthetic_match(frames=1)
    try:build_viewer_payload(t,b,build_report(t,b,min_windows=9),max_frames=0)
    except ValueError:pass
    else:raise AssertionError("Must reject zero frames")

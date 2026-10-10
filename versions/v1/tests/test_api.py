import pandas as pd
import pytest
from fastapi import HTTPException
from taj import api
from taj.schema import synthetic_match
from taj.ingest import write_bundle

def test_api_analyze_report_and_viewer(tmp_path,monkeypatch):
    monkeypatch.setattr(api,"DATA_ROOT",tmp_path)
    t,b=synthetic_match(frames=5)
    write_bundle(t,b,pd.DataFrame(),tmp_path/"demo")
    r=api.analyze(api.AnalysisRequest(dataset="demo",samples_target=15))
    assert r["schema_version"]=="1.1"
    viewer=api.viewer(api.AnalysisRequest(dataset="demo"),max_frames=3)
    assert viewer["format"]=="taj-viewer-v1"
    assert len(viewer["frames"])==3

def test_api_refuses_unsafe_dataset(tmp_path,monkeypatch):
    monkeypatch.setattr(api,"DATA_ROOT",tmp_path)
    with pytest.raises(HTTPException) as e:
        api.analyze(api.AnalysisRequest(dataset="../etc"))
    assert e.value.status_code==400

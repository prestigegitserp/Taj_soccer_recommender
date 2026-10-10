"""Cross-language parity tests for real TFLite football features."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
file_path=ROOT/"scripts/train_litert_soccer.py"
spec=importlib.util.spec_from_file_location("taj_soccer_training",file_path)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def game(i,league="eng.1"):
    at=(dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=i*3))
    home=str(i%4+1)
    away=str((i+2)%4+1)
    return {
        "league":league,"id":str(i),"kickoff":at.isoformat().replace("+00:00","Z"),
        "state":"post",
        "home":{"id":home,"name":"Club "+home,"score":int(i%4)},
        "away":{"id":away,"name":"Club "+away,"score":int((i*3+1)%3)},
    }

def test_same_game_python_js_features():
    if not shutil.which("node"):pytest.skip("Node not available")
    previous=[game(i) for i in range(40)]
    next_game={
        "league":"eng.1","id":"future","kickoff":"2025-06-15T19:00:00Z","state":"pre",
        "home":{"id":"1","name":"Club 1","score":None},
        "away":{"id":"3","name":"Club 3","score":None},
    }
    python_result=module.extract(next_game,previous)
    assert python_result is not None and len(python_result)==20
    js="""
      import {featuresFor} from './docs/football-features.mjs';
      let buffer='';for await(const chunk of process.stdin)buffer+=chunk;
      const data=JSON.parse(buffer);
      console.log(JSON.stringify(featuresFor(data.match,data.previous)));
    """
    r=subprocess.run(["node","--input-type=module","-e",js],
        cwd=ROOT,input=json.dumps({"match":next_game,"previous":previous}),
        text=True,capture_output=True,check=True)
    browser=json.loads(r.stdout)
    assert browser is not None
    assert browser==pytest.approx(python_result,abs=1e-6)

def test_venue_feature_respects_home_and_away():
    games=[game(i) for i in range(40)]
    next_game={
        "league":"eng.1","id":"next","kickoff":"2025-06-20T12:00:00Z","state":"pre",
        "home":{"id":"1","score":None},"away":{"id":"3","score":None}
    }
    features=module.extract(next_game,games)
    assert features and len(features)==20
    assert features[18]>=0 and features[19]>=0
    assert module.FEATURES[18]=="home_home_gf6"

def test_no_form_if_future_result():
    history=[game(i) for i in range(30)]
    next_game={"league":"eng.1","id":"test","kickoff":"2025-02-01T00:00:00Z",
        "home":{"id":"1"},"away":{"id":"3"}}
    # A real caller must supply only games strictly before kickoff.
    past=[x for x in history if module.parse_time(x["kickoff"])<module.parse_time(next_game["kickoff"])]
    assert len(past)<16
    assert module.extract(next_game,past) is None

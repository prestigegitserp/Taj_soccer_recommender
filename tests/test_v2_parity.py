"""V2 scientific implementation parity: features and REAL exported LightGBM trees."""
from __future__ import annotations
import importlib.util
import json
import math
from pathlib import Path
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
path=ROOT/"versions/v2/features.py"
spec=importlib.util.spec_from_file_location("v2features",path)
features=importlib.util.module_from_spec(spec)
spec.loader.exec_module(features)

def prior(i):
    from datetime import datetime,timedelta,timezone
    t=(datetime(2025,1,1,tzinfo=timezone.utc)+timedelta(days=i*2)).isoformat().replace("+00:00","Z")
    home=str(i%12+1)
    away=str((i+5)%12+1)
    return {"id":str(i),"league":"eng.1","kickoff":t,"state":"post",
        "home":{"id":home,"name":"Team "+home,"score":i%4},
        "away":{"id":away,"name":"Team "+away,"score":(i+1)%3}}

def run_js(source,payload):
    p=subprocess.run(["node","--input-type=module","-e",source],
        cwd=ROOT,input=json.dumps(payload),text=True,check=True,capture_output=True)
    return json.loads(p.stdout)

def test_v2_recency_features_identical_in_python_and_browser():
    games=[prior(i) for i in range(145)]
    target={"id":"test","league":"eng.1","state":"pre",
       "kickoff":"2025-12-22T19:00:00Z",
       "home":{"id":"1","name":"Team 1"},"away":{"id":"6","name":"Team 6"}}
    expected=features.build_extra(target,games)
    assert expected is not None and len(expected)==14
    script="""
      import {v2RawFeatures,EXTRA_NAMES} from "./docs/v2/features.mjs";
      let text=''; for await (const piece of process.stdin) text+=piece;
      const data=JSON.parse(text);
      console.log(JSON.stringify({
        value:v2RawFeatures(data.target,data.games),names:EXTRA_NAMES
      }));
    """
    output=run_js(script,{"target":target,"games":games})
    assert output["names"]==features.EXTRA_NAMES
    assert output["value"]==pytest.approx(expected,abs=1e-7)
    # The target score does not exist. A result published AFTER kickoff
    # must never affect pre-kickoff V2 features.
    future=prior(400)
    future["kickoff"]="2025-12-29T19:00:00Z"
    future["home"]["score"]=25
    future["away"]["score"]=0
    assert features.build_extra(target,games+[future])==pytest.approx(expected)

def test_real_lightgbm_export_matches_independent_python_evaluator():
    forest_path=ROOT/"docs/v2/models/forest.json"
    if not forest_path.exists():pytest.skip("Generated V2 artifact not in this revision")
    forest=json.loads(forest_path.read_text(encoding="utf-8"))
    assert forest["schema"]=="taj-v2-lgbm-v1"
    assert forest["features"][-14:]==features.EXTRA_NAMES
    assert len(forest["features"])==56
    values=[((i%9)-4)*.3 for i in range(56)]
    logits=[0.,0.,0.]
    for i,node in enumerate(forest["trees"]):
        while "v" not in node:
            v=values[node["f"]]
            node=node["l"] if (v<=node["t"] if math.isfinite(v) else node["d"]) else node["r"]
        logits[i%3]+=node["v"]
    z=[k/forest["temperature"] for k in logits]
    m=max(z)
    ex=[math.exp(x-m) for x in z]
    expected=[v/sum(ex) for v in ex]
    script="""
      globalThis.self={};
      const {evaluateForest}=await import("./docs/v2/worker.mjs");
      let text='';for await (const s of process.stdin)text+=s;
      const payload=JSON.parse(text);
      console.log(JSON.stringify(evaluateForest(payload.values,payload.forest)));
    """
    actual=run_js(script,{"values":values,"forest":forest})
    assert actual==pytest.approx(expected,abs=1e-10)
    assert sum(actual)==pytest.approx(1.,abs=1e-12)

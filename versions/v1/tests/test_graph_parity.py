import datetime as dt
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("graph_features",ROOT/"scripts/graph_features.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def fixture(i,league="eng.1"):
    day=(dt.datetime(2025,1,1,tzinfo=dt.timezone.utc)+dt.timedelta(days=i*2))
    h=str(i%12+1);a=str((i+5)%12+1)
    return {"league":league,"id":str(i),"kickoff":day.isoformat().replace("+00:00","Z"),
       "state":"post","home":{"id":h,"name":"Team "+h,"score":i%4},
       "away":{"id":a,"name":"Team "+a,"score":(i+1)%3}}

def get_case():
    history=[fixture(i) for i in range(130)]
    next_={"id":"target","league":"eng.1","kickoff":"2025-12-01T15:00:00Z",
       "home":{"id":"1","name":"Team 1"},"away":{"id":"6","name":"Team 6"},"state":"pre"}
    return next_,history

def test_graph_feature_parity_and_future_leakage():
    match,history=get_case()
    python=module.extract_graph(match,history)
    assert python is not None and len(python)==22
    assert all(-20<float(x)<20 for x in python)
    assert shutil.which("node")
    js="""
      import {buildGraph,GRAPH_NAMES} from './docs/graph-features.mjs';
      let json='';for await(const b of process.stdin)json+=b;
      const x=JSON.parse(json);
      const graph=buildGraph(x.match,x.history);
      console.log(JSON.stringify({features:graph?.features,n:GRAPH_NAMES.length}));
    """
    proc=subprocess.run(["node","--input-type=module","-e",js],
      cwd=ROOT,input=json.dumps({"match":match,"history":history}),
      capture_output=True,text=True,check=True)
    result=json.loads(proc.stdout)
    assert result["n"]==22
    assert result["features"]==pytest.approx(python,abs=1e-7)
    invalid=fixture(500)
    invalid["kickoff"]="2025-12-15T15:00:00Z"
    invalid["home"]["score"]=22
    assert module.extract_graph(match,history+[invalid])==pytest.approx(python,abs=1e-8)

def test_graph_abstains_without_rival_history():
    match,history=get_case()
    assert module.extract_graph(match,history[:10]) is None

def test_direct_head_to_head_depends_on_real_games():
    match,history=get_case()
    result=module.extract_graph(match,history)
    assert result is not None and result[15]>=0

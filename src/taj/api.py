"""Local FastAPI for prepared datasets; no implicit outbound data access."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from .ingest import read_bundle
from .analysis import build_report
from .viewer import build_viewer_payload
from .prediction import predict as predict_results

app=FastAPI(title="TAJ Football Tactical Intelligence",version="0.3.0",
    description="Evidence-driven scouting research with explicit dataset availability.")
DATA_ROOT=Path("data")

class AnalysisRequest(BaseModel):
    dataset:str="demo"
    defending:str="Home"
    goal:int|None=None
    direction_by_period:dict[int,int]|None=None
    min_windows:int=Field(default=4,ge=1,le=100)
    risk_threshold:float=Field(default=.36,ge=0,le=1)
    samples_target:int=Field(default=180,ge=10,le=2000)

class PredictionRequest(BaseModel):
    results_dataset:str
    home:str
    away:str
    as_of:str

def _safe_name(name:str):
    if not name or name!=Path(name).name or name.startswith(".") or "/" in name or "\\" in name:
        raise HTTPException(status_code=400,detail="Invalid dataset name")
    return name

def _bundle(name):
    name=_safe_name(name)
    folder=DATA_ROOT/name
    if not (folder/"tracking.parquet").exists():
        raise HTTPException(status_code=404,detail="Dataset not found")
    return read_bundle(folder)

@app.get("/health")
def health():return {"status":"ok","version":"0.3.0","prediction_live":False}

@app.get("/datasets")
def datasets():
    if not DATA_ROOT.exists():return {"datasets":[]}
    return {"datasets":[p.name for p in DATA_ROOT.iterdir()
                        if p.is_dir() and (p/"tracking.parquet").exists()]}

@app.post("/analyze")
def analyze(request:AnalysisRequest):
    if request.defending not in ("Home","Away") or request.goal not in (None,-1,1):
        raise HTTPException(400,"Invalid team or goal")
    t,b,e=_bundle(request.dataset)
    count=t[["period","frame_id"]].drop_duplicates().shape[0]
    stride=max(1,(count+request.samples_target-1)//request.samples_target)
    try:
        return build_report(t,b,e,defending=request.defending,goal=request.goal,
            direction_by_period=request.direction_by_period,min_windows=request.min_windows,
            high_risk=request.risk_threshold,stride=stride)
    except ValueError as exc:raise HTTPException(422,str(exc))

@app.post("/viewer")
def viewer(request:AnalysisRequest,max_frames:int=100):
    if max_frames<1 or max_frames>400:raise HTTPException(400,"max_frames outside 1..400")
    report=analyze(request)
    t,b,_=_bundle(request.dataset)
    return build_viewer_payload(t,b,report,max_frames=max_frames,step_m=7)

@app.post("/predict")
def predict(request:PredictionRequest):
    name=_safe_name(request.results_dataset)
    path=DATA_ROOT/"results"/(name+".csv")
    if not path.exists():raise HTTPException(404,"Historical CSV not found under data/results/")
    try:
        return predict_results(pd.read_csv(path),request.home,request.away,request.as_of)
    except ValueError as exc:raise HTTPException(422,str(exc))

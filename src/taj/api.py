"""Optional local API: only reads user-generated analyses, no live prediction claims."""
from __future__ import annotations
import json
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel
from .ingest import read_bundle
from .analysis import build_report

app=FastAPI(title="Taj Soccer Recommender API",version="0.2.0")
DATA_ROOT=Path("data")

class AnalysisRequest(BaseModel):
    dataset: str="demo"
    defending: str="Home"
    goal: int=1

@app.get("/health")
def health():return {"status":"ok","model":"heuristic spatial analysis"}

@app.get("/datasets")
def datasets():
    return {"datasets":[x.name for x in DATA_ROOT.iterdir() if x.is_dir() and (x/"tracking.parquet").exists()]} if DATA_ROOT.exists() else {"datasets":[]}

@app.post("/analyze")
def analyze(request:AnalysisRequest):
    if request.defending not in ("Home","Away") or request.goal not in (-1,1):
        raise HTTPException(400,"Invalid team or goal")
    # Avoid path traversal when selecting local datasets.
    if request.dataset != Path(request.dataset).name or request.dataset.startswith("."):
        raise HTTPException(400,"Invalid dataset name")
    folder=DATA_ROOT/request.dataset
    if not (folder/"tracking.parquet").exists():raise HTTPException(404,"Dataset not found")
    tracking,ball,events=read_bundle(folder)
    return build_report(tracking,ball,events,defending=request.defending,goal=request.goal)

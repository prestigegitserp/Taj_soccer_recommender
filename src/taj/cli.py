"""Reproducible Taj CLI, optimized for Colab CPU and optional real-data downloads."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
from .ingest import load_idsse,load_skillcorner_local,write_bundle,read_bundle
from .analysis import create_demo,build_report,export_report
from .export import export_html
from .viewer import build_viewer_payload,export_viewer_json

def parse_directions(text:str|None):
    """Example: '1:-1,2:1'."""
    if not text:return None
    result={}
    for pair in text.split(","):
        try:
            a,b=pair.split(":");a=int(a.strip());b=int(b.strip())
        except (ValueError,TypeError):
            raise argparse.ArgumentTypeError("Expected --directions '1:-1,2:1'")
        if b not in (-1,1):raise argparse.ArgumentTypeError("Directions must be -1 or +1")
        if a in result:raise argparse.ArgumentTypeError("Duplicate period")
        result[a]=b
    return result

def add_analysis_options(parser):
    parser.add_argument("--data",default="data/demo",help="Parquet bundle directory")
    parser.add_argument("--defending",choices=["Home","Away"],default="Home")
    parser.add_argument("--goal",type=int,choices=[-1,1],default=None,
                        help="Apply one sign to all periods (IDSSE default inferred from normalized orientation)")
    parser.add_argument("--directions",type=str,default=None,
                        help="Explicit mapping, e.g. 1:-1,2:1, mandatory for raw SkillCorner unless --goal given")
    parser.add_argument("--sample-target",type=int,default=220)
    parser.add_argument("--min-windows",type=int,default=4)
    parser.add_argument("--risk-threshold",type=float,default=.36)

def report_from_args(args,frames,balls,events):
    directions=parse_directions(args.directions)
    count=frames[["period","frame_id"]].drop_duplicates().shape[0]
    stride=max(1,(count+args.sample_target-1)//args.sample_target)
    return build_report(frames,balls,events,defending=args.defending,goal=args.goal,
        direction_by_period=directions,stride=stride,
        min_windows=args.min_windows,high_risk=args.risk_threshold)

def main(argv=None):
    p=argparse.ArgumentParser(prog="taj",description="TAJ Football Spatial Intelligence")
    sub=p.add_subparsers(dest="cmd",required=True)
    demo=sub.add_parser("demo",help="Generate explicitly SYNTHETIC data to smoke test environment")
    demo.add_argument("--out",default="data/demo")
    ingest=sub.add_parser("ingest",help="Normalize provider data into Parquet")
    ingest.add_argument("--provider",choices=["idsse","skillcorner"],required=True)
    ingest.add_argument("--match",default=None)
    ingest.add_argument("--source-dir",default="external/opendata")
    ingest.add_argument("--out",default=None)
    ingest.add_argument("--limit",type=int,default=2000)
    ingest.add_argument("--sample-rate",type=float,default=2.)
    ingest.add_argument("--no-events",action="store_true")
    ingest.add_argument("--sample-all",action="store_true",
                        help="Process full match; ignores frame limit for IDSSE / SkillCorner")
    a=sub.add_parser("analyze",help="Generate JSON with model provenance, risk and evidence")
    add_analysis_options(a);a.add_argument("--out",default=None)
    v=sub.add_parser("viewer-export",help="Export player positions and heatmap for GitHub Pages viewer")
    add_analysis_options(v);v.add_argument("--out",default=None)
    v.add_argument("--max-frames",type=int,default=110);v.add_argument("--grid",type=float,default=6.)
    h=sub.add_parser("html-export",help="Export standalone interactive Plotly HTML")
    add_analysis_options(h);h.add_argument("--out",default=None)
    args=p.parse_args(argv)
    if args.cmd=="demo":
        _,_,r=create_demo(args.out)
        print(json.dumps({"path":args.out,"synthetic":True,"findings":len(r["findings"])},ensure_ascii=False))
        return
    if args.cmd=="ingest":
        if args.limit<=0 or args.sample_rate<=0:p.error("--limit and --sample-rate must be positive")
        match=args.match or ("J03WMX" if args.provider=="idsse" else "1925299")
        limit=None if args.sample_all else args.limit
        if args.provider=="idsse":
            frames,balls,events=load_idsse(match,limit=limit,sample_rate=args.sample_rate,
                                          with_events=not args.no_events)
        else:
            stride=max(1,round(10/args.sample_rate))
            frames,balls,events=load_skillcorner_local(match,args.source_dir,limit=limit,stride=stride)
            if args.no_events:events=pd.DataFrame()
        output=args.out or "data/"+args.provider+"_"+match
        q=write_bundle(frames,balls,events,output)
        print(json.dumps({"path":output,"quality":q,"events":len(events)},ensure_ascii=False))
        return
    frames,balls,events=read_bundle(args.data)
    report=report_from_args(args,frames,balls,events)
    if args.cmd=="analyze":
        out=args.out or str(Path(args.data)/"report.json")
        export_report(report,out)
        print(json.dumps({"report":out,"hypotheses":len(report["findings"]),
                          "recommendations":len(report["recommendations"]),
                          "warnings":report["warnings"]},ensure_ascii=False))
    elif args.cmd=="viewer-export":
        payload=build_viewer_payload(frames,balls,report,max_frames=args.max_frames,step_m=args.grid)
        out=args.out or str(Path(args.data)/"viewer.json")
        export_viewer_json(payload,out)
        print(json.dumps({"viewer":out,"frames":len(payload["frames"]),
                          "synthetic":payload["synthetic"]},ensure_ascii=False))
    elif args.cmd=="html-export":
        out=args.out or str(Path(args.data)/"scouting_report.html")
        export_html(report,frames,balls,out)
        print(json.dumps({"html":out},ensure_ascii=False))

if __name__=="__main__":main()

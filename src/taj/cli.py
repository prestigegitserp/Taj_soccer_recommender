"""Command-line reproducible workflows."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from .ingest import load_idsse,load_skillcorner_local,write_bundle,read_bundle
from .analysis import create_demo,build_report,export_report

def main(argv=None):
    p=argparse.ArgumentParser(prog="taj",description="Taj Football Spatial Intelligence")
    sub=p.add_subparsers(dest="cmd",required=True)
    demo=sub.add_parser("demo",help="Create labeled synthetic data for offline smoke testing")
    demo.add_argument("--out",default="data/demo")
    ing=sub.add_parser("ingest",help="Download/read open data and normalize as Parquet")
    ing.add_argument("--provider",choices=["idsse","skillcorner"],required=True)
    ing.add_argument("--match",default=None)
    ing.add_argument("--source-dir",default="external/opendata")
    ing.add_argument("--out",default=None)
    ing.add_argument("--limit",type=int,default=2000)
    ing.add_argument("--sample-rate",type=float,default=2.)
    ing.add_argument("--no-events",action="store_true")
    ana=sub.add_parser("analyze")
    ana.add_argument("--data",default="data/demo")
    ana.add_argument("--out",default=None)
    ana.add_argument("--defending",choices=["Home","Away"],default="Home")
    ana.add_argument("--goal",type=int,choices=[-1,1],default=1)
    ana.add_argument("--min-windows",type=int,default=4)
    args=p.parse_args(argv)
    if args.cmd=="demo":
        _,_,r=create_demo(args.out)
        print(json.dumps({"path":args.out,"findings":len(r["findings"]),
                          "synthetic":r["synthetic"]},ensure_ascii=False))
    elif args.cmd=="ingest":
        match=args.match or ("J03WMX" if args.provider=="idsse" else "1925299")
        if args.provider=="idsse":
            frames,ball,events=load_idsse(match,limit=args.limit,
                sample_rate=args.sample_rate,with_events=not args.no_events)
        else:
            frames,ball,events=load_skillcorner_local(match,args.source_dir,
                limit=args.limit,stride=max(1,round(10/args.sample_rate)))
        dest=args.out or f"data/{args.provider}_{match}"
        q=write_bundle(frames,ball,events,dest)
        print(json.dumps({"path":dest, "quality":q},ensure_ascii=False))
    elif args.cmd=="analyze":
        frames,ball,events=read_bundle(args.data)
        rep=build_report(frames,ball,events,defending=args.defending,
                         goal=args.goal,min_windows=args.min_windows)
        out=args.out or str(Path(args.data)/"report.json")
        export_report(rep,out)
        print(json.dumps({"report":out,"hypotheses":len(rep["findings"]),
                          "recommendations":len(rep["recommendations"])},ensure_ascii=False))

if __name__=="__main__":main()

#!/usr/bin/env python3
"""Append-only live prospective prediction ledger for actual upcoming soccer.

Records predictions BEFORE kickoff and never changes their probability vector,
using only earlier completed games. The original prediction time and model hash
are fixed; final actual score may be attached after ESPN reports a completed
fixture. This is for future measurement, NOT a historical backtest.
"""
from __future__ import annotations
from datetime import datetime,timedelta,timezone
import hashlib,json,math,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(Path(__file__).resolve().parent)]
from train_litert_soccer import extract
from graph_features import extract_graph
from features import build_extra

FEED=ROOT/"docs/data/feed.json"
HISTORY=ROOT/"docs/v2/data/history.json"
FOREST=ROOT/"docs/v2/models/forest.json"
REPORT=ROOT/"docs/v2/data/evaluation.json"
LEDGER=ROOT/"docs/v2/data/prospective.json"
def date(s):return datetime.fromisoformat(str(s).replace("Z","+00:00"))
def scores(g):
    return g.get("state")=="post" and type(g.get("home",{}).get("score")) is int and type(g.get("away",{}).get("score")) is int
def softmax(values):
    highest=max(values);a=[math.exp(v-highest) for v in values];s=sum(a)
    return [v/s for v in a]
def forest_predict(row,forest):
    logits=[0.,0.,0.]
    if len(row)!=len(forest["features"]):raise ValueError("Forest feature schema mismatch")
    for i,node in enumerate(forest["trees"]):
        while "v" not in node:
            v=row[node["f"]]
            node=node["l"] if (v<=node["t"] if math.isfinite(v) else node["d"]) else node["r"]
        logits[i%3]+=node["v"]
    return softmax([v/forest["temperature"] for v in logits])
def poisson_probs(row,rho=0):
    bhome,baway=row[0],row[1]
    base=max(.45,(bhome+baway)/2)
    nh,na=row[7]*6,row[11]*6
    shrink=lambda rate,n:(n*rate+6*base)/(n+6)
    lh=min(3.7,max(.25,(shrink(row[4],nh)*shrink(row[9],na)/base)*(bhome/base)))
    la=min(3.7,max(.25,(shrink(row[8],na)*shrink(row[5],nh)/base)*(baway/base)))
    probs=[0.,0.,0.];total=0.;ph=math.exp(-lh)
    for h in range(11):
        if h:ph*=lh/h
        pa=math.exp(-la)
        for a in range(11):
            if a:pa*=la/a
            tau=1.
            if h==0 and a==0:tau=1-lh*la*rho
            elif h==0 and a==1:tau=1+lh*rho
            elif h==1 and a==0:tau=1+la*rho
            elif h==1 and a==1:tau=1-rho
            if tau<=0:raise ValueError("invalid rho")
            p=ph*pa*tau
            probs[0 if h>a else 1 if h==a else 2]+=p;total+=p
    return [v/total for v in probs]
def once(now=None):
    now=now or datetime.now(timezone.utc)
    feed=json.loads(FEED.read_text())
    archive=json.loads(HISTORY.read_text())
    forest=json.loads(FOREST.read_text())
    evaluation=json.loads(REPORT.read_text())
    ledger=json.loads(LEDGER.read_text()) if LEDGER.exists() else {
      "schema":"taj-v2-prospective-v1","games":{}}
    if ledger.get("schema")!="taj-v2-prospective-v1":
        raise RuntimeError("Prospective ledger version mismatch")
    known=ledger.setdefault("games",{})
    published=[g for block in feed["leagues"].values() for g in block["events"]]
    previous={}
    for g in archive["events"]+published:
        if scores(g):previous[g["league"]+":"+g["id"]]=g
    history=sorted(previous.values(),key=lambda g:(g["kickoff"],g["id"]))
    nowiso=now.isoformat()
    predictions=0;resolved=0
    def export(probs):
        assert len(probs)==3 and abs(sum(probs)-1)<1e-5
        return [round(float(v),8) for v in probs]
    for g in published:
        event_id=g["league"]+":"+g["id"]
        if event_id in known:
            r=known[event_id]
            if scores(g) and "final" not in r:
                r["final"]={"home_goals":g["home"]["score"],
                    "away_goals":g["away"]["score"],
                    "outcome":(0 if g["home"]["score"]>g["away"]["score"]
                        else 1 if g["home"]["score"]==g["away"]["score"] else 2),
                    "observed_at":nowiso}
                resolved+=1
            continue
        kickoff=date(g["kickoff"])
        # Never backfill forecasts after kickoff, or after the 15-minute buffer.
        if g["state"]!="pre" or kickoff<=now+timedelta(minutes=15) or kickoff>now+timedelta(days=10):
            continue
        past=[x for x in history if x["league"]==g["league"] and x["kickoff"]<g["kickoff"]]
        base=extract(g,past)
        graph=extract_graph(g,past) if base is not None else None
        extra=build_extra(g,past) if graph is not None else None
        if base is None or graph is None or extra is None:continue
        vector=base+graph+extra
        reference=export(poisson_probs(vector))
        boosted=export(forest_predict(vector,forest))
        rho=float(forest.get("production_rho",0))
        correlated=poisson_probs(vector,rho)
        w=float(forest.get("production_weight",0))
        challenger=export([w*boosted[i]+(1-w)*correlated[i] for i in range(3)])
        promoted=bool(evaluation.get("promote_for_accuracy"))
        official=challenger if promoted else reference
        known[event_id]={
          "id":event_id,"league":g["league"],
          "home":g["home"]["name"],"away":g["away"]["name"],
          "kickoff":g["kickoff"],"first_logged_at":nowiso,
          "min_pre_kickoff_minutes":15,
          "model_version":"2.0.0",
          "forest_sha256":hashlib.sha256(FOREST.read_bytes()).hexdigest(),
          "source_feed_generated_at":feed.get("generated_at"),
          "probabilities":{"poisson":reference,"lgb56":boosted,
            "challenger":challenger,"official":official},
          "promoted":promoted
        }
        predictions+=1
    resolved_games=[g for g in known.values() if g.get("final")]
    metrics={}
    for k in ("poisson","lgb56","challenger","official"):
        if resolved_games:
            n=len(resolved_games);correct=0;brier=0.;loss=0.
            for g in resolved_games:
                p=g["probabilities"][k];y=g["final"]["outcome"]
                correct+=int(p.index(max(p))==y)
                brier+=sum((p[i]-(i==y))**2 for i in range(3))
                loss-=math.log(max(1e-7,p[y]))
            metrics[k]={"n":n,"correct":correct,
              "accuracy":round(correct/n,5),"brier":round(brier/n,5),
              "log_loss":round(loss/n,5)}
    ledger.update({
      "updated_at":nowiso,
      "source":"ESPN public completed scoreboards and scheduled fixtures",
      "predict_before_kickoff":True,
      "games":known,"locked_predictions":len(known),
      "finalized_results":len(resolved_games),"metrics":metrics,
      "research_note":"Prospective predictions stored only before kickoff, never overwritten; exact accuracy depends on future verified final scores."
    })
    LEDGER.parent.mkdir(parents=True,exist_ok=True)
    LEDGER.write_text(json.dumps(ledger,ensure_ascii=False,separators=(",",":")))
    print("V2 REAL PROSPECTIVE",json.dumps({
      "new":predictions,"resolved_now":resolved,"locked":len(known),
      "settled":len(resolved_games),"metrics":metrics},ensure_ascii=False))
if __name__=="__main__":once()

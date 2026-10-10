#!/usr/bin/env python3
"""TAJ V2: Leakage-controlled challengers, 3-fold walk-forward and static export.

V1 never changes. V2 competitors:
  - V1 Poisson (gold-standard reference of this project)
  - time-calibrated Dixon-Coles low-score correction, rho validation-selected
  - two separately trained LightGBM multiclass models: 42 and 56 features
  - validation-selected conservative blend of selected LightGBM & Dixon-Coles
  - V1 20-feature neural / graph baselines as reported separately by V1

Only completed ESPN games; NO fabricated player lineups/xG. All candidates
are retrained on prefix data and evaluated on later disjoint fixtures.
The browser uses an exported LightGBM JSON forest interpreted by pure JS.
"""
from __future__ import annotations

import json,math,os,random,sys
from collections import defaultdict
from datetime import datetime,timezone
from itertools import groupby
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"scripts"),str(Path(__file__).resolve().parent)]
from train_graph_ai import load_history
from train_litert_soccer import FEATURES,extract,poisson_baseline,metrics
from graph_features import GRAPH_NAMES,extract_graph
from features import EXTRA_NAMES,build_extra

V2=ROOT/"docs/v2"
REPORT=V2/"data/evaluation.json"
WEIGHTS=V2/"models/forest.json"
HISTORY=V2/"data/history.json"
MATCHES=ROOT/"versions/v2/backtest-predictions.json"
NAME42=FEATURES+GRAPH_NAMES
NAMES=NAME42+EXTRA_NAMES
SEED=20261010
FOLDS=[(.40,.50,.65),(.55,.65,.80),(.70,.80,1.0)]
RHO=[-.16,-.12,-.08,-.04,0.]
BLEND=[0,.25,.5,.75,1.]
TEMP=[.8,1.,1.2,1.5,1.8]
def softmax(logits):
    z=np.asarray(logits,dtype=float)
    z=z-np.max(z,axis=1,keepdims=True)
    ex=np.exp(z)
    return ex/ex.sum(axis=1,keepdims=True)
def scale_probs(p,t):
    z=np.log(np.clip(p,1e-8,1))/t
    return softmax(z)
def numeric(p,y):
    q=np.clip(np.asarray(p,dtype=float),1e-12,1)
    one=np.eye(3)[y]
    return {"n":len(y),"accuracy":round(float((q.argmax(1)==y).mean()),5),
        "correct":int((q.argmax(1)==y).sum()),
        "brier":round(float(((q-one)**2).sum(1).mean()),5),
        "log_loss":round(float(-np.log(q[np.arange(len(y)),y]).mean()),5)}
def poisson_rates(x):
    bh,ba=float(x[0]),float(x[1])
    base=max(.45,(bh+ba)/2)
    nh,na=float(x[7])*6,float(x[11])*6
    sh=lambda r,n:(n*r+6*base)/(n+6)
    lam=min(3.7,max(.25,(sh(x[4],nh)*sh(x[9],na)/base)*(bh/base)))
    mu=min(3.7,max(.25,(sh(x[8],na)*sh(x[5],nh)/base)*(ba/base)))
    return lam,mu
def dc_one(x,rho):
    lh,la=poisson_rates(x)
    home=draw=away=total=0.
    ph=math.exp(-lh)
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
            if tau<=0: return None
            p=ph*pa*tau
            total+=p
            if h>a:home+=p
            elif h==a:draw+=p
            else:away+=p
    return [home/total,draw/total,away/total]
def dc_probs(x,rho):
    data=[dc_one(row,rho) for row in x]
    if any(item is None for item in data):raise RuntimeError("DC rho outside valid tau")
    return np.array(data,dtype=float)
def samples():
    games=load_history()
    history=defaultdict(list)
    features=[];targets=[];fixtures=[];kickoffs=[]
    for kickoff,group in groupby(games,key=lambda g:g["kickoff"]):
        simultaneous=list(group)
        for g in simultaneous:
            previous=history[g["league"]]
            tab=extract(g,previous)
            graph=extract_graph(g,previous) if tab is not None else None
            extra=build_extra(g,previous) if graph is not None else None
            if tab is None or graph is None or extra is None:continue
            row=tab+graph+extra
            if len(row)!=len(NAMES) or not np.all(np.isfinite(row)):
                raise RuntimeError("V2 invalid feature parity")
            features.append(row);fixtures.append(g);kickoffs.append(kickoff)
            h,a=g["home"]["score"],g["away"]["score"]
            targets.append(0 if h>a else 1 if h==a else 2)
        for g in simultaneous:history[g["league"]].append(g)
    X=np.asarray(features,dtype="float32");y=np.asarray(targets,dtype=int)
    if len(X)<1500:raise RuntimeError("Historical ESPN match sample too small")
    return games,X,y,fixtures,kickoffs
def split(dates,frac):
    i=min(len(dates),round(len(dates)*frac))
    if i==len(dates):return i
    while i>0 and dates[i-1]==dates[i]:i-=1
    return i
def fit_boost(X,y,trainEnd,valEnd,kickoffs,dim,leaves=7,seed=SEED):
    import lightgbm as lgb
    cutoff=datetime.fromisoformat(kickoffs[trainEnd-1].replace("Z","+00:00"))
    def days(s):
        return (cutoff-datetime.fromisoformat(s.replace("Z","+00:00"))).total_seconds()/86400
    weighted=np.asarray([max(.16,math.exp(-math.log(2)*max(0,days(t))/400))
                         for t in kickoffs[:trainEnd]])
    model=lgb.LGBMClassifier(
      objective="multiclass",num_class=3,num_leaves=leaves,max_depth=4,
      learning_rate=.035,n_estimators=250,min_child_samples=44,
      colsample_bytree=.85,reg_lambda=7,reg_alpha=.1,
      max_bin=127,verbosity=-1,random_state=seed,n_jobs=2,
      deterministic=True,force_col_wise=True)
    model.fit(X[:trainEnd,:dim],y[:trainEnd],sample_weight=weighted,
       eval_set=[(X[trainEnd:valEnd,:dim],y[trainEnd:valEnd])],
       eval_metric="multi_logloss",
       callbacks=[lgb.early_stopping(30,verbose=False)])
    if model.best_iteration_ is None:raise RuntimeError("No best iteration")
    return model
def save_forest(model,features,temperature):
    booster=model.booster_
    dump=booster.dump_model(num_iteration=model.best_iteration_)
    def convert(node):
        if "leaf_value" in node:return {"v":round(float(node["leaf_value"]),10)}
        if node.get("decision_type")!="<=":raise RuntimeError("Non-numeric LightGBM split")
        return {"f":int(node["split_feature"]),"t":float(node["threshold"]),
          "d":bool(node.get("default_left",True)),
          "l":convert(node["left_child"]),"r":convert(node["right_child"])}
    trees=[convert(x["tree_structure"]) for x in dump["tree_info"]]
    if not trees or len(trees)%3:raise RuntimeError("Invalid LightGBM tree ordering")
    return {"schema":"taj-v2-lgbm-v1","features":features,
      "num_classes":3,"num_trees":len(trees),"num_tree_per_iteration":3,
      "temperature":temperature,"trees":trees}
def predict_exported(blob,X):
    def traverse(node,row):
        while "v" not in node:
            value=row[node["f"]]
            goes_left=node["d"] if not math.isfinite(float(value)) else float(value)<=node["t"]
            node=node["l"] if goes_left else node["r"]
        return node["v"]
    accum=np.zeros((len(X),3),dtype=float)
    for tree_i,node in enumerate(blob["trees"]):
        for i,row in enumerate(X):accum[i,tree_i%3]+=traverse(node,row)
    return softmax(accum)
def bootstrap(rows,n=1200):
    rng=np.random.default_rng(SEED)
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row["kickoff"][:10]].append(i)
    byday=list(groups.values())
    differences=[]
    for r in rows:
        truth=np.eye(3)[r["outcome"]]
        differences.append(sum((np.array(r["prob"]["v2_auto"])-truth)**2)-
                           sum((np.array(r["prob"]["v1_poisson"])-truth)**2))
    arr=np.array(differences)
    d=[]
    for _ in range(n):
        indexes=[i for k in rng.integers(0,len(byday),size=len(byday)) for i in byday[k]]
        d.append(float(arr[indexes].mean()))
    ci=np.quantile(d,[.025,.975])
    return {"delta_brier":round(float(arr.mean()),6),
        "ci_95":[round(float(ci[0]),6),round(float(ci[1]),6)],
        "paired_by":"UTC match day","bootstrap_draws":n,"significant_95":bool(ci[1]<0 or ci[0]>0)}
def holdout():
    games,x,y,records,dates=samples()
    allrows=[];folds=[]
    lastEnd=0
    for f,(a,b,c) in enumerate(FOLDS,1):
        tr,va,te=split(dates,a),split(dates,b),split(dates,c)
        if not(tr<va<te and va>=lastEnd and dates[tr-1]<dates[tr] and dates[va-1]<dates[va]):
            raise RuntimeError("Overlapping OOS or bad chronological boundary")
        lastEnd=te
        print(f"V2 FOLD {f}: train {tr}, validation {va-tr}, unseen test {te-va}",flush=True)
        poisson_val=poisson_baseline(x[tr:va,:20])
        poisson_test=poisson_baseline(x[va:te,:20])
        dc_valid_candidates={rho:dc_probs(x[tr:va,:20],rho) for rho in RHO}
        rho=min(RHO,key=lambda rho:numeric(dc_valid_candidates[rho],y[tr:va])["log_loss"])
        dc_val=dc_valid_candidates[rho];dc_test=dc_probs(x[va:te,:20],rho)
        graph=fit_boost(x,y,tr,va,dates,42,seed=SEED+f*11)
        expanded=fit_boost(x,y,tr,va,dates,56,seed=SEED+f*11+1)
        candidates={}
        for key,fit,dim in [("lgb42",graph,42),("lgb56",expanded,56)]:
            v=fit.predict_proba(x[tr:va,:dim])
            temp=min(TEMP,key=lambda t:numeric(scale_probs(v,t),y[tr:va])["log_loss"])
            candidates[key]={"model":fit,"dim":dim,"temp":temp,
                "val":scale_probs(v,temp),
                "test":scale_probs(fit.predict_proba(x[va:te,:dim]),temp)}
        # One candidate + conservative blend. Selection uses ONLY validation.
        choices={"v1_poisson":(poisson_val,poisson_test),
            "dixon_coles":(dc_val,dc_test)}
        for key,obj in candidates.items():
            choices[key]=(obj["val"],obj["test"])
            for w in [.25,.5,.75]:
                choices[key+"_mix"+str(w)]=(w*obj["val"]+(1-w)*dc_val,
                                              w*obj["test"]+(1-w)*dc_test)
        selected=min(choices,key=lambda key:numeric(choices[key][0],y[tr:va])["log_loss"])
        winner=choices[selected][1]
        # Gate is fixed before testing. A false discovery must not displace Poisson.
        valPoisson=numeric(poisson_val,y[tr:va])
        valWinner=numeric(choices[selected][0],y[tr:va])
        improved=(valWinner["log_loss"]<valPoisson["log_loss"]-.001
             and valWinner["brier"]<valPoisson["brier"]-.001)
        if not improved:selected="v1_poisson";winner=poisson_test
        output={"v1_poisson":poisson_test,"dixon_coles":dc_test,
                "lgb42":candidates["lgb42"]["test"],
                "lgb56":candidates["lgb56"]["test"],
                "v2_auto":winner}
        group=[]
        for idx,g in enumerate(records[va:te]):
            row={"id":g["league"]+":"+g["id"],"kickoff":g["kickoff"],"league":g["league"],
                "home":g["home"]["name"],"away":g["away"]["name"],
                "score":[g["home"]["score"],g["away"]["score"]],"outcome":int(y[va+idx]),
                "fold":f,"choice":selected,
                "prob":{key:[round(float(z),8) for z in value[idx]] for key,value in output.items()}}
            group.append(row);allrows.append(row)
        summaries={key:numeric(value,y[va:te]) for key,value in output.items()}
        folds.append({"fold":f,"train_n":tr,"validation_n":va-tr,"test_n":te-va,
            "train_until":dates[tr-1],"validation_until":dates[va-1],
            "test_from":dates[va],"test_through":dates[te-1],
            "selected_without_test_outcomes":selected,
            "valid_poisson":valPoisson,"valid_selected":valWinner,
            "rho":rho,"models":summaries,
            "g42_best_iteration":graph.best_iteration_,
            "g56_best_iteration":expanded.best_iteration_})
        print("V2 FOLD",f,"VALIDATED",selected,"TEST",json.dumps(summaries),flush=True)
    if len({r["id"] for r in allrows})!=len(allrows):
        raise RuntimeError("Repeated unseen match across folds")
    names=list(allrows[0]["prob"])
    overall={key:numeric(np.array([r["prob"][key] for r in allrows]),
                         np.array([r["outcome"] for r in allrows])) for key in names}
    delta=bootstrap(allrows)
    print("V2 OVERALL",json.dumps(overall),"CI",delta,flush=True)
    per_league={}
    for league in sorted({r["league"] for r in allrows}):
        matches=[r for r in allrows if r["league"]==league]
        labels=np.array([r["outcome"] for r in matches])
        per_league[league]={"count":len(matches),"v1_poisson":numeric(
          [r["prob"]["v1_poisson"] for r in matches],labels),
          "v2_auto":numeric([r["prob"]["v2_auto"] for r in matches],labels)}
    # Production model: train on completed historical outcomes ONLY;
    # reserve most recent 15% for tuning; old 85% for weights.
    prod_train=split(dates,.85);prod_val=len(x)
    production=fit_boost(x,y,prod_train,prod_val,dates,56,seed=SEED+203)
    production_val=production.predict_proba(x[prod_train:,:56])
    production_temp=min(TEMP,key=lambda t:numeric(scale_probs(production_val,t),
                          y[prod_train:])["log_loss"])
    export=save_forest(production,NAMES,production_temp)
    predictions=predict_exported(export,x[-24:,:56])
    reference=production.predict_proba(x[-24:,:56])
    deviation=float(np.max(np.abs(predictions-reference)))
    if deviation>1e-5:raise RuntimeError("V2 browser LightGBM export mismatch "+str(deviation))
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
    HISTORY.parent.mkdir(parents=True,exist_ok=True)
    MATCHES.parent.mkdir(parents=True,exist_ok=True)
    report={"schema":"taj-v2-evaluation-v1","version":"2.0.0",
       "created_at":datetime.now(timezone.utc).isoformat(),
       "sourced":"ESPN completed real match scores, no synthetic outcomes",
       "raw_historical_games":len(games),"eligible_games":len(x),
       "out_of_sample_games":len(allrows),
       "from":allrows[0]["kickoff"],"through":allrows[-1]["kickoff"],
       "architecture":"V2 temporal 56-feature LightGBM/42-feature LightGBM + Dixon-Coles; validation-only champion gating",
       "features":NAMES,"folds":folds,"overall":overall,
       "per_league":per_league,"paired_95pct":delta,
       "production_forest":{"file":"models/forest.json","tree_count":export["num_trees"],
          "features":len(NAMES),"temperature":production_temp,
          "python_js_numerical_max_error":deviation,
          "trained_until":dates[prod_train-1],
          "tuned_until":dates[-1]},
       "research_warning":"Architecture iterated after seeing V1 backtest; not a pristine preregistered blind benchmark.",
       "promote_for_accuracy":bool(delta["ci_95"][1]<0
          and overall["v2_auto"]["log_loss"]<=overall["v1_poisson"]["log_loss"]),
       "explanations":[
          "Test games never in gradient boosting fit or validation for their own fold",
          "Prediction features use finished games strictly BEFORE target kickoff",
          "Non-overlapping walkforward test folds; model reinitialized every fold",
          "Weights/calibration chosen on validation only; reported CI paired by calendar day",
          "Current registered rosters NOT used for historical lineup reconstruction",
          "Retrospective public ESPN snapshots, not prospective logged live forecasts",
          "Raw historical real data excludes live injury availability and tracking",
          "V2 experimental tree forest is visible, yet conservative Poisson reference stays available"
       ]}
    WEIGHTS.write_text(json.dumps(export,separators=(",",":")))
    REPORT.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    MATCHES.write_text(json.dumps({"schema":"taj-v2-predictions-v1","matches":allrows},
       ensure_ascii=False,separators=(",",":")))
    HISTORY.write_text(json.dumps({
      "schema":"taj-v2-history-v1",
      "updated_at":report["created_at"],
      "events":games},ensure_ascii=False,separators=(",",":")))
    print("V2 OUTPUTS:",[str(v)+" "+str(v.stat().st_size)+" bytes"
         for v in (WEIGHTS,REPORT,MATCHES,HISTORY)],flush=True)
    return report

if __name__=="__main__":
    random.seed(SEED);np.random.seed(SEED)
    holdout()

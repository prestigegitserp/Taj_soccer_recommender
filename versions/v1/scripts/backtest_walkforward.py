#!/usr/bin/env python3
"""Auditable, rolling-origin, genuinely out-of-sample football backtest.

The full dataset is REAL completed ESPN matches. Features at kickoff t are
computed from matches with kickoff < t. At each independently retrained fold:
  train (older) -> validation (newer) -> TEST (strictly later)
Train-only scaling; train/validation-only early stopping, calibration and
ensemble weights. Fold TEST outcomes never influence training or selection.
Test windows are disjoint. Do NOT use a single production model to retrospectively
score fixtures which that production model may have been trained on.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from itertools import groupby
import json
import math
import os
from pathlib import Path
import random
from statistics import NormalDist

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","2")
os.environ.setdefault("OMP_NUM_THREADS","2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS","2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS","2")
import numpy as np
import tensorflow as tf

from train_graph_ai import load_history, make_examples, network, NAMES
from train_litert_soccer import poisson_baseline, metrics, calibrate, FEATURES, ARCHIVE

BASE=Path(__file__).resolve().parents[1]
PUBLIC=BASE/"docs/data/backtest.json"
FULL=BASE/"training/backtest-predictions.json"
SEED=20261010
CLASSES=["home","draw","away"]

# Locked a priori and tested only against chronologically later dates.
WINDOW_FRACTIONS=[
  (.40,.50,.65),
  (.55,.65,.80),
  (.70,.80,1.00),
]
WEIGHTS=[0.,.25,.50,.75,1.]
TEMPERATURES=[round(float(t),3) for t in np.arange(.80,2.51,.10)]

def align_start(times, fraction):
    """Align folds to the first fixture of a shared kickoff time to avoid overlap."""
    raw=min(len(times),max(0,int(round(len(times)*fraction))))
    if raw>=len(times):return len(times)
    t=times[raw]
    while raw>0 and times[raw-1]==t:raw-=1
    return raw

def construct_model(dim,seed,graph=False):
    tf.keras.utils.set_random_seed(seed)
    if graph:
        return network()
    return tf.keras.Sequential([
      tf.keras.layers.Input(shape=(dim,),dtype=tf.float32),
      tf.keras.layers.Dense(32,activation="relu"),
      tf.keras.layers.Dense(16,activation="relu"),
      tf.keras.layers.Dense(3,activation="softmax"),
    ])

def train_one(x,y,train_end,valid_end,dim,seed,graph=False):
    mean=x[:train_end,:dim].mean(axis=0)
    std=np.maximum(x[:train_end,:dim].std(axis=0),1e-5)
    normalized=np.clip((x[:,:dim]-mean)/std,-5,5).astype("float32")
    tf.keras.backend.clear_session()
    model=construct_model(dim,seed,graph)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.0015 if graph else .002),
                  loss="sparse_categorical_crossentropy")
    stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",
             patience=9,restore_best_weights=True)
    model.fit(normalized[:train_end],y[:train_end],
       validation_data=(normalized[train_end:valid_end],y[train_end:valid_end]),
       batch_size=48,epochs=60,callbacks=[stop],verbose=0,shuffle=True)
    validraw=model.predict(normalized[train_end:valid_end],verbose=0)
    if np.any(~np.isfinite(validraw)):
        raise RuntimeError("Invalid validation probabilities")
    temps=[metrics(calibrate(validraw,y[train_end:valid_end],t),
                   y[train_end:valid_end])["log_loss"] for t in TEMPERATURES]
    temperature=TEMPERATURES[int(np.argmin(temps))]
    return model,normalized,temperature,validraw

def brier_per_match(probs,actual):
    truth=np.eye(3,dtype=np.float64)[actual]
    return ((probs-truth)**2).sum(axis=1)

def confidence_interval_accuracy(correct_count,n,z=1.9599639845):
    """Wilson 95% interval for the accuracy of the argmax prediction."""
    if n==0:return [None,None]
    p=correct_count/n
    den=1+z*z/n
    center=(p+z*z/(2*n))/den
    margin=(z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)))/den
    return [round(center-margin,5),round(center+margin,5)]

def bootstrap_day_blocks(rows,models=("graph_ensemble","poisson"),seed=SEED,n_boot=2000):
    """Resample calendar-day blocks, not individual games, for dependence robustness."""
    groups=defaultdict(list)
    for i,row in enumerate(rows):groups[row["date"][:10]].append(i)
    days=list(groups.values())
    rng=np.random.default_rng(seed)
    diffs=[]
    observed=np.array([
      sum((np.asarray(r["probabilities"][models[0]])-np.eye(3)[r["outcome"]])**2)
      -sum((np.asarray(r["probabilities"][models[1]])-np.eye(3)[r["outcome"]])**2)
      for r in rows],dtype=float)
    for _ in range(n_boot):
        indexes=[idx for j in rng.integers(0,len(days),size=len(days)) for idx in days[j]]
        diffs.append(float(observed[indexes].mean()))
    estimate=float(observed.mean())
    ci=np.quantile(diffs,[.025,.975])
    return {"delta_brier":round(estimate,5),
      "ci_95":[round(float(v),5) for v in ci],
      "confidence_description":"paired block bootstrap, grouped by fixture UTC calendar day",
      "bootstrap_samples":n_boot,
      "interpretation":"negative is better for ensemble",
      "statistically_resolved_at_95pct":bool(ci[1]<0 or ci[0]>0)}

def detailed_metrics(rows,key):
    p=np.asarray([r["probabilities"][key] for r in rows])
    actual=np.asarray([r["outcome"] for r in rows])
    m=metrics(p,actual)
    hits=int(np.sum(p.argmax(axis=1)==actual))
    m["correct"]=hits
    m["accuracy_ci_95_wilson"]=confidence_interval_accuracy(hits,len(actual))
    return m

def summarize_calibration(rows,key):
    probs=np.asarray([r["probabilities"][key] for r in rows])
    y=np.asarray([r["outcome"] for r in rows])
    confidence=probs.max(axis=1)
    correct=(probs.argmax(axis=1)==y)
    bins=[]
    for lo,hi in [(0,0.4),(.4,.5),(.5,.6),(.6,.7),(.7,1.000001)]:
        indices=(confidence>=lo)&(confidence<hi)
        n=int(indices.sum())
        bins.append({"range":[lo,min(1,hi)],"n":n,
         "mean_predicted_confidence":round(float(confidence[indices].mean()),4) if n else None,
         "real_accuracy":round(float(correct[indices].mean()),4) if n else None})
    return bins

def run():
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    games=load_history()
    x,y,dates,records=make_examples(games,with_matches=True)
    n=len(y)
    assert len(NAMES)==42 and len(FEATURES)==20
    if n<1500:raise RuntimeError("Not enough genuine complete matches for this backtest")
    # Historical data is sourced from the immutable-ish ESPN completed snapshots
    # held in the repository. No artificial final scores are generated.
    index={f"{g['league']}:{g['id']}":g for g in games}
    assert len({f"{g['league']}:{g['id']}" for g in records})==len(records)
    periods=[]
    all_rows=[]
    previous_test_end=0
    for fold,(train_frac,val_frac,test_frac) in enumerate(WINDOW_FRACTIONS,1):
        tr=align_start(dates,train_frac)
        va=align_start(dates,val_frac)
        te=align_start(dates,test_frac)
        if not(700<tr<va<te<=n):
            raise RuntimeError("Invalid fold boundaries")
        if not(dates[tr-1]<dates[tr]<=dates[va-1]<dates[va]
          <=dates[te-1] or te==n):
            # Explicit comparisons checked individually below; condition
            # above is redundant safety against unexpectedly unsorted dates.
            raise RuntimeError("Nonchronological fold")
        assert dates[tr-1]<dates[tr]
        assert dates[va-1]<dates[va]
        if te<n:assert dates[te-1]<dates[te]
        if va<previous_test_end:raise RuntimeError("TEST windows overlap!")
        previous_test_end=te
        print(f"FOLD {fold}: train {tr}, validation {va-tr}, test {te-va}, "
              f"test {dates[va]} → {dates[te-1]}",flush=True)

        # No model sees TEST labels, including normalization/early-stop/temp/blend.
        base,baseX,baseTemp,baseValid=train_one(x,y,tr,va,20,SEED+fold*4,False)
        graph,graphX,graphTemp,graphValid=train_one(x,y,tr,va,42,SEED+fold*4+1,True)
        poisson_validation=poisson_baseline(x[tr:va,:20])
        graph_validation=calibrate(graphValid,y[tr:va],graphTemp)
        losses=[
          metrics(w*graph_validation+(1-w)*poisson_validation,y[tr:va])["log_loss"]
          for w in WEIGHTS
        ]
        weight=float(WEIGHTS[int(np.argmin(losses))])
        p_base=calibrate(base.predict(baseX[va:te],verbose=0),y[va:te],baseTemp)
        p_graph=calibrate(graph.predict(graphX[va:te],verbose=0),y[va:te],graphTemp)
        p_poisson=poisson_baseline(x[va:te,:20])
        p_ensemble=weight*p_graph+(1-weight)*p_poisson
        freq=np.bincount(y[:tr],minlength=3)/tr
        p_frequency=np.broadcast_to(freq,(te-va,3))
        predictions={
            "mlp_20":p_base,"graph_42":p_graph,"poisson":p_poisson,
            "graph_ensemble":p_ensemble,"frequency":p_frequency}
        for arr in predictions.values():
            assert arr.shape==(te-va,3)
            assert np.max(np.abs(arr.sum(axis=1)-1))<1e-4
        rows=[]
        for k,g in enumerate(records[va:te]):
            outcomes={
              "home_goals":g["home"]["score"],
              "away_goals":g["away"]["score"]
            }
            rows.append({
              "fold":fold,
              "date":g["kickoff"],
              "id":g["league"]+":"+g["id"],
              "league":g["league"],
              "home":{"id":g["home"]["id"],"name":g["home"].get("name","")},
              "away":{"id":g["away"]["id"],"name":g["away"].get("name","")},
              "score":outcomes,
              "outcome":int(y[va+k]),
              "probabilities":{name:[round(float(v),7) for v in arr[k]] for name,arr in predictions.items()},
            })
        all_rows.extend(rows)
        period={
          "fold":fold,"train_n":tr,"validation_n":va-tr,"test_n":te-va,
          "train_through":dates[tr-1],"validation_from":dates[tr],
          "validation_through":dates[va-1],"test_from":dates[va],
          "test_through":dates[te-1],
          "models":{key:detailed_metrics(rows,key) for key in predictions},
          "graph_temperature":graphTemp,
          "mlp_temperature":baseTemp,
          "validated_graph_ensemble_weight":weight,
        }
        periods.append(period)
        print("FOLD",fold, "WEIGHT",weight,"METRICS",json.dumps(period["models"]),flush=True)
        del base,graph
        tf.keras.backend.clear_session()

    ids=[r["id"] for r in all_rows]
    if len(ids)!=len(set(ids)):
        raise RuntimeError("Match occurs in multiple test windows: invalid OOS backtest")
    overall={key:detailed_metrics(all_rows,key)
             for key in ["frequency","poisson","mlp_20","graph_42","graph_ensemble"]}
    per_league={}
    for league,group in groupby(sorted(all_rows,key=lambda z:z["league"]),
                              key=lambda z:z["league"]):
        block=list(group)
        per_league[league]={
            "n":len(block),
            "models":{key:detailed_metrics(block,key)
               for key in ["poisson","graph_42","graph_ensemble"]}
        }
    cross=bootstrap_day_blocks(all_rows)
    confusion=[[0]*3 for _ in range(3)]
    for row in all_rows:
        guess=int(np.argmax(row["probabilities"]["graph_ensemble"]))
        confusion[row["outcome"]][guess]+=1
    kickoff_dates=[r["date"] for r in all_rows]
    source=json.loads(ARCHIVE.read_text())
    report={
      "schema":"taj-walkforward-v1",
      "generated_at":datetime.now(timezone.utc).isoformat(),
      "source":"Actual completed football ESPN scoreboard scores captured in repository training/history.json",
      "archive_updated_at":source.get("updated_at"),
      "input_dataset_games":len(games),
      "eligible_pre_match_examples":n,
      "oos_unique_fixtures":len(all_rows),
      "oos_from":min(kickoff_dates),
      "oos_to":max(kickoff_dates),
      "classes":CLASSES,
      "formula":"multiclass Brier sum of squared errors across 3 outcomes, unscaled",
      "temporal_protocol":{
         "folds":3,
         "strategy":"expanding-window rolling-origin; repeated fresh training and validation; non-overlapping test windows",
         "fractions":WINDOW_FRACTIONS,
         "features":"20 strictly pre-kickoff + 22 graph of completed prior fixture edges",
         "train_only_scaler":True,
         "validation_only_hyperparameters":["early stopping","temperature","ensemble weight"],
         "test_label_used_for_training":False,
         "same_kickoff_batched":True,
         "no_current_player_roster_as_historical_covariates":True,
         "snapshot_caveat":"Historical ESPN completed scoreboards are retrospective. We reconstruct what was known before kickoff by timestamps; original historical API snapshots were not recorded live at those times.",
      },
      "overall":overall,
      "folds":periods,
      "per_league":per_league,
      "paired_uncertainty":cross,
      "ensemble_calibration":summarize_calibration(all_rows,"graph_ensemble"),
      "ensemble_confusion":confusion,
      "saved_game_level_predictions":"training/backtest-predictions.json",
      "conclusion_rule":"Statistical superiority requires paired CI entirely below zero (ensemble - Poisson).",
      "limitations":[
        "Not a live prospective forward test; historical fixtures and final scores were retrieved retrospectively.",
        "No player starting XI, player injuries, xG or odds in the model features.",
        "ESPN public feed is unofficial and may contain retrospective corrections.",
        "Three folds share an underlying history; confidence intervals are approximate, grouped by match day.",
        "Model/architecture were prototyped before this backtest; treat as research development, not a preregistered unbiased benchmark.",
        "Do not use these metrics as a guarantee or wagering advice."
      ],
    }
    FULL.parent.mkdir(parents=True,exist_ok=True)
    PUBLIC.parent.mkdir(parents=True,exist_ok=True)
    FULL.write_text(json.dumps({"schema":"taj-backtest-game-predictions-v1",
        "generated_at":report["generated_at"],"matches":all_rows},
        ensure_ascii=False,separators=(",",":")))
    PUBLIC.write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print("OOS BACKTEST COMPLETE",json.dumps({
      "n":len(all_rows),"from":report["oos_from"],"to":report["oos_to"],
      "overall":overall,"paired":cross},ensure_ascii=False),flush=True)
    return report

if __name__=="__main__":run()

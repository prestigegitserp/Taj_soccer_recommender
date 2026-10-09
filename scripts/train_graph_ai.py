#!/usr/bin/env python3
"""Train a genuinely graph-conditioned, 4-layer football network.

The knowledge graph comprises dated (team)-[played]-(team) edges, and message
aggregation across one/two opponent hops; 22 reproducible graph features
complement 20 already validated pre-match tabular features. This is NOT a GNN
with learned edge message weights: it is a neural graph-feature model.
Splits are chronological and frozen. Calibrated mixing with a Poisson baseline
is selected ONLY from validation; true test remains untouched.
"""
from __future__ import annotations
import datetime as dt
from itertools import groupby
import json
from pathlib import Path
import random
import numpy as np
import tensorflow as tf
from train_litert_soccer import (
  ARCHIVE,MODELS,FEATURES,LEAGUES,parse_time,extract,metrics,calibrate,
  poisson_baseline,UTC,
)
from graph_features import GRAPH_NAMES,extract_graph

NAMES=FEATURES+GRAPH_NAMES
OUT=MODELS/"graph_1x2.tflite"
CARD=MODELS/"graph-card.json"
def load_history():
    store=json.loads(ARCHIVE.read_text())
    if store.get("schema")!="taj-history-v1":raise RuntimeError("Historical archive schema invalid")
    all_games={}
    for block in store["months"].values():
        for g in block:
            if g.get("state")=="post" and isinstance(g.get("home",{}).get("score"),int) and isinstance(g.get("away",{}).get("score"),int):
                all_games[g["league"]+":"+g["id"]]=g
    return sorted(all_games.values(),key=lambda g:(g["kickoff"],g["id"]))

def make_examples(games, with_matches=False):
    history={league:[] for league in LEAGUES}
    data=[];labels=[];dates=[];match_metadata=[]
    for stamp,group in groupby(games,key=lambda g:g["kickoff"]):
        simultaneous=list(group)
        for g in simultaneous:
            prior=history[g["league"]]
            tab=extract(g,prior)
            graph=extract_graph(g,prior) if tab is not None else None
            if tab is not None and graph is not None:
                row=tab+graph
                if not np.all(np.isfinite(row)):
                    raise RuntimeError("Nonfinite graph feature")
                h,a=g["home"]["score"],g["away"]["score"]
                data.append(row);labels.append(0 if h>a else 1 if h==a else 2)
                dates.append(stamp)
                match_metadata.append(g)
        for g in simultaneous:history[g["league"]].append(g)
    if len(data)<400:raise RuntimeError(f"Not enough dated graph examples: {len(data)}")
    result=(np.array(data,dtype=np.float32),np.array(labels,dtype=np.int32),dates)
    return (*result,match_metadata) if with_matches else result

def network(batch_size=None):
    model=tf.keras.Sequential([
      tf.keras.layers.Input(shape=(len(NAMES),),batch_size=batch_size,dtype=tf.float32),
      tf.keras.layers.Dense(96,activation="relu"),
      tf.keras.layers.Dense(64,activation="relu"),
      tf.keras.layers.Dense(32,activation="relu"),
      tf.keras.layers.Dense(3,activation="softmax"),
    ])
    return model

def run():
    seed=20261010
    tf.keras.utils.set_random_seed(seed)
    random.seed(seed);np.random.seed(seed)
    games=load_history()
    x,y,dates=make_examples(games)
    n=len(x);ntrain=int(n*.70);nvalid=int(n*.85)
    if min(ntrain,nvalid-ntrain,n-nvalid)<55:raise RuntimeError("Holdouts too small")
    mean=x[:ntrain].mean(axis=0);std=np.maximum(x[:ntrain].std(axis=0),1e-5)
    z=np.clip((x-mean)/std,-5,5).astype(np.float32)
    net=network()
    net.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=.0015),
      loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    stop=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=12,restore_best_weights=True)
    net.fit(z[:ntrain],y[:ntrain],validation_data=(z[ntrain:nvalid],y[ntrain:nvalid]),
      batch_size=48,epochs=85,verbose=0,callbacks=[stop])
    validraw=net.predict(z[ntrain:nvalid],verbose=0)
    choices=np.arange(.75,2.51,.05)
    temps=[metrics(calibrate(validraw,y[ntrain:nvalid],t),y[ntrain:nvalid])["log_loss"] for t in choices]
    temperature=float(choices[int(np.argmin(temps))])
    valmodel=calibrate(validraw,y[ntrain:nvalid],temperature)
    valpoisson=poisson_baseline(x[ntrain:nvalid,:20])
    # Convex mixture is CHOSEN on validation only; never tuned on test.
    candidates=[0,.25,.5,.75,1]
    loss=[]
    for weight in candidates:
        prob=weight*valmodel+(1-weight)*valpoisson
        loss.append(metrics(prob,y[ntrain:nvalid])["log_loss"])
    best_weight=float(candidates[int(np.argmin(loss))])
    testmodel=calibrate(net.predict(z[nvalid:],verbose=0),y[nvalid:],temperature)
    testpoisson=poisson_baseline(x[nvalid:,:20])
    testblend=best_weight*testmodel+(1-best_weight)*testpoisson
    testfreq=np.bincount(y[:ntrain],minlength=3)/ntrain
    card_metrics={
      "graph_model_holdout":metrics(testmodel,y[nvalid:]),
      "poisson_holdout":metrics(testpoisson,y[nvalid:]),
      "validation_blend_holdout":metrics(testblend,y[nvalid:]),
      "frequency_holdout":metrics(np.broadcast_to(testfreq,testpoisson.shape),y[nvalid:]),
    }
    net_fixed=network(batch_size=1)
    net_fixed.set_weights(net.get_weights())
    converter=tf.lite.TFLiteConverter.from_keras_model(net_fixed)
    binary=converter.convert()
    runner=tf.lite.Interpreter(model_content=binary)
    runner.allocate_tensors()
    details=runner.get_input_details()[0];out=runner.get_output_details()[0]
    if details["shape_signature"].tolist()!=[1,len(NAMES)] or out["shape_signature"].tolist()!=[1,3]:
        raise RuntimeError("LiteRT browser requires fixed [1,42] input and [1,3] output")
    diffs=[]
    for row,ref in zip(z[nvalid:nvalid+10],net.predict(z[nvalid:nvalid+10],verbose=0)):
        runner.set_tensor(details["index"],row.reshape(1,-1))
        runner.invoke()
        diffs.append(float(np.max(np.abs(runner.get_tensor(out["index"])[0]-ref))))
    if max(diffs)>3e-4:raise RuntimeError("TFLite conversion drift: "+str(max(diffs)))
    MODELS.mkdir(parents=True,exist_ok=True)
    OUT.write_bytes(binary)
    info={
      "schema":"taj-graph-ai-v1",
      "architecture":"graph-conditioned-MLP-42-96-64-32-3",
      "graph_topology":"team nodes, timestamped played-vs edges, opponent strength + 2-hop aggregation; no learned GNN message pass",
      "created_at":dt.datetime.now(UTC).isoformat(),
      "input_features":NAMES,"graph_features":GRAPH_NAMES,
      "input_shape":[1,len(NAMES)],"output_shape":[1,3],
      "means":[float(v) for v in mean],
      "stds":[float(v) for v in std],
      "temperature":round(temperature,4),
      "validated_blend_graph_weight":best_weight,
      "train_n":int(ntrain),"validation_n":int(nvalid-ntrain),"test_n":int(n-nvalid),
      "train_until":dates[ntrain-1],"validation_until":dates[nvalid-1],
      "test_from":dates[nvalid],
      "metrics":card_metrics,"max_conversion_difference":max(diffs),
      "provenance":"ESPN completed match scorelines only, 25-month window",
      "missing_sources":["verified actual pre-match lineups","player tracking","xG","injury reports"],
      "caveat":"experimental performance; no guarantee of winning against Poisson, no transfer of future team knowledge",
    }
    CARD.write_text(json.dumps(info,ensure_ascii=False,indent=2))
    print("GRAPH AI MODEL trained:",n,"games, 42 graph/scoreline features, ",len(binary),"bytes")
    print("GRAPH AI HONEST CHRONO HOLDOUT:",json.dumps(card_metrics))
    print("GRAPH AI validation-only blend weight:",best_weight)
    return info

if __name__=="__main__":run()

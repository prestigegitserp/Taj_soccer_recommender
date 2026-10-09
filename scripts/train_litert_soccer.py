#!/usr/bin/env python3
"""Build a reproducible football 1X2 Keras model and LiteRT .tflite artifact.

Data: historical completed ESPN scoreboard events, merged per calendar month.
Features: strictly pre-kickoff team form + rolling league rates, not postmatch
stats. Split chronologically into train/validation/test. Never score train on
test or promote performance claims without explicit holdout measurement.
GitHub Actions does the infrequent training, the visitor only runs inference.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from build_public_feed import LEAGUES, fetch_json, normalize  # noqa: E402

BASE=Path(__file__).resolve().parents[1]
ARCHIVE=BASE/"training/history.json"
MODELS=BASE/"docs/models"
FEATURES=[
    "league_home_goals","league_away_goals","league_home_win","league_draw",
    "home_gf6","home_ga6","home_ppg6","home_n6",
    "away_gf6","away_ga6","away_ppg6","away_n6",
    "home_gf3","home_ga3","away_gf3","away_ga3",
    "home_rest20","away_rest20","home_home_gf6","away_away_gf6",
]
CATEGORIES=["home","draw","away"]
UTC=dt.timezone.utc

def parse_time(s):
    return dt.datetime.fromisoformat(s.replace("Z","+00:00")).astimezone(UTC)

def valid(g):
    return (g.get("state")=="post"
        and type(g.get("home",{}).get("score")) is int
        and type(g.get("away",{}).get("score")) is int
        and 0<=g["home"]["score"]<=25 and 0<=g["away"]["score"]<=25)

def months_ending(now,months=25):
    serial=now.year*12+now.month-1
    for i in range(months-1,-1,-1):
        n=serial-i
        y,m=divmod(n,12)
        yield f"{y:04d}{m+1:02d}"

def read_archive():
    try:
        a=json.loads(ARCHIVE.read_text())
        return a if a.get("schema")=="taj-history-v1" else {"schema":"taj-history-v1","months":{}}
    except (OSError,ValueError):
        return {"schema":"taj-history-v1","months":{}}

def collect(now=None,fetcher=fetch_json):
    now=now or dt.datetime.now(UTC)
    archive=read_archive()
    existing=archive.get("months",{})
    next_months={}
    fetched,failed=0,0
    for league in LEAGUES:
        entries={}
        for period in months_ending(now):
            key=league+":"+period
            previous=existing.get(key)
            # Immutable old scoreboards should be retrieved only once.
            recent=period>=f"{(now-dt.timedelta(days=67)):%Y%m}"
            if previous is not None and not recent:
                next_months[key]=previous
                continue
            url=(f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard"
                 f"?dates={period}&limit=500")
            try:
                body=fetcher(url)
                if not isinstance(body.get("events"),list):
                    raise ValueError("Expected events list")
                games=[normalize(raw,league) for raw in body["events"]]
                cleaned=[x for x in games if x and x["id"] and valid(x)]
                next_months[key]=cleaned
                fetched+=1
            except (OSError,ValueError,RuntimeError) as e:
                failed+=1
                if previous is not None:
                    next_months[key]=previous
                else:
                    print(f"WARNING {key}: {str(e)[:140]}",flush=True)
        print(f"{league}: {sum(len(next_months.get(league+':'+m,[])) for m in months_ending(now))} results",flush=True)
    archive={"schema":"taj-history-v1","updated_at":now.isoformat(),"months":next_months,
             "retrievals":fetched,"fetch_errors":failed}
    all_games={}
    for batch in next_months.values():
        for g in batch:
            if valid(g):all_games[g["league"]+":"+g["id"]]=g
    games=sorted(all_games.values(),key=lambda g:(g["kickoff"],g["id"]))
    if len(games)<650:
        raise RuntimeError(f"Only {len(games)} historical results; insufficient to train model")
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    tmp=ARCHIVE.with_suffix(".tmp")
    tmp.write_text(json.dumps(archive,ensure_ascii=False,separators=(",",":")))
    tmp.replace(ARCHIVE)
    print(f"Historical archive: {len(games)} completed, {fetched} refreshed monthly sources, {failed} warnings",flush=True)
    return games

def average(seq,default):
    return sum(seq)/len(seq) if seq else default

def form(team,history,at):
    recent=[g for g in history if g["home"]["id"]==team or g["away"]["id"]==team][-6:][::-1]
    values=[]
    for g in recent:
        home=g["home"]["id"]==team
        gf=g["home"]["score"] if home else g["away"]["score"]
        ga=g["away"]["score"] if home else g["home"]["score"]
        values.append((gf,ga,3 if gf>ga else 1 if gf==ga else 0,home,parse_time(g["kickoff"])))
    if not values:return None
    gf=average([x[0] for x in values],1.4)
    ga=average([x[1] for x in values],1.4)
    venue=[x[0] for x in values if x[3] == (at["venue"]=="home")]
    last=values[:3]
    rest=min(20,max(0,(parse_time(at["kickoff"])-values[0][4]).total_seconds()/86400))/20 if isinstance(at,dict) else 0
    return [gf,ga,average([x[2] for x in values],1.3),len(values)/6,
            average([x[0] for x in last],gf),average([x[1] for x in last],ga),
            rest,average(venue,gf)]

def extract(match,prior):
    prior_league=[g for g in prior if g["league"]==match["league"]]
    league=prior_league[-80:]
    if len(league)<16:return None
    baseh=average([g["home"]["score"] for g in league],1.4)
    basea=average([g["away"]["score"] for g in league],1.1)
    home_win=average([1 if g["home"]["score"]>g["away"]["score"] else 0 for g in league],.45)
    draw=average([1 if g["home"]["score"]==g["away"]["score"] else 0 for g in league],.27)
    h=form(match["home"]["id"],prior_league,{**match,"venue":"home"})
    a=form(match["away"]["id"],prior_league,{**match,"venue":"away"})
    if h is None or a is None or h[3]<.5 or a[3]<.5:return None
    return [baseh,basea,home_win,draw,
        h[0],h[1],h[2],h[3],a[0],a[1],a[2],a[3],
        h[4],h[5],a[4],a[5],h[6],a[6],h[7],a[7]]

def make_examples(games):
    # Games at identical kickoff timestamps are one temporal batch.
    # Otherwise a finished result in one 15:00 fixture could leak into
    # features of another 15:00 fixture during retrospective training.
    by_league={k:[] for k in LEAGUES}
    x,y,dates,identifiers=[],[],[],[]
    from itertools import groupby
    for stamp, group in groupby(games,key=lambda g:g["kickoff"]):
        simultaneous=list(group)
        pending=[]
        for g in simultaneous:
            hist=by_league[g["league"]]
            features=extract(g,hist)
            if features:
                home,away=g["home"]["score"],g["away"]["score"]
                target=0 if home>away else 1 if home==away else 2
                x.append(features);y.append(target);dates.append(g["kickoff"])
                identifiers.append(g["league"]+":"+g["id"])
            pending.append(g)
        for g in pending:by_league[g["league"]].append(g)
    if len(x)<430:raise RuntimeError(f"Insufficient chronology-safe examples: {len(x)}")
    return np.array(x,dtype="float32"),np.array(y,dtype="int32"),dates,identifiers

def metrics(prob,labels):
    p=np.clip(prob[np.arange(len(labels)),labels],1e-7,1)
    log_loss=float(-np.log(p).mean())
    onehot=np.eye(3)[labels]
    brier=float(((prob-onehot)**2).sum(axis=1).mean())
    accuracy=float((prob.argmax(axis=1)==labels).mean())
    return {"n":int(len(labels)),"log_loss":round(log_loss,5),
            "brier":round(brier,5),"accuracy":round(accuracy,5)}

def calibrate(probs,labels,temp):
    p=np.maximum(probs,1e-6)**(1/float(temp))
    return p/p.sum(axis=1,keepdims=True)

def poisson_baseline(feature_matrix):
    """Chronology-safe version of the existing site Poisson for fair comparison."""
    output=[]
    for x in feature_matrix:
        bh,ba=float(x[0]),float(x[1])
        base=max(.45,(bh+ba)/2)
        nh,na=float(x[7])*6,float(x[11])*6
        shrink=lambda mean,n:(n*float(mean)+6*base)/(n+6)
        lh=min(3.7,max(.25,(shrink(x[4],nh)*shrink(x[9],na)/base)*(bh/base)))
        la=min(3.7,max(.25,(shrink(x[8],na)*shrink(x[5],nh)/base)*(ba/base)))
        home=draw=away=total=0.
        ph=math.exp(-lh)
        for h in range(11):
            if h:ph*=lh/h
            pa=math.exp(-la)
            for a in range(11):
                if a:pa*=la/a
                p=ph*pa
                total+=p
                if h>a:home+=p
                elif h==a:draw+=p
                else:away+=p
        output.append([home/total,draw/total,away/total])
    return np.array(output,dtype="float64")

def train(games):
    import tensorflow as tf

    seed=20261010
    random.seed(seed);np.random.seed(seed);tf.random.set_seed(seed)
    x,y,dates,identifiers=make_examples(games)
    n=len(x)
    ntrain=int(n*.70);nvalid=int(n*.85)
    if min(ntrain,nvalid-ntrain,n-nvalid)<50:raise RuntimeError("Split too small")
    mean=x[:ntrain].mean(axis=0)
    std=np.maximum(x[:ntrain].std(axis=0),1e-5)
    xz=np.clip((x-mean)/std,-5,5).astype("float32")
    tf.keras.backend.clear_session()
    model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),),dtype=tf.float32),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(3,activation="softmax"),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.002),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    cb=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)
    model.fit(xz[:ntrain],y[:ntrain],validation_data=(xz[ntrain:nvalid],y[ntrain:nvalid]),
              epochs=70,batch_size=48,verbose=0,callbacks=[cb])
    validation=model.predict(xz[ntrain:nvalid],verbose=0)
    temperatures=np.linspace(.7,2.5,37)
    scores=[metrics(calibrate(validation,y[ntrain:nvalid],t),y[ntrain:nvalid])["log_loss"] for t in temperatures]
    temperature=float(temperatures[int(np.argmin(scores))])
    test_raw=model.predict(xz[nvalid:],verbose=0)
    test_pred=calibrate(test_raw, y[nvalid:],temperature)
    baseline_probs=np.bincount(y[:ntrain],minlength=3)/len(y[:ntrain])
    baseline=np.broadcast_to(baseline_probs,(len(y)-nvalid,3))
    result=metrics(test_pred,y[nvalid:])
    baseline_result=metrics(baseline,y[nvalid:])
    poisson_result=metrics(poisson_baseline(x[nvalid:]),y[nvalid:])
    print("HELD-OUT TEST / Poisson:",poisson_result,flush=True)
    print("HELD-OUT TEST / model:",result,flush=True)
    print("HELD-OUT TEST / train-freq baseline:",baseline_result,flush=True)
    # Avoid a dynamic [-1,20] batch dimension in the published TFLite
    # artifact. Rebuild the already-trained network with a fixed batch=1
    # Keras input and COPY its learned weights. Converting this Keras model
    # freezes the variables; converting a raw tf.function instead produced
    # READ_VARIABLE operator failure in TensorFlow Lite (real CI finding).
    fixed_model=tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURES),),batch_size=1,dtype=tf.float32),
        tf.keras.layers.Dense(32,activation="relu"),
        tf.keras.layers.Dense(16,activation="relu"),
        tf.keras.layers.Dense(3,activation="softmax"),
    ])
    fixed_model.set_weights(model.get_weights())
    converter=tf.lite.TFLiteConverter.from_keras_model(fixed_model)
    compiled=converter.convert()
    # Sanity check TFLite output against TensorFlow before publication.
    runner=tf.lite.Interpreter(model_content=compiled)
    runner.allocate_tensors()
    i=runner.get_input_details()[0];o=runner.get_output_details()[0]
    expected_shape=[1,len(FEATURES)]
    if i["shape_signature"].tolist()!=expected_shape:
        raise RuntimeError("Browser incompatible dynamic shape: "+
            str(i["shape_signature"].tolist()))
    if o["shape_signature"].tolist()!=[1,3]:
        raise RuntimeError("Browser incompatible output shape: "+
            str(o["shape_signature"].tolist()))
    diffs=[]
    for row,reference in zip(xz[nvalid:nvalid+12],test_raw[:12]):
        runner.set_tensor(i["index"],row.reshape(1,-1))
        runner.invoke()
        diffs.append(float(np.max(np.abs(runner.get_tensor(o["index"])[0]-reference))))
    max_delta=max(diffs)
    if max_delta>3e-4:raise RuntimeError(f"TFLite/TF numerical mismatch {max_delta}")
    MODELS.mkdir(parents=True,exist_ok=True)
    (MODELS/"football_1x2.tflite").write_bytes(compiled)
    meta={
        "schema":"taj-litert-1x2-v1","model":"historical-match-mlp-v1",
        "created_at":dt.datetime.now(UTC).isoformat(),
        "input_features":FEATURES,"classes":CATEGORIES,
        "input_shape":[1,len(FEATURES)],"output_shape":[1,3],
        "mean":[round(float(v),8) for v in mean],
        "std":[round(float(v),8) for v in std],
        "temperature":round(temperature,4),
        "train_n":ntrain,"validation_n":nvalid-ntrain,
        "holdout":result,"frequency_baseline_holdout":baseline_result,
        "poisson_baseline_holdout":poisson_result,
        "train_until":dates[ntrain-1],"validation_until":dates[nvalid-1],
        "test_from":dates[nvalid],
        "data_note":"Historical ESPN results, no live lineup, tracking, xG, tactical labels",
        "model_note":"Experimental, unverified probability calibration; avoid betting/guarantees",
        "max_tf_litert_difference":max_delta,
        "train_leagues":list(LEAGUES),
    }
    (MODELS/"model-card.json").write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    print(f"Exported TFLite {len(compiled)} bytes; {len(FEATURES)} input features. max delta={max_delta}",flush=True)
    return meta

def main():
    games=collect()
    card=train(games)
    print(f"TRAIN COMPLETE: {card['train_n']} train, {card['validation_n']} valid, {card['holdout']['n']} test",flush=True)

if __name__=="__main__":main()

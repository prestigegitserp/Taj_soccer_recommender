"""Optional pre-match baseline using user-provided past match results.

Deliberately separate from tactical model: no claimed tactical causality and
no live lineups/injuries. Bayesian-shrunk home/away goal rates + independent
Poisson; validated with chronological walk-forward Brier score.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import exp,factorial
import numpy as np
import pandas as pd

REQUIRED={"date","home_team","away_team","home_goals","away_goals"}

def prepare_matches(df:pd.DataFrame):
    missing=REQUIRED-set(df.columns)
    if missing:raise ValueError("Missing columns: "+", ".join(sorted(missing)))
    x=df.copy()
    x["date"]=pd.to_datetime(x.date,utc=True,errors="coerce")
    for c in ("home_goals","away_goals"):x[c]=pd.to_numeric(x[c],errors="coerce")
    x=x.dropna(subset=["date","home_team","away_team","home_goals","away_goals"]).copy()
    if (x[["home_goals","away_goals"]]<0).any().any():
        raise ValueError("Negative goals invalid")
    if not ((x[["home_goals","away_goals"]] %1)==0).all().all():
        raise ValueError("Goal counts must be integers")
    return x.sort_values("date").reset_index(drop=True)

def poisson_probs(mu,max_goals=10):
    if mu<=0:raise ValueError("Goals expectation must be positive")
    return np.array([exp(-mu)*mu**i/factorial(i) for i in range(max_goals+1)])

def predict(history:pd.DataFrame,home:str,away:str,as_of:str,
            *,prior_games=8,min_games=30,min_team_games=3,max_goals=10):
    if home==away:raise ValueError("Two distinct teams required")
    cutoff=pd.to_datetime(as_of,utc=True)
    data=prepare_matches(history)
    train=data.loc[data.date<cutoff].copy()
    if len(train)<min_games:raise ValueError(f"Only {len(train)} historical games before {cutoff.date()}; require {min_games}")
    home_data=train[train.home_team.astype(str)==home]
    away_data=train[train.away_team.astype(str)==away]
    if len(home_data)<min_team_games or len(away_data)<min_team_games:
        raise ValueError("Not enough historical home/away samples for these teams")
    avg_h=max(.2,float(train.home_goals.mean()))
    avg_a=max(.2,float(train.away_goals.mean()))
    def shrunk(series,prior):
        return (float(series.sum())+prior_games*prior)/(len(series)+prior_games)
    home_attack=shrunk(home_data.home_goals,avg_h)/avg_h
    home_defence=shrunk(home_data.away_goals,avg_a)/avg_a
    away_attack=shrunk(away_data.away_goals,avg_a)/avg_a
    away_defence=shrunk(away_data.home_goals,avg_h)/avg_h
    mu_h=float(np.clip(avg_h*home_attack*away_defence,.15,5))
    mu_a=float(np.clip(avg_a*away_attack*home_defence,.15,5))
    p_h=poisson_probs(mu_h,max_goals)
    p_a=poisson_probs(mu_a,max_goals)
    scores=np.outer(p_h,p_a)
    scores/=scores.sum() # condition on goal counts <= max_goals
    p_home=float(np.tril(scores,-1).sum())
    p_away=float(np.triu(scores,1).sum())
    p_draw=float(np.trace(scores))
    idx=np.unravel_index(np.argmax(scores),scores.shape)
    return {
      "home":home,"away":away,"as_of":str(cutoff.date()),"trained_matches":int(len(train)),
      "home_samples":int(len(home_data)),"away_samples":int(len(away_data)),
      "expected_home_goals":round(mu_h,3),"expected_away_goals":round(mu_a,3),
      "probabilities":{"home_win":round(p_home,5),"draw":round(p_draw,5),"away_win":round(p_away,5)},
      "most_likely_score":{"home_goals":int(idx[0]),"away_goals":int(idx[1]),
                           "probability":round(float(scores[idx]),5)},
      "model":"home/away shrunk goal rates + independent Poisson v1",
      "limitations":"No injury/lineup/tactical event features, not calibrated; train uses only matches strictly before as_of."
    }

def walk_forward_brier(matches:pd.DataFrame,min_games=50,min_team_games=3,prior_games=8):
    data=prepare_matches(matches)
    briers=[];examples=[]
    for idx in range(min_games,len(data)):
        row=data.iloc[idx]
        try:
            pred=predict(data.iloc[:idx],str(row.home_team),str(row.away_team),
                         str(row.date),prior_games=prior_games,min_games=min_games,
                         min_team_games=min_team_games)
        except ValueError:continue
        p=pred["probabilities"]
        actual=np.array([int(row.home_goals>row.away_goals),
                         int(row.home_goals==row.away_goals),
                         int(row.home_goals<row.away_goals)])
        predicted=np.array([p["home_win"],p["draw"],p["away_win"]])
        briers.append(float(np.sum((predicted-actual)**2)))
        examples.append({"date":str(row.date.date()),"home":row.home_team,"away":row.away_team,
                         "brier":round(briers[-1],5)})
    return {"n_predictions":len(briers),"mean_multiclass_brier":round(float(np.mean(briers)),5) if briers else None,
            "examples":examples[-10:],
            "warning":"Temporal hold-out; no claims about performance until enough matches across seasons."}

import numpy as np
import pandas as pd
import pytest
from taj.prediction import predict,walk_forward_brier,prepare_matches

def example():
    dates=pd.date_range("2024-01-01",periods=100,freq="D")
    return pd.DataFrame([{"date":str(t.date()),
      "home_team":"Alpha" if i%2==0 else "Beta",
      "away_team":"Beta" if i%2==0 else "Alpha",
      "home_goals":i%4,"away_goals":i%3} for i,t in enumerate(dates)])

def test_temporal_predict_and_probabilities():
    h=example()
    p=predict(h,"Alpha","Beta","2024-04-11")
    assert p["trained_matches"]==100
    assert abs(sum(p["probabilities"].values())-1)<.0001
    assert p["expected_home_goals"]>0 and p["expected_away_goals"]>0

def test_future_rows_never_leak():
    h=example();p=predict(h,"Alpha","Beta","2024-02-20")
    copy=h.copy()
    copy.loc[copy.index>=50,"home_goals"]=100
    p2=predict(copy,"Alpha","Beta","2024-02-20")
    assert p["probabilities"]==p2["probabilities"]

def test_walk_forward():
    report=walk_forward_brier(example(),min_games=40)
    assert report["n_predictions"]>0
    assert 0 <= report["mean_multiclass_brier"]<=2

def test_missing_team_rejected():
    with pytest.raises(ValueError):predict(example(),"Unknown","Beta","2024-05-01")

"""Independent standard-library audit of REAL unseen football match predictions."""
from collections import defaultdict
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/"docs/data/backtest.json"
DETAIL=ROOT/"training/backtest-predictions.json"
CLASSES=("home","draw","away")

def datasets():
    return (json.loads(REPORT.read_text(encoding="utf-8")),
            json.loads(DETAIL.read_text(encoding="utf-8"))["matches"])

def recompute(rows,key):
    n=len(rows)
    accuracy=sum(max(range(3),key=lambda i:r["probabilities"][key][i])==r["outcome"] for r in rows)/n
    brier=sum(sum((r["probabilities"][key][i]-(i==r["outcome"]))**2 for i in range(3)) for r in rows)/n
    loss=-sum(math.log(max(r["probabilities"][key][r["outcome"]],1e-7)) for r in rows)/n
    return accuracy,brier,loss

def test_all_original_outcomes_and_no_duplicate_or_future_leakage():
    report,rows=datasets()
    assert report["schema"]=="taj-walkforward-v1"
    assert len(rows)==report["oos_unique_fixtures"]>1500
    assert len({r["id"] for r in rows})==len(rows)
    assert len(report["folds"])==3
    previous_last=None
    for fold in report["folds"]:
        assert fold["train_through"]<fold["validation_from"]
        assert fold["validation_through"]<fold["test_from"]
        assert fold["test_from"]<=fold["test_through"]
        if previous_last is not None:assert previous_last<fold["test_from"]
        previous_last=fold["test_through"]
        block=[r for r in rows if r["fold"]==fold["fold"]]
        assert len(block)==fold["test_n"]
        assert all(fold["test_from"]<=r["date"]<=fold["test_through"] for r in block)
    for row in rows:
        goals=row["score"]
        real=0 if goals["home_goals"]>goals["away_goals"] else 1 if goals["home_goals"]==goals["away_goals"] else 2
        assert row["outcome"]==real
        assert isinstance(row["home"]["name"],str) and row["home"]["id"]
        assert isinstance(row["away"]["name"],str) and row["away"]["id"]
        for probs in row["probabilities"].values():
            assert len(probs)==3
            assert all(0<=p<=1 and math.isfinite(p) for p in probs)
            assert abs(sum(probs)-1)<1e-5

def test_report_metrics_match_individual_real_results():
    report,rows=datasets()
    for key,summary in report["overall"].items():
        accuracy,brier,loss=recompute(rows,key)
        assert abs(accuracy-summary["accuracy"])<.00002,(key,accuracy)
        assert abs(brier-summary["brier"])<.00002,(key,brier)
        assert abs(loss-summary["log_loss"])<.00002,(key,loss)
        assert summary["correct"]==round(accuracy*len(rows))
    for fold in report["folds"]:
        block=[r for r in rows if r["fold"]==fold["fold"]]
        for key,s in fold["models"].items():
            acc,brier,loss=recompute(block,key)
            assert abs(acc-s["accuracy"])<.00002
            assert abs(brier-s["brier"])<.00002
            assert abs(loss-s["log_loss"])<.00002

def test_league_coverage_and_ensemble_weight_fixed_within_each_fold():
    report,rows=datasets()
    leagues=defaultdict(list)
    for row in rows:leagues[row["league"]].append(row)
    assert len(leagues)==6
    assert sum(entry["n"] for entry in report["per_league"].values())==len(rows)
    for league,items in leagues.items():
        score=report["per_league"][league]
        assert score["n"]==len(items)
        for name,m in score["models"].items():
            a,b,l=recompute(items,name)
            assert abs(b-m["brier"])<.00002
    for fold in report["folds"]:
        w=fold["validated_graph_ensemble_weight"]
        assert w in (0,.25,.5,.75,1)
        for r in (r for r in rows if r["fold"]==fold["fold"]):
            graph=r["probabilities"]["graph_42"]
            poisson=r["probabilities"]["poisson"]
            ensemble=r["probabilities"]["graph_ensemble"]
            for i in range(3):assert abs(ensemble[i]-(w*graph[i]+(1-w)*poisson[i]))<1e-6

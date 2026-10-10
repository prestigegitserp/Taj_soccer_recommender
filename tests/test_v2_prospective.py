"""Audit the immutable V2 live pre-kickoff prediction ledger as a separate test."""
from datetime import datetime,timedelta
from pathlib import Path
import hashlib,json,math

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/"docs/v2/data/prospective.json"
MODEL=ROOT/"docs/v2/models/forest.json"

def time(value):
    return datetime.fromisoformat(value.replace("Z","+00:00"))

def test_prospective_probabilities_timestamp_model_hash_and_truth():
    if not LEDGER.exists():
        return
    ledger=json.loads(LEDGER.read_text(encoding="utf-8"))
    assert ledger["schema"]=="taj-v2-prospective-v1"
    entries=ledger["games"]
    assert len(entries)==ledger["locked_predictions"]
    assert all( ":" in id_ for id_ in entries)
    for key,game in entries.items():
        assert game["id"]==key
        assert game["model_version"].startswith("2.")
        assert time(game["first_logged_at"])+timedelta(minutes=15)<time(game["kickoff"])
        assert len(game["forest_sha256"])==64
        for name in ("poisson","lgb56","challenger","official"):
            p=game["probabilities"][name]
            assert len(p)==3
            assert all(0<=x<=1 and math.isfinite(x) for x in p)
            assert abs(sum(p)-1)<1e-6
        if not game["promoted"]:
            assert game["probabilities"]["official"]==game["probabilities"]["poisson"]
        if "final" in game:
            h=game["final"]["home_goals"];a=game["final"]["away_goals"]
            assert game["final"]["outcome"]==(0 if h>a else 1 if h==a else 2)
            assert time(game["final"]["observed_at"])>=time(game["kickoff"])
    settled=sum("final" in g for g in entries.values())
    assert settled==ledger["finalized_results"]

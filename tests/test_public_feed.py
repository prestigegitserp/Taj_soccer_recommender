import datetime as dt
from scripts.build_public_feed import normalize,build

def event(i="1",state="pre",date="2026-10-11T20:00:00Z"):
    return {"id":i,"date":date,"status":{"type":{"state":state}},
       "competitions":[{"competitors":[
          {"homeAway":"away","team":{"id":"20","displayName":"Away FC"},"score":None},
          {"homeAway":"home","team":{"id":"10","displayName":"Home FC"},"score":None}]}]}

def test_home_away_never_assumed_positional():
    r=normalize(event(),"eng.1")
    assert r["home"]["name"]=="Home FC" and r["away"]["id"]=="20"
    assert r["kickoff"].startswith("2026-10-11")

def test_unknown_missing_source_doesnt_invent_game():
    assert normalize({"id":"x","date":"2026-10-11T00:00:00Z"},"eng.1") is None

def test_valid_source_feeds():
    x=build(now=dt.datetime(2026,10,10,tzinfo=dt.timezone.utc),
       fetcher=lambda _:{"events":[event()]},
       previous={})
    assert len(x["leagues"])==6
    assert all(len(v["events"])==1 for v in x["leagues"].values())

def test_failed_source_preserves_previous_without_fake_refresh():
    earlier={"leagues":{"eng.1":{"name":"Premier League","updated_at":"2026-10-05T00:00:00Z","events":[normalize(event(),"eng.1")]}}}
    x=build(now=dt.datetime(2026,10,10,tzinfo=dt.timezone.utc),
       fetcher=lambda _: (_ for _ in ()).throw(RuntimeError("network error")),
       previous=earlier)
    assert x["live_source_count"]==0
    assert x["leagues"]["eng.1"]["updated_at"]=="2026-10-05T00:00:00Z"
    assert "esp.1" in x["errors"]

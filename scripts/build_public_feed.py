#!/usr/bin/env python3
"""Fetch verified public soccer fixtures/results for a browser-only GitHub Pages app.

The GitHub runner only collects and normalizes public scoreboard JSON. ALL matchup
feature engineering, probability modeling, selection, and UI rendering happen in
the visitor's browser. No server, paid API, or Google Colab is required.

ESPN site APIs are unofficial and may change. On partial failures preserve the
previous snapshot per competition; never synthesize events or results.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request

LEAGUES = {
    "eng.1":"Premier League",
    "esp.1":"La Liga",
    "ger.1":"Bundesliga",
    "ita.1":"Serie A",
    "fra.1":"Ligue 1",
    "uefa.champions":"Champions League",
}
OUTPUT=Path("docs/data/feed.json")
UTC=dt.timezone.utc
def iso_now():return dt.datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00","Z")
def to_dt(s):
    if not s:return None
    try:return dt.datetime.fromisoformat(str(s).replace("Z","+00:00")).astimezone(UTC)
    except (ValueError,TypeError):return None
def score_value(v):
    if isinstance(v,dict):v=v.get("value",v.get("displayValue"))
    try:return int(v) if v is not None and str(v)!="" else None
    except (ValueError,TypeError):return None

def normalize(raw,league):
    """Fail closed on ambiguous dates and teams."""
    comp=(raw.get("competitions") or [{}])[0]
    sides={}
    for item in comp.get("competitors",[]):
        if item.get("homeAway") in ("home","away"):sides[item["homeAway"]]=item
    if set(sides)!={"home","away"}:return None
    kickoff=to_dt(raw.get("date") or comp.get("date"))
    if not kickoff:return None
    state=(raw.get("status") or comp.get("status") or {}).get("type",{}).get("state","pre")
    if state not in ("pre","in","post"):state="pre"
    def team(item):
        t=item.get("team") or {}
        logo=t.get("logo")
        if not logo and t.get("logos"):logo=t["logos"][0].get("href")
        return {
            "id":str(t.get("id","")),"name":str(t.get("displayName") or t.get("name") or "Unknown"),
            "short":str(t.get("shortDisplayName") or t.get("abbreviation") or t.get("name") or "Unknown"),
            "logo":logo if isinstance(logo,str) and logo.startswith("https://") else None,
            "score":score_value(item.get("score")),
        }
    h,a=team(sides["home"]),team(sides["away"])
    if not h["id"] or not a["id"] or h["id"]==a["id"]:return None
    st=raw.get("status",{}) or comp.get("status",{}) or {}
    return {
        "id":str(raw.get("id") or comp.get("id") or ""),
        "league":league,"kickoff":kickoff.isoformat().replace("+00:00","Z"),
        "state":state,"detail":str((st.get("type") or {}).get("shortDetail") or ""),
        "home":h,"away":a,
        "venue":str(((comp.get("venue") or {}).get("fullName") or ""))[:160],
    }

def fetch_json(url,attempts=3):
    last=None
    for attempt in range(attempts):
        try:
            req=urllib.request.Request(url,headers={
                "User-Agent":"python-urllib/3.11",
                "Accept":"application/json",
                "Accept-Encoding":"identity",
            })
            with urllib.request.urlopen(req,timeout=20) as resp:
                if resp.status!=200:raise RuntimeError(f"HTTP {resp.status}")
                return json.load(resp)
        except (OSError,ValueError,RuntimeError,urllib.error.HTTPError) as exc:
            last=exc
            if attempt+1<attempts:time.sleep(1.3*(attempt+1))
    raise RuntimeError(f"ESPN fetch failed: {str(last)[:130]}")

def fetch_league(league,now,fetcher=fetch_json):
    # One future window and one historical form window. Select matches in the
    # BROWSER from a merged, never fabricated set of competitions/periods.
    windows=[(now-dt.timedelta(days=55),now+dt.timedelta(days=1)),
             (now-dt.timedelta(days=1),now+dt.timedelta(days=47))]
    found={}
    for lo,hi in windows:
        interval=f"{lo:%Y%m%d}-{hi:%Y%m%d}"
        url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/scoreboard?"+urllib.parse.urlencode({"dates":interval,"limit":500})
        data=fetcher(url)
        if not isinstance(data,dict) or not isinstance(data.get("events"),list):
            raise ValueError("ESPN scoreboard format missing events")
        for ev in data["events"]:
            if not isinstance(ev,dict):continue
            game=normalize(ev,league)
            if game and game["id"]:
                old=found.get(game["id"])
                # Prefer updated in-play/final states to older prerelease snapshots.
                if old is None or (old["state"]=="pre" and game["state"]!="pre"):
                    found[game["id"]]=game
    return sorted(found.values(),key=lambda g:(g["kickoff"],g["id"]))

def existing_snapshot(path=OUTPUT):
    try:return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError,ValueError):return {}

def build(now=None,fetcher=fetch_json,previous=None):
    now=now or dt.datetime.now(UTC)
    previous=previous if previous is not None else existing_snapshot()
    prior=(previous.get("leagues") or {}) if isinstance(previous,dict) else {}
    packet={
        "schema":"taj-fixtures-v1",
        "generated_at":iso_now(),
        "source":"ESPN public scoreboard endpoints (unofficial; not licensed tracking data)",
        "freshness":"Scheduled refresh approximately every 30 minutes; actual latency may be longer",
        "leagues":{},"errors":{},
    }
    ok=0
    for league,label in LEAGUES.items():
        try:
            events=fetch_league(league,now,fetcher)
            # A genuinely empty response can occur in off-season: preserve zero.
            packet["leagues"][league]={"name":label,"updated_at":iso_now(),"events":events}
            ok+=1
            print(f"{league:17} {len(events):3} events")
        except (ValueError,RuntimeError,OSError) as exc:
            packet["errors"][league]=str(exc)[:200]
            if league in prior:
                packet["leagues"][league]=prior[league]
                print(f"{league:17} STALE ({exc})")
            else:
                print(f"{league:17} FAILED ({exc})")
    if ok==0 and not prior:
        raise RuntimeError("No live source succeeded, refusing to publish empty fabricated feed")
    packet["live_source_count"]=ok
    return packet

def main():
    packet=build()
    OUTPUT.parent.mkdir(parents=True,exist_ok=True)
    tmp=OUTPUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(packet,ensure_ascii=False,separators=(",",":")),encoding="utf-8")
    tmp.replace(OUTPUT)
    total=sum(len(v["events"]) for v in packet["leagues"].values())
    print(f"Saved {OUTPUT}: {total} real events from {len(packet['leagues'])} competitions. New sources {packet['live_source_count']}.")

if __name__=="__main__":main()

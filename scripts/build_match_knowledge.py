#!/usr/bin/env python3
"""Evidence-only optional match intelligence for five upcoming fixtures.

Fetch public, unofficial ESPN team squad rosters and recent match boxscores.
This is not an official lineup feed. A roster is NOT a starting eleven.
Never fill missing possession/shots/player identities with estimated values.
Results are displayed with coverage and observation source in the browser.

Cache previous successful requests in GitHub repo, so recurring Actions are
incremental and do not repeatedly poll old full-match summaries.
"""
from __future__ import annotations
from datetime import datetime,timezone,timedelta
import json
from pathlib import Path
import re
import urllib.parse
from build_public_feed import fetch_json
BASE=Path(__file__).resolve().parents[1]
FEED=BASE/"docs/data/feed.json"
OUT=BASE/"docs/data/match-knowledge.json"
CACHE=BASE/"training/knowledge-cache.json"
def now_utc():return datetime.now(timezone.utc)
def parsed(s):
    try:return datetime.fromisoformat(str(s).replace("Z","+00:00"))
    except (ValueError,TypeError):return None
def read(path):
    try:return json.loads(path.read_text(encoding="utf-8"))
    except (OSError,ValueError):return {}
def safe_number(value):
    if isinstance(value,dict):value=value.get("value",value.get("displayValue"))
    if isinstance(value,bool) or value is None:return None
    text=str(value).replace(",","").replace("%","").strip()
    match=re.fullmatch(r"-?\d+(?:\.\d+)?",text)
    if not match:return None
    n=float(text)
    return n if -100<=n<=100000 else None
STAT_NAMES={
  "possession":["possessionpct","possession","possessionpercentage","ballpossession"],
  "shots":["totalshots","shots","shotattempts","totalshotattempts"],
  "shots_on_target":["shotsontarget","ontarget","shotson"],
  "corners":["cornerkicks","corners"],
  "fouls":["fouls","totalfouls"],
  "yellow_cards":["yellowcards","yellowcard"],
}
def normalize_stats(items):
    raw={}
    for item in items or []:
        if not isinstance(item,dict):continue
        key=re.sub("[^a-z0-9]","",str(item.get("name") or item.get("abbreviation") or item.get("label") or "").lower())
        value=safe_number(item.get("value",item.get("displayValue")))
        if key and value is not None:raw[key]=value
    result={}
    for name,aliases in STAT_NAMES.items():
        for alias in aliases:
            if alias in raw:result[name]=raw[alias];break
    return result
def parse_summary(summary):
    box=summary.get("boxscore") or {}
    teams=box.get("teams") or []
    result={}
    for team in teams:
        if not isinstance(team,dict):continue
        t=team.get("team") or {}
        id_=str(t.get("id") or team.get("id") or "")
        if id_:result[id_]=normalize_stats(team.get("statistics"))
    return result
def parse_roster(body):
    groups=body.get("athletes") or []
    players=[]
    if not isinstance(groups,list):return []
    for group in groups:
        if not isinstance(group,dict):continue
        candidates=group.get("items") or group.get("athletes")
        if candidates is None and ("id" in group or "displayName" in group):candidates=[group]
        if not isinstance(candidates,list):continue
        for person in candidates:
            if not isinstance(person,dict):continue
            athlete=person.get("athlete") or person
            if not isinstance(athlete,dict):continue
            id_=str(athlete.get("id") or "")
            name=str(athlete.get("displayName") or athlete.get("fullName") or "").strip()
            if not id_ or not name:continue
            pos=athlete.get("position") or person.get("position") or {}
            if isinstance(pos,dict):position=str(pos.get("abbreviation") or pos.get("name") or "")
            else:position=str(pos)
            jersey=athlete.get("jersey") or person.get("jersey")
            players.append({"id":id_,"name":name[:90],"position":position[:32],
                "jersey":str(jersey)[:5] if jersey is not None else None})
    dedup={p["id"]:p for p in players}
    return sorted(dedup.values(),key=lambda x:(x["position"],x["name"]))[:40]
def select_matches(feed,now):
    games=[g for record in (feed.get("leagues") or {}).values()
      for g in record.get("events",[]) if g.get("state")=="pre"
      and parsed(g.get("kickoff")) and parsed(g["kickoff"])>=now-timedelta(minutes=2)]
    return sorted(games,key=lambda g:g["kickoff"])[:5]
def build(fetcher=fetch_json, now=None):
    now=now or now_utc()
    feed=read(FEED)
    if feed.get("schema")!="taj-fixtures-v1":raise RuntimeError("Missing valid scoreboards")
    selected=select_matches(feed,now)
    cache=read(CACHE)
    squads=cache.get("squads") or {}
    boxscores=cache.get("boxscores") or {}
    games=[g for l in feed["leagues"].values() for g in l.get("events",[])]
    fixtures={}
    observed=0;failures=[]
    for game in selected:
        league=game["league"];h=game["home"];a=game["away"]
        sides={}
        for team in (h,a):
            id_=str(team["id"]);key=league+":"+id_
            old=squads.get(key) or {}
            updated=parsed(old.get("updated_at"))
            if not updated or (now-updated).total_seconds()>8*3600:
                try:
                    url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/teams/{id_}/roster"
                    roster=parse_roster(fetcher(url))
                    if roster:
                        old={"updated_at":now.isoformat(),"players":roster,"source":"ESPN team roster; not a confirmed lineup"}
                        squads[key]=old;observed+=1
                except (OSError,RuntimeError,ValueError) as exc:
                    failures.append("roster "+key+": "+str(exc)[:75])
            sides[id_]={"roster":old.get("players") or [],
                "roster_updated_at":old.get("updated_at"),
                "roster_source":old.get("source") or "Not available"}
        for team in (h,a):
            id_=str(team["id"])
            completed=sorted((g for g in games if g["league"]==league
              and g.get("state")=="post" and g["home"].get("score") is not None
              and g["away"].get("score") is not None
              and g["kickoff"]<game["kickoff"]
              and (g["home"]["id"]==id_ or g["away"]["id"]==id_)),
              key=lambda g:g["kickoff"],reverse=True)[:4]
            observations={}
            for recent in completed:
                key=league+":"+recent["id"]
                old=boxscores.get(key)
                if old is None:
                    try:
                        url=f"https://site.api.espn.com/apis/site/v2/sports/soccer/{league}/summary?event="+urllib.parse.quote(str(recent["id"]))
                        old={"updated_at":now.isoformat(),"sides":parse_summary(fetcher(url))}
                        boxscores[key]=old;observed+=1
                    except (OSError,RuntimeError,ValueError) as exc:
                        failures.append("summary "+key+": "+str(exc)[:75])
                        continue
                own=(old.get("sides") or {}).get(id_) or {}
                if own:observations[key]=own
            means={}
            for stat in STAT_NAMES:
                values=[g[stat] for g in observations.values() if stat in g]
                if values:means[stat]={"value":round(sum(values)/len(values),2),"observations":len(values)}
            sides[id_]["recent_boxscore"]=means
            sides[id_]["games_checked"]=len(completed)
        fixtures[league+":"+game["id"]]={
          "league":league,"id":game["id"],"kickoff":game["kickoff"],
          "home_id":h["id"],"away_id":a["id"],
          "teams":sides,
        }
    newcache={"updated_at":now.isoformat(),"squads":squads,"boxscores":boxscores}
    CACHE.parent.mkdir(parents=True,exist_ok=True)
    CACHE.write_text(json.dumps(newcache,ensure_ascii=False,separators=(",",":")))
    out={"schema":"taj-knowledge-v1","generated_at":now.isoformat(),
      "sources":["ESPN public team rosters","ESPN public completed-match boxscores"],
      "lineup_quality":"No confirmed starting XI; listed players are registered squad only",
      "matches":fixtures,"warnings":failures[:12]}
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(out,ensure_ascii=False,separators=(",",":")))
    print("Verified match knowledge:",len(fixtures),"upcoming games",
       sum(len(t["roster"]) for g in fixtures.values() for t in g["teams"].values()),"squad-player records",
       "fresh public requests",observed,"source errors",len(failures),flush=True)
    print("Sample missing/stats coverage:",[(k,[(t["games_checked"],len(t["roster"]),
      list(t["recent_boxscore"])) for t in v["teams"].values()]) for k,v in list(fixtures.items())[:2]],flush=True)
    return out
if __name__=="__main__":build()

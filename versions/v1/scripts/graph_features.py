"""Temporal football knowledge graph features.

Nodes: verified team IDs. Edges: finished league matches with score + date.
Metrics: causal, rolling Elo opponent strength, 1-hop and 2-hop neighborhoods,
mutual opponents, direct rival meetings, recent opponent quality.
Every feature is computed only from games STRICTLY before target kickoff.
No player data is imputed.
"""
from __future__ import annotations
from collections import defaultdict
from math import pow

GRAPH_NAMES=[
 "graph_home_rating","graph_away_rating","graph_rating_gap",
 "graph_home_opp_rating","graph_away_opp_rating",
 "graph_home_quality_form","graph_away_quality_form",
 "graph_home_opp_goals_for","graph_away_opp_goals_for",
 "graph_home_opp_goals_against","graph_away_opp_goals_against",
 "graph_home_twohop","graph_away_twohop",
 "graph_common_opponent_ratio","graph_common_opponent_count",
 "graph_h2h_n","graph_h2h_home_points","graph_h2h_goal_diff",
 "graph_home_distinct_opponents","graph_away_distinct_opponents",
 "graph_home_recent_elo_trend","graph_away_recent_elo_trend",
]
def elo_chance(a,b,home_advantage=0):
    return 1/(1+10**((b-a-home_advantage)/400))
def mean(arr,default=0.):
    return sum(arr)/len(arr) if arr else default
def score_for(g,team):
    host=g["home"]["id"]==team
    own=g["home"]["score"] if host else g["away"]["score"]
    against=g["away"]["score"] if host else g["home"]["score"]
    opponent=g["away"]["id"] if host else g["home"]["id"]
    return own,against,opponent,host
def extract_graph(match,history):
    home=match["home"]["id"];away=match["away"]["id"]
    games=sorted(
      [g for g in history if g.get("league")==match.get("league") and
       g.get("state")=="post" and isinstance(g["home"].get("score"),int)
       and isinstance(g["away"].get("score"),int) and
       g["kickoff"]<match["kickoff"]],
      key=lambda g:(g["kickoff"],g["id"]))[-80:]
    if len(games)<16:return None
    rating=defaultdict(lambda:1500.)
    matches=defaultdict(list)
    rating_paths=defaultdict(list)
    i=0
    while i<len(games):
        stamp=games[i]["kickoff"];block=[]
        while i<len(games) and games[i]["kickoff"]==stamp:
            block.append(games[i]);i+=1
        pending=defaultdict(float)
        for g in block:
            h=g["home"]["id"];a=g["away"]["id"]
            rh=rating[h];ra=rating[a]
            expected=elo_chance(rh,ra,55.)
            hs=g["home"]["score"];as_=g["away"]["score"]
            actual=1. if hs>as_ else .5 if hs==as_ else 0.
            change=26*(actual-expected)
            pending[h]+=change;pending[a]-=change
            matches[h].append(g);matches[a].append(g)
        for team,change in pending.items():rating[team]+=change
        for team in pending:rating_paths[team].append(rating[team])
    # Require at least three observed games per team.
    if len(matches[home])<3 or len(matches[away])<3:return None
    def neighbor_profile(team):
        recent=matches[team][-6:]
        opponents=[]
        strength=[]
        quality=[]
        for g in recent:
            gf,ga,opponent,is_home=score_for(g,team)
            opponents.append(opponent)
            strength.append(rating[opponent])
            # Residual score result against opponent's strength: graph-derived
            # opponent-adjusted form, not a calibrated match win probability.
            expected=elo_chance(rating[team],rating[opponent],55 if is_home else -55)
            outcome=1 if gf>ga else .5 if gf==ga else 0
            quality.append(outcome-expected)
        opp_recent={}
        for other in set(opponents):
            vals=[score_for(g,other) for g in matches[other][-6:]]
            opp_recent[other]=(mean([v[0] for v in vals],1.4),
                              mean([v[1] for v in vals],1.4))
        weighted_for=mean([opp_recent[o][0] for o in opponents],1.4)
        weighted_against=mean([opp_recent[o][1] for o in opponents],1.4)
        twohop=mean([
            mean([rating[score_for(g,o)[2]] for g in matches[o][-6:]],1500)
            for o in opponents],1500)
        track=rating_paths[team]
        trend=rating[team]-(track[-6] if len(track)>=6 else 1500.)
        return dict(neighbors=set(opponents),opponent_rating=mean(strength,1500),
          quality=mean(quality,0),for_=weighted_for,against=weighted_against,
          twohop=twohop,degree=len({score_for(g,team)[2] for g in matches[team]}),
          trend=trend)
    h=neighbor_profile(home);a=neighbor_profile(away)
    common=h["neighbors"]&a["neighbors"]
    direct=[g for g in games if
       {g["home"]["id"],g["away"]["id"]}=={home,away}][-6:]
    h2h=[]
    for g in direct:
        gf,ga,_,_=score_for(g,home)
        h2h.append((gf,ga,1 if gf>ga else .5 if gf==ga else 0))
    out=[
        (rating[home]-1500)/250,
        (rating[away]-1500)/250,
        (rating[home]-rating[away])/400,
        (h["opponent_rating"]-1500)/250,
        (a["opponent_rating"]-1500)/250,
        h["quality"],a["quality"],
        h["for_"]/3,a["for_"]/3,
        h["against"]/3,a["against"]/3,
        (h["twohop"]-1500)/250,(a["twohop"]-1500)/250,
        len(common)/max(1,len(h["neighbors"]|a["neighbors"])),
        min(1.,len(common)/6),
        len(direct)/6,
        mean([x[2] for x in h2h],.5),
        mean([(x[0]-x[1]) for x in h2h],0)/3,
        min(1.,h["degree"]/16),min(1.,a["degree"]/16),
        h["trend"]/200,a["trend"]/200,
    ]
    assert len(out)==len(GRAPH_NAMES)
    return out

def graph_context(match,history):
    """Human-explainable verified edges for graph explorer."""
    cutoff=match["kickoff"];league=match["league"]
    matches=sorted([g for g in history if
      g.get("league")==league and g.get("state")=="post" and
      g["kickoff"]<cutoff and isinstance(g["home"].get("score"),int) and
      isinstance(g["away"].get("score"),int)],key=lambda x:x["kickoff"])
    h,a=match["home"]["id"],match["away"]["id"]
    result={}
    for team in (h,a):
        recent=[g for g in matches if team in (g["home"]["id"],g["away"]["id"])][-6:][::-1]
        result[team]=[{
          "id":g["id"],"opponent":score_for(g,team)[2],
          "opponent_name":g["away"]["name"] if g["home"]["id"]==team else g["home"]["name"],
          "gf":score_for(g,team)[0],"ga":score_for(g,team)[1],
          "kickoff":g["kickoff"],
        } for g in recent]
    return result

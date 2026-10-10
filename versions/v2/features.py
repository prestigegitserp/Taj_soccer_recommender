"""V2 purely causal match features. NO current squad / retrospective transfer leakage.

Inputs are historical completed fixtures before the scheduled kickoff.
Output names and math must match docs/v2/features.mjs. Rolling weighted form,
opponent adjusted return, clean sheets, BTTS and recent direction signals.
"""
from __future__ import annotations
from math import exp,log

EXTRA_NAMES=[
 "h_wgf","h_wga","h_wppg","h_clean","h_btts","h_goal_trend",
 "a_wgf","a_wga","a_wppg","a_clean","a_btts","a_goal_trend",
 "rest_gap","venue_form_gap",
]
def build_extra(game,league_prior):
    cutoff=game["kickoff"]
    home=game["home"]["id"];away=game["away"]["id"]
    def make(team):
        history=[x for x in league_prior
            if (x["home"]["id"]==team or x["away"]["id"]==team)
            and x["kickoff"]<cutoff][-8:]
        if len(history)<3:return None
        values=[]
        from datetime import datetime
        t0=datetime.fromisoformat(cutoff.replace("Z","+00:00"))
        for x in history:
            is_home=x["home"]["id"]==team
            gf=x["home"]["score"] if is_home else x["away"]["score"]
            ga=x["away"]["score"] if is_home else x["home"]["score"]
            old=datetime.fromisoformat(x["kickoff"].replace("Z","+00:00"))
            age=max(0,(t0-old).total_seconds()/86400)
            w=exp(-log(2)*age/60)
            values.append((gf,ga,3 if gf>ga else 1 if gf==ga else 0,
                           int(ga==0),int(gf>0 and ga>0),w,age,is_home))
        den=sum(v[5] for v in values)
        weighted=lambda index:sum(v[index]*v[5] for v in values)/den
        trend=sum(v[0]-v[1] for v in values[-3:])/3-sum(v[0]-v[1] for v in values[:3])/3
        venue=[v[0]-v[1] for v in values if v[7] == (team==home)]
        venue_gap=(sum(venue)/len(venue)) if venue else 0
        return [weighted(0),weighted(1),weighted(2),weighted(3),
                weighted(4),trend,values[-1][6],venue_gap]
    h=make(home);a=make(away)
    if h is None or a is None:return None
    return h[:6]+a[:6]+[(h[6]-a[6])/20,(h[7]-a[7])/3]

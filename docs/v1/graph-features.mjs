/**
 * Pre-kickoff football knowledge graph. Identical to scripts/graph_features.py.
 * Teams are nodes; completed matches are dated edges. Elo is recalculated
 * chronologically from the last 80 completed games in the competition.
 * NO POST-KICKOFF match result or imagined squad statistic is used.
 */
export const GRAPH_NAMES=[
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
];
const average=(a,fallback=0)=>a.length?a.reduce((x,y)=>x+y,0)/a.length:fallback;
const eloChance=(a,b,adv=0)=>1/(1+10**((b-a-adv)/400));
const valid=g=>g.state==="post"&&Number.isInteger(g.home?.score)&&Number.isInteger(g.away?.score);
const scoreFor=(g,id)=>{
  const host=g.home.id===id;
  return {gf:host?g.home.score:g.away.score,ga:host?g.away.score:g.home.score,
    opponent:host?g.away.id:g.home.id,host};
};
export function buildGraph(match,all){
  if(!match?.home?.id||!match?.away?.id||!match?.kickoff)return null;
  const games=all.filter(g=>g.league===match.league&&valid(g)&&
      Date.parse(g.kickoff)<Date.parse(match.kickoff))
    .sort((a,b)=>Date.parse(a.kickoff)-Date.parse(b.kickoff)||String(a.id).localeCompare(String(b.id)))
    .slice(-80);
  if(games.length<16)return null;
  const ratings=new Map();
  const matches=new Map();
  const paths=new Map();
  const rating=id=>ratings.get(id)??1500;
  const arr=(store,id)=>{if(!store.has(id))store.set(id,[]);return store.get(id);};
  for(let i=0;i<games.length;){
    const stamp=games[i].kickoff;
    const changes=new Map();
    while(i<games.length&&games[i].kickoff===stamp){
      const g=games[i++],h=g.home.id,a=g.away.id;
      const expected=eloChance(rating(h),rating(a),55);
      const actual=g.home.score>g.away.score?1:g.home.score===g.away.score?.5:0;
      const diff=26*(actual-expected);
      changes.set(h,(changes.get(h)||0)+diff);
      changes.set(a,(changes.get(a)||0)-diff);
      arr(matches,h).push(g);arr(matches,a).push(g);
    }
    for(const [id,change] of changes){
      ratings.set(id,rating(id)+change);
    }
    for(const [id] of changes){arr(paths,id).push(rating(id));}
  }
  const home=match.home.id,away=match.away.id;
  if((matches.get(home)?.length||0)<3||(matches.get(away)?.length||0)<3)return null;
  function profile(team){
    const recent=(matches.get(team)||[]).slice(-6);
    const opponents=recent.map(g=>scoreFor(g,team).opponent);
    const quality=recent.map(g=>{
      const {gf,ga,opponent,host}=scoreFor(g,team);
      return (gf>ga?1:gf===ga?.5:0)-eloChance(rating(team),rating(opponent),host?55:-55);
    });
    const rates=new Map();
    for(const id of new Set(opponents)){
      const recentOther=(matches.get(id)||[]).slice(-6).map(g=>scoreFor(g,id));
      rates.set(id,{gf:average(recentOther.map(g=>g.gf),1.4),
        ga:average(recentOther.map(g=>g.ga),1.4)});
    }
    const twohop=average(opponents.map(id=>{
      return average((matches.get(id)||[]).slice(-6).map(g=>rating(scoreFor(g,id).opponent)),1500);
    }),1500);
    const track=paths.get(team)||[];
    return {neighbors:new Set(opponents),
      opponentRating:average(opponents.map(id=>rating(id)),1500),
      quality:average(quality),
      opponentFor:average(opponents.map(id=>rates.get(id).gf),1.4),
      opponentAgainst:average(opponents.map(id=>rates.get(id).ga),1.4),
      twohop,
      degree:new Set((matches.get(team)||[]).map(g=>scoreFor(g,team).opponent)).size,
      trend:rating(team)-(track.length>=6?track[track.length-6]:1500)};
  }
  const h=profile(home),a=profile(away);
  const common=[...h.neighbors].filter(x=>a.neighbors.has(x));
  const union=new Set([...h.neighbors,...a.neighbors]);
  const direct=games.filter(g=>(g.home.id===home&&g.away.id===away)||(g.home.id===away&&g.away.id===home)).slice(-6);
  const h2h=direct.map(g=>{const x=scoreFor(g,home);return {...x,score:x.gf>x.ga?1:x.gf===x.ga?.5:0};});
  const vector=[
    (rating(home)-1500)/250,(rating(away)-1500)/250,
    (rating(home)-rating(away))/400,
    (h.opponentRating-1500)/250,(a.opponentRating-1500)/250,
    h.quality,a.quality,
    h.opponentFor/3,a.opponentFor/3,
    h.opponentAgainst/3,a.opponentAgainst/3,
    (h.twohop-1500)/250,(a.twohop-1500)/250,
    common.length/Math.max(1,union.size),Math.min(1,common.length/6),
    direct.length/6,average(h2h.map(x=>x.score),.5),
    average(h2h.map(x=>x.gf-x.ga))/3,
    Math.min(1,h.degree/16),Math.min(1,a.degree/16),
    h.trend/200,a.trend/200,
  ];
  if(vector.length!==GRAPH_NAMES.length||vector.some(x=>!Number.isFinite(x)))return null;
  return {features:vector,ratings:{home:rating(home),away:rating(away)},
    teams:{home:h,away:a},common,direct:direct.map(g=>({id:g.id,kickoff:g.kickoff,...scoreFor(g,home)})),
    games};
}
export function graphNeighbors(match,all){
  const graph=buildGraph(match,all);
  if(!graph)return null;
  const result={};
  for(const t of [match.home,match.away]){
    const matches=graph.games.filter(g=>g.home.id===t.id||g.away.id===t.id).slice(-6).reverse();
    result[t.id]=matches.map(g=>{
      const side=scoreFor(g,t.id);
      return {...side,id:g.id,kickoff:g.kickoff,
        opponentName:g.home.id===t.id?g.away.name:g.home.name};
    });
  }
  return {match,ratings:graph.ratings,common:graph.common,neighbors:result,
    features:graph.features,direct:graph.direct};
}

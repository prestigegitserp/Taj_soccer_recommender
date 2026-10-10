/**
 * V2 features. Exact parity with versions/v2/features.py.
 * All event timestamps precede the target fixture. Registered rosters are NOT
 * used as historical features because current membership leaks transfers.
 */
import {featuresFor,FEATURE_NAMES} from "../football-features.mjs";
import {buildGraph,GRAPH_NAMES} from "../graph-features.mjs";
export const EXTRA_NAMES=[
 "h_wgf","h_wga","h_wppg","h_clean","h_btts","h_goal_trend",
 "a_wgf","a_wga","a_wppg","a_clean","a_btts","a_goal_trend",
 "rest_gap","venue_form_gap",
];
export const V2_NAMES=[...FEATURE_NAMES,...GRAPH_NAMES,...EXTRA_NAMES];
function extras(game,previous){
  const cutoff=Date.parse(game.kickoff);
  if(!Number.isFinite(cutoff))return null;
  const one=(team)=>{
    const history=previous.filter(x=>x.league===game.league
      &&Date.parse(x.kickoff)<cutoff&&x.state==="post"
      &&Number.isInteger(x.home.score)&&Number.isInteger(x.away.score)
      &&(x.home.id===team||x.away.id===team))
      .sort((a,b)=>Date.parse(a.kickoff)-Date.parse(b.kickoff)).slice(-8);
    if(history.length<3)return null;
    const values=history.map(x=>{
      const home=x.home.id===team;
      const gf=home?x.home.score:x.away.score;
      const ga=home?x.away.score:x.home.score;
      const age=Math.max(0,(cutoff-Date.parse(x.kickoff))/86400000);
      const w=Math.exp(-Math.log(2)*age/60);
      return {gf,ga,pt:gf>ga?3:gf===ga?1:0,clean:ga===0?1:0,
        btts:gf>0&&ga>0?1:0,w,age,home};
    });
    const denom=values.reduce((a,v)=>a+v.w,0);
    const weighted=key=>values.reduce((a,v)=>a+v[key]*v.w,0)/denom;
    const recent=values.slice(-3),older=values.slice(0,3);
    const differential=v=>v.reduce((a,x)=>a+x.gf-x.ga,0)/3;
    const trend=differential(recent)-differential(older);
    const atHome=team===game.home.id;
    const venue=values.filter(v=>v.home===atHome);
    const venueGap=venue.length?venue.reduce((a,v)=>a+v.gf-v.ga,0)/venue.length:0;
    return [weighted("gf"),weighted("ga"),weighted("pt"),weighted("clean"),
      weighted("btts"),trend,values[values.length-1].age,venueGap];
  };
  const h=one(game.home.id),a=one(game.away.id);
  if(!h||!a)return null;
  return [...h.slice(0,6),...a.slice(0,6),(h[6]-a[6])/20,(h[7]-a[7])/3];
}
export function v2Vector(game,history){
  if(!game||!Array.isArray(history))return null;
  const base=featuresFor(game,history);
  const graph=buildGraph(game,history);
  const extra=extras(game,history);
  if(!base||!graph?.features||!extra)return null;
  const value=[...base,...graph.features,...extra];
  if(value.length!==56||value.some(v=>!Number.isFinite(v)))return null;
  return value;
}
export function v2RawFeatures(game,history){return extras(game,history);}

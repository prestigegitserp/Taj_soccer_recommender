/**
 * Versioned feature specification paired with train_litert_soccer.py.
 * Features are COMPUTED ON THE USER'S DEVICE from completed matches only,
 * before the target kickoff. No tracking or tactical properties invented.
 */
export const FEATURE_NAMES=[
  "league_home_goals","league_away_goals","league_home_win","league_draw",
  "home_gf6","home_ga6","home_ppg6","home_n6",
  "away_gf6","away_ga6","away_ppg6","away_n6",
  "home_gf3","home_ga3","away_gf3","away_ga3",
  "home_rest20","away_rest20","home_home_gf6","away_away_gf6",
];
const timestamp=s=>{const n=Date.parse(s);return Number.isFinite(n)?n:NaN;};
const mean=(arr,otherwise)=>arr.length?arr.reduce((a,b)=>a+b,0)/arr.length:otherwise;
const valid=g=>g&&g.state==="post"&&Number.isInteger(g.home?.score)
  &&Number.isInteger(g.away?.score)&&g.home.score>=0&&g.away.score>=0
  &&g.home.score<=25&&g.away.score<=25;
function form(id,completed,kickoff,isHome){
  const games=completed.filter(g=>g.home.id===id||g.away.id===id).slice(-6).reverse();
  if(!games.length)return null;
  const entries=games.map(g=>{
    const ownHome=g.home.id===id;
    const gf=ownHome?g.home.score:g.away.score;
    const ga=ownHome?g.away.score:g.home.score;
    return {gf,ga,points:gf>ga?3:gf===ga?1:0,ownHome,
      t:timestamp(g.kickoff)};
  });
  const gf=mean(entries.map(g=>g.gf),1.4);
  const ga=mean(entries.map(g=>g.ga),1.4);
  const recent=entries.slice(0,3);
  const sameVenue=entries.filter(x=>x.ownHome===isHome).map(x=>x.gf);
  const restDays=Math.min(20,Math.max(0,(timestamp(kickoff)-entries[0].t)/86400000))/20;
  return {n:entries.length,features:[
    gf,ga,mean(entries.map(x=>x.points),1.3),entries.length/6,
    mean(recent.map(x=>x.gf),gf),mean(recent.map(x=>x.ga),ga),
    restDays,mean(sameVenue,gf)
  ]};
}
export function featuresFor(game,all){
  if(!game?.home?.id||!game?.away?.id||!game?.league)return null;
  const cutoff=timestamp(game.kickoff);
  if(!Number.isFinite(cutoff))return null;
  const previous=all.filter(g=>g.league===game.league&&valid(g)
    &&timestamp(g.kickoff)<cutoff).sort((a,b)=>timestamp(a.kickoff)-timestamp(b.kickoff));
  const league=previous.slice(-80);
  if(league.length<16)return null;
  const bh=mean(league.map(x=>x.home.score),1.4);
  const ba=mean(league.map(x=>x.away.score),1.1);
  const homeWin=mean(league.map(x=>x.home.score>x.away.score?1:0),.45);
  const draw=mean(league.map(x=>x.home.score===x.away.score?1:0),.27);
  const h=form(game.home.id,previous,game.kickoff,true);
  const a=form(game.away.id,previous,game.kickoff,false);
  if(!h||!a||h.n<3||a.n<3)return null;
  const hf=h.features,af=a.features;
  const vector=[
    bh,ba,homeWin,draw,
    hf[0],hf[1],hf[2],hf[3],af[0],af[1],af[2],af[3],
    hf[4],hf[5],af[4],af[5],hf[6],af[6],hf[7],af[7],
  ];
  if(vector.length!==FEATURE_NAMES.length||vector.some(v=>!Number.isFinite(v)))return null;
  return vector;
}
export function prepareInput(features,card){
  if(!Array.isArray(features)||features.length!==FEATURE_NAMES.length
    ||card?.schema!=="taj-litert-1x2-v1"
    ||JSON.stringify(card.input_features)!==JSON.stringify(FEATURE_NAMES)
    ||card.mean?.length!==FEATURE_NAMES.length
    ||card.std?.length!==FEATURE_NAMES.length)return null;
  return new Float32Array(features.map((x,i)=>{
    const z=(x-card.mean[i])/Math.max(card.std[i],1e-5);
    return Math.max(-5,Math.min(5,z));
  }));
}
export function temperatureAdjust(values,temp){
  const t=Number(temp);
  if(!Array.isArray(values)||values.length!==3||!Number.isFinite(t)||t<=0
     ||values.some(x=>!Number.isFinite(x)||x<0))return null;
  const powers=values.map(x=>Math.pow(Math.max(x,1e-6),1/t));
  const den=powers.reduce((a,b)=>a+b,0);
  return powers.map(x=>x/den);
}

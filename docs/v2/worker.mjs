/**
 * V2: pure-JavaScript evaluator of the Python-trained LightGBM forest.
 * The 56-feature vector is computed in this worker from REAL, finished prior
 * matches. No model inference requests to any server.
 */
import {v2Vector,V2_NAMES} from "./features.mjs";
import {graphNeighbors} from "../graph-features.mjs";
const max=(a,b)=>a>b?a:b;
const min=(a,b)=>a<b?a:b;
const send=(type,obj={})=>self.postMessage({type,...obj});
let history=null,model=null,report=null,initPromise=null;
function softmax(values){
  const m=Math.max(...values);
  const v=values.map(x=>Math.exp(x-m));
  const sum=v.reduce((a,b)=>a+b,0);
  return v.map(x=>x/sum);
}
export function evaluateForest(row,forest){
  if(!forest||forest.schema!=="taj-v2-lgbm-v1"||forest.features?.length!==56
    ||row?.length!==56)return null;
  const logits=[0,0,0];
  for(let k=0;k<forest.trees.length;k++){
    let n=forest.trees[k];
    while(n.v===undefined){
      const v=row[n.f];
      n=(Number.isFinite(v)?v<=n.t:n.d)?n.l:n.r;
    }
    logits[k%3]+=n.v;
  }
  const t=forest.temperature;
  if(!(t>0))return null;
  return softmax(logits.map(x=>x/t));
}
const factorial=[1,1];for(let i=2;i<=12;i++)factorial.push(factorial[i-1]*i);
function lambda(row){
  const bh=row[0],ba=row[1],base=max(.45,(bh+ba)/2);
  const nh=row[7]*6,na=row[11]*6;
  const shrink=(r,n)=>(n*r+6*base)/(n+6);
  return [
    min(3.7,max(.25,(shrink(row[4],nh)*shrink(row[9],na)/base)*(bh/base))),
    min(3.7,max(.25,(shrink(row[8],na)*shrink(row[5],nh)/base)*(ba/base)))
  ];
}
export function scoreProb(row,rho=0){
  const [h,a]=lambda(row);
  const p=[0,0,0];let den=0;
  let ph=Math.exp(-h);
  for(let i=0;i<=10;i++){
    if(i)ph*=h/i;
    let pa=Math.exp(-a);
    for(let j=0;j<=10;j++){
      if(j)pa*=a/j;
      let tau=1;
      if(i===0&&j===0)tau=1-h*a*rho;
      else if(i===0&&j===1)tau=1+h*rho;
      else if(i===1&&j===0)tau=1+a*rho;
      else if(i===1&&j===1)tau=1-rho;
      if(tau<=0)return null;
      const v=ph*pa*tau;
      p[i>j?0:i===j?1:2]+=v;den+=v;
    }
  }
  return p.map(v=>v/den);
}
function sourceArchive(payload){
  const indexed=new Map();
  for(const g of payload.events||[]){
    if(g?.state==="post" && Number.isInteger(g.home?.score)
      && Number.isInteger(g.away?.score))indexed.set(g.league+":"+g.id,g);
  }
  return indexed;
}
async function init(){
  if(model&&history&&report)return;
  if(initPromise)return initPromise;
  initPromise=(async()=>{
  send("status",{message:"در حال بارگیری آرشیو واقعی مسابقات و مدل آموزش‌دیده..."});
  const [fm,fh,fr]=await Promise.all([
    fetch("./models/forest.json",{cache:"force-cache"}),
    fetch("./data/history.json",{cache:"force-cache"}),
    fetch("./data/evaluation.json",{cache:"no-store"})
  ]);
  if(!fm.ok||!fh.ok||!fr.ok)throw Error("V2 model artifacts are not published");
  const [forest,archive,evalReport]=await Promise.all([fm.json(),fh.json(),fr.json()]);
  if(forest.schema!=="taj-v2-lgbm-v1"||JSON.stringify(forest.features)!==JSON.stringify(V2_NAMES)
     ||archive.schema!=="taj-v2-history-v1"||evalReport.schema!=="taj-v2-evaluation-v1")
    throw Error("V2 model, history and evaluation version mismatch");
  model=forest;history=sourceArchive(archive);report=evalReport;
  send("ready",{metrics:report.overall,confidence:report.paired_95pct,
    promoted:report.promote_for_accuracy,rawHistory:history.size,
    production:report.production_forest});
  })();
  try{await initPromise;}
  catch(err){initPromise=null;throw err;}
}
self.onmessage=async ({data})=>{
  const message=data||{};
  try{
    if(message.type==="init"){await init();return;}
    if(message.type==="prediction"){
      await init();
      const game=message.game;
      if(!game?.league||!game.home?.id||!game.away?.id)return;
      const all=new Map(history);
      for(const g of (message.recent||[])){
        if(g?.state==="post" && Number.isInteger(g.home?.score)
          && Number.isInteger(g.away?.score))all.set(g.league+":"+g.id,g);
      }
      const prior=[...all.values()];
      const vector=v2Vector(game,prior);
      const graphContext=graphNeighbors(game,prior);
      if(!vector){send("unavailable",{key:message.key,message:"Not enough verified earlier matches for V2 prediction"});return;}
      const poisson=scoreProb(vector,0);
      const rho=Number.isFinite(model.production_rho)?model.production_rho:0;
      const dc=scoreProb(vector,rho);
      const boosted=evaluateForest(vector,model);
      if(!boosted||!poisson||!dc)throw Error("Nonfinite probability vector");
      const w=Number.isFinite(model.production_weight)?model.production_weight:0;
      const trial=boosted.map((v,i)=>v*w+dc[i]*(1-w));
      // Prespecified research safeguard: do not silently promote a challenger
      // with no statistically supported out-of-sample win.
      const deployed=report.promote_for_accuracy?trial:poisson;
      send("result",{key:message.key,poisson,dixon:dc,boosted,candidate:trial,
        deployed,vector,graphContext,weight:w,rho,official:report.promote_for_accuracy?"v2":"v1-poisson"});
    }
  }catch(err){send("error",{key:message.key,message:String(err?.message||err).slice(0,400)});}
};

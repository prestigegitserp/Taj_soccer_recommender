/**
 * TAJ V2 UI, with real fixtures and verified completed-game knowledge.
 * Predictions are entirely executed inside the V2 model Web Worker.
 */
const el=id=>document.getElementById(id);
let fixtures=[],events=[],knowledge=null,worker=null,chosen=null,report=null;
const cache=new Map();
const key=g=>g.league+":"+g.id;
const name=t=>t.short||t.name||"Team";
const time=v=>new Date(v).toLocaleString("fa-IR",{month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"});
function child(parent,tag,value,cls){
 const node=document.createElement(tag);
 if(value!==undefined)node.textContent=String(value);
 if(cls)node.className=cls;
 parent.appendChild(node);return node;
}
function status(message){el("status").textContent=message;}
function flatten(feed){return Object.values(feed?.leagues||{}).flatMap(l=>l.events||[]).filter(x=>x?.id&&x.home?.id&&x.away?.id);}
function upcoming(){
 return events.filter(x=>x.state==="pre"&&Date.parse(x.kickoff)>Date.now()-120000)
  .sort((a,b)=>Date.parse(a.kickoff)-Date.parse(b.kickoff)).slice(0,5);
}
function drawBars(root,p){
 root.replaceChildren();
 for(let i=0;i<3;i++){
  const row=child(root,"div",undefined,"probRow");
  child(row,"span",["برد میزبان","مساوی","برد میهمان"][i]);
  const track=child(row,"div",undefined,"track");
  const fill=child(track,"div",undefined,"trackFill"+(i===1?" draw":i===2?" away":""));
  fill.style.width=(p[i]*100).toFixed(2)+"%";
  child(row,"b",(p[i]*100).toFixed(1)+"٪");
 }
}
function schedule(){
 const root=el("fixtures");root.replaceChildren();
 if(!fixtures.length){child(root,"div","پنج بازی آینده در داده معتبر فعلی پیدا نشد.","loading");return;}
 for(const g of fixtures){
  const btn=child(root,"button",undefined,"fixture"+(chosen===key(g)?" selected":""));
  btn.type="button";
  const top=child(btn,"div",undefined,"fixtureTop");
  child(top,"span",g.league);child(top,"span",time(g.kickoff));
  const names=child(btn,"div",undefined,"fixtureTitle");
  child(names,"span",name(g.home));child(names,"span","VS","vs");child(names,"span",name(g.away));
  btn.addEventListener("click",()=>choose(g));
 }
}
function squads(g){
 const root=el("squads");root.replaceChildren();
 const k=knowledge?.matches?.[key(g)];
 for(const team of [g.home,g.away]){
  const card=child(root,"article",undefined,"squadCard");
  child(card,"h3",name(team));
  const evidence=k?.teams?.[team.id];
  if(!evidence){child(card,"p","آمار ثبت‌شده برای این بازی موجود نیست؛ داده فرضی جایگزین نمی‌شود.","footnote");continue;}
  const block=child(card,"div",undefined,"squadValues");
  for(const [id,label] of Object.entries({possession:"مالکیت توپ",shots:"شوت",shots_on_target:"شوت در چارچوب",yellow_cards:"کارت زرد"})){
   const v=evidence.recent_boxscore?.[id];if(!v)continue;
   const tile=child(block,"div",undefined,"squadMetric");
   child(tile,"b",v.value+(id==="possession"?"٪":""));
   child(tile,"small",label+" · "+v.observations+" مسابقه");
  }
  const members=evidence.roster||[];
  const detail=child(card,"details");
  child(detail,"summary",members.length+" بازیکن ثبت‌شده (نه ترکیب اصلی)");
  const list=child(detail,"div",undefined,"players");
  for(const p of members)child(list,"span",(p.jersey||"—")+" · "+p.name+" / "+p.position,"player");
 }
}
function graph(result,g){
 const root=el("rivalGraph");root.replaceChildren();
 const facts=el("graphMetrics");facts.replaceChildren();
 const data=result.graphContext;
 if(!data){child(root,"div","اطلاعات کافی برای گراف دو تیم پیدا نشد.","loading");return;}
 const ns="http://www.w3.org/2000/svg";
 const svg=(tag,props={})=>{
   const e=document.createElementNS(ns,tag);
   for(const [k,v] of Object.entries(props))e.setAttribute(k,String(v));
   return e;
 };
 const scene=svg("svg",{viewBox:"0 0 780 360",role:"img","aria-label":"گراف حریفان واقعی دو تیم"});
 const center={home:[195,170],away:[585,170]};
 const positions=new Map(),common=new Set(data.common||[]);
 for(const side of ["home","away"]){
  const members=data.neighbors[g[side].id]||[];
  const unique=[...new Set(members.map(x=>x.opponent))].slice(0,6);
  unique.forEach((id,i)=>{
   if(positions.has(id))return;
   positions.set(id,{x:common.has(id)?390:(side==="home"?70:710),
     y:common.has(id)?40+65*(positions.size%5):42+260*(i+1)/(unique.length+1),
     name:members.find(m=>m.opponent===id)?.opponentName||id});
  });
 }
 for(const side of ["home","away"]){
  const [x,y]=center[side];
  for(const e of data.neighbors[g[side].id]||[]){
   const n=positions.get(e.opponent);if(!n)continue;
   const line=svg("path",{d:"M "+x+" "+y+" L "+n.x+" "+n.y,
     stroke:e.gf>e.ga?"#6de8bb":e.gf===e.ga?"#e0c281":"#e29897",
     "stroke-width":common.has(e.opponent)?2:1.2,opacity:".65"});
   const title=svg("title");
   title.textContent=name(g[side])+" "+e.gf+"-"+e.ga+" "+e.opponentName;
   line.appendChild(title);scene.appendChild(line);
  }
 }
 function plot(x,y,text,isMain=false,isCommon=false){
  scene.appendChild(svg("circle",{cx:x,cy:y,r:isMain?45:isCommon?26:21,
   fill:isMain?"#225d54":isCommon?"#354d70":"#243c48",
   stroke:isMain?"#8cf2bd":"#7fa2ae","stroke-width":isMain?2.5:1.2}));
  const t=svg("text",{x,y:y+4,"font-size":isMain?13:10,fill:"#f0f9f6",
   "font-weight":"bold","text-anchor":"middle"});
  t.textContent=text.length>15?text.slice(0,12)+"…":text;scene.appendChild(t);
 }
 for(const [id,n] of positions)plot(n.x,n.y,n.name,false,common.has(id));
 plot(...center.home,name(g.home),true);
 plot(...center.away,name(g.away),true);
 root.appendChild(scene);
 for(const [label,v] of [["قدرت میزبان",data.ratings.home.toFixed(0)],
   ["حریف مشترک",data.common.length],["قدرت میهمان",data.ratings.away.toFixed(0)]]){
  const n=child(facts,"div");
  child(n,"small",label);child(n,"b",v);
 }
}
function compare(response){
 const root=el("comparison");root.replaceChildren();
 for(const [title,p,note] of [
  ["Poisson",response.poisson,"خط مبنای نسخه ۱"],
  ["Dixon–Coles",response.dixon,"اصلاح کم‌گل · rho "+Number(response.rho).toFixed(2)],
  ["LightGBM · 56F",response.boosted,"مدل یادگیری ماشین آموزش‌دیده"],
  ["Validated Blend",response.candidate,"وزن AI: "+Math.round(response.weight*100)+"٪"]]){
   const card=child(root,"article",undefined,"algo");
   child(card,"div","MODEL COMPARISON","algoTag");
   child(card,"h3",title);
   drawBars(child(card,"div",undefined,"outcomeBars"),p);
   child(card,"p",note,"detail");
 }
}
function result(data){
 cache.set(data.key,data);
 if(chosen!==data.key)return;
 const game=fixtures.find(g=>key(g)===chosen);
 if(!game)return;
 drawBars(el("probabilities"),data.deployed);
 el("predictionNote").textContent="خروجی واقعی شبکه آموزش‌دیده؛ محاسبه داخل مرورگر انجام شد.";
 const msg=el("qualityMessage");
 msg.className="interpretation"+(data.official==="v2"?"":" warning");
 msg.textContent=data.official==="v2"
  ?"V2 معیار ارتقا را در آزمون زمانی گذرانده؛ احتمال به‌معنای قطعیت نیست."
  :"مزیت پایدار مدل جدید هنوز اثبات نشده؛ احتمال رسمی با Poisson محافظت می‌شود و خروجی AI جدید جداگانه نمایش داده می‌شود.";
 el("modelDisclosure").textContent="LightGBM: "+data.boosted.map(x=>(100*x).toFixed(1)+"٪").join(" / ")
   +" · مدل مرجع: "+data.poisson.map(x=>(100*x).toFixed(1)+"٪").join(" / ");
 compare(data);graph(data,game);
}
function choose(g){
 chosen=key(g);schedule();squads(g);
 const header=el("matchHeader");header.replaceChildren();
 const team=child(header,"div");
 child(team,"div",name(g.home)+" × "+name(g.away));
 child(team,"small",g.league+" · "+time(g.kickoff));
 child(header,"span","T-K","versionCapsule");
 if(cache.has(chosen)){result(cache.get(chosen));return;}
 el("probabilities").replaceChildren();
 el("predictionNote").textContent="در حال محاسبه محلی با آرشیو واقعی بازی‌ها...";
 if(worker)worker.postMessage({type:"prediction",game:g,key:chosen,
   recent:events.filter(x=>x.state==="post")});
}
async function main(){
 try{
  const [feedRes,knowRes]=await Promise.all([
   fetch("../data/feed.json",{cache:"no-store"}),
   fetch("../data/match-knowledge.json",{cache:"no-store"})
  ]);
  if(!feedRes.ok)throw Error("منبع مسابقات در دسترس نیست");
  const feed=await feedRes.json();
  events=flatten(feed);fixtures=upcoming();
  if(knowRes.ok)knowledge=await knowRes.json();
  el("sourceTime").textContent=feed.generated_at?time(feed.generated_at):"ESPN";
  el("historyCount").textContent=events.filter(x=>x.state==="post").length.toLocaleString("fa-IR");
  schedule();
  worker=new Worker(new URL("./worker.mjs",import.meta.url),{type:"module"});
  worker.onmessage=({data})=>{
   if(data.type==="status")status(data.message);
   if(data.type==="ready"){
    status("AI فعال · محلی");
    report=data;
    el("historyCount").textContent=data.rawHistory.toLocaleString("fa-IR");
    el("researchStatus").textContent=(100*data.metrics.v2_auto.accuracy).toFixed(1)+"٪";
    el("researchStatCaption").textContent="روی "+data.metrics.v2_auto.n+" مسابقه تست؛ Brier "+data.metrics.v2_auto.brier;
    el("promotionStatement").textContent=data.promoted
      ?"نسخه ۲ در بک‌تست معیار محافظه‌کار ارتقا را گذرانده است."
      :"نسخه ۲ هنوز برتری معنادار خود را ثابت نکرده است؛ Poisson مرجع باقی می‌ماند و مدل‌های جدید برای مقایسه علمی ارائه می‌شوند.";
    const game=fixtures.find(g=>key(g)===chosen);
    if(game)choose(game);
   }else if(data.type==="result")result(data);
   else if(data.type==="unavailable"&&chosen===data.key)el("predictionNote").textContent="تاریخچه معتبر کافی برای پیش‌بینی موجود نیست.";
   else if(data.type==="error"){
    status("خطای AI");
    el("predictionNote").textContent="خطای مدل محلی: "+data.message;
   }
  };
  worker.onerror=()=>status("Worker مدل پشتیبانی نشد");
  worker.postMessage({type:"init"});
  if(fixtures.length)choose(fixtures[0]);
 }catch(error){
  status("داده در دسترس نیست");
  el("fixtures").textContent=String(error.message||error);
 }
}
window.addEventListener("pagehide",()=>worker?.terminate());
main();

/**
 * Browser knowledge graph explorer and a REAL graph-conditioned neural model.
 * Edges come from completed matches; roster players and boxscores from verified
 * ESPN snapshots, NOT inferred lineups. No graph/training backend runs here.
 */
import {activeGameEvidence} from "./app.mjs";
import {graphNeighbors} from "./graph-features.mjs";
import {featuresFor} from "./football-features.mjs";

const byId=id=>document.getElementById(id);
const add=(parent,tag,cls,content)=>{
  const child=document.createElement(tag);
  if(cls)child.className=cls;
  if(content!==undefined)child.textContent=String(content);
  parent.appendChild(child);return child;
};
const average=(array)=>array.reduce((a,b)=>a+b,0)/array.length;
const percent=value=>(100*value).toFixed(1)+"٪";
const model={
 knowledge:null,
 lastKey:null,
 lastGraph:null,
 predictions:new Map(),
};
function freshKey(game){return game.league+":"+game.id;}
function probabilityFromBase(x){
  if(!x)return null;
  const bh=x[0],ba=x[1],base=Math.max(.45,(bh+ba)/2);
  const nh=x[7]*6,na=x[11]*6;
  const shrink=(rate,n)=>(n*rate+6*base)/(n+6);
  const home=Math.min(3.7,Math.max(.25,(shrink(x[4],nh)*shrink(x[9],na)/base)*(bh/base)));
  const away=Math.min(3.7,Math.max(.25,(shrink(x[8],na)*shrink(x[5],nh)/base)*(ba/base)));
  const probabilities=[0,0,0];let total=0;
  let ph=Math.exp(-home);
  for(let h=0;h<=10;h++){
    if(h)ph*=home/h;
    let pa=Math.exp(-away);
    for(let a=0;a<=10;a++){
      if(a)pa*=away/a;
      const p=ph*pa;total+=p;
      probabilities[h>a?0:h===a?1:2]+=p;
    }
  }
  return probabilities.map(v=>v/total);
}
function forecastBar(parent,name,value,cls){
  const root=add(parent,"div","graphForecastRow");
  add(root,"span","graphForecastLabel",name);
  const track=add(root,"div","graphForecastTrack");
  const fill=add(track,"div","graphForecastFill "+cls);
  fill.style.width=100*Math.max(0,Math.min(1,value))+"%";
  add(root,"strong","",percent(value));
}
function forecast(detail){
  const item=activeGameEvidence();
  if(!item||freshKey(item.game)!==detail.id)return;
  const root=byId("graphModelResults");
  if(!root)return;
  root.replaceChildren();
  const data=detail.probabilities;
  const card=detail.card;
  if(!Array.isArray(data)||data.length!==3)return;
  const modelSection=add(root,"div","graphPredictionBox");
  add(modelSection,"h4","","پیش‌بینی مستقل شبکه ۴۲ویژگی گرافی");
  forecastBar(modelSection,"میزبان",data[0],"home");
  forecastBar(modelSection,"مساوی",data[1],"draw");
  forecastBar(modelSection,"میهمان",data[2],"away");
  const base=featuresFor(item.game,item.results);
  const poisson=probabilityFromBase(base);
  if(poisson){
    const weight=Number(card.validated_blend_graph_weight);
    const w=Number.isFinite(weight)?Math.max(0,Math.min(1,weight)):0;
    const combo=poisson.map((p,i)=>w*data[i]+(1-w)*p);
    const mix=add(root,"div","graphPredictionBox graphBlend");
    add(mix,"h4","","Ensemble · مدل گرافی + Poisson");
    add(mix,"p","graphSubline","وزن مدل گرافی "+Math.round(w*100)+"٪؛ انتخاب‌شده فقط روی داده اعتبارسنجی");
    forecastBar(mix,"برد میزبان",combo[0],"home");
    forecastBar(mix,"مساوی",combo[1],"draw");
    forecastBar(mix,"برد میهمان",combo[2],"away");
  }
  const metrics=card.metrics||{};
  const line=add(root,"div","graphModelEvidence");
  const g=metrics.graph_model_holdout,bl=metrics.validation_blend_holdout,p=metrics.poisson_holdout;
  if(g&&bl&&p){
    add(line,"span","","آزمون زمانی: "+card.test_n+" مسابقه (بدون استفاده در آموزش)");
    add(line,"span","","Brier: شبکه "+g.brier+" · ترکیبی "+bl.brier+" · Poisson "+p.brier);
    add(line,"span","","Log loss: شبکه "+g.log_loss+" · ترکیبی "+bl.log_loss+" · Poisson "+p.log_loss);
    if(bl.brier>=p.brier)add(line,"strong","graphWarning","این نسخه در آزمون Brier بهتر از Poisson نیست؛ قطعیت ادعا نمی‌شود.");
  }
  add(root,"p","graphCaveat","این مدل نتیجه‌ها و ساختار حریفان را می‌بیند؛ هنوز ترکیب قطعی ۱۱ نفره، مصدومیت و Tracking را به‌عنوان ورودی یاد نگرفته است.");
}
const svgNS="http://www.w3.org/2000/svg";
const svgElement=(name,attrs={})=>{
  const n=document.createElementNS(svgNS,name);
  for(const [key,value] of Object.entries(attrs))n.setAttribute(key,String(value));
  return n;
};
function drawGraph(parent,graph){
  const {match,neighbors,common,ratings}=graph;
  const nameById=new Map([[match.home.id,match.home.short||match.home.name],
    [match.away.id,match.away.short||match.away.name]]);
  for(const t of [match.home,match.away]){
    for(const edge of neighbors[t.id]){
      if(!nameById.has(edge.opponent))nameById.set(edge.opponent,edge.opponentName);
    }
  }
  const nHome=neighbors[match.home.id],nAway=neighbors[match.away.id];
  const homeIDs=[...new Set(nHome.map(x=>x.opponent).filter(x=>!common.includes(x)))].slice(0,6);
  const awayIDs=[...new Set(nAway.map(x=>x.opponent).filter(x=>!common.includes(x)))].slice(0,6);
  const commonIDs=common.slice(0,6);
  const nodes=new Map();
  const scatter=(ids,x,top,bottom)=>{
    ids.forEach((id,i)=>nodes.set(id,{id,name:nameById.get(id)||id,
      x:x+(i%2?24:-24),y:top+(bottom-top)*(i+1)/(ids.length+1),common:false}));
  };
  scatter(homeIDs,87,15,315);scatter(awayIDs,673,15,315);
  commonIDs.forEach((id,i)=>nodes.set(id,{
    id,name:nameById.get(id)||id,x:380,y:25+(i+1)*280/(commonIDs.length+1),common:true}));
  const shell=add(parent,"div","graphCanvas");
  const svg=svgElement("svg",{viewBox:"0 0 760 340",role:"img",
    "aria-label":"گراف روابط واقعی تیم‌ها با حریفان اخیر و حریفان مشترک"});
  const centerHome={x:220,y:170},centerAway={x:540,y:170};
  for(const t of [match.home,match.away]){
    const center=t.id===match.home.id?centerHome:centerAway;
    const sides=neighbors[t.id];
    for(const edge of sides){
      const pos=nodes.get(edge.opponent);
      if(!pos)continue;
      const color=edge.gf>edge.ga?"#54f1b5":edge.gf===edge.ga?"#8498ab":"#db877c";
      const line=svgElement("path",{d:"M "+center.x+" "+center.y+" Q "+((center.x+pos.x)/2)+
        " "+(Math.min(center.y,pos.y)-10)+" "+pos.x+" "+pos.y,
        stroke:color,"stroke-width":pos.common?2.3:1.4,"stroke-opacity":.70,fill:"none"});
      const title=svgElement("title");
      title.textContent=(t.short||t.name)+" "+edge.gf+" – "+edge.ga+" "+edge.opponentName+" · "+edge.kickoff;
      line.appendChild(title);svg.appendChild(line);
    }
  }
  // Genuinely grounded rival linkage: only draw a direct edge if H2H exists.
  if(graph.direct.length){
    const r=svgElement("path",{d:"M 263 170 L 497 170",stroke:"#d9b785",
      "stroke-dasharray":"6 5","stroke-width":"2",opacity:".65"});
    svg.appendChild(r);
  }
  const plot=(n,central=false)=>{
    const group=svgElement("g");
    const circle=svgElement("circle",{cx:n.x,cy:n.y,r:central?44:n.common?24:19,
      fill:central?"#1c7166":n.common?"#26556e":"#1e3847",
      stroke:central?"#74f5c3":n.common?"#94c1ed":"#638898",
      "stroke-width":central?2.5:1.4});
    group.appendChild(circle);
    const t=svgElement("text",{x:n.x,y:n.y+(central?3:2),fill:"#effcf9",
      "text-anchor":"middle","font-size":central?12:10,
      "font-weight":central?"800":"600","font-family":"Arial,sans-serif"});
    const name=String(n.name||n.id);
    t.textContent=name.length>(central?18:13)?name.slice(0,central?15:11)+"…":name;
    group.appendChild(t);
    const label=svgElement("title");
    label.textContent=n.name+(n.common?" · حریف مشترک":"");
    group.appendChild(label);
    svg.appendChild(group);
  };
  for(const n of nodes.values())plot(n);
  plot({x:centerHome.x,y:centerHome.y,name:match.home.short||match.home.name},true);
  plot({x:centerAway.x,y:centerAway.y,name:match.away.short||match.away.name},true);
  shell.appendChild(svg);
  const legend=add(parent,"div","graphLegend");
  for(const [c,text] of [["#54f1b5","برد"],["#8498ab","مساوی"],["#db877c","باخت"],["#94c1ed","حریف مشترک"]]){
    const item=add(legend,"span");const dot=add(item,"i");
    dot.style.background=c;add(item,"span","",text);
  }
  const summary=add(parent,"div","graphFacts");
  const h=add(summary,"div");add(h,"small","","قدرت شبکه‌ای "+(match.home.short||match.home.name));
  add(h,"strong","",ratings.home.toFixed(0));
  const commonCell=add(summary,"div");add(commonCell,"small","","حریف مشترک در ۶ بازی");
  add(commonCell,"strong","",common.length);
  const away=add(summary,"div");add(away,"small","","قدرت شبکه‌ای "+(match.away.short||match.away.name));
  add(away,"strong","",ratings.away.toFixed(0));
  add(parent,"p","graphCaveat","رتبه‌ها Elo محاسبه‌شده از ۸۰ مسابقه ثبت‌شده این لیگ هستند؛ امتیاز رسمی تیم یا رنکینگ جهانی نیستند.");
}
function drawSquadGraph(parent,match,knowledge){
  const home=knowledge?.teams?.[match.home.id]?.roster||[];
  const away=knowledge?.teams?.[match.away.id]?.roster||[];
  if(!home.length&&!away.length){
    add(parent,"p","graphEmpty","هنوز فهرست معتبر بازیکنان دریافت نشده است.");
    return;
  }
  const wrap=add(parent,"div","graphCanvas squadGraphCanvas");
  const svg=svgElement("svg",{viewBox:"0 0 820 480",role:"img",
    "aria-label":"گراف واقعی ارتباط باشگاه‌ها با فهرست بازیکنان ثبت‌شده در چهار گروه پستی"});
  const roles=[["G","دروازه‌بان"],["D","مدافع"],["M","هافبک"],["F","مهاجم"]];
  function link(x1,y1,x2,y2,color){
    svg.appendChild(svgElement("path",{d:"M "+x1+" "+y1+" L "+x2+" "+y2,
      stroke:color,"stroke-width":1.3,opacity:.55,fill:"none"}));
  }
  function text(x,y,str,size=10,color="#eef9fa",weight=600){
    const node=svgElement("text",{x,y,"text-anchor":"middle",fill:color,
      "font-family":"Arial,sans-serif","font-size":size,"font-weight":weight});
    node.textContent=String(str).slice(0,24);svg.appendChild(node);
  }
  function club(team,players,x,color,right){
    const title=team.short||team.name;
    const node=svgElement("rect",{x:x-75,y:14,width:150,height:52,rx:18,
      fill:"#183d44",stroke:color,"stroke-width":2});
    svg.appendChild(node);text(x,45,title.slice(0,18),14,color,700);
    roles.forEach(([pos,label],i)=>{
      const filtered=players.filter(p=>p.position===pos);
      const y=120+i*106;
      link(x,66,x,y-24,color);
      svg.appendChild(svgElement("rect",{x:x-69,y:y-24,width:138,height:47,
        rx:12,fill:"#183c49",stroke:color,opacity:.9}));
      text(x,y-4,label+" · "+filtered.length,11,"#d8f5f0",700);
      text(x,y+12,"عضو فهرست",9,"#94bbc0",500);
      // Two actual roster members shown per role. Full named squad remains
      // available below the graph; these are NOT predicted starters.
      filtered.slice(0,2).forEach((person,index)=>{
        const px=right?x-203:x+203,py=y-14+index*31;
        link(right?x-69:x+69,y+(index===0?-5:5),right?px+55:px-55,py,color);
        const rect=svgElement("rect",{x:px-79,y:py-15,width:158,height:25,
          rx:8,fill:"#142e3c",stroke:"#466775","stroke-width":1});
        svg.appendChild(rect);
        const abbreviated=person.name.length>21?person.name.slice(0,19)+"…":person.name;
        text(px,py+1,abbreviated,9,"#e0f0f2",500);
        const tip=svgElement("title");
        tip.textContent=person.name+" · "+label+" · شماره "+(person.jersey||"نامشخص");
        rect.appendChild(tip);
      });
    });
  }
  club(match.home,home,180,"#76e9bc",false);
  club(match.away,away,640,"#9ebefa",true);
  wrap.appendChild(svg);
  add(parent,"p","graphCaveat","پیوندها «بازیکن عضو فهرست باشگاه» هستند؛ مدل هیچ ترکیب اصلی یا دقایق بازی را پیش‌بینی نکرده است. نام‌های نمایش‌داده‌شده نمونه‌ای از کل فهرست واقعی‌اند.");
}

function statsSummary(team){
  const box=team?.recent_boxscore||{};
  const labels={
    possession:"مالکیت توپ",shots:"شوت در بازی",shots_on_target:"شوت در چارچوب",
    corners:"کرنر",fouls:"خطا",yellow_cards:"کارت زرد"};
  return Object.entries(labels).filter(([name])=>box[name]).map(([name,label])=>({
    name,label,value:box[name].value,n:box[name].observations,
  }));
}
function rosterPanel(parent,team,info){
  const root=add(parent,"article","squadCard");
  add(root,"h4","",team.short||team.name);
  const roster=info?.roster||[];
  const summary=statsSummary(info);
  if(summary.length){
    const rows=add(root,"div","squadStats");
    for(const field of summary){
      const chip=add(rows,"div","squadStat");
      add(chip,"strong","",field.value+(field.name==="possession"?"٪":""));
      add(chip,"small","",field.label);
      add(chip,"small","squadCoverage",field.n+" بازی");
    }
  }else add(root,"p","graphCaveat","آمار Boxscore معتبر در داده فعلی موجود نیست.");
  const details=add(root,"details","squadPlayers");
  add(details,"summary","",roster.length+" بازیکن ثبت‌شده در فهرست باشگاه");
  if(!roster.length){
    add(details,"p","graphCaveat","فهرست بازیکنان از منبع عمومی دریافت نشد.");
    return;
  }
  const positions=[["G","دروازه‌بان"],["D","مدافع"],["M","هافبک"],["F","مهاجم"]];
  for(const [pos,label] of positions){
    const members=roster.filter(p=>p.position===pos);
    if(!members.length)continue;
    const group=add(details,"div","squadGroup");
    add(group,"h5","",label+" · "+members.length);
    const players=add(group,"div","squadPlayerGrid");
    for(const p of members){
      const el=add(players,"div","squadPlayer");
      add(el,"b","squadJersey",p.jersey||"—");
      add(el,"span","",p.name);
    }
  }
}
function renderGraphMatch(){
  const evidence=activeGameEvidence();
  if(!evidence)return;
  const match=evidence.game,key=freshKey(match);
  if(key===model.lastKey)return;
  model.lastKey=key;
  const explorer=byId("knowledgeGraph");
  const context=byId("teamKnowledge");
  if(!explorer||!context)return;
  explorer.replaceChildren();context.replaceChildren();
  const modes=add(explorer,"div","graphModes");
  for(const [id,label] of [["rivals","روابط تیم‌ها و حریفان"],["players","باشگاه ← پست ← بازیکن"]]){
    const button=add(modes,"button",model.mode===id?"selected":"",label);
    button.type="button";
    button.addEventListener("click",()=>{
      if(model.mode===id)return;
      model.mode=id;model.lastKey=null;renderGraphMatch();
    });
  }
  const graph=graphNeighbors(match,evidence.results);
  model.lastGraph=graph;
  const current=model.knowledge?.matches?.[key];
  if(model.mode==="players")drawSquadGraph(explorer,match,current);
  else if(graph)drawGraph(explorer,graph);
  else add(explorer,"p","graphEmpty","مسابقات تاریخی کافی برای ساخت گراف این رقابت ثبت نشده است.");
  if(current){
    const sides=add(context,"div","squadColumns");
    rosterPanel(sides,match.home,current.teams?.[match.home.id]);
    rosterPanel(sides,match.away,current.teams?.[match.away.id]);
    add(context,"p","graphCaveat","منبع: فهرست باشگاهی ESPN و Boxscore بازی‌های پایان‌یافته. بازیکنان فوق ترکیب ۱۱ نفره اعلام‌شده نیستند.");
  }else add(context,"p","graphEmpty","داده تکمیلی این بازی هنوز تأیید یا منتشر نشده است.");
  const stored=model.predictions.get(key);
  const result=byId("graphModelResults");
  if(result)result.replaceChildren();
  if(stored)forecast(stored);
}
async function start(){
  window.addEventListener("taj:graphPrediction",event=>{
    if(!event.detail?.id)return;
    model.predictions.set(event.detail.id,event.detail);
    forecast(event.detail);
  });
  const area=byId("matches");if(!area)return;
  const observe=new MutationObserver(()=>queueMicrotask(renderGraphMatch));
  observe.observe(area,{childList:true});
  renderGraphMatch();
  try{
    const response=await fetch("./data/match-knowledge.json",{cache:"no-store"});
    if(response.ok){
      const result=await response.json();
      if(result.schema==="taj-knowledge-v1"){
        model.knowledge=result;
        model.lastKey=null;
        renderGraphMatch();
      }
    }
  }catch{}
  document.addEventListener("visibilitychange",()=>{
    if(!document.hidden){model.lastKey=null;renderGraphMatch();}
  });
  window.addEventListener("pagehide",()=>observe.disconnect());
}
if(typeof document!=="undefined")start();

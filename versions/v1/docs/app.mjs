// TAJ Live Match Intelligence. Zero dependencies, browser-side inference.
export const LEAGUES={"eng.1":"Premier League","esp.1":"La Liga","ger.1":"Bundesliga","ita.1":"Serie A","fra.1":"Ligue 1","uefa.champions":"Champions League"};
const scoreOk=g=>Number.isInteger(g?.home?.score)&&Number.isInteger(g?.away?.score);
const time=s=>{const x=Date.parse(s);return Number.isFinite(x)?x:0;};
const average=(array)=>array.length?array.reduce((a,b)=>a+b,0)/array.length:0;
const clamp=(v,lo,hi)=>Math.min(hi,Math.max(lo,v));
const two=n=>Number(n).toFixed(2);
export function flatten(feed){
  const out=new Map();
  for(const [league,bundle] of Object.entries(feed?.leagues||{})){
    for(const match of bundle.events||[]){
      if(!match?.id||!match?.home?.id||!match?.away?.id||!time(match.kickoff))continue;
      const key=league+":"+match.id;
      const old=out.get(key);
      if(!old||(old.state==="pre"&&match.state!=="pre"))out.set(key,{...match,league});
    }
  }
  return [...out.values()].sort((a,b)=>time(a.kickoff)-time(b.kickoff));
}
export function nextFive(games,league="all",at=Date.now()){
  return games.filter(g=>g.state==="pre"&&time(g.kickoff)>=at-120000&&(league==="all"||g.league===league))
    .sort((a,b)=>time(a.kickoff)-time(b.kickoff)).slice(0,5);
}
export function historyFor(id,games,cutoff){
  return games.filter(g=>g.state==="post"&&scoreOk(g)&&time(g.kickoff)<cutoff
    &&(g.home.id===id||g.away.id===id))
    .sort((a,b)=>time(b.kickoff)-time(a.kickoff)).slice(0,6);
}
export function describe(id,hist){
  if(!hist.length)return null;
  const matches=hist.map(g=>{
    const host=g.home.id===id;
    const scored=host?g.home.score:g.away.score;
    const conceded=host?g.away.score:g.home.score;
    return {scored,conceded,points:scored>conceded?3:scored===conceded?1:0,
      result:scored>conceded?"W":scored===conceded?"D":"L",
      opponent:host?g.away.name:g.home.name,date:g.kickoff};
  });
  return {n:matches.length,scored:average(matches.map(m=>m.scored)),
    conceded:average(matches.map(m=>m.conceded)),ppg:average(matches.map(m=>m.points)),matches};
}
function poisson(mu,k){let x=Math.exp(-mu);for(let i=1;i<=k;i++)x=x*mu/i;return x;}
export function scoreProbabilities(h,a){
  let home=0,away=0,draw=0,total=0,best={h:0,a:0,p:0};
  for(let i=0;i<=10;i++)for(let j=0;j<=10;j++){
    const p=poisson(h,i)*poisson(a,j);total+=p;
    if(i>j)home+=p;else if(i===j)draw+=p;else away+=p;
    if(p>best.p)best={h:i,a:j,p};
  }
  return {home:home/total,draw:draw/total,away:away/total,h,a,score:best};
}
export function analyze(match,all,scenario="base"){
  const completed=all.filter(g=>g.state==="post"&&scoreOk(g)&&time(g.kickoff)<time(match.kickoff));
  const league=completed.filter(g=>g.league===match.league);
  const h=describe(match.home.id,historyFor(match.home.id,completed,time(match.kickoff)));
  const a=describe(match.away.id,historyFor(match.away.id,completed,time(match.kickoff)));
  const valid=Boolean(h&&a&&h.n>=3&&a.n>=3&&league.length>=8);
  if(!valid)return {home:h,away:a,leagueN:league.length,forecast:null};
  const baseH=average(league.map(x=>x.home.score));
  const baseA=average(league.map(x=>x.away.score));
  const base=Math.max(.45,(baseH+baseA)/2);
  const shrink=(rate,n)=>(n*rate+6*base)/(n+6);
  const hmu=clamp(shrink(h.scored,h.n)*shrink(a.conceded,a.n)/base*(baseH/base)*(scenario==="home"?1.10:1),.25,3.7);
  const amu=clamp(shrink(a.scored,a.n)*shrink(h.conceded,h.n)/base*(baseA/base)*(scenario==="away"?1.10:1),.25,3.7);
  return {home:h,away:a,leagueN:league.length,forecast:scoreProbabilities(hmu,amu),
    quality:Math.round(100*Math.min(h.n,a.n,6)/6*Math.min(league.length,18)/18)};
}
export function advice(analysis){
  const h=analysis.home,a=analysis.away,out=[];
  if(!h||!a)return [{title:"شواهد ناکافی",reason:"برای ارزیابی، سابقه نتایج بیشتری لازم است.",caveat:"آمار تیمی جای Tracking را نمی‌گیرد."}];
  if(h.conceded>=1.45)out.push({title:"بازبینی امنیت دفاعی میزبان",reason:"میانگین گل خورده میزبان "+two(h.conceded)+" در "+h.n+" مسابقه اخیر بوده است.",caveat:"علت و منطقه آسیب‌پذیری از نتایج مشخص نمی‌شود."});
  if(a.conceded>=1.45)out.push({title:"بررسی فرصت حمله مقابل میهمان",reason:"میهمان در "+a.n+" مسابقه اخیر میانگین "+two(a.conceded)+" گل دریافت کرده است.",caveat:"نمی‌توان بدون ویدیو یا Event محل نفوذ را تعیین کرد."});
  if(h.scored>=1.50)out.push({title:"رصد گلزنی میزبان",reason:"میانگین "+two(h.scored)+" گل در بازی‌های اخیر میزبان ثبت شده است.",caveat:"این معیار سبک پرس یا الگوی پاس تیم را اثبات نمی‌کند."});
  if(a.scored>=1.50)out.push({title:"آمادگی برای تهدید تهاجمی میهمان",reason:"میانگین گلزنی میهمان "+two(a.scored)+" بوده است.",caveat:"نوع ضدحمله و برتری فضایی از این آمار قابل استخراج نیست."});
  if(!out.length)out.push({title:"اولویت مشاهده مسابقات اخیر",reason:"برتری آماری قوی در نمونه فعلی وجود ندارد؛ تحلیلگران باید صحنه‌ها را بررسی کنند.",caveat:"هیچ تاکتیک قطعی از تعداد گل‌ها نتیجه نمی‌شود."});
  return out.slice(0,3);
}
const el=(tag,cls,text)=>{
  const node=document.createElement(tag);
  if(cls)node.className=cls;
  if(text!==undefined)node.textContent=String(text);
  return node;
};
const add=(parent,tag,cls,text)=>{const node=el(tag,cls,text);parent.appendChild(node);return node;};
const root=id=>document.getElementById(id);
const dateFa=s=>new Intl.DateTimeFormat("fa-IR",{weekday:"short",month:"short",day:"numeric",hour:"2-digit",minute:"2-digit"}).format(new Date(s));
const percent=n=>(100*n).toFixed(1)+"٪";
const labelLeague=id=>LEAGUES[id]||id;
const countdown=s=>{
  const mins=Math.max(0,Math.floor((time(s)-Date.now())/60000));
  const days=Math.floor(mins/1440);
  return days?days+" روز تا شروع":Math.floor(mins/60)+" ساعت و "+mins%60+" دقیقه تا شروع";
};
function avatar(team){
  const wrap=el("div","avatar");
  if(team.logo&&String(team.logo).startsWith("https://a.espncdn.com/")){
    const img=el("img");img.src=team.logo;img.alt="";img.loading="lazy";img.referrerPolicy="no-referrer";wrap.appendChild(img);
  }else add(wrap,"span","initial",String(team.short||team.name).slice(0,2));
  return wrap;
}
function teamFace(team){
  const face=el("div","teamFace");face.appendChild(avatar(team));
  add(face,"span","teamName",team.short||team.name);return face;
}
const state={feed:null,all:[],filter:"all",selected:null,scenario:"base",polling:false};
export function activeGameEvidence(){
  const game=state.all.find(g=>g.league+":"+g.id===state.selected);
  return game?{game,results:state.all,feedUpdated:state.feed?.generated_at||null}:null;
}
function status(text,warn=false){
  root("statusLabel").textContent=text;
  root("statusPoint").className=warn?"statusPoint warn":"statusPoint";
}
function card(match){
  const key=match.league+":"+match.id;
  const button=el("button","matchCard"+(key===state.selected?" active":""));
  button.setAttribute("aria-pressed",String(key===state.selected));
  const row=add(button,"div","cardHeading");
  add(row,"span","leagueTag",labelLeague(match.league));
  add(row,"span","dateTag",dateFa(match.kickoff));
  const teams=add(button,"div","cardTeams");
  teams.appendChild(teamFace(match.home));
  add(teams,"span","vs","VS");
  teams.appendChild(teamFace(match.away));
  const bottom=add(button,"div","cardFooter");
  add(bottom,"span","countdown",countdown(match.kickoff));
  add(bottom,"span","cardLink","تحلیل ←");
  button.addEventListener("click",()=>{
    state.selected=key;state.scenario="base";render();
    if(window.innerWidth<900)root("intelligence").scrollIntoView({behavior:"smooth"});
  });
  return button;
}
function addForm(parent,label,form){
  const shell=add(parent,"div","formBox");
  add(shell,"h4","",label);
  if(!form){add(shell,"p","dim","نتیجه کافی موجود نیست.");return;}
  const marks=add(shell,"div","formMarks");
  form.matches.slice(0,5).forEach(m=>{
    const badge=add(marks,"span","formResult "+m.result,m.result==="W"?"برد":m.result==="D"?"مساوی":"باخت");
    badge.title=m.opponent+" "+m.scored+"-"+m.conceded;
  });
  const stats=add(shell,"div","miniStats");
  for(const [label,value] of [["گل زده/بازی",form.scored],["گل خورده/بازی",form.conceded],["امتیاز/بازی",form.ppg]]){
    const box=add(stats,"div","miniStat");add(box,"b","",two(value));add(box,"span","",label);
  }
  add(shell,"p","dim","نمونه: "+form.n+" بازی پایان‌یافته ثبت‌شده");
}
function bar(parent,label,p,cls){
  const line=add(parent,"div","probRow");
  add(line,"span","probName",label);
  const track=add(line,"div","probTrack");
  const fill=add(track,"div","probFill "+cls);fill.style.width=Math.max(0,Math.min(100,p*100))+"%";
  add(line,"strong","probValue",percent(p));
}
function intelligence(match){
  const dest=root("intelligence");dest.replaceChildren();
  const analysis=analyze(match,state.all,state.scenario);
  const heading=add(dest,"section","insightHero");
  const hrow=add(heading,"div","insightLabel");
  add(hrow,"span","labelGlow","MATCH INTELLIGENCE");
  add(hrow,"span","hint","Online Results · Browser AI");
  const teams=add(heading,"div","focusTeams");
  teams.appendChild(teamFace(match.home));add(teams,"span","focusVs","VS");teams.appendChild(teamFace(match.away));
  add(heading,"p","insightMeta",labelLeague(match.league)+" · "+dateFa(match.kickoff));
  const probability=add(dest,"section","analysisSection");
  const top=add(probability,"div","sectionHeader");
  add(top,"h3","","پیش‌بینی آماری مسابقه");
  add(top,"span","minorTag","Experimental · Poisson");
  if(!analysis.forecast){
    const empty=add(probability,"div","insufficient");
    add(empty,"strong","","برای برآورد درصدها شواهد کافی نداریم.");
    add(empty,"p","","حداقل ۳ بازی گذشته برای هر تیم و ۸ نتیجه برای این لیگ لازم است؛ احتمال ساختگی نمی‌سازیم.");
  }else{
    add(probability,"p","modelDisclaimer","برآورد آزمایشی از گل‌های اخیر؛ پیش‌بینی تأییدشده یا xG نیست.");
    const block=add(probability,"div","probabilities");
    bar(block,"برد میزبان",analysis.forecast.home,"mint");
    bar(block,"مساوی",analysis.forecast.draw,"gold");
    bar(block,"برد میهمان",analysis.forecast.away,"blue");
    const goals=add(probability,"div","goalForecast");
    for(const [name,value] of [
      ["گل مدل · میزبان",two(analysis.forecast.h)],
      ["نتیجه پرتکرار مدل",analysis.forecast.score.h+" – "+analysis.forecast.score.a],
      ["گل مدل · میهمان",two(analysis.forecast.a)]
    ]){
      const x=add(goals,"div","goalBox");add(x,"span","",name);add(x,"b","",value);
    }
  }
  add(probability,"p","methodInfo","نمونه لیگ: "+analysis.leagueN+" · میزبان: "+(analysis.home?.n||0)+" · میهمان: "+(analysis.away?.n||0)+
    (analysis.quality===undefined?"":" · شاخص پوشش: "+analysis.quality+"/۱۰۰"));
  const form=add(dest,"section","analysisSection");
  add(form,"h3","","فرم شش مسابقه اخیر موجود");
  const formTwo=add(form,"div","formTwo");
  addForm(formTwo,match.home.short,analysis.home);addForm(formTwo,match.away.short,analysis.away);
  const playbook=add(dest,"section","analysisSection");
  const phead=add(playbook,"div","sectionHeader");
  add(phead,"h3","","سناریوهای قابل بررسی");
  add(phead,"span","minorTag","شواهد نتیجه‌ای، نه Tracking");
  advice(analysis).forEach((s,i)=>{
    const box=add(playbook,"article","suggestion");
    const title=add(box,"div","suggestTitle");
    add(title,"span","index","0"+(i+1));add(title,"strong","",s.title);
    add(box,"p","",s.reason);add(box,"div","risk","محدودیت: "+s.caveat);
  });
  const sensitivity=add(dest,"section","analysisSection");
  add(sensitivity,"h3","","What-if · حساسیت مدل");
  add(sensitivity,"p","dim","فقط فرض نرخ گل را ۱۰٪ تغییر بده؛ واکنش واقعی بازیکنان شبیه‌سازی نمی‌شود.");
  const options=add(sensitivity,"div","switches");
  for(const [key,label] of [["base","پایه"],["home","میزبان +۱۰٪"],["away","میهمان +۱۰٪"]]){
    const b=add(options,"button","switch"+(state.scenario===key?" chosen":""),label);
    b.addEventListener("click",()=>{state.scenario=key;render();});
  }
  const foot=add(dest,"div","analysisFooter");
  add(foot,"span","","منبع: ESPN · پردازش: مرورگر · Tracking آینده موجود نیست");
  const a=add(foot,"a","","جزئیات بازی در ESPN ↗");
  a.href="https://www.espn.com/soccer/match/_/gameId/"+encodeURIComponent(match.id);
  a.target="_blank";a.rel="noopener noreferrer";
}
function render(){
  const fixtures=nextFive(state.all,state.filter);
  if(!fixtures.some(g=>state.selected===g.league+":"+g.id)){
    state.selected=fixtures[0]?fixtures[0].league+":"+fixtures[0].id:null;
    state.scenario="base";
  }
  root("fixtureCount").textContent=String(fixtures.length);
  root("fixtureSubtitle").textContent=fixtures.length===5?"پنج بازی نزدیک‌تر که در منبع ثبت شده‌اند":"تعداد بازی‌های تأییدشده: "+fixtures.length+" (بازی فرضی ساخته نمی‌شود)";
  const grid=root("matches");grid.replaceChildren();
  if(fixtures.length)fixtures.forEach(g=>grid.appendChild(card(g)));
  else{
    const nothing=add(grid,"div","emptyState");
    add(nothing,"h3","","بازی آینده تأییدشده پیدا نشد");
    add(nothing,"p","","ممکن است تعطیلات مسابقات باشد یا تأمین‌کننده داده موقتاً در دسترس نباشد. لیگ دیگری انتخاب کن.");
  }
  const selected=state.all.find(g=>g.league+":"+g.id===state.selected);
  if(selected)intelligence(selected);
  else{root("intelligence").replaceChildren();add(root("intelligence"),"p","emptyState","در انتظار مسابقه ثبت‌شده...");}
  const live=state.all.filter(x=>x.state==="in");
  const ticker=root("liveTicker");ticker.replaceChildren();
  ticker.hidden=!live.length;
  if(live.length){
    add(ticker,"b","","● "+live.length+" بازی در جریان (طبق منبع)");
    live.slice(0,5).forEach(g=>add(ticker,"span","",g.home.short+" "+(g.home.score??"-")+" : "+(g.away.score??"-")+" "+g.away.short));
  }
}
function applyFeed(feed){
  if(feed?.schema!=="taj-fixtures-v1"||!feed.leagues)throw Error("ساختار منبع نامعتبر است.");
  state.feed=feed;state.all=flatten(feed);
  root("allCount").textContent=String(state.all.length);
  root("leagueCount").textContent=String(Object.keys(feed.leagues).length);
  const updates=Object.values(feed.leagues).map(v=>time(v.updated_at)).filter(Boolean).sort((a,b)=>b-a);
  const age=updates.length?Math.max(0,Date.now()-updates[0]):Infinity;
  root("age").textContent=age<60000?"همین لحظه":age<3600000?Math.round(age/60000)+" دقیقه":Math.round(age/3600000)+" ساعت";
  status("منبع معتبر: "+state.all.length+" مسابقه · آخرین به‌روزرسانی "+root("age").textContent,age>75*60000);
  root("dataDisclosure").textContent="داده‌ها از API عمومی غیررسمی ESPN گردآوری می‌شوند. پردازش مدل در دستگاه شماست. زمان آخرین جمع‌آوری: "+(feed.generated_at||"نامعلوم");
  render();
}
async function load(){
  try{
    const response=await fetch("./data/feed.json?t="+Date.now(),{cache:"no-store"});
    if(!response.ok)throw Error("خطای دریافت "+response.status);
    const packet=await response.json();applyFeed(packet);
    try{localStorage.setItem("taj-public-feed",JSON.stringify(packet));}catch{}
  }catch(err){
    let cached=null;
    try{cached=JSON.parse(localStorage.getItem("taj-public-feed")||"null");}catch{}
    if(cached){applyFeed(cached);status("نسخه کش‌شده مرورگر؛ ممکن است قدیمی باشد",true);}
    else{
      status("داده معتبر فعلاً در دسترس نیست. دوباره تلاش کن.",true);
      root("matches").replaceChildren();add(root("matches"),"p","emptyState","هیچ بازی فرضی نمایش نمی‌دهیم.");
    }
  }
}
export function normalizeEspn(raw,league){
  const comp=raw?.competitions?.[0];if(!comp)return null;
  const group=Object.fromEntries((comp.competitors||[]).filter(x=>["home","away"].includes(x.homeAway)).map(x=>[x.homeAway,x]));
  if(!group.home||!group.away||!time(raw.date||comp.date))return null;
  const side=x=>{
    const t=x.team||{};let score=x.score;
    if(typeof score==="object"&&score!==null)score=score.value??score.displayValue;
    return {id:String(t.id||""),name:String(t.displayName||t.name||""),short:String(t.shortDisplayName||t.abbreviation||t.name||""),
      logo:typeof t.logo==="string"?t.logo:null,
      score:score===null||score===undefined||score===""?null:Number(score)};
  };
  const status=raw?.status?.type?.state||"pre";
  return {id:String(raw.id||""),league,kickoff:new Date(raw.date||comp.date).toISOString(),
    state:["pre","in","post"].includes(status)?status:"pre",home:side(group.home),away:side(group.away)};
}
async function tryLive(){
  if(state.polling||!state.feed)return;
  state.polling=true;root("refresh").disabled=true;
  const ids=state.filter==="all"?Object.keys(LEAGUES):[state.filter];
  const stamp=d=>String(d.getUTCFullYear())+String(d.getUTCMonth()+1).padStart(2,"0")+String(d.getUTCDate()).padStart(2,"0");
  const today=new Date(),end=new Date(Date.now()+35*86400000);
  const range=stamp(today).slice(0,6);
  const responses=await Promise.allSettled(ids.map(async league=>{
    const controller=new AbortController(),id=setTimeout(()=>controller.abort(),9000);
    try{
      const url="https://site.web.api.espn.com/apis/site/v2/sports/soccer/"+encodeURIComponent(league)+"/scoreboard?dates="+range+"&limit=250";
      const res=await fetch(url,{signal:controller.signal,mode:"cors",cache:"no-store"});
      if(!res.ok)throw Error(String(res.status));
      const data=await res.json();if(!Array.isArray(data.events))throw Error("Invalid events");
      return {league,events:data.events.map(x=>normalizeEspn(x,league)).filter(Boolean)};
    }finally{clearTimeout(id)}
  }));
  let success=0;
  for(const x of responses){
    if(x.status!=="fulfilled")continue;
    const {league,events}=x.value;success++;
    const bundle=state.feed.leagues[league]||{name:LEAGUES[league],events:[]};
    const merged=new Map((bundle.events||[]).map(m=>[m.id,m]));
    events.forEach(m=>merged.set(m.id,m));
    state.feed.leagues[league]={...bundle,updated_at:new Date().toISOString(),events:[...merged.values()]};
  }
  if(success){
    state.all=flatten(state.feed);
    status("دریافت مستقیم تازه از "+success+" رقابت؛ محاسبات در مرورگر",false);render();
  }else{
    status("دسترسی مستقیم مرورگر مسدود یا ناموفق بود؛ آخرین داده تأییدشده نمایش داده می‌شود.",true);
  }
  state.polling=false;root("refresh").disabled=false;
}
function bootstrap(){
  root("leagueTabs").querySelectorAll("button[data-league]").forEach(b=>{
    b.addEventListener("click",()=>{
      state.filter=b.dataset.league;state.selected=null;state.scenario="base";
      root("leagueTabs").querySelectorAll("button").forEach(x=>x.classList.toggle("current",x===b));
      render();
    });
  });
  root("refresh").addEventListener("click",async()=>{await load();await tryLive();});
  load();
  setInterval(()=>{if(!document.hidden&&state.feed)tryLive();},300000);
  setInterval(()=>{if(state.feed)render();},60000);
  document.addEventListener("visibilitychange",()=>{if(!document.hidden)load();});
}
if(typeof document!=="undefined")bootstrap();

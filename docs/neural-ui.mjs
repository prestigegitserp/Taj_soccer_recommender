/**
 * Neural match forecasts: static site + actual LiteRT model weights.
 * Nothing is sent to a model server. Static JSON and model can be CDN cached;
 * feature extraction and trained neural forward-pass happen on visitor device.
 */
import {activeGameEvidence} from "./app.mjs";
import {featuresFor,prepareInput} from "./football-features.mjs";

const byId=id=>document.getElementById(id);
const state={worker:null,ready:false,starting:false,card:null,
  data:new Map(),active:null,locked:false};
function status(t){const e=byId("neuralStatus");if(e)e.textContent=t;}
function node(tag,className,text){
  const el=document.createElement(tag);
  if(className)el.className=className;
  if(text!==undefined)el.textContent=text;
  return el;
}
function bar(parent,label,value,name){
  const row=node("div","neuralProbability");
  row.appendChild(node("span","",label));
  const track=node("div","neuralTrack");
  const fill=node("div","neuralFill "+name);
  fill.style.width=(Math.max(0,Math.min(1,value))*100).toFixed(2)+"%";
  track.appendChild(fill);row.appendChild(track);
  row.appendChild(node("strong","",(value*100).toFixed(1)+"٪"));
  parent.appendChild(row);
}
function report(prediction){
  const root=byId("neuralResults");if(!root)return;
  root.replaceChildren();
  const p=prediction.probabilities;
  if(!p||p.length!==3){status("خروجی مدل معتبر نیست.");return;}
  bar(root,"برد میزبان",p[0],"mint");
  bar(root,"مساوی",p[1],"gold");
  bar(root,"برد میهمان",p[2],"blue");
  const card=state.card||prediction.card;
  const footer=node("div","neuralMetrics");
  footer.appendChild(node("span","",
    "مدل: "+String(card?.model||"LiteRT MLP")+" · "+String(card?.train_n||"—")+" نمونه آموزش"));
  const result=card?.holdout,base=card?.frequency_baseline_holdout;
  if(result&&base){
    footer.appendChild(node("span","",
      "Brier آزمون زمانی: "+result.brier+" | خط مبنای فراوانی: "+base.brier));
    footer.appendChild(node("span","",
      "Log loss آزمون زمانی: "+result.log_loss+" | خط مبنا: "+base.log_loss));
    const poisson=card?.poisson_baseline_holdout;
    if(poisson){
      footer.appendChild(node("span","",
        "Poisson در همان آزمون: Brier "+poisson.brier+" | Log loss "+poisson.log_loss));
      if(result.brier>=poisson.brier){
        footer.appendChild(node("span","neuralCaution",
          "هشدار: مدل عصبی روی آزمون تاریخی از Poisson بهتر نبود؛ دو خروجی را با احتیاط مقایسه کن."));
      }else{
        footer.appendChild(node("span","",
          "در این بازه آزمون، Brier مدل عصبی کمتر از Poisson بوده است؛ تضمینی برای بازی‌های آینده نیست."));
      }
    }
    if(result.brier>=base.brier){
      footer.appendChild(node("span","neuralCaution",
        "در همین آزمون، مدل از خط مبنای ساده فراوانی هم بهتر نشده است."));
    }
  }
  root.appendChild(footer);
}
function chosen(){
  const evidence=activeGameEvidence();
  if(!evidence)return null;
  const id=evidence.game.league+":"+evidence.game.id;
  return {...evidence,id};
}
function ask(){
  const item=chosen();
  if(!item||!state.ready||!state.card)return;
  const raw=featuresFor(item.game,item.results);
  if(!raw){status("برای مدل عصبی، نتایج قبلی این مسابقه کافی نیست؛ مدل Poisson همچنان در دسترس است.");byId("neuralResults")?.replaceChildren();return;}
  const vector=prepareInput(raw,state.card);
  if(!vector){status("نسخه ویژگی‌های سایت با نسخه مدل همخوانی ندارد؛ پیش‌بینی عصبی غیرفعال شد.");return;}
  state.active=item.id;
  if(state.data.has(item.id)){
    report(state.data.get(item.id));
    status("مدل آموزش‌دیده LiteRT.js · استنتاج محلی روی دستگاه");
  }else if(state.worker){
    status("در حال محاسبه احتمال‌های AI در مرورگر...");
    state.worker.postMessage({type:"predict",id:item.id,vector:Array.from(vector)});
  }
}
async function ensureModel(){
  if(state.starting||state.worker)return;
  const evidence=chosen();
  if(!evidence){status("ابتدا داده پنج بازی آینده را بارگیری می‌کنیم.");return;}
  const v=featuresFor(evidence.game,evidence.results);
  if(!v){status("نمونه نتایج تاریخی برای این مسابقه کافی نیست.");return;}
  state.starting=true;
  try{
    const resp=await fetch("./models/model-card.json",{cache:"force-cache"});
    if(!resp.ok)throw Error("مدل LiteRT هنوز در سایت منتشر نشده است.");
    const metadata=await resp.json();
    if(metadata.schema!=="taj-litert-1x2-v1")throw Error("نسخه مدل ناشناخته است.");
    state.card=metadata;
    state.worker=new Worker(new URL("./litert-worker.mjs",import.meta.url),{type:"classic"});
    state.worker.onmessage=({data})=>{
      if(data.type==="loading"){status(data.message);}
      if(data.type==="ready"){
        state.ready=true;state.card=data.card;status("مدل عصبی آماده شد؛ پیش‌بینی واقعی در مرورگر انجام می‌شود.");ask();
      }
      if(data.type==="prediction"){
        state.data.set(data.id,data);
        if(chosen()?.id===data.id){report(data);status("پیش‌بینی با مدل آموزش‌دیده LiteRT.js · محاسبه محلی");}
      }
      if(data.type==="error"){
        status("LiteRT اجرا نشد: "+String(data.message||"خطای نامشخص")+
          " · مدل آماری Poisson سایت همچنان فعال است.");
      }
    };
    state.worker.onerror=()=>{
      status("Worker مدل روی این دستگاه پشتیبانی نشد؛ Poisson همچنان کار می‌کند.");
      state.worker?.terminate();state.worker=null;state.ready=false;state.starting=false;
    };
    state.worker.postMessage({type:"init"});
  }catch(err){
    status("موتور AI هنوز آماده انتشار نیست: "+String(err.message||err));
    state.starting=false;
  }
}
let lastKey=null;
function refresh(){
  const item=chosen();
  if(!item)return;
  const key=item.id+"|"+String(item.feedUpdated||"");
  if(lastKey===key)return;
  lastKey=key;
  byId("neuralResults")?.replaceChildren();
  if(state.ready)ask();
  else ensureModel();
}
export function mount(){
  const area=byId("matches");
  if(!area||!byId("neuralPrediction"))return;
  // Watch the normal app's match rerenders. Does not touch its rendering engine.
  const observer=new MutationObserver(()=>queueMicrotask(refresh));
  observer.observe(area,{childList:true});
  const retry=byId("neuralRetry");
  retry?.addEventListener("click",()=>{
    state.worker?.terminate();state.worker=null;state.ready=false;
    state.starting=false;lastKey=null;refresh();
  });
  window.addEventListener("pagehide",()=>{observer.disconnect();state.worker?.terminate();});
  refresh();
}
if(typeof document!=="undefined")mount();

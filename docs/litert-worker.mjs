/**
 * Only numerical inference is performed here. Browser loads the exact trained
 * TFLite model, executes it with LiteRT.js (CPU/WASM as most reliable default)
 * and sends back probabilities. No API inference or server model call.
 */
// LiteRT's WASM loader uses importScripts. This script deliberately runs
// inside a CLASSIC Worker, so avoid top-level ESM imports/import.meta.
function temperatureAdjust(values,temp){
  if(!Array.isArray(values)||values.length!==3)return null;
  const t=Number(temp);
  if(!Number.isFinite(t)||t<=0||values.some(v=>!Number.isFinite(v)||v<0))return null;
  const q=values.map(v=>Math.pow(Math.max(v,1e-6),1/t));
  const total=q.reduce((a,b)=>a+b,0);
  return q.map(v=>v/total);
}
let runtime=null,model=null,card=null,graphModel=null,graphCard=null;
const send=(type,value={})=>self.postMessage({type,...value});
async function initialize(){
  if(model&&card)return;
  const response=await fetch("./models/model-card.json",{cache:"no-store"});
  if(!response.ok)throw new Error("مدل AI آموزش‌دیده هنوز منتشر نشده است.");
  const info=await response.json();
  if(info.schema!=="taj-litert-1x2-v1"||info.input_features?.length!==20)
    throw new Error("مدل منتشرشده با نسخه سایت سازگار نیست.");
  send("loading",{message:"در حال دانلود موتور LiteRT.js و وزن‌های مدل واقعی..."});
  // The import and WASM download are lazy and only begin after model card exists.
  // Emscripten resolves the WASM file against worker.location by default,
  // NOT the imported glue script! The URL must be redirected explicitly.
  self.Module=self.Module||{};
  self.Module.locateFile=name=>new URL("./vendor/litert/wasm/"+name,self.location.href).href;
  runtime=await import("https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm");
  await runtime.loadLiteRt(new URL("./vendor/litert/wasm/",self.location.href).href);
  model=await runtime.loadAndCompile(new URL("./models/football_1x2.tflite",self.location.href).href,
     {accelerator:"wasm"});
  card=info;
  send("ready",{card});
}
async function initializeGraph(){
  if(graphModel&&graphCard)return;
  await initialize();
  const response=await fetch("./models/graph-card.json",{cache:"no-store"});
  if(!response.ok)throw new Error("مدل گراف هنوز منتشر نشده است.");
  const metadata=await response.json();
  if(metadata.schema!=="taj-graph-ai-v1"||metadata.input_features?.length!==42)
    throw new Error("نسخه مدل گراف با مرورگر ناسازگار است.");
  send("graphLoading",{message:"در حال دریافت شبکه عصبی چندلایه گراف..."});
  graphModel=await runtime.loadAndCompile(
    new URL("./models/graph_1x2.tflite",self.location.href).href,{accelerator:"wasm"});
  graphCard=metadata;
  send("graphReady",{card:graphCard});
}
async function graphInference(message){
  let input,results;
  try{
    await initializeGraph();
    const vector=message.vector;
    if(!Array.isArray(vector)||vector.length!==graphCard.input_features.length
      ||vector.some(x=>!Number.isFinite(x)))
      throw new Error("بردار گراف معتبر نیست.");
    input=new runtime.Tensor(new Float32Array(vector),[1,42]);
    results=await graphModel.run([input]);
    const raw=await results[0].data();
    const probabilities=temperatureAdjust(Array.from(raw),graphCard.temperature);
    if(!probabilities||Math.abs(probabilities.reduce((a,b)=>a+b,0)-1)>1e-5)
      throw new Error("خروجی احتمالات گراف معتبر نیست.");
    send("graphPrediction",{id:message.id,probabilities,card:graphCard});
  }catch(err){
    send("graphError",{id:message.id,message:String(err?.message||err).slice(0,260)});
  }finally{
    try{input?.delete();}catch{}
    try{for(const t of results||[])t.delete();}catch{}
  }
}
self.onmessage=async ({data})=>{
  const m=data||{};
  if(m.type==="init"){
    try{await initialize();}
    catch(err){send("error",{message:String(err?.message||err).slice(0,260)});}
    return;
  }
  if(m.type==="initGraph"){
    try{await initializeGraph();}
    catch(err){send("graphError",{message:String(err?.message||err).slice(0,260)});}
    return;
  }
  if(m.type==="predictGraph"){await graphInference(m);return;}
  if(m.type!=="predict")return;
  let input,results;
  try{
    await initialize();
    if(!Array.isArray(m.vector)||m.vector.length!==card.input_features.length)
      throw new Error("بردار ویژگی ناقص است.");
    const numbers=new Float32Array(m.vector);
    if([...numbers].some(v=>!Number.isFinite(v)))throw new Error("ورودی غیرعددی دریافت شد.");
    input=new runtime.Tensor(numbers,[1,card.input_features.length]);
    results=await model.run([input]);
    const raw=await results[0].data();
    const calibrated=temperatureAdjust(Array.from(raw),card.temperature);
    if(!calibrated||Math.abs(calibrated.reduce((a,b)=>a+b,0)-1)>1e-5)
      throw new Error("خروجی احتمالات معتبر نبود.");
    send("prediction",{id:m.id,probabilities:calibrated,
      model:card.model,card});
  }catch(err){
    send("error",{id:m.id,message:String(err?.message||err).slice(0,300)});
  }finally{
    try{input?.delete();}catch{}
    try{if(results){for(const t of results)t.delete();}}catch{}
  }
};

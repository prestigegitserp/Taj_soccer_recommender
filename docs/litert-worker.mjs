/**
 * Only numerical inference is performed here. Browser loads the exact trained
 * TFLite model, executes it with LiteRT.js (CPU/WASM as most reliable default)
 * and sends back probabilities. No API inference or server model call.
 */
import {temperatureAdjust} from "./football-features.mjs";
let runtime=null,model=null,card=null;
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
  runtime=await import("https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm");
  await runtime.loadLiteRt("https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/wasm/");
  model=await runtime.loadAndCompile(new URL("./models/football_1x2.tflite",import.meta.url).href,
     {accelerator:"wasm"});
  card=info;
  send("ready",{card});
}
self.onmessage=async ({data})=>{
  const m=data||{};
  if(m.type==="init"){
    try{await initialize();}
    catch(err){send("error",{message:String(err?.message||err).slice(0,260)});}
    return;
  }
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
    if(!calibrated)throw new Error("خروجی احتمالات معتبر نبود.");
    send("prediction",{id:m.id,probabilities:calibrated,
      model:card.model,card});
  }catch(err){
    send("error",{id:m.id,message:String(err?.message||err).slice(0,300)});
  }finally{
    try{input?.delete();}catch{}
    try{if(results){for(const t of results)t.delete();}}catch{}
  }
};

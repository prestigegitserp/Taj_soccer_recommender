/**
 * Optional LOCAL language model worker. No prompt/inference request goes to an
 * AI provider. Download model files from CDN/Hugging Face only after user consent.
 * WebGPU is required; no hidden CPU attempt that freezes low-power phones.
 */
let engine=null;
const MODEL="Qwen2.5-0.5B-Instruct-q4f16_1-MLC";
const say=(type,data={})=>self.postMessage({type,...data});

self.onmessage=async event=>{
  const msg=event.data;
  if(msg?.type==="init"){
    if(engine){say("ready",{model:MODEL});return;}
    try{
      say("progress",{text:"در حال آماده‌سازی موتور WebLLM...",value:0});
      // Official WebLLM CDN ESM entry point. Loaded only when requested.
      const webllm=await import("https://esm.run/@mlc-ai/web-llm");
      engine=await webllm.CreateMLCEngine(MODEL,{
        initProgressCallback:p=>say("progress",{
          text:String(p.text||"در حال دانلود و آماده‌سازی مدل..."),
          value:Number.isFinite(p.progress)?p.progress:0
        })
      });
      say("ready",{model:MODEL});
    }catch(error){
      engine=null;
      say("error",{message:"مدل در این مرورگر فعال نشد: "+String(error?.message||error).slice(0,350)});
    }
    return;
  }
  if(msg?.type==="generate"){
    if(!engine){say("error",{message:"ابتدا مدل محلی را فعال کن."});return;}
    const evidence=msg.snapshot;
    if(!evidence?.game||!evidence?.analysis){say("error",{message:"اول یک مسابقه معتبر انتخاب کن."});return;}
    try{
      const context=JSON.stringify(evidence).slice(0,6500);
      const question=String(msg.question||"نقاط قوت و ضعف آماری دو تیم چیست؟").slice(0,420);
      const system=[
        "You are TAJ, a cautious football research analyst. Reply in Persian.",
        "Only use JSON match evidence provided in this prompt. No browsing or fabricated facts.",
        "When no data exists, explicitly say it is unknown.",
        "No claim of actual tracking, player movements, formations, line-ups, xG, injuries, pressing traps or causal tactical weaknesses without the relevant fields.",
        "Poisson values, if present, are uncalibrated experimental estimates, not facts or guaranteed predictions.",
        "Brief answer: observations with concrete numbers, 2 conditional scouting hypotheses, limitations.",
        "Always distinguish the model's computed estimates from observed scores."
      ].join(" ");
      const response=await engine.chat.completions.create({
        messages:[
          {role:"system",content:system},
          {role:"user",content:"Evidence JSON:\n"+context+"\n\nQuestion:\n"+question}
        ],
        temperature:0.15,max_tokens:330
      });
      const output=response?.choices?.[0]?.message?.content;
      if(typeof output!=="string"||!output.trim())throw new Error("پاسخ متنی از مدل دریافت نشد.");
      say("answer",{text:output.slice(0,5000)});
    }catch(error){
      say("error",{message:"در تولید تحلیل محلی خطا رخ داد: "+String(error?.message||error).slice(0,300)});
    }
  }
};

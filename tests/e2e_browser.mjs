import {chromium} from "playwright";
import fs from "node:fs/promises";

const site="http://127.0.0.1:8765/";
const browser=await chromium.launch({headless:true,args:["--no-sandbox"]});
await fs.mkdir("test-results",{recursive:true});
const messages=[];
async function check(page,viewport){
  page.on("console",message=>{
    if(message.type()==="error")messages.push("console: "+message.text().slice(0,400));
  });
  page.on("pageerror",error=>messages.push("pageerror: "+String(error.message).slice(0,400)));
  page.on("response",response=>{
    const url=response.url();
    if(url.includes("litert")||url.endsWith(".wasm")){
      const record="asset "+response.status()+" "+url;
      console.log(record);messages.push(record);
    }
  });
  page.on("requestfailed",request=>{
    if(request.url().includes("litert")||request.url().includes(".wasm"))
      messages.push("request failed: "+request.url()+" "+request.failure()?.errorText);
  });
  await page.goto(site,{waitUntil:"domcontentloaded",timeout:60000});
  await page.locator(".matchCard").first().waitFor({timeout:40000});
  let count=await page.locator(".matchCard").count();
  if(count!==5)throw Error("Expected 5 real upcoming fixtures; got "+count);
  const names=await page.locator(".matchCard .teamName").allTextContents();
  console.log("Fixtures:",names.slice(0,10).join(" / "));
  // Trigger WebWorker & LiteRT.js WASM, not only model-card download.
  await page.waitForFunction(()=>{
    const status=document.getElementById("neuralStatus")?.textContent||"";
    return status.includes("پیش‌بینی با مدل")||status.includes("LiteRT اجرا نشد")
      ||status.includes("موتور AI هنوز")||status.includes("Worker مدل")||status.includes("ویژگی‌های سایت");
  },null,{timeout:165000});
  const status=await page.locator("#neuralStatus").textContent();
  console.log("Neural status:",status);
  if(!status.includes("پیش‌بینی با مدل"))throw Error("LiteRT numerical inference did not succeed: "+status+"\n"+messages.join("\n"));
  const raw=await page.locator(".neuralProbability strong").allTextContents();
  if(raw.length!==3)throw Error("Expected three local neural forecast bars, got "+raw.length);
  const probs=raw.map(s=>Number(s.replace("٪","").trim()));
  if(probs.some(x=>!Number.isFinite(x))||Math.abs(probs.reduce((a,b)=>a+b,0)-100)>.25)
    throw Error("Neural probabilities invalid: "+JSON.stringify(probs));
  await page.locator("#knowledgeGraph svg").first().waitFor({timeout:40000});
  const modeButtons=page.locator("#knowledgeGraph .graphModes button");
  if(await modeButtons.count()!==2)throw Error("Both graph projection modes are required");
  await modeButtons.nth(1).click();
  await page.locator("#knowledgeGraph .squadGraphCanvas svg").waitFor({timeout:12000});
  if(!(await page.locator("#knowledgeGraph .squadGraphCanvas svg").textContent()).trim())
    throw Error("Club-position-player graph is empty");
  await page.locator("#knowledgeGraph .graphModes button").first().click();
  await page.locator("#knowledgeGraph .graphCanvas svg").waitFor({timeout:12000});
  console.log("Verified opponent and squad knowledge graph modes");
  const rosterPlayers=await page.locator(".squadPlayer").count();
  if(rosterPlayers<20)throw Error("Real squad roster comparison missing: "+rosterPlayers);
  await page.waitForFunction(()=>{
    const rows=document.querySelectorAll("#graphModelResults .graphForecastRow").length;
    const status=document.querySelector("#graphStatus")?.textContent||"";
    return rows===6||status.includes("مدل گرافی در دسترس نیست");
  },null,{timeout:120000});
  const graphRowCount=await page.locator("#graphModelResults .graphForecastRow").count();
  const graphStatus=await page.locator("#graphStatus").textContent();
  console.log("GRAPH AI status:",graphStatus,"forecast rows:",graphRowCount,"squad player count:",rosterPlayers);
  if(graphRowCount!==6)throw Error("Actual second LiteRT graph model failed: "+graphStatus+" "+messages.join("\n"));
  const graphValues=await page.locator("#graphModelResults .graphForecastRow strong").allTextContents();
  const triplets=[graphValues.slice(0,3),graphValues.slice(3,6)];
  for(const values of triplets){
    const p=values.map(x=>Number(x.replace("٪","").trim()));
    if(p.some(x=>!Number.isFinite(x))||Math.abs(p.reduce((a,b)=>a+b,0)-100)>.25)
      throw Error("Graph AI probabilities invalid: "+JSON.stringify(p));
  }
  console.log("GRAPH AI local",graphValues);
  await page.screenshot({path:"test-results/taj-"+viewport+".png",fullPage:true});
  console.log("Verified browser LiteRT CPU forecast:",probs,"sum:",probs.reduce((a,b)=>a+b,0));

  // Once the first real LiteRT inference has initialized the model, cut off
  // the internet. Switching match cards must still infer locally in the same
  // worker with the downloaded model and saved match history.
  await page.context().setOffline(true);
  for(let i=1;i<5;i++){
    await page.locator(".matchCard").nth(i).click();
    await page.waitForFunction(() => {
      const text=document.querySelector("#neuralStatus")?.textContent||"";
      return text.includes("پیش‌بینی با مدل") &&
        document.querySelectorAll(".neuralProbability strong").length===3;
    },null,{timeout:35000});
    const rows=await page.locator(".neuralProbability strong").allTextContents();
    const probability=rows.map(s=>Number(s.replace("٪","").trim()));
    if(Math.abs(probability.reduce((a,b)=>a+b,0)-100)>.25||
       probability.some(x=>!Number.isFinite(x))) {
      throw Error("Offline LiteRT prediction invalid for match "+i+": "+JSON.stringify(probability));
    }
    await page.waitForFunction(()=>{
      return document.querySelectorAll("#graphModelResults .graphForecastRow strong").length===6;
    },null,{timeout:35000});
    const rawGraph=await page.locator("#graphModelResults .graphForecastRow strong").allTextContents();
    const gp=rawGraph.slice(0,3).map(x=>Number(x.replace("٪","").trim()));
    if(gp.some(x=>!Number.isFinite(x))||Math.abs(gp.reduce((a,b)=>a+b,0)-100)>.25)
      throw Error("Offline Graph AI invalid for match "+i+": "+JSON.stringify(gp));
    console.log("OFFLINE GRAPH match",i+1,"LiteRT 42-feature forecast",gp);
    console.log("OFFLINE match",i+1,"LiteRT forecast",probability);
  }
  await page.context().setOffline(false);
}
try{
  const desktop=await browser.newPage({viewport:{width:1440,height:1000},locale:"fa-IR"});
  await check(desktop,"desktop");
  await desktop.close();
  const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,locale:"fa-IR"});
  await check(mobile,"mobile");
  await mobile.close();
}finally{await browser.close();}

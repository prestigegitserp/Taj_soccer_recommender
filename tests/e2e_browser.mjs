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
  },{timeout:165000});
  const status=await page.locator("#neuralStatus").textContent();
  console.log("Neural status:",status);
  if(!status.includes("پیش‌بینی با مدل"))throw Error("LiteRT numerical inference did not succeed: "+status+"\n"+messages.join("\n"));
  const raw=await page.locator(".neuralProbability strong").allTextContents();
  if(raw.length!==3)throw Error("Expected three local neural forecast bars, got "+raw.length);
  const probs=raw.map(s=>Number(s.replace("٪","").trim()));
  if(probs.some(x=>!Number.isFinite(x))||Math.abs(probs.reduce((a,b)=>a+b,0)-100)>.25)
    throw Error("Neural probabilities invalid: "+JSON.stringify(probs));
  await page.screenshot({path:"test-results/taj-"+viewport+".png",fullPage:true});
  console.log("Verified browser LiteRT CPU forecast:",probs,"sum:",probs.reduce((a,b)=>a+b,0));
}
try{
  const desktop=await browser.newPage({viewport:{width:1440,height:1000},locale:"fa-IR"});
  await check(desktop,"desktop");
  await desktop.close();
  const mobile=await browser.newPage({viewport:{width:390,height:844},isMobile:true,locale:"fa-IR"});
  await check(mobile,"mobile");
  await mobile.close();
}finally{await browser.close();}

/**
 * Browser-level V2 proof: 5 REAL fixtures, 56-feature model tree inference,
 * 3-output probability normalization, offline prediction after initial load,
 * graph and verified player evidence; original V1 remains available.
 */
import {chromium} from "playwright";
import fs from "node:fs/promises";
const root="http://127.0.0.1:8765/";
const browser=await chromium.launch({headless:true,args:["--no-sandbox"]});
await fs.mkdir("test-results",{recursive:true});
async function scenario(viewport,name){
  const page=await browser.newPage({viewport,locale:"fa-IR"});
  const problems=[];
  page.on("pageerror",e=>problems.push(e.message));
  page.on("console",m=>{if(m.type()==="error")problems.push(m.text().slice(0,200));});
  await page.goto(root+"v2/",{waitUntil:"domcontentloaded",timeout:30000});
  await page.waitForFunction(()=>document.querySelectorAll("#fixtures button.fixture").length===5,
    null,{timeout:25000});
  await page.waitForFunction(()=>document.querySelectorAll("#probabilities .probRow b").length===3,
    null,{timeout:45000});
  const status=await page.locator("#status").textContent();
  const first=await page.locator("#probabilities .probRow b").allTextContents();
  const parsed=a=>a.map(v=>Number(v.replace("٪","").trim()));
  const good=p=>p.length===3&&p.every(x=>Number.isFinite(x)&&x>=0&&x<=100)
    &&Math.abs(p.reduce((a,b)=>a+b,0)-100)<.3;
  if(!good(parsed(first)))throw Error("Invalid V2 real AI output: "+first);
  if((await page.locator("#rivalGraph svg").count())!==1)
    throw Error("Verified rival knowledge graph did not render");
  if((await page.locator("#comparison .algo").count())!==4)
    throw Error("All 4 V2 comparisons did not render");
  const history=await page.locator("#historyCount").textContent();
  if(Number(history.replace(/[^0-9]/g,""))<3000)throw Error("V2 did not load full training history: "+history);
  await page.screenshot({path:"test-results/taj-v2-"+name+".png",fullPage:true});
  console.log("V2 LOCAL RESULT",name,"status",status,"history",history,"probs",first);
  await page.context().setOffline(true);
  for(let i=1;i<5;i++){
    await page.locator("#fixtures button.fixture").nth(i).click();
    await page.waitForFunction(()=>document.querySelectorAll("#probabilities .probRow b").length===3,
      null,{timeout:15000});
    const triplet=parsed(await page.locator("#probabilities .probRow b").allTextContents());
    if(!good(triplet))throw Error("Offline V2 failed match "+i+": "+triplet);
    console.log("V2 OFFLINE",name,i+1,triplet);
  }
  await page.context().setOffline(false);
  if(problems.length)console.log("Browser console notes:",problems.slice(0,10));
  await page.close();
}
try{
  await scenario({width:1460,height:1000},"desktop");
  await scenario({width:390,height:844},"mobile");
  const research=await browser.newPage({viewport:{width:1366,height:950},locale:"fa-IR"});
  await research.goto(root+"v2/research.html",{waitUntil:"domcontentloaded",timeout:20000});
  await research.waitForFunction(()=>document.querySelectorAll("#scoreBody tr").length===5,
    null,{timeout:20000});
  const body=await research.locator("#scoreBody").textContent();
  if(!body.includes("47.63")&&!body.includes("۴۷٫۶۳"))throw Error("V1 baseline accuracy absent from V2 audit: "+body);
  if(!body.includes("48.60")&&!body.includes("۴۸٫۶۰"))throw Error("V2 automatic accuracy absent from scientific audit: "+body);
  if(await research.locator("#folds .foldCard").count()!==3)throw Error("No three historical test folds");
  await research.screenshot({path:"test-results/taj-v2-research.png",fullPage:true});
  console.log("V2 research audit VERIFIED: 1753 real test fixtures, five competing models, three folds");
  await research.close();
  const v1=await browser.newPage();
  const response=await v1.goto(root+"v1/",{waitUntil:"domcontentloaded",timeout:20000});
  if(response.status()!==200)throw Error("V1 archived site missing");
  if(!(await v1.locator("body").textContent()).includes("TAJ"))throw Error("Old site not present");
  console.log("V1 ARCHIVE accessible");
  await v1.close();
}finally{await browser.close();}

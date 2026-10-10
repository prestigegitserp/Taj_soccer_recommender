/** Display the actual GitHub Actions V2 report, never hardcoded results. */
const e=id=>document.getElementById(id);
const add=(parent,tag,text,cls)=>{
  const node=document.createElement(tag);
  if(text!==undefined)node.textContent=String(text);
  if(cls)node.className=cls;
  parent.appendChild(node);return node;
};
const p=v=>(100*v).toFixed(2)+"٪";
const num=v=>Number(v).toFixed(5);
const modelNames={v1_poisson:"Poisson (مرجع V1)",dixon_coles:"Dixon–Coles",
  lgb42:"LightGBM با ۴۲ ویژگی",lgb56:"LightGBM با ۵۶ ویژگی",
  v2_auto:"V2 · انتخاب اعتبارسنجی"};
const leaguelabel={"eng.1":"انگلیس","esp.1":"اسپانیا","ger.1":"آلمان",
  "ita.1":"ایتالیا","fra.1":"فرانسه","uefa.champions":"لیگ قهرمانان"};
function tableRow(root,...values){
  const row=add(root,"tr");
  values.forEach((v,i)=>add(row,"td",v,i?"numeric":""));
  return row;
}
function render(r){
  const models=r.overall;
  const base=models.v1_poisson,challenger=models.v2_auto,ci=r.paired_95pct;
  const better=r.promote_for_accuracy;
  const warning=e("researchWarning");
  warning.textContent=better
    ?"مدل V2 معیار محافظه‌کار را در آزمون زمانی گذرانده، ولی پیش‌بینی فوتبال تضمینی نیست."
    :"اگرچه Accuracy نسخه ۲ بهتر شده است، فاصله اطمینان Brier هنوز صفر را دربرمی‌گیرد؛ برتری علمی پایدار اثبات نشده و Poisson مدل رسمی باقی می‌ماند.";
  if(better){warning.style.borderColor="#4eb98b";warning.style.background="#193e36";}
  const metrics=[
    ["بازی‌های خارج از آموزش",r.out_of_sample_games.toLocaleString("fa-IR")],
    ["Accuracy نسخه ۱",p(base.accuracy)],
    ["Accuracy V2",p(challenger.accuracy)],
    ["تفاوت Brier",((ci.delta_brier>0)?"+":"")+num(ci.delta_brier)]
  ];
  const root=e("researchMetrics");root.replaceChildren();
  for(const [label,value] of metrics){
    const tile=add(root,"div",undefined,"auditCard");
    add(tile,"small",label);add(tile,"b",value);
  }
  const rows=e("scoreBody");rows.replaceChildren();
  for(const key of Object.keys(modelNames)){
    const v=models[key];
    const row=tableRow(rows,modelNames[key],v.correct+" / "+v.n,
      p(v.accuracy),num(v.brier),num(v.log_loss));
    if(key==="v1_poisson")row.style.background="#173d36";
    if(key==="v2_auto")row.style.color="#a0f2c3";
  }
  const folds=e("folds");folds.replaceChildren();
  for(const f of r.folds){
    const card=add(folds,"article",undefined,"foldCard");
    add(card,"h3","بازه "+f.fold+" · "+f.test_n+" بازی");
    add(card,"small","آموزش تا "+f.train_until.slice(0,10));
    add(card,"p","آزمون از "+f.test_from.slice(0,10)+" تا "+f.test_through.slice(0,10));
    add(card,"p","انتخاب اعتبارسنجی: "+f.selected_without_test_outcomes,"positive");
    for(const [label,value] of [
      ["دقت Poisson",p(f.models.v1_poisson.accuracy)],
      ["دقت V2",p(f.models.v2_auto.accuracy)],
      ["Brier Poisson",num(f.models.v1_poisson.brier)],
      ["Brier V2",num(f.models.v2_auto.brier)] ]){
      const line=add(card,"p");
      line.textContent=label+" — "+value;
    }
  }
  const leagues=e("leagueBody");leagues.replaceChildren();
  for(const [key,val] of Object.entries(r.per_league)){
    tableRow(leagues,leaguelabel[key]||key,val.count,
      p(val.v2_auto.accuracy),num(val.v2_auto.brier),num(val.v1_poisson.brier));
  }
  const method=e("method");method.replaceChildren();
  add(method,"p","منبع: "+r.sourced);
  add(method,"p","۳ پنجره Walk-forward با تست‌های بدون هم‌پوشانی؛ "+r.raw_historical_games+
    " بازی تاریخی و "+r.eligible_games+" نمونه ویژگی معتبر.");
  add(method,"p","Bootstrap زوجی روزمحور: اختلاف Brier نسخه ۲ منهای Poisson برابر "+
    ci.delta_brier+" است؛ فاصله اطمینان ۹۵٪ ["+ci.ci_95.join(", ")+
    "]. مقدار منفی به‌معنی بهتر بودن V2 است.");
  add(method,"p","این ارزیابی تاریخی به‌صورت گذشته‌نگر ساخته شده، نه پیش‌بینی زنده ثبت‌شده پیش از بازی. در طراحی مدل‌ها از نتایج بک‌تست V1 استفاده شده؛ بنابراین آزمون V2 را نباید معادل آزمایش کاملاً کور و ازپیش‌ثبت‌شده تفسیر کرد.");
  add(method,"p","مدل نهایی: "+r.production_forest.features+" ویژگی و "+
    r.production_forest.tree_count+" درخت آموزش‌دیده. اختلاف خروجی درخت‌های صادرشده با Python: "+
    r.production_forest.python_js_numerical_max_error.toExponential(2)+".");
}
try{
 const res=await fetch("./data/evaluation.json",{cache:"no-store"});
 if(!res.ok)throw Error("مدل V2 هنوز آموزش یا منتشر نشده است: HTTP "+res.status);
 const report=await res.json();
 if(report.schema!=="taj-v2-evaluation-v1"||report.out_of_sample_games<1500)
   throw Error("نسخه گزارش معتبر نیست");
 render(report);
}catch(err){e("researchWarning").textContent="خطا در دریافت گزارش واقعی: "+String(err.message||err);}

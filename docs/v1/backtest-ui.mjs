/** Transparent human-readable report sourced ONLY from saved walk-forward JSON. */
const $=id=>document.getElementById(id);
const pretty=(v,dec=2)=>Number.isFinite(v)?Number(v).toFixed(dec):"—";
const perc=(v,dec=2)=>Number.isFinite(v)?pretty(v*100,dec)+"٪":"—";
const label={
  frequency:"خط مبنای فراوانی",
  poisson:"Poisson",
  mlp_20:"شبکه عصبی ۲۰ویژگی",
  graph_42:"شبکه گرافی ۴۲ویژگی",
  graph_ensemble:"Ensemble گراف + Poisson"
};
const modelKeys=["frequency","mlp_20","graph_42","graph_ensemble","poisson"];
const leagueNames={
  "eng.1":"لیگ برتر انگلیس","esp.1":"لالیگا اسپانیا","ger.1":"بوندس‌لیگا آلمان",
  "ita.1":"سری آ ایتالیا","fra.1":"لیگ ۱ فرانسه","uefa.champions":"لیگ قهرمانان اروپا"
};
const create=(tag,text,className)=>{
  const el=document.createElement(tag);
  if(className)el.className=className;
  if(text!==undefined)el.textContent=String(text);
  return el;
};
function children(root,...items){root.append(...items);return root;}
function tableRow(...values){
  const tr=create("tr");
  for(const value of values){tr.appendChild(create("td",value));}
  return tr;
}
function render(report){
  $("sourceInfo").textContent="منبع: نتایج واقعی پایان‌یافته ESPN · "
    +report.input_dataset_games.toLocaleString("fa-IR")+" مسابقه خام · "
    +report.eligible_pre_match_examples.toLocaleString("fa-IR")
    +" نمونه قابل مدل‌سازی · "
    +report.oos_unique_fixtures.toLocaleString("fa-IR")
    +" مسابقه آزمونِ کاملاً خارج از آموزش · تاریخ آزمون: "
    +report.oos_from.slice(0,10)+" تا "+report.oos_to.slice(0,10)
    +" · تاریخ تهیه گزارش: "+new Date(report.generated_at).toLocaleString("fa-IR");
  const overall=report.overall;
  const p=overall.poisson,b=overall.graph_ensemble;
  const delta=report.paired_uncertainty.delta_brier;
  const metrics=[
    [report.oos_unique_fixtures.toLocaleString("fa-IR"),"مسابقه واقعی، آزمون بدون آموزش"],
    [perc(b.accuracy),"پیش‌بینی صحیح "+b.correct+" بازی"],
    [pretty(b.brier,5),"مدل ترکیبی، کمتر بهتر"],
    [(delta>0?"+":"")+pretty(delta,5),"Ensemble − Poisson؛ مقدار مثبت بدتر"]
  ];
  const metricsRoot=$("keyMetrics");
  metricsRoot.replaceChildren();
  for(const [value,details] of metrics){
    const item=create("div",undefined,"metric");
    children(item,create("small",details),create("strong",value));
    metricsRoot.appendChild(item);
  }
  const modelBody=$("modelTable").querySelector("tbody");
  modelBody.replaceChildren();
  for(const key of modelKeys){
    const m=overall[key];
    const row=tableRow(label[key],perc(m.accuracy),pretty(m.brier,5),
      pretty(m.log_loss,5),m.correct+" / "+m.n);
    if(key==="poisson")row.className="winner";
    for(const cell of [...row.children].slice(1))cell.className="ltr";
    if(key==="poisson"){
      const tag=create("span"," · بهترین Brier در این آزمون","tag");
      row.firstChild.appendChild(tag);
    }
    modelBody.appendChild(row);
  }
  const barRoot=$("modelBars");
  barRoot.replaceChildren();
  for(const key of modelKeys){
    const entry=create("div",undefined,"barrow");
    const track=create("div",undefined,"track");
    const fill=create("div",undefined,"fill");
    // A visual comparison of 1 - multiclass Brier/2. All values fall in [0,1].
    fill.style.width=pretty(100*(1-overall[key].brier/2),3)+"%";
    track.appendChild(fill);
    children(entry,create("span",label[key]),track,
      create("strong",perc(overall[key].accuracy)));
    barRoot.appendChild(entry);
  }
  const ci=report.paired_uncertainty;
  const reliable=ci.statistically_resolved_at_95pct;
  const favorable=ci.ci_95[1]<0;
  const ciBox=create("div",undefined,"ci "+(reliable&&favorable?"ok":"warning"));
  children(ciBox,
    create("span",reliable&&favorable?"برتری آماری تأییدشده":"برتری آماری اثبات نشده","tag"),
    create("strong",(delta>=0?"+":"")+pretty(delta,5)),
    create("p","فاصله اطمینان ۹۵٪: ["+ci.ci_95.map(v=>(v>=0?"+":"")+pretty(v,5)).join(" , ")+"]"),
    create("p","روش: "+ci.bootstrap_samples.toLocaleString("fa-IR")
      +" بار نمونه‌گیری Bootstrap از روزهای مسابقه، با مقایسه جفتی روی همان بازی‌ها."),
    create("p",reliable&&favorable
      ?"در این ارزیابی، احتمالاً Ensemble از Poisson بهتر است."
      :"چون فاصله اطمینان صفر را دربرمی‌گیرد، بهبود قابل اتکایی ثابت نشده است.")
  );
  $("uncertainty").replaceChildren(ciBox);
  const cal=$("calibration");
  cal.replaceChildren();
  for(const item of report.ensemble_calibration){
    const box=create("div",undefined,"calItem");
    children(box,create("span",Math.round(item.range[0]*100)+"–"+Math.round(item.range[1]*100)+"٪ اعتماد"),
      create("b",item.n?perc(item.real_accuracy,1):"—"),
      create("span","درست از "+item.n+" پیش‌بینی"),
      create("span","اعتماد متوسط: "+(item.n?perc(item.mean_predicted_confidence,1):"—"))
    );
    cal.appendChild(box);
  }
  const folds=$("foldCards");
  folds.replaceChildren();
  for(const fold of report.folds){
    const f=create("article",undefined,"fold");
    children(f,create("h3","بازه "+fold.fold+" · آزمون از "+fold.test_from.slice(0,10)),
      create("p","پایان آموزش: "+fold.train_through.slice(0,10)
        +" · پایان اعتبارسنجی: "+fold.validation_through.slice(0,10)
        +" · پایان آزمون: "+fold.test_through.slice(0,10)));
    for(const [key,value] of [
      ["آموزش",fold.train_n+" بازی"],
      ["اعتبارسنجی",fold.validation_n+" بازی"],
      ["آزمون کاملاً دیده‌نشده",fold.test_n+" بازی"],
      ["دقت ترکیبی",perc(fold.models.graph_ensemble.accuracy)],
      ["دقت Poisson",perc(fold.models.poisson.accuracy)],
      ["Brier ترکیبی",pretty(fold.models.graph_ensemble.brier,5)],
      ["Brier Poisson",pretty(fold.models.poisson.brier,5)],
      ["وزن انتخاب‌شده شبکه گرافی",perc(fold.validated_graph_ensemble_weight,0)],
    ]){
      const row=create("div",undefined,"datum");
      children(row,create("span",key),create("strong",value));
      f.appendChild(row);
    }
    folds.appendChild(f);
  }
  const leagues=$("leagueTable").querySelector("tbody");
  leagues.replaceChildren();
  for(const [key,info] of Object.entries(report.per_league).sort((a,b)=>b[1].n-a[1].n)){
    const row=tableRow(leagueNames[key]||key,info.n,perc(info.models.graph_ensemble.accuracy),
      pretty(info.models.graph_ensemble.brier,5),pretty(info.models.poisson.brier,5));
    for(const cell of [...row.children].slice(1))cell.className="ltr";
    leagues.appendChild(row);
  }
  const limitations=$("limitations");
  limitations.replaceChildren();
  const paragraph=create("p","این آزمایش ارزیابی تاریخیِ رو‌به‌جلو است، نه پیش‌بینی ثبت‌شده در روز وقوع مسابقه. مدل در هر بازه از نو آموزش دیده است و نتایج آزمون به تنظیم وزن و کالیبراسیون همان بازه وارد نشده‌اند.");
  limitations.appendChild(paragraph);
  const list=create("ul");
  for(const limit of report.limitations||[]){
    list.appendChild(create("li",limit));
  }
  limitations.appendChild(list);
}
try{
  const resp=await fetch("./data/backtest.json",{cache:"no-store"});
  if(!resp.ok)throw Error("گزارش بک‌تست هنوز در نسخه منتشرشده وجود ندارد (HTTP "+resp.status+").");
  const data=await resp.json();
  if(data?.schema!=="taj-walkforward-v1"||data.oos_unique_fixtures<1000)
    throw Error("نسخه یا داده گزارش معتبر نیست.");
  render(data);
}catch(e){
  $("error").textContent="مشکل در نمایش گزارش: "+String(e.message||e);
  $("sourceInfo").textContent="فایل JSON منتشرشده را بررسی کنید.";
}

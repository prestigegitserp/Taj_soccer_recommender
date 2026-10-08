"""Deterministic explainable recommendation ranker, no LLM hallucination."""
from __future__ import annotations

PLANS = {
  "left": ("Switch into left half-space","جابجایی سریع توپ به نیم‌فضای چپ",
           "بالانس دفاع انتقالی و پوشش پشت فول‌بک"),
  "centre": ("Third-man central combinations","ترکیب نفر سوم در کانال مرکزی",
           "فشردگی دفاع حریف و ریسک ازدست‌دادن توپ در مرکز"),
  "right": ("Attack right channel","حمله با برتری عددی از کانال راست",
           "جلوگیری از جا ماندن مدافع کناری"),
}

def recommend(findings:list[dict], *, risk_aversion:float=.4, min_windows:int=4):
    """Only evidence-backed candidates; scores are relative heuristic priorities (0..100)."""
    result=[]
    for f in findings:
        if f.get("kind")!="spatial_exposure" or int(f.get("n_independent_windows",0))<min_windows:
            continue
        zone=f.get("zone")
        if zone not in PLANS:continue
        en,fa,risk=PLANS[zone]
        access=float(f["severity"])
        n=int(f["n_independent_windows"])
        support=min(1.0,n/10)
        score=100*max(0,min(1,.65*access+.35*support-risk_aversion*.12))
        result.append(dict(plan_id="plan-"+zone,title=en,title_fa=fa,
            priority=round(score,1),confidence_label="exploratory",
            risk_fa=risk,reason_fa=f["statement_fa"],evidence=f["evidence"],
            sample_windows=n,expected_goal_lift=None,
            caveat_fa="امتیاز، احتمال موفقیت یا افزایش قطعی xG نیست؛ نیازمند بررسی ویدیویی و اعتبارسنجی است."))
    return sorted(result,key=lambda x:x["priority"],reverse=True)

"""Evidence-linked recommender. Outputs candidates, NEVER predicted probability of success."""
from __future__ import annotations
from typing import Any

# Maps opposing access in our defended third to actions for the *attacking* side.
# Avoid claiming this tactic must work; rank within a reviewed, explainable playbook.
PLANS={
 "left":("Attack accessible left lane","نفوذ هدفمند از کانال چپ",
   "در صورت از دست‌دادن توپ، پوشش پشت مدافع کناری حفظ شود.",
   "overload_and_switch"),
 "central":("Third-man combinations centrally","ترکیب نفر سوم در فضاهای مرکزی",
   "تراکم مرکز و خطر ضدحمله پس از اشتباه در پاس.",
   "third_man_run"),
 "centre":("Third-man combinations centrally","ترکیب نفر سوم در فضاهای مرکزی",
   "تراکم مرکز و خطر ضدحمله پس از اشتباه در پاس.",
   "third_man_run"),
 "right":("Attack accessible right lane","ایجاد برتری عددی در کانال راست",
   "مدافع کناری باید پوشش انتقال دفاعی داشته باشد.",
   "wide_overload"),
}

def recommend(findings:list[dict],*,risk_aversion:float=.4,min_windows:int=4,
              max_candidates:int=6):
    if not 0<=risk_aversion<=1:raise ValueError("risk_aversion must be within [0,1]")
    result=[]
    for f in findings:
        if f.get("kind")!="spatial_exposure" or int(f.get("n_independent_windows",0))<min_windows:
            continue
        zone=f.get("zone")
        if zone not in PLANS:continue
        en,fa,risk,playbook=PLANS[zone]
        severity=max(0,min(1,float(f.get("severity",0))))
        n=int(f["n_independent_windows"])
        support=min(1.0,n/12.0)
        priority=100*max(0,min(1,.62*severity+.38*support-risk_aversion*.12))
        result.append({"plan_id":playbook+"-"+zone,"title":en,"title_fa":fa,
            "priority":round(priority,1),"confidence_label":"exploratory",
            "risk_fa":risk,"reason_fa":f.get("statement_fa","Spatial exposure hypothesis"),
            "evidence":f.get("evidence",[])[:60],"sample_windows":n,
            "expected_goal_lift":None,"estimated_win_probability":None,
            "caveat_fa":"رتبه نسبی و نه احتمال موفقیت؛ نیازمند تأیید تحلیلگر و اعتبارسنجی مسابقات مستقل."})
    return sorted(result,key=lambda r:(r["priority"],r["sample_windows"]),reverse=True)[:max_candidates]

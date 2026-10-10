"""Export portable, self-contained interactive scouting report, sanitizing text."""
from __future__ import annotations
import html,json
from pathlib import Path
from .plotting import pitch_figure
from .schema import at_frame

def export_html(report:dict, tracking, ball, output="outputs/tactical_report.html"):
    sample=report["sample_frame"]
    frame=at_frame(tracking,int(sample["period"]),str(sample["frame_id"]))
    b=ball[(ball.period==int(sample["period"])) &
            (ball.frame_id.astype(str)==str(sample["frame_id"]))] if ball is not None and not ball.empty else None
    fig=pitch_figure(frame,b,title="Taj Tactical Intelligence — Evidence sample")
    viz=fig.to_html(full_html=False,include_plotlyjs=True)
    esc=lambda x:html.escape(str(x),quote=True)
    cards=[]
    for r in report["recommendations"]:
        snippets="".join(
            f"<span class='stamp'>{esc(e.get('match_id'))} / P{esc(e.get('period'))} / {esc(e.get('t_s'))}s</span>"
            for e in r.get("evidence",[])[:12])
        cards.append(f"""<article><h3>{esc(r.get('title_fa'))}</h3>
          <p>{esc(r.get('reason_fa'))}</p><p class='risk'>ریسک: {esc(r.get('risk_fa'))}</p>
          <small>رتبه نسبی {esc(r.get('priority'))}/100 · {esc(r.get('sample_windows'))} بازه</small>
          <div>{snippets}</div></article>""")
    warning="".join(f"<p>⚠ {esc(w)}</p>" for w in report.get("warnings",[]))
    html_page=f"""<!doctype html><html lang='fa' dir='rtl'><meta charset='utf-8'>
    <meta name='viewport' content='width=device-width,initial-scale=1'>
    <title>TAJ — {esc(report['match_id'])}</title>
    <style>body{{font:15px system-ui,Tahoma;background:#06101d;color:#e7faf6;margin:0;padding:22px}}
    main{{max-width:1200px;margin:auto}}h1{{color:#26e0b5}}p{{line-height:1.9}}.panel,article{{background:#102438;
    border:1px solid #255267;padding:18px;margin:16px 0;border-radius:16px}}
    .risk{{color:#f2c183}}.stamp{{display:inline-block;border:1px solid #315465;padding:6px;margin:4px;
    border-radius:10px;color:#afdae0}}.muted{{color:#b5c8d6}}</style>
    <main><h1>TAJ · Tactical Intelligence</h1>
    <p class='muted'>Dataset: {esc(report['provider'])} · Match: {esc(report['match_id'])} ·
      {'SYNTHETIC DEMO' if report.get('synthetic') else 'HISTORICAL TRACKING'}</p>
    <section class='panel'>{warning}<p>{esc(report.get('disclaimer_fa',''))}</p></section>
    {viz}<h2>پیشنهادهای قابل بازبینی</h2>
    {''.join(cards) or '<p>شواهد کافی برای پیشنهاد وجود ندارد.</p>'}
    <h2>اطلاعات روش‌شناسی</h2><pre class='panel' dir='ltr' style='white-space:pre-wrap'>{esc(json.dumps(report.get('quality',{}),ensure_ascii=False,indent=2))}</pre>
    </main></html>"""
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(html_page,encoding="utf-8")
    return output

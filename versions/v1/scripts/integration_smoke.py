"""Opt-in network integration smoke test; run on Colab / Python with network access."""
from __future__ import annotations
from taj.ingest import load_idsse,write_bundle
from taj.events import link_events,event_summary
from taj.analysis import build_report

def main():
    tracking,ball,events=load_idsse("J03WMX",limit=60,sample_rate=1.,with_events=True)
    q=write_bundle(tracking,ball,events,"data/integration_idsse")
    print("Tracking QC:",q)
    print("Ball positions:",len(ball),"IDSSE event rows:",len(events),"columns:",list(events.columns)[:25])
    assert len(events)>0
    report=build_report(tracking,ball,events,min_windows=4)
    print("Output schema:",report["schema_version"],"findings:",len(report["findings"]))
    assert len(tracking)>0 and set(tracking.team)=={"Home","Away"}

if __name__=="__main__":main()

"""Network smoke test for a real SkillCorner JSONL LFS match.
Run after cloning https://github.com/SkillCorner/opendata into external/opendata.
"""
from taj.ingest import load_skillcorner_local,write_bundle
from taj.analysis import build_report

def main():
    players,ball,events=load_skillcorner_local("1925299","external/opendata",limit=100,stride=10)
    stats=write_bundle(players,ball,events,"data/integration_skillcorner")
    print("SkillCorner QC:",stats)
    print("Ball positions:",len(ball),"dynamic events:",len(events))
    assert stats["rows"]>0 and {"Home","Away"}.issubset(set(stats["teams"]))
    assert stats["observed_rate"] is not None
    assert len(events)>0
    report=build_report(players,ball,events,
                        defending="Home",direction_by_period={1:-1},
                        min_windows=4)
    print("Report schema:",report["schema_version"],"findings:",len(report["findings"]))

if __name__=="__main__":main()

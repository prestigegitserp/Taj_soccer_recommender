"""Evidence/provenance requirements for public football squad and boxscores."""
import importlib.util
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]/"scripts"
spec=importlib.util.spec_from_file_location("build_match_knowledge",HERE/"build_match_knowledge.py")
import sys
sys.path.insert(0,str(HERE))
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_roster_is_squad_not_eleven():
    fixture={"athletes":[{"position":"D","items":[
      {"id":"1","displayName":"Player A","position":{"abbreviation":"D"},"jersey":"2"},
      {"id":"2","displayName":"Player B","position":{"abbreviation":"F"},"jersey":"9"}]}]}
    roster=module.parse_roster(fixture)
    assert len(roster)==2
    assert roster[0]["name"]=="Player A"
    assert roster[1]["position"]=="F"

def test_missing_boxscore_not_imputed():
    assert module.parse_summary({})=={}
    assert module.normalize_stats([])=={}
    assert module.safe_number(None) is None
    assert module.safe_number("not recorded") is None
    assert module.safe_number("51.2%")==51.2

def test_summary_team_statistics_are_sourced_by_ids():
    payload={"boxscore":{"teams":[
      {"team":{"id":"001"},"statistics":[
          {"name":"possessionPct","displayValue":"54.2%"},
          {"name":"totalShots","displayValue":"14"},
          {"name":"shotsOnTarget","displayValue":"7"}]},
      {"team":{"id":"002"},"statistics":[{"name":"shotsOnTarget","displayValue":"2"}]}
    ]}}
    stats=module.parse_summary(payload)
    assert stats["001"]["possession"]==54.2
    assert stats["001"]["shots"]==14
    assert stats["001"]["shots_on_target"]==7
    assert "possession" not in stats["002"]
    assert stats["002"]["shots_on_target"]==2

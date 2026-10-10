import json
import pytest
from pathlib import Path
from taj.ingest import load_skillcorner_local

def test_skillcorner_lfs_pointer_refused(tmp_path):
    folder=tmp_path/"1925299";folder.mkdir()
    (folder/"1925299_match.json").write_text("{}")
    (folder/"1925299_tracking_extrapolated.jsonl").write_text("version https://git-lfs.github.com/spec/v1\n")
    with pytest.raises(ValueError,match="Git LFS pointer"):
        load_skillcorner_local("1925299",tmp_path)

def test_skillcorner_detected_flag_and_trackable_id(tmp_path):
    folder=tmp_path/"1925299";folder.mkdir()
    metadata={"home_team":{"id":1},"away_team":{"id":2},
              "players":[{"id":99,"trackable_object":1001,"team_id":1},
                         {"id":88,"trackable_object":2001,"team_id":2}]}
    (folder/"1925299_match.json").write_text(json.dumps(metadata))
    lines=[]
    for idx in range(4):
        lines.append(json.dumps({"frame":idx,"period":1,"timestamp":idx*.1,
            "ball_data":{"x":0.,"y":0.},
            "player_data":[{"player_id":1001,"x":-10.,"y":2.,"is_detected":True},
                           {"player_id":2001,"x":10.,"y":-2.,"is_detected":False}]}))
    # Ensure >500 bytes to emulate an actual dataset instead of a Git LFS pointer.
    (folder/"1925299_tracking_extrapolated.jsonl").write_text("\n".join(lines)+"\n")
    a,b,e=load_skillcorner_local("1925299",tmp_path,limit=4,stride=1)
    assert set(a.team)=={"Home","Away"}
    assert a[a.team=="Away"].is_detected.eq(False).all()
    assert a[a.team=="Home"].is_detected.eq(True).all()
    assert len(b)==4

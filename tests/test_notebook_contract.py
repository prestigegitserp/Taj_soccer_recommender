from pathlib import Path
import json

def test_notebook_is_structurally_valid_and_has_colab_real_data_path():
    nb=json.loads(Path("notebooks/Taj_Colab.ipynb").read_text(encoding="utf-8"))
    assert nb["nbformat"]==4
    ids=[c["id"] for c in nb["cells"]]
    assert len(ids)==len(set(ids))
    all_source="\n".join("".join(c["source"]) for c in nb["cells"])
    assert "RUN_IDSSE" in all_source
    assert "RUN_SKILLCORNER" in all_source
    assert "build_viewer_payload" in all_source
    assert "walk_forward_brier" in all_source

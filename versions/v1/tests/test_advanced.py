import numpy as np
import pytest
from taj.advanced import SpatialConfig,pitch_control,pass_interception,zone_risk
from taj.episodes import analyse_frames
from taj.schema import synthetic_match,at_frame

def test_advanced_map_finite_bounded_and_shape():
    tr,_=synthetic_match(frames=2)
    result=pitch_control(at_frame(tr,1,"0"),SpatialConfig(grid_m=7))
    a=result["home_access"]
    assert a.ndim==2 and a.min()>=0 and a.max()<=1
    assert a.shape==(len(result["y"]),len(result["x"]))
    assert result["coverage"]["home"]==11

def test_symmetry_for_mirrored_teams():
    tr,_=synthetic_match(frames=1)
    frame=at_frame(tr,1,"0")
    pc=pitch_control(frame,SpatialConfig(grid_m=5))["home_access"]
    clone=frame.copy()
    clone.team=clone.team.map({"Home":"Away","Away":"Home"})
    mirror=pitch_control(clone,SpatialConfig(grid_m=5))["home_access"]
    np.testing.assert_allclose(pc+mirror,1,atol=1e-8)

def test_zone_risk_and_direction_contract():
    tr,_=synthetic_match(frames=1)
    control=pitch_control(at_frame(tr,1,"0"))
    assert len(zone_risk(control,"Home",-1))==3
    with pytest.raises(ValueError):zone_risk(control,"Home",0)

def test_interception_margin_stable():
    tr,_=synthetic_match(frames=1)
    f=at_frame(tr,1,"0")
    out=pass_interception(f,(-10,-4),(25,10),"Away")
    assert np.isfinite(out["interception_margin_s"])
    assert out["distance_m"]>30

def test_episodes_requires_explicit_direction():
    tr,_=synthetic_match(frames=8)
    with pytest.raises(ValueError,match="direction"):analyse_frames(tr)
    data,findings=analyse_frames(tr,direction_by_period={1:-1},frame_stride=4,min_windows=10)
    assert len(data)>0 and findings==[]

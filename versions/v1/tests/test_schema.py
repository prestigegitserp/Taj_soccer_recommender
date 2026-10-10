import numpy as np
import pandas as pd
import pytest
from taj.schema import synthetic_match,validate_tracking,add_velocity,at_frame

def test_demo_has_all_players_and_valid_coordinates():
    tracking,ball=synthetic_match(frames=5)
    q=validate_tracking(tracking)
    assert q["rows"]==5*22
    assert q["frames"]==5
    assert q["observed_rate"]==1
    assert len(ball)==5

def test_invalid_out_of_bounds_rejected():
    tracking,_=synthetic_match(frames=1)
    tracking.loc[tracking.index[0],"x"]=1000
    with pytest.raises(ValueError,match="outside"):validate_tracking(tracking)

def test_velocity_never_crosses_period_boundaries():
    d=pd.DataFrame({"match_id":["x"]*4,"period":[1,1,2,2],
        "team":["Home"]*4,"player_id":["p"]*4,"t_s":[0.,.2,0.,.2],
        "x":[0.,1.,25.,26.],"y":[0.]*4})
    out=add_velocity(d,max_speed=10.)
    assert np.isnan(out.iloc[2].vx)
    assert np.isclose(out.iloc[3].vx,5)

def test_missing_frame_is_error():
    tr,_=synthetic_match(frames=1)
    with pytest.raises(ValueError,match="Frame"):at_frame(tr,2,"0")

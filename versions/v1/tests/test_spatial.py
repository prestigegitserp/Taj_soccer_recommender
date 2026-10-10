import numpy as np
import pandas as pd
from taj.schema import synthetic_match,at_frame
from taj.spatial import pitch_access,passing_lane,shape_metrics,zonal_access,grid

def test_pitch_access_has_complementary_indices():
    tr,_=synthetic_match(frames=2)
    frame=at_frame(tr,1,"0")
    home,away,X,Y=pitch_access(frame,step=7)
    assert home.shape==X.shape
    assert np.all(np.isfinite(home))
    assert np.all((home>=0)&(home<=1))
    np.testing.assert_allclose(home+away,1)
    assert len(zonal_access(home,X,Y,team="Home",direction=1))==3

def test_home_is_stronger_at_home_players():
    tr,_=synthetic_match(frames=1)
    fr=at_frame(tr,1,"0")
    h,a,X,Y=pitch_access(fr,step=2.)
    player=fr[fr.team=="Home"].iloc[0]
    idx=np.argmin((X-player.x)**2+(Y-player.y)**2)
    assert h.ravel()[idx]>.5

def test_pass_lane_detects_blocker():
    defenders=pd.DataFrame([{"x":5.0,"y":0.1}])
    blocked=passing_lane((0,0),(10,0),defenders)
    assert not blocked["open"] and blocked["blockers"]==1
    assert passing_lane((0,0),(0,10),defenders)["open"]

def test_team_shape_has_width_length():
    tr,_=synthetic_match(frames=1)
    shp=shape_metrics(at_frame(tr,1,"0"))
    assert set(shp)=={"Home","Away"}
    assert shp["Home"]["width_m"]>30

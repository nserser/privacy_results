import numpy as np
from scalecert.privacy import chi2_mass, privacy_profile_mass, epsilon_for_delta

def test_identical():
    q=np.array([.2,.3,.5])
    assert chi2_mass(q,q)==0
    assert max(x['delta_symmetric'] for x in privacy_profile_mass(q,q,[0,.1,1]))==0
    assert epsilon_for_delta(q,q,1e-5)==0

from scalecert.privacy import privacy_profile_with_tv_defect, epsilon_upper_for_delta_with_tv_defect

def test_tv_defect_makes_privacy_more_conservative():
    p=np.array([.51,.49]); q=np.array([.49,.51])
    base=privacy_profile_mass(p,q,[.5])[0]['delta_symmetric']
    up=privacy_profile_with_tv_defect(p,q,[.5],1e-4,2e-4)[0]['delta_symmetric_upper']
    assert up >= base
    e0=epsilon_for_delta(p,q,1e-2)
    eu=epsilon_upper_for_delta_with_tv_defect(p,q,1e-2,1e-5,1e-5)
    assert eu >= e0

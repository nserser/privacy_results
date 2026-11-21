import math
from scalecert.theory_certificate import contraction_constants, finite1_chi2_upper, symmetric_delta_from_forward_chi2


def test_contractfix_constants_nonvacuous():
    c=contraction_constants(.5,.55,.05,1.0)
    assert 0<c['gamma']<1
    assert c['lambda']>0
    assert abs(c['rho_lsi_lower']-.5*math.exp(-1.1))<1e-14


def test_finite1_bound_positive_and_scales():
    a=finite1_chi2_upper(32,1,2,.5,.55,.05,1.0)['chi2_final_upper']
    b=finite1_chi2_upper(64,1,2,.5,.55,.05,1.0)['chi2_final_upper']
    assert a>0 and b>0 and b<a


def test_symmetric_dp_transfer():
    assert symmetric_delta_from_forward_chi2(1e-4,.5)>0

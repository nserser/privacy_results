import numpy as np
from scalecert.cc_solver import Grid1D, ContinuousLangevin1D
from scalecert.privacy import chi2_mass


def test_generator_conservative_and_positive_offdiag():
    g=Grid1D.make(8,401); m=ContinuousLangevin1D(g,.5,.55,.05,1.0)
    d=m.diagnostics(2)
    for c in ['0','1','2']:
        assert d[c]['max_abs_column_sum']<1e-9
        assert d[c]['min_offdiag']>=-1e-12


def test_numerical_stationary_is_machine_stationary():
    g=Grid1D.make(10,501); m=ContinuousLangevin1D(g,.5,.55,.05,1.0)
    q=m.numerical_stationary_mass(); q1,e=m.transition(q,0)
    assert e==0
    assert abs(q1.sum()-1)<1e-10
    assert np.abs(q1-q).sum()<1e-11


def test_positive_tails_preserved_reverse_chi2_finite():
    g=Grid1D.make(12,1201); m=ContinuousLangevin1D(g,.5,.55,.05,1.0,substeps=16)
    q=m.numerical_stationary_mass(); p,_=m.transition(q,1); q1,_=m.transition(q,0)
    assert np.all(p>0) and np.all(q1>0)
    assert np.isfinite(chi2_mass(q1/q1.sum(),p/p.sum()))

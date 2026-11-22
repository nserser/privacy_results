import numpy as np
from scalecert.cc_solver import Grid1D, ContinuousLangevin1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D, sgld_noise_std, dpsgd_noise_std, normal_interval_prob


def test_discrete_probability_ledger():
    g=Grid1D.make(8,201)
    k=DiscreteGaussianKernel1D(g,.5,.55,.05,.1,sgld_noise_std(.1))
    d=k.diagnostics(1)
    assert d['0']['max_column_probability_error']<1e-12
    q=ContinuousLangevin1D(g,.5,.55,.05,1).numerical_stationary_mass()
    out,lost=k.transition(q,0)
    assert abs(out.sum()+lost-1)<1e-12


def test_noise_multiplier_mapping():
    assert abs(dpsgd_noise_std(.1,.05,2.0)-.01)<1e-15


def test_tail_interval_probabilities_are_positive():
    lo=np.array([8.0,9.0,-10.0]); hi=lo+.1
    p=normal_interval_prob(lo,hi,0.0,1.0)
    assert np.all(p>0)

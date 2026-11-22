from scalecert.cc_solver import Grid1D, ContinuousLangevin1D
from scalecert.hidden_support import evolve_hidden_epoch, evolve_absent_epoch
from scalecert.privacy import normalize_if_close, chi2_mass, fit_power_exponent


def test_corrected_hidden_support_symmetric_scaling():
    fwd=[]; rev=[]
    for M in [16,32,64]:
        g=Grid1D.make(8,301); model=ContinuousLangevin1D(g,.5,.55,.05,1.0); q0=model.numerical_stationary_mass()
        p=evolve_hidden_epoch(q0,model.transition,M,1,1).mass
        q=evolve_absent_epoch(q0,model.transition,M).mass
        p,_=normalize_if_close(p,1e-7); q,_=normalize_if_close(q,1e-7)
        fwd.append(chi2_mass(p,q)); rev.append(chi2_mass(q,p))
    assert fit_power_exponent([16,32,64],fwd)['beta']>1.8
    assert fit_power_exponent([16,32,64],rev)['beta']>1.8

from __future__ import annotations

import math
import numpy as np


def impulse_curve(initial_mass, transition, max_age:int, chi2_fn):
    """One c=1 interval followed by r common c=0 intervals, r=0..max_age-1."""
    p,_=transition(np.asarray(initial_mass,float),1)
    q,_=transition(np.asarray(initial_mass,float),0)
    d=[]
    for _ in range(int(max_age)):
        d.append(float(chi2_fn(p,q)))
        p,_=transition(p,0); q,_=transition(q,0)
    return np.asarray(d,float)


def adaptive_impulse_curve(initial_mass, transition, initial_age:int, chi2_fn,
                           tail_ratio_tol:float=1e-6, max_age:int=4096,
                           tail_window:int=4):
    """Extend impulse response until the privacy-memory tail is negligible.

    The stopping certificate requires the last ``tail_window`` values to be at
    most ``tail_ratio_tol*d0``.  The state is evolved incrementally, so extension
    does not repeat earlier work.
    """
    initial_age=max(1,int(initial_age)); max_age=max(initial_age,int(max_age))
    p,_=transition(np.asarray(initial_mass,float),1)
    q,_=transition(np.asarray(initial_mass,float),0)
    vals=[]; target=initial_age
    while True:
        while len(vals) < target:
            vals.append(float(chi2_fn(p,q)))
            p,_=transition(p,0); q,_=transition(q,0)
        d=np.asarray(vals,float); d0=max(float(d[0]),1e-300)
        n=min(tail_window,len(d))
        tail_ratio=float(np.max(d[-n:])/d0)
        complete=(len(d)>=initial_age and tail_ratio <= tail_ratio_tol)
        if complete or len(d)>=max_age:
            return d,{"tail_complete":bool(complete),"tail_ratio":tail_ratio,
                      "tail_ratio_tol":float(tail_ratio_tol),"max_age_used":len(d),
                      "max_age_cap":int(max_age),"tail_window":int(tail_window)}
        target=min(max_age,max(target+1,2*target))


def empirical_memory_length(d_r:np.ndarray,path_chi2:float)->float:
    if path_chi2<=0: raise ValueError
    return float(np.sum(np.sqrt(np.maximum(d_r,0.0)))**2/path_chi2)


def continuous_path_chi2(B:float,tau:float)->float:
    mu2=B*B*tau/2.0
    return float(math.expm1(mu2))


def discrete_path_chi2(eta:float,B:float,noise_std:float)->float:
    mu=abs(eta*B/noise_std)
    return float(math.expm1(mu*mu))

from __future__ import annotations

import math
import numpy as np
from scipy.optimize import brentq


def mass(v: np.ndarray) -> float:
    """Vectors in this package represent probability mass per grid cell."""
    return float(np.sum(np.asarray(v, float)))


def require_probability(v: np.ndarray, tol: float = 1e-8, name: str = "distribution") -> np.ndarray:
    v = np.asarray(v, float)
    if np.min(v) < -tol:
        raise FloatingPointError(f"{name} has negative mass {np.min(v)}")
    s = mass(v)
    if not math.isfinite(s) or abs(s-1.0) > tol:
        raise FloatingPointError(f"{name} mass={s}, expected 1±{tol}")
    out = np.maximum(v, 0.0)
    return out / out.sum()


def normalize_if_close(v: np.ndarray, max_mass_error: float = 1e-8) -> tuple[np.ndarray, float]:
    v = np.asarray(v, float)
    s = mass(v)
    err = 1.0 - s
    if not math.isfinite(s) or abs(err) > max_mass_error:
        raise FloatingPointError(f"mass error {err} exceeds tolerance {max_mass_error}")
    v = np.maximum(v, 0.0)
    return v/v.sum(), float(err)


def chi2_mass(p: np.ndarray, q: np.ndarray, floor: float = 0.0) -> float:
    p = require_probability(p, name="p")
    q = require_probability(q, name="q")
    if floor > 0:
        mask = q > floor
        if np.any((~mask) & (p > max(1e-15, floor))):
            return float("inf")
    else:
        mask = q > 0
        if np.any((~mask) & (p > 0)):
            return float("inf")
    return float(np.sum((p[mask]-q[mask])**2/q[mask]))


def d2_from_chi2(v: float) -> float:
    return float(np.log1p(v))


def privacy_profile_mass(p: np.ndarray, q: np.ndarray, eps_values) -> list[dict]:
    p = require_probability(p, name="p")
    q = require_probability(q, name="q")
    out=[]
    for eps in eps_values:
        a = math.exp(float(eps))
        f = float(np.maximum(p-a*q, 0.0).sum())
        r = float(np.maximum(q-a*p, 0.0).sum())
        out.append({"epsilon":float(eps),"delta_forward":f,"delta_reverse":r,"delta_symmetric":max(f,r)})
    return out


def symmetric_delta_at_epsilon(p: np.ndarray, q: np.ndarray, eps: float) -> float:
    return privacy_profile_mass(p,q,[eps])[0]["delta_symmetric"]


def epsilon_for_delta(p: np.ndarray, q: np.ndarray, delta: float, eps_max: float = 100.0) -> float:
    """Numerically invert the exact discrete-grid symmetric privacy profile."""
    p=require_probability(p,name="p"); q=require_probability(q,name="q")
    target=float(delta)
    if target < 0 or target >= 1:
        raise ValueError("delta must be in [0,1)")
    if symmetric_delta_at_epsilon(p,q,0.0) <= target:
        return 0.0
    hi=1.0
    while hi < eps_max and symmetric_delta_at_epsilon(p,q,hi) > target:
        hi*=2
    if hi >= eps_max and symmetric_delta_at_epsilon(p,q,hi) > target:
        return float("inf")
    return float(brentq(lambda e: symmetric_delta_at_epsilon(p,q,e)-target, 0.0, hi, xtol=1e-10, rtol=1e-10))



def privacy_profile_with_tv_defect(p: np.ndarray, q: np.ndarray, eps_values, tau_p: float, tau_q: float) -> list[dict]:
    """Conservative symmetric hockey-stick profile with distributional defects.

    If TV(P,P_tilde)<=tau_p and TV(Q,Q_tilde)<=tau_q, then
      H_a(P,Q) <= H_a(P_tilde,Q_tilde) + tau_p + a*tau_q
    and symmetrically in the reverse direction.
    """
    base=privacy_profile_mass(p,q,eps_values)
    tp=max(0.0,float(tau_p)); tq=max(0.0,float(tau_q))
    out=[]
    for row in base:
        eps=float(row["epsilon"]); a=math.exp(eps)
        f=min(1.0,float(row["delta_forward"])+tp+a*tq)
        r=min(1.0,float(row["delta_reverse"])+tq+a*tp)
        out.append({**row,"delta_forward_upper":f,"delta_reverse_upper":r,"delta_symmetric_upper":max(f,r),"tau_p":tp,"tau_q":tq})
    return out


def epsilon_upper_for_delta_with_tv_defect(p: np.ndarray, q: np.ndarray, delta: float, tau_p: float, tau_q: float, eps_max: float = 100.0) -> float:
    """Smallest epsilon found whose conservative defect-adjusted profile is <=delta.

    The transfer bound contains exp(epsilon)*tau and is not globally monotone, so
    we locate the first certified crossing on an adaptive grid and refine it by bisection.
    Returns +inf if no certified epsilon exists.
    """
    p=require_probability(p,name="p"); q=require_probability(q,name="q")
    target=float(delta); tp=max(0.0,float(tau_p)); tq=max(0.0,float(tau_q))
    if not (0 <= target < 1): raise ValueError("delta must be in [0,1)")
    def bound(e):
        row=privacy_profile_mass(p,q,[float(e)])[0]; a=math.exp(float(e))
        f=float(row["delta_forward"])+tp+a*tq
        r=float(row["delta_reverse"])+tq+a*tp
        return min(1.0,max(f,r))
    if bound(0.0) <= target: return 0.0
    # Once exp(e)*min-positive-defect alone exceeds target substantially, a later
    # crossing is impossible after the privacy profile has vanished.  Still scan
    # conservatively up to eps_max or a defect-informed cap.
    pos=[x for x in (tp,tq) if x>0]
    cap=float(eps_max)
    if pos:
        cap=min(cap,max(2.0,math.log(max(target,1e-300)/min(pos))+2.0))
    cap=max(cap,1.0)
    grid=np.linspace(0.0,cap,2001)
    prev_e=0.0; prev_v=bound(0.0)-target
    for e in grid[1:]:
        v=bound(float(e))-target
        if v <= 0 < prev_v:
            lo,hi=prev_e,float(e)
            for _ in range(70):
                mid=(lo+hi)/2
                if bound(mid) <= target: hi=mid
                else: lo=mid
            return float(hi)
        prev_e=float(e); prev_v=v
    return float("inf")

def fit_power_exponent(M, leakage):
    M=np.asarray(M,float); y=np.asarray(leakage,float)
    mask=np.isfinite(M)&np.isfinite(y)&(M>0)&(y>0)
    if mask.sum()<3:
        return {"beta":float("nan"),"intercept":float("nan"),"r2":float("nan"),"n":int(mask.sum())}
    x=np.log(M[mask]); z=np.log(y[mask])
    slope,intercept=np.polyfit(x,z,1)
    pred=slope*x+intercept
    ssr=float(np.sum((z-pred)**2)); sst=float(np.sum((z-z.mean())**2))
    return {"beta":float(-slope),"intercept":float(intercept),"r2":1.0-ssr/sst if sst else 1.0,"n":int(mask.sum())}

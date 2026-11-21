from __future__ import annotations

import math
from typing import Iterable


def lsi_rho_lower(m: float, A: float) -> float:
    """Holley--Stroock lower bound for U(x)=m x^2/2 + A cos(x).

    Convention: Ent_pi(g^2) <= (2/rho) E_pi |grad g|^2.
    The Gaussian base has rho=m and osc(A cos)=2|A|.
    """
    m=float(m); A=float(A)
    if m <= 0:
        raise ValueError("m must be positive")
    return m * math.exp(-2.0 * abs(A))


def contraction_constants(m: float, A: float, B: float, tau: float, theta: float = 1.0) -> dict:
    theta=float(theta)
    if not (0.0 < theta < 2.0):
        raise ValueError("theta must lie in (0,2)")
    if tau <= 0:
        raise ValueError("tau must be positive")
    rho=lsi_rho_lower(m,A)
    kappa=(2.0-theta)*rho/2.0
    gamma=math.exp(-kappa*float(tau))
    lam=(float(B)**2)/(theta*kappa) * (1.0-gamma)
    return {
        "rho_lsi_lower":rho,
        "theta":theta,
        "kappa":kappa,
        "gamma":gamma,
        "lambda":lam,
    }


def geometric_sum(x: float, M: int) -> float:
    if M < 1:
        raise ValueError
    if abs(1.0-x) < 1e-14:
        return float(M)
    return (1.0-x**M)/(1.0-x)


def finite1_chi2_upper(M:int,b:int,k:int,m:float,A:float,B:float,tau:float,theta:float=1.0) -> dict:
    """Finite-amplitude collision-aware hidden-support upper bound.

    Bound used in the frozen FINITE1 theorem:
      chi2 <= (exp(lambda k^2)-1) * [ A_M/M + sqrt((k-1)(b-1) B_M /(k M(Mb-1))) ]^2
    where A_M=sum gamma^(r/2), B_M=sum gamma^r.
    """
    M=int(M); b=int(b); k=int(k)
    if M<1 or b<1 or k<1 or k>M*b:
        raise ValueError("invalid M,b,k")
    c=contraction_constants(m,A,B,tau,theta)
    gamma=c["gamma"]; lam=c["lambda"]
    A_M=geometric_sum(math.sqrt(gamma),M)
    B_M=geometric_sum(gamma,M)
    collision=0.0
    if k>1 and b>1:
        collision=math.sqrt((k-1)*(b-1)*B_M/(k*M*(M*b-1)))
    bracket=A_M/M + collision
    # expm1 may overflow only in regimes where the certificate is intentionally useless.
    x=lam*k*k
    amp=math.expm1(x) if x < 700 else float("inf")
    upper=amp*bracket*bracket
    l_fa_asym=(1.0/(1.0-math.sqrt(gamma)) +
               math.sqrt(((k-1)/k)*((b-1)/b)/(1.0-gamma)) if k>1 and b>1 else
               1.0/(1.0-math.sqrt(gamma)))
    # Above conditional expression duplicates first term in the b=1 branch; fix explicitly.
    if k>1 and b>1:
        l_fa_asym=(1.0/(1.0-math.sqrt(gamma)) +
                   math.sqrt(((k-1)/k)*((b-1)/b)/(1.0-gamma)))**2
    else:
        l_fa_asym=(1.0/(1.0-math.sqrt(gamma)))**2
    return {
        **c,
        "A_M":A_M,
        "B_M":B_M,
        "amplitude_lambda_k2":x,
        "L_FA_asymptotic":l_fa_asym,
        "chi2_final_upper":upper,
    }


def one_shock_d2_envelope(max_age:int,m:float,A:float,B:float,tau:float,theta:float=1.0) -> list[float]:
    c=contraction_constants(m,A,B,tau,theta)
    return [c["lambda"]*(c["gamma"]**r) for r in range(int(max_age))]


def symmetric_delta_from_forward_chi2(chi2_forward: float, eps: float) -> float:
    """Certified symmetric approximate-DP transfer from one-way chi^2.

    eps=0 uses TV <= sqrt(chi2)/2.  For eps>0 the reverse hockey-stick
    coefficient is the limiting direction, with the proof-freeze constant.
    """
    v=max(0.0,float(chi2_forward)); eps=float(eps)
    if eps <= 0:
        return 0.5*math.sqrt(v)
    a=math.exp(eps)
    if a <= 2.0:
        c=a*a/(4.0*(a-1.0))
    else:
        c=1.0
    return min(1.0,c*v)


def certify_profile(actual_profile: Iterable[dict], chi2_upper: float, slack: float=1e-10) -> dict:
    rows=[]; ok=True
    for r in actual_profile:
        bound=symmetric_delta_from_forward_chi2(chi2_upper,float(r["epsilon"]))
        actual=float(r["delta_symmetric"])
        passed=actual <= bound*(1.0+slack)+slack
        ok &= passed
        rows.append({"epsilon":float(r["epsilon"]),"actual_delta_symmetric":actual,
                     "theory_delta_upper":bound,"pass":bool(passed)})
    return {"pass":bool(ok),"rows":rows}

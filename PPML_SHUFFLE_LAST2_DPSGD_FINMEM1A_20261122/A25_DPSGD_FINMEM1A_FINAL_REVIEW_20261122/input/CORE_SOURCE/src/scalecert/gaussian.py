from __future__ import annotations

import math
import numpy as np
from scipy.special import gammaln


def transcript_chi2_exact(M: int, b: int, k: int, mu: float) -> float:
    """Exact coefficient-DP oracle for true minibatch occupancy."""
    if not 0 <= k <= M*b:
        raise ValueError
    local=[]
    for a in range(min(b,k)+1):
        for d in range(min(b,k)+1):
            local.append((a,d,math.comb(b,a)*math.comb(b,d)*math.exp(mu*mu*a*d)))
    dp=np.zeros((k+1,k+1),dtype=np.float64); dp[0,0]=1.0
    for _ in range(M):
        nd=np.zeros_like(dp)
        for i in range(k+1):
            for j in range(k+1):
                base=dp[i,j]
                if base==0: continue
                for a,d,w in local:
                    if i+a<=k and j+d<=k:
                        nd[i+a,j+d]+=base*w
        dp=nd
    denom=float(math.comb(M*b,k))**2
    return float(dp[k,k]/denom-1.0)


def transcript_chi2_b1(M: int, k: int, mu: float) -> float:
    den=math.comb(M,k); s=0.0
    for h in range(max(0,2*k-M),k+1):
        p=math.comb(k,h)*math.comb(M-k,k-h)/den
        s += p*math.exp(mu*mu*h)
    return float(s-1.0)


def geometric_weights(M: int, rho: float) -> np.ndarray:
    return float(rho)**np.arange(M,dtype=float)


def memory_length(weights: np.ndarray) -> float:
    w=np.asarray(weights,float); s2=float(w@w)
    return float(w.sum()**2/s2) if s2>0 else 0.0


def _moment_dp(M: int, b: int, k: int, weights: np.ndarray, order: int) -> np.ndarray:
    """
    Exact raw moments E[(w^T C)^n], n=0..order, under true occupancy.
    Uses exponential-generating-function coefficients m_n/n! to make shift
    updates simple convolutions.
    """
    w=np.asarray(weights,float)
    if len(w)!=M: raise ValueError
    # egf[u,n] stores sum combinatorial_weight * z^n/n! after processed batches
    egf=np.zeros((k+1,order+1),dtype=np.float64); egf[0,0]=1.0
    inv_fact=np.exp(-np.array([gammaln(n+1) for n in range(order+1)]))
    for j in range(M):
        new=np.zeros_like(egf)
        for c in range(min(b,k)+1):
            shift=c*w[j]
            e=(shift**np.arange(order+1))*inv_fact
            comb=math.comb(b,c)
            for u in range(k-c+1):
                if not np.any(egf[u]): continue
                conv=np.convolve(egf[u],e)[:order+1]
                new[u+c]+=comb*conv
        egf=new
    denom=float(math.comb(M*b,k))
    factorial=np.exp(np.array([gammaln(n+1) for n in range(order+1)]))
    return egf[k]/denom*factorial


def final_chi2_exact_moments(M: int, b: int, k: int, weights: np.ndarray, mu: float, tol: float=1e-13, max_order: int=160) -> tuple[float, dict]:
    """
    Deterministic exact-series oracle:
      chi2 = sum_{n>=1} (t^n/n!) E[Z^n]^2,
      Z=w^T C, t=mu^2/||w||^2.
    Series truncation is certified against exp(x) with x=t*(k*wmax)^2.
    """
    w=np.asarray(weights,float); s2=float(w@w)
    if s2<=0: raise ValueError("zero weights")
    t=float(mu*mu/s2)
    xmax=t*(k*float(np.max(np.abs(w))))**2
    # Pick order using scalar exp-series tail upper bound.
    scalar=1.0; term=1.0; order=0
    target=tol*max(1.0,math.exp(min(xmax,700.0)))
    while order < max_order:
        order += 1
        term *= xmax/order
        scalar += term
        rem = math.exp(xmax)-scalar if xmax < 700 else float("inf")
        if rem <= target and order>=8:
            break
    if order>=max_order and xmax < 700 and math.exp(xmax)-scalar > target:
        raise RuntimeError(f"series max_order={max_order} insufficient for xmax={xmax}")
    moments=_moment_dp(M,b,k,w,order)
    terms=[]; total=0.0
    logfact=0.0
    for n in range(1,order+1):
        logfact += math.log(n)
        val=(t**n)*moments[n]*moments[n]/math.exp(logfact)
        total += val; terms.append(val)
    return float(total), {"series_order":order,"xmax":xmax,"last_term":float(terms[-1] if terms else 0.0),"scalar_tail_bound":float(max(0.0, math.exp(xmax)-scalar) if xmax<700 else float('inf'))}


def final_chi2_k1_closed(M: int, weights: np.ndarray, mu: float) -> float:
    w=np.asarray(weights,float); s2=float(w@w)
    return float(np.exp(mu*mu*np.outer(w,w)/s2).mean()-1.0)

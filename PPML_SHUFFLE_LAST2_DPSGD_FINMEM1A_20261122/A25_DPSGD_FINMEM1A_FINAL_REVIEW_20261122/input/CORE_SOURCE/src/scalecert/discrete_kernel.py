from __future__ import annotations

import math
import numpy as np
from scipy.special import log_ndtr, ndtr
from scipy.sparse import csc_matrix

from .cc_solver import Grid1D, common_drift


def _logdiffexp(loga: np.ndarray, logb: np.ndarray) -> np.ndarray:
    """log(exp(loga)-exp(logb)), requiring loga>=logb."""
    d=np.minimum(logb-loga,0.0)
    return loga + np.log1p(-np.exp(d))


def normal_interval_prob(lo: np.ndarray, hi: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Stable N(mu,sigma^2) probabilities on [lo,hi]."""
    lo=np.asarray(lo,float); hi=np.asarray(hi,float)
    zl=(lo-mu)/sigma; zh=(hi-mu)/sigma
    out=np.empty_like(zl)
    left=zh <= 0
    right=zl >= 0
    mid=~(left|right)
    if np.any(left):
        la=log_ndtr(zh[left]); lb=log_ndtr(zl[left])
        out[left]=np.exp(_logdiffexp(la,lb))
    if np.any(right):
        # Phi(-zl)-Phi(-zh)
        la=log_ndtr(-zl[right]); lb=log_ndtr(-zh[right])
        out[right]=np.exp(_logdiffexp(la,lb))
    if np.any(mid):
        out[mid]=ndtr(zh[mid])-ndtr(zl[mid])
    return np.maximum(out,0.0)


class DiscreteGaussianKernel1D:
    """Sparse finite-domain Gaussian transition for a DP-SGD/SGLD-style step.

    X' = X + eta*(-U'(X)+B*c) + noise_std*Z.

    No renormalization is performed.  Probability outside the finite domain AND
    the explicitly controlled Gaussian stencil tail are returned as lost mass.
    Headline runs must certify this loss is negligible and stable under grid/domain
    refinement.  Tiny positive probabilities within the retained stencil are never
    hard-thresholded to zero.
    """
    def __init__(self, grid: Grid1D, m: float, A: float, B: float, eta: float,
                 noise_std: float, drop_tol: float=0.0, tail_sigma: float=10.0):
        self.grid=grid; self.m=float(m); self.A=float(A); self.B=float(B)
        self.eta=float(eta); self.noise_std=float(noise_std); self.drop_tol=float(drop_tol)
        self.tail_sigma=float(tail_sigma)
        if self.noise_std <= 0: raise ValueError("noise_std must be positive")
        if self.tail_sigma < 6: raise ValueError("tail_sigma must be >=6 for certification")
        self._cache={}

    def kernel(self,c:int):
        c=int(c)
        if c in self._cache: return self._cache[c]
        x=self.grid.x; dx=self.grid.dx; n=len(x)
        edges=np.empty(n+1,float)
        edges[1:-1]=0.5*(x[:-1]+x[1:])
        edges[0]=x[0]-dx/2; edges[-1]=x[-1]+dx/2
        means=x+self.eta*(common_drift(x,self.m,self.A)+self.B*c)
        rows=[]; cols=[]; vals=[]
        lost=np.empty(n,float); domain_escape=np.empty(n,float); stencil_loss=np.empty(n,float)
        for i,mu in enumerate(means):
            # Exact domain escape using stable one-sided Gaussian probabilities.
            zl=(edges[0]-mu)/self.noise_std; zr=(edges[-1]-mu)/self.noise_std
            left=float(ndtr(zl)); right=float(ndtr(-zr))
            de=max(0.0,min(1.0,left+right)); domain_escape[i]=de
            lo_val=mu-self.tail_sigma*self.noise_std
            hi_val=mu+self.tail_sigma*self.noise_std
            j0=max(0,int(np.searchsorted(edges,lo_val,side="right")-1))
            j1=min(n-1,int(np.searchsorted(edges,hi_val,side="left")))
            idx=np.arange(j0,j1+1,dtype=int)
            probs=normal_interval_prob(edges[idx],edges[idx+1],mu,self.noise_std)
            if self.drop_tol>0:
                keep=probs>self.drop_tol
                dropped=float(probs[~keep].sum())
                idx=idx[keep]; probs=probs[keep]
            else:
                dropped=0.0
            kept=float(probs.sum())
            inside=max(0.0,1.0-de)
            sl=max(0.0,inside-kept)
            # Numerical dropped mass is already included in inside-kept; retain for diagnostics.
            stencil_loss[i]=sl
            lost[i]=min(1.0,de+sl)
            rows.extend(idx.tolist()); cols.extend([i]*len(idx)); vals.extend(probs.tolist())
        K=csc_matrix((vals,(rows,cols)),shape=(n,n))
        self._cache[c]=(K,lost,domain_escape,stencil_loss)
        return self._cache[c]

    def transition(self,mass:np.ndarray,c:int)->tuple[np.ndarray,float]:
        mass=np.asarray(mass,float)
        K,lost,_,_=self.kernel(c)
        out=np.asarray(K@mass,float)
        escaped=float(np.dot(lost,mass))
        return out,escaped

    def diagnostics(self,max_c:int=1)->dict:
        out={}
        for c in range(max_c+1):
            K,lost,de,sl=self.kernel(c)
            sums=np.asarray(K.sum(axis=0)).ravel()+lost
            out[str(c)]={
                "max_column_probability_error":float(np.max(np.abs(sums-1.0))),
                "max_one_step_total_lost_probability":float(np.max(lost)),
                "max_one_step_domain_escape_probability":float(np.max(de)),
                "max_one_step_stencil_loss_probability":float(np.max(sl)),
                "noise_std_over_dx":float(self.noise_std/self.grid.dx),
                "tail_sigma":self.tail_sigma,
            }
        return out


def sgld_noise_std(eta:float)->float:
    return math.sqrt(2.0*eta)


def dpsgd_noise_std(eta:float,B:float,noise_multiplier:float)->float:
    """Parameter-noise std for sensitivity-normalized DP-SGD bridge.

    A one-record drift has parameter shift eta*B.  The conventional noise
    multiplier z therefore corresponds to std = eta*B*z and standardized
    one-record Gaussian signal mu=1/z.
    """
    z=float(noise_multiplier)
    if z<=0: raise ValueError("noise_multiplier must be positive")
    return abs(float(eta)*float(B))*z


class RepeatedTransition:
    """Apply a base discrete Gaussian transition n_sub times for each logical batch."""
    def __init__(self, base: DiscreteGaussianKernel1D, n_sub: int):
        self.base=base; self.n_sub=int(n_sub)
        if self.n_sub < 1: raise ValueError
    def transition(self,mass:np.ndarray,c:int)->tuple[np.ndarray,float]:
        v=np.asarray(mass,float); lost=0.0
        for _ in range(self.n_sub):
            v,e=self.base.transition(v,c); lost+=e
        return v,float(lost)
    def diagnostics(self,max_c:int=1)->dict:
        d=self.base.diagnostics(max_c); d["n_sub"]=self.n_sub; return d

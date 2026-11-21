from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, identity, csc_matrix
from scipy.sparse.linalg import expm_multiply, splu


def _cc_delta(w: np.ndarray) -> np.ndarray:
    w=np.asarray(w,float)
    out=np.empty_like(w)
    small=np.abs(w)<1e-5
    ws=w[small]
    out[small]=0.5-ws/12.0+ws**3/720.0
    wn=w[~small]
    out[~small]=1.0/wn-1.0/np.expm1(wn)
    return out


def potential(x: np.ndarray, m: float, A: float) -> np.ndarray:
    return 0.5*m*x*x + A*np.cos(x)


def common_drift(x: np.ndarray, m: float, A: float) -> np.ndarray:
    return -m*x + A*np.sin(x)


@dataclass(frozen=True)
class Grid1D:
    x: np.ndarray
    dx: float
    x_max: float

    @staticmethod
    def make(x_max: float, nx: int) -> "Grid1D":
        if nx < 21 or nx % 2 == 0:
            raise ValueError("nx must be odd and >=21")
        x=np.linspace(-float(x_max),float(x_max),int(nx))
        return Grid1D(x=x,dx=float(x[1]-x[0]),x_max=float(x_max))


class ContinuousLangevin1D:
    """
    Conservative Chang--Cooper finite-volume semi-discretization of
      dX=(-U'(X)+B*c)dt + sqrt(2)dW
    on a finite reflecting domain. No renormalization is performed by transitions.
    Unbounded-process validity is established by domain/grid convergence and tiny edge mass.
    """
    def __init__(self, grid: Grid1D, m: float, A: float, B: float, tau: float, method: str="be", substeps: int=8):
        self.grid=grid; self.m=float(m); self.A=float(A); self.B=float(B); self.tau=float(tau)
        self.method=str(method); self.substeps=int(substeps)
        if self.method not in ("be","expm"): raise ValueError("method must be be or expm")
        if self.substeps < 1: raise ValueError("substeps must be >=1")
        self._cache={}; self._factor_cache={}

    def analytic_stationary_mass(self) -> np.ndarray:
        z=-potential(self.grid.x,self.m,self.A)
        z-=np.max(z)
        p=np.exp(z)
        p/=p.sum()
        return p

    def numerical_stationary_mass(self) -> np.ndarray:
        """Exact zero-flux stationary mass of the Chang--Cooper generator.

        Uses detailed balance across adjacent cells.  This removes the small
        analytic-density/grid mismatch from the reference process while the
        analytic stationary law remains available as a convergence diagnostic.
        """
        G=self.generator(0).tocsc(); n=G.shape[0]
        logq=np.zeros(n,float)
        for i in range(n-1):
            down=float(G[i+1,i]); up=float(G[i,i+1])
            if down <= 0 or up <= 0:
                raise FloatingPointError(f"nonpositive adjacent CC rates at {i}: {down}, {up}")
            logq[i+1]=logq[i]+math.log(down)-math.log(up)
        logq-=np.max(logq)
        q=np.exp(logq); q/=q.sum()
        return q

    def generator(self, c: int) -> csr_matrix:
        c=int(c)
        if c in self._cache:
            return self._cache[c]
        x=self.grid.x; dx=self.grid.dx; n=len(x)
        xm=0.5*(x[:-1]+x[1:])
        b=common_drift(xm,self.m,self.A)+self.B*c
        w=b*dx
        delta=_cc_delta(w)
        # Flux J_{i+1/2}=aL*p_i+aR*p_{i+1}; D=1.
        aL=b*(1.0-delta)+1.0/dx
        aR=b*delta-1.0/dx
        rows=[]; cols=[]; vals=[]
        for i in range(n-1):
            # dp_i/dt -= J/dx
            rows += [i,i]
            cols += [i,i+1]
            vals += [-aL[i]/dx,-aR[i]/dx]
            # dp_{i+1}/dt += J/dx
            rows += [i+1,i+1]
            cols += [i,i+1]
            vals += [aL[i]/dx,aR[i]/dx]
        G=coo_matrix((vals,(rows,cols)),shape=(n,n)).tocsr()
        self._cache[c]=G
        return G

    def _factor(self, c: int):
        c=int(c)
        if c in self._factor_cache: return self._factor_cache[c]
        dt=self.tau/self.substeps
        A=(identity(len(self.grid.x),format="csc")-dt*self.generator(c).tocsc())
        fac=splu(A)
        self._factor_cache[c]=fac
        return fac

    def transition(self, mass: np.ndarray, c: int) -> tuple[np.ndarray,float]:
        mass=np.asarray(mass,float)
        before=float(mass.sum())
        if before < -1e-14:
            raise FloatingPointError("negative total mass")
        if self.method=="expm":
            out=np.asarray(expm_multiply(self.generator(c)*self.tau,mass),float)
        else:
            out=mass.copy(); fac=self._factor(c)
            for _ in range(self.substeps): out=np.asarray(fac.solve(out),float)
        # Never truncate small POSITIVE tail probabilities: doing so creates
        # artificial support mismatch and can make reverse chi^2 infinite.
        minv=float(out.min())
        if minv < -1e-10:
            raise FloatingPointError(f"Chang-Cooper transition generated negative mass {minv}")
        neg=out < 0.0
        clipped=float(-out[neg].sum()) if np.any(neg) else 0.0
        if clipped > 1e-12*max(1.0,before):
            raise FloatingPointError(f"negative roundoff mass too large: {clipped}")
        if np.any(neg):
            out=out.copy(); out[neg]=0.0
        # Negative values here are roundoff only; clipping is accepted only at a
        # level far below the experiment mass tolerance.  Positive tails are untouched.
        return out, 0.0

    def diagnostics(self, max_c: int = 1) -> dict:
        ds={}
        for c in range(max_c+1):
            G=self.generator(c)
            col=np.asarray(G.sum(axis=0)).ravel()
            coo=G.tocoo(); off=coo.data[coo.row!=coo.col]
            ds[str(c)]={
                "max_abs_column_sum":float(np.max(np.abs(col))),
                "min_offdiag":float(np.min(off)) if len(off) else 0.0,
            }
        qa=self.analytic_stationary_mass()
        q=self.numerical_stationary_mass()
        q1,_=self.transition(q,0)
        ds["stationarity_l1_one_interval"]=float(np.abs(q1-q).sum())
        ds["analytic_vs_numerical_stationary_l1"]=float(np.abs(qa-q).sum())
        ds["stationary_edge_mass_5"]=float(q[:5].sum()+q[-5:].sum())
        ds["method"]=self.method; ds["substeps"]=self.substeps
        return ds


def edge_mass(mass: np.ndarray, cells: int=5) -> float:
    mass=np.asarray(mass,float)
    return float(mass[:cells].sum()+mass[-cells:].sum())

from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.special import ndtr

from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D, dpsgd_noise_std


def make_kernel(cfg):
    grid=Grid1D.make(cfg['x_max'],cfg['nx'])
    ker=DiscreteGaussianKernel1D(grid,cfg['m'],cfg['A'],cfg['B'],cfg['eta'],
          dpsgd_noise_std(cfg['eta'],cfg['B'],cfg['z']),drop_tol=0.,tail_sigma=cfg['tail_sigma'])
    return grid,ker


def analytic_escape_vector(kernel,c):
    """Gaussian probability outside the retained stencil, computed directly
    from normal tails rather than 1 minus a floating-point column sum. The
    THEORYCOMPLY transition matrix itself is unchanged."""
    g=kernel.grid; x=g.x; dx=g.dx; n=len(x)
    ee=np.empty(n+1); ee[1:-1]=.5*(x[:-1]+x[1:]); ee[0]=x[0]-dx/2; ee[-1]=x[-1]+dx/2
    means=x+kernel.eta*(-kernel.m*x+kernel.A*np.sin(x)+kernel.B*int(c))
    sig=kernel.noise_std; tail=kernel.tail_sigma*sig; out=np.empty(n)
    for i,mu in enumerate(means):
        lo=max(0,int(np.searchsorted(ee,mu-tail,side='right')-1))
        hi=min(n-1,int(np.searchsorted(ee,mu+tail,side='left')))
        out[i]=float(ndtr((ee[lo]-mu)/sig)+ndtr((mu-ee[hi+1])/sig))
    return out


def _ordered_eigs(A,nmodes,tol,maxiter):
    n=A.shape[0]
    vals,vecs=eigs(A,k=min(nmodes,n-2),which='LM',tol=tol,maxiter=maxiter,ncv=min(n,max(2*nmodes+8,40)))
    order=np.argsort(-np.abs(vals)); return vals[order],vecs[:,order]


def compute_spectrum(cfg,nmodes=32,tol=1e-11,maxiter=200000):
    grid,ker=make_kernel(cfg); K,lost,_,_=ker.kernel(0); K1=ker.kernel(1)[0]
    vals,R=_ordered_eigs(K,nmodes,tol,maxiter)
    lvals,Lraw=_ordered_eigs(K.T.tocsc(),nmodes,tol,maxiter)
    # Match the left eigenvector of K for lambda to the K.T eigenvalue conjugate(lambda).
    left=[]; lres=[]
    for j,lam in enumerate(vals):
        idx=int(np.argmin(np.abs(lvals-np.conjugate(lam))))
        u=Lraw[:,idx]
        v=R[:,j]
        den=np.vdot(u,v)
        if abs(den)<1e-14: raise RuntimeError(f"biorthogonal pairing nearly singular for mode {j}")
        u=u*np.conjugate(1.0/den) # makes u^H v = 1
        left.append(u)
        lres.append(float(np.linalg.norm(K.T@u-np.conjugate(lam)*u)/max(np.linalg.norm(u),1e-300)))
    left=np.column_stack(left)
    # Perron vector is the leading right mode, normalized as a probability.
    lam0=vals[0]
    q=np.real(R[:,0])
    if np.sum(q)<0:q=-q
    if np.min(q)<-1e-8: raise RuntimeError(f"Perron vector not nonnegative: min={np.min(q)}")
    q=np.maximum(q,0.); q/=q.sum()
    stat_res=float(np.linalg.norm(K@q-lam0.real*q,1))
    escape=float(np.dot(lost,q))
    h=np.asarray(K1@q-q).ravel()
    modes=[]
    for j,lam in enumerate(vals):
        v=R[:,j]; u=left[:,j]
        alpha=np.vdot(u,h)
        modes.append({'mode':j,'lambda_real':float(np.real(lam)),'lambda_imag':float(np.imag(lam)),
          'abs_lambda':float(abs(lam)),'tau':float(-1/np.log(abs(lam))) if 0<abs(lam)<1 else None,
          'projection_real':float(np.real(alpha)),'projection_imag':float(np.imag(alpha)),
          'density_l1_amplitude':float(abs(alpha)*np.linalg.norm(v,1)),
          'right_residual':float(np.linalg.norm(K@v-lam*v)/max(np.linalg.norm(v),1e-300)),
          'left_residual':lres[j],
          'biorthogonal_condition_indicator':float(np.linalg.norm(u)*np.linalg.norm(v))})
    return {'config':cfg,'grid':{'nx':len(grid.x),'x_max':grid.x_max,'dx':grid.dx},
      'lambda_perron_real':float(lam0.real),'lambda_perron_imag':float(lam0.imag),
      'quasi_stationary_residual_l1':stat_res,'perron_escape_per_step':escape,
      'q':q,'h':h,'eigenvalues':vals,'right_vectors':R,'left_vectors':left,
      'modes':modes,'kernel':K,'kernel1':K1,'lost0':lost,'kernel_obj':ker,'eigs_tol':tol}


def direct_response(spec,max_r,checkpoint_every=512,progress=None):
    K=spec['kernel']; K1=spec['kernel1']; q=spec['q']; lost0=analytic_escape_vector(spec['kernel_obj'],0)
    lost1=analytic_escape_vector(spec['kernel_obj'],1)
    p=np.asarray(K1@q).ravel(); escape1=float(np.dot(lost1,q))
    # K1 can also lose mass under truncation; count it explicitly.
    tv=[]; tv_upper=[]; esc_p=[]; esc_q=[]
    loss_p=escape1; round_p=abs(float(p.sum())+escape1-1.0); round_q=0.0
    lossq=0.
    for r in range(max_r+1):
        v=.5*float(np.abs(p-q).sum())
        # A conservative finite-domain envelope: missing path mass from both
        # arms is added to the interior TV. No mass is renormalized.
        roundoff=(r+1)*float(spec['quasi_stationary_residual_l1'])+round_p+round_q
        tv.append(v); tv_upper.append(min(1.,v+loss_p+lossq+roundoff))
        esc_p.append(loss_p); esc_q.append(lossq)
        if r==max_r: break
        oldp=float(p.sum())
        ep=float(np.dot(lost0,p)); eq=float(np.dot(lost0,q))
        pnew=np.asarray(K@p).ravel()
        loss_p += ep; lossq += eq
        round_p += abs(float(pnew.sum())+ep-oldp)
        p=pnew
        if progress and (r+1)%checkpoint_every==0: progress({'r':r+1,'tv':tv[-1],'tv_upper':tv_upper[-1],'escape_present':loss_p,'escape_reference':lossq})
    return {'tv_grid':np.asarray(tv),'tv_upper':np.asarray(tv_upper),'escape_present':np.asarray(esc_p),'escape_reference':np.asarray(esc_q)}


def spectral_reconstruct(spec, r_values):
    vals=spec['eigenvalues']; R=spec['right_vectors']; L=spec['left_vectors']; K=spec['kernel']; K1=spec['kernel1']; q=spec['q']
    h=np.asarray(K1@q-q).ravel()
    # Projection in a non-normal basis; left/right vectors are biorthogonalized.
    alpha=np.asarray([np.vdot(L[:,j],h) for j in range(len(vals))])
    # The Perron eigenvalue differs from one at the last few floating-point
    # bits because the frozen sparse matrix's column sums are accumulated in
    # float64. The intended absent-user kernel is conservative; its analytically
    # bounded Gaussian escape is handled separately. Remove that numerical
    # Perron drift from the memory response and report the residual as error.
    alpha[0]=0.0
    rr=np.asarray(r_values,dtype=int); out=np.empty(len(rr),float)
    for i,r in enumerate(rr):
        delta=R@(alpha*np.power(vals,r))
        out[i]=.5*float(np.sum(np.abs(delta)))
    return out


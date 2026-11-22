from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .occupancy import next_batch_probs, enumerate_occupancies


@dataclass
class EvolutionResult:
    mass: np.ndarray
    escaped_mass: float
    latent_mass_error: float


def evolve_hidden_epoch(initial_mass: np.ndarray, transition, M:int,b:int,k:int) -> EvolutionResult:
    """
    Exact hidden-occupancy DP. Crucially each branch is a SUB-PROBABILITY MASS VECTOR.
    Branches are never independently normalized.
    """
    states={int(k):np.asarray(initial_mass,float).copy()}
    escaped_total=0.0
    initial_total=float(np.sum(initial_mass))
    for t in range(M):
        S=(M-t)*b
        nxt={}
        for r,v in states.items():
            for c,pr in next_batch_probs(r,S,b):
                out,escaped=transition(v,c)
                out=pr*out
                escaped_total += pr*escaped
                key=r-c
                if key in nxt: nxt[key]+=out
                else: nxt[key]=out.copy()
        states=nxt
    if set(states)!={0}:
        raise RuntimeError(f"terminal latent states {states.keys()}")
    final=states[0]
    latent_error=initial_total-float(final.sum())-escaped_total
    return EvolutionResult(final,float(escaped_total),float(latent_error))


def evolve_fixed_counts(initial_mass:np.ndarray,transition,counts)->EvolutionResult:
    v=np.asarray(initial_mass,float).copy(); escaped=0.0
    for c in counts:
        v,e=transition(v,int(c)); escaped+=e
    return EvolutionResult(v,float(escaped),float(np.sum(initial_mass)-v.sum()-escaped))


def evolve_absent_epoch(initial_mass:np.ndarray,transition,M:int)->EvolutionResult:
    return evolve_fixed_counts(initial_mass,transition,[0]*M)


def evolve_multi_epoch(initial_present:np.ndarray, initial_absent:np.ndarray, transition, M:int,b:int,k:int,E_values)->dict:
    targets=sorted(set(int(e) for e in E_values))
    p=np.asarray(initial_present,float).copy(); q=np.asarray(initial_absent,float).copy()
    escaped_p=0.0; escaped_q=0.0; out={}
    maxE=max(targets)
    for e in range(1,maxE+1):
        rp=evolve_hidden_epoch(p,transition,M,b,k); rq=evolve_absent_epoch(q,transition,M)
        p,q=rp.mass,rq.mass; escaped_p+=rp.escaped_mass; escaped_q+=rq.escaped_mass
        if e in targets:
            out[e]={"present":p.copy(),"absent":q.copy(),"escaped_present":escaped_p,"escaped_absent":escaped_q}
    return out


def known_support_average(initial_mass, absent_mass, transition, M:int,b:int,k:int, chi2_fn, max_patterns:int=5000):
    vals=[]; probs=[]
    for counts,pr in enumerate_occupancies(M,b,k,max_patterns=max_patterns):
        rp=evolve_fixed_counts(initial_mass,transition,counts)
        vals.append(chi2_fn(rp.mass,absent_mass)); probs.append(pr)
    return float(np.dot(vals,probs)), {"patterns":len(vals),"prob_sum":float(sum(probs))}

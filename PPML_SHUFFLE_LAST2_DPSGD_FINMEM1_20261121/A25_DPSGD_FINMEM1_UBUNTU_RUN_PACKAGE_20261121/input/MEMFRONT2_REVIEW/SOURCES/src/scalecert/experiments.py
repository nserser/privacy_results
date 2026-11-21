from __future__ import annotations

import json, math, os, platform, socket, time, hashlib
import numpy as np

from .gaussian import (
    transcript_chi2_exact, transcript_chi2_b1, geometric_weights,
    memory_length, final_chi2_exact_moments, final_chi2_k1_closed,
)
from .occupancy import expected_collision_pairs
from .cc_solver import Grid1D, ContinuousLangevin1D, edge_mass
from .discrete_kernel import DiscreteGaussianKernel1D, RepeatedTransition, sgld_noise_std, dpsgd_noise_std
from .hidden_support import evolve_hidden_epoch, evolve_absent_epoch, evolve_multi_epoch, known_support_average
from .memory import adaptive_impulse_curve, empirical_memory_length, continuous_path_chi2
from .privacy import normalize_if_close, chi2_mass, d2_from_chi2, privacy_profile_mass, epsilon_for_delta, privacy_profile_with_tv_defect, epsilon_upper_for_delta_with_tv_defect
from .theory_certificate import finite1_chi2_upper, one_shock_d2_envelope, certify_profile


def canonical_id(family: str, cfg: dict) -> str:
    raw=json.dumps({"family":family,"config":cfg},sort_keys=True,separators=(",",":")).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def base_record(family: str, cfg: dict) -> dict:
    return {
        "experiment_id":canonical_id(family,cfg),"family":family,"config":cfg,
        "status":"RUNNING","start_time":time.time(),"host":socket.gethostname(),
        "python":platform.python_version(),"pid":os.getpid(),"metrics":{},
        "diagnostics":{},"warnings":[],
    }


def finish(rec:dict,status="OK"):
    rec["status"]=status; rec["end_time"]=time.time(); rec["elapsed_s"]=rec["end_time"]-rec["start_time"]
    return rec


def _safe_prob(v,tol): return normalize_if_close(v,max_mass_error=tol)


def run_gaussian(cfg:dict)->dict:
    rec=base_record("gaussian",cfg)
    M,b,k=int(cfg["M"]),int(cfg["b"]),int(cfg["k"]); mu=float(cfg["mu"]); rho=float(cfg["rho"])
    tr=transcript_chi2_exact(M,b,k,mu); w=geometric_weights(M,rho)
    fin,diag=final_chi2_exact_moments(M,b,k,w,mu,tol=float(cfg.get("series_tol",1e-13)))
    if k==1: rec["diagnostics"]["k1_closed_abs_error"]=abs(fin-final_chi2_k1_closed(M,w,mu))
    if b==1: rec["diagnostics"]["b1_transcript_abs_error"]=abs(tr-transcript_chi2_b1(M,k,mu))
    rec["metrics"]={
        "transcript_chi2":tr,"transcript_d2":d2_from_chi2(tr),"final_chi2":fin,"final_d2":d2_from_chi2(fin),
        "gap_chi2":tr/fin if fin>0 else float("inf"),"gap_d2":d2_from_chi2(tr)/d2_from_chi2(fin) if fin>0 else float("inf"),
        "l_priv":memory_length(w),"collision_pair_expectation":expected_collision_pairs(M,b,k),
    }
    rec["diagnostics"].update(diag); return finish(rec)


def _continuous_model(cfg):
    grid=Grid1D.make(float(cfg["x_max"]),int(cfg["nx"]))
    model=ContinuousLangevin1D(grid,float(cfg["m"]),float(cfg["A"]),float(cfg["B"]),float(cfg["tau"]),
                               method=str(cfg.get("cc_method","be")),substeps=int(cfg.get("cc_substeps",8)))
    return grid,model


def _continuous_reference(model:ContinuousLangevin1D):
    return model.numerical_stationary_mass()


def run_nonlinear(cfg:dict)->dict:
    rec=base_record("nonlinear",cfg); M,b,k=int(cfg["M"]),int(cfg["b"]),int(cfg["k"])
    mass_tol=float(cfg.get("mass_tol",1e-7)); grid,model=_continuous_model(cfg); q0=_continuous_reference(model)
    rp=evolve_hidden_epoch(q0,model.transition,M,b,k); rq=evolve_absent_epoch(q0,model.transition,M)
    p,ep=_safe_prob(rp.mass,mass_tol); q,eq=_safe_prob(rq.mass,mass_tol)
    cf=chi2_mass(p,q); cr=chi2_mass(q,p); mu=abs(float(cfg["B"]))*math.sqrt(float(cfg["tau"])/2.0)
    tr=transcript_chi2_exact(M,b,k,mu); eps_grid=cfg.get("epsilon_grid",[0.0,0.1,0.25,0.5,1.0,2.0]); deltas=cfg.get("delta_targets",[1e-3,1e-5])
    profile=privacy_profile_mass(p,q,eps_grid)
    theta=float(cfg.get("theory_theta",1.0)); th=finite1_chi2_upper(M,b,k,float(cfg["m"]),float(cfg["A"]),float(cfg["B"]),float(cfg["tau"]),theta)
    profile_cert=certify_profile(profile,th["chi2_final_upper"])
    rec["metrics"]={
        "final_chi2_forward":cf,"final_chi2_reverse":cr,"final_chi2_symmetric_max":max(cf,cr),
        "final_d2_forward":d2_from_chi2(cf),"final_d2_reverse":d2_from_chi2(cr),
        "transcript_chi2":tr,"transcript_d2":d2_from_chi2(tr),
        "gap_chi2":tr/cf if cf>0 else float("inf"),"gap_d2":d2_from_chi2(tr)/d2_from_chi2(cf) if cf>0 else float("inf"),
        "privacy_profile":profile,"epsilon_at_delta":{str(d):epsilon_for_delta(p,q,float(d)) for d in deltas},
        "collision_pair_expectation":expected_collision_pairs(M,b,k),
        "finite1_chi2_upper":th["chi2_final_upper"],
        "finite1_bound_ratio_to_observed":th["chi2_final_upper"]/cf if cf>0 else float("inf"),
        "theory_profile_certificate":profile_cert,
    }
    rec["diagnostics"]={
        "present_mass_error_before_metric":float(ep),"absent_mass_error_before_metric":float(eq),
        "present_latent_mass_error":rp.latent_mass_error,"absent_latent_mass_error":rq.latent_mass_error,
        "present_edge_mass_5":edge_mass(p),"absent_edge_mass_5":edge_mass(q),"solver":model.diagnostics(max_c=min(k,b)),
        "theory":th,
    }
    if cf > th["chi2_final_upper"]*(1+float(cfg.get("theory_slack",0.03)))+1e-14:
        rec["warnings"].append("observed forward chi2 exceeds FINITE1 theorem envelope")
        return finish(rec,"THEORY_CERTIFICATE_FAIL")
    if not profile_cert["pass"]:
        rec["warnings"].append("symmetric privacy profile exceeds theorem transfer envelope")
        return finish(rec,"THEORY_CERTIFICATE_FAIL")
    if not np.isfinite(cr):
        rec["warnings"].append("reverse chi2 is non-finite")
        return finish(rec,"NUMERICAL_INVALID")
    if cfg.get("known_support",False):
        try:
            avg,kdiag=known_support_average(q0,q,model.transition,M,b,k,lambda a,b_: chi2_mass(_safe_prob(a,mass_tol)[0],b_),max_patterns=int(cfg.get("known_support_max_patterns",5000)))
            rec["metrics"]["known_support_mean_chi2"]=avg; rec["diagnostics"]["known_support"]=kdiag
        except ValueError as e: rec["warnings"].append(f"known support skipped: {e}")
    return finish(rec)


def run_memory(cfg:dict)->dict:
    rec=base_record("memory",cfg); M=int(cfg["M"]); mass_tol=float(cfg.get("mass_tol",1e-7))
    grid,model=_continuous_model(cfg); q0=_continuous_reference(model)
    def c2(a,b):
        aa,_=_safe_prob(a,mass_tol); bb,_=_safe_prob(b,mass_tol); return chi2_mass(aa,bb)
    d,tail=adaptive_impulse_curve(q0,model.transition,M,c2,tail_ratio_tol=float(cfg.get("memory_tail_tol",1e-6)),
                                  max_age=int(cfg.get("memory_max_age",4096)),tail_window=int(cfg.get("memory_tail_window",4)))
    path=continuous_path_chi2(float(cfg["B"]),float(cfg["tau"])); l=empirical_memory_length(d,path)
    env=np.asarray(one_shock_d2_envelope(len(d),float(cfg["m"]),float(cfg["A"]),float(cfg["B"]),float(cfg["tau"]),float(cfg.get("theory_theta",1.0))))
    d2=np.log1p(d); slack=float(cfg.get("theory_slack",0.03)); env_ok=bool(np.all(d2 <= env*(1+slack)+1e-14))
    rec["metrics"]={"d_r":d.tolist(),"d2_r":d2.tolist(),"path_chi2":path,"l_emp":l,"d0":float(d[0]),"d_last":float(d[-1]),
                    "theory_d2_envelope":env.tolist(),"theory_envelope_pass":env_ok}
    rec["diagnostics"]={"solver":model.diagnostics(max_c=1),"memory_tail":tail}
    if not tail["tail_complete"]: return finish(rec,"MEMORY_TAIL_INCOMPLETE")
    if not env_ok: return finish(rec,"THEORY_CERTIFICATE_FAIL")
    return finish(rec)


def run_multiepoch(cfg:dict)->dict:
    rec=base_record("multiepoch",cfg); M,b,k=int(cfg["M"]),int(cfg["b"]),int(cfg["k"]); E_values=[int(x) for x in cfg["E_values"]]
    mass_tol=float(cfg.get("mass_tol",1e-7)); grid,model=_continuous_model(cfg); q0=_continuous_reference(model)
    series=evolve_multi_epoch(q0,q0,model.transition,M,b,k,E_values)
    mu=abs(float(cfg["B"]))*math.sqrt(float(cfg["tau"])/2.0); tr1=transcript_chi2_exact(M,b,k,mu); d2tr1=d2_from_chi2(tr1)
    rows=[]
    for E in sorted(series):
        rr=series[E]; p,ep=_safe_prob(rr["present"],mass_tol); q,eq=_safe_prob(rr["absent"],mass_tol)
        cf=chi2_mass(p,q); cr=chi2_mass(q,p)
        rows.append({"E":E,"final_chi2_forward":cf,"final_chi2_reverse":cr,"final_d2_forward":d2_from_chi2(cf),"final_d2_reverse":d2_from_chi2(cr),
                     "transcript_d2":E*d2tr1,"gap_d2":E*d2tr1/d2_from_chi2(cf) if cf>0 else float("inf"),
                     "present_mass_error":ep,"absent_mass_error":eq,"present_edge_mass_5":edge_mass(p),"absent_edge_mass_5":edge_mass(q)})
    rec["metrics"]={"rows":rows,"transcript_one_epoch_chi2":tr1,"transcript_one_epoch_d2":d2tr1}
    rec["diagnostics"]={"solver":model.diagnostics(max_c=min(k,b))}; return finish(rec)


def _discrete_transition(cfg,mode:str):
    grid=Grid1D.make(float(cfg["x_max"]),int(cfg["nx"])); eta=float(cfg["eta"]); B=float(cfg["B"])
    tail_sigma=float(cfg.get("tail_sigma",10.0))
    if mode=="sgld":
        tau=float(cfg["tau"]); n_sub=int(round(tau/eta))
        if abs(n_sub*eta-tau)>1e-10: raise ValueError("eta must divide tau for SGLD bridge")
        base=DiscreteGaussianKernel1D(grid,float(cfg["m"]),float(cfg["A"]),B,eta,sgld_noise_std(eta),drop_tol=float(cfg.get("drop_tol",0.0)),tail_sigma=tail_sigma)
        trans=RepeatedTransition(base,n_sub); mu=abs(B)*math.sqrt(tau/2.0); noise_multiplier=None
    elif mode=="dpsgd":
        z=float(cfg["noise_multiplier"]); ns=dpsgd_noise_std(eta,B,z)
        base=DiscreteGaussianKernel1D(grid,float(cfg["m"]),float(cfg["A"]),B,eta,ns,drop_tol=float(cfg.get("drop_tol",0.0)),tail_sigma=tail_sigma)
        trans=base; mu=1.0/z; noise_multiplier=z
    else: raise ValueError(mode)
    return grid,trans,mu,noise_multiplier


def run_bridge(cfg:dict,mode:str)->dict:
    rec=base_record(f"bridge_{mode}",cfg); M,b,k=int(cfg["M"]),int(cfg["b"]),int(cfg["k"]); mass_tol=float(cfg.get("mass_loss_tol",1e-8))
    grid,trans,mu,z=_discrete_transition(cfg,mode)
    ref=ContinuousLangevin1D(grid,float(cfg["m"]),float(cfg["A"]),float(cfg["B"]),float(cfg.get("tau",1.0)),method="be",substeps=8)
    q0=ref.numerical_stationary_mass()
    rp=evolve_hidden_epoch(q0,trans.transition,M,b,k); rq=evolve_absent_epoch(q0,trans.transition,M)
    if rp.escaped_mass>mass_tol or rq.escaped_mass>mass_tol:
        rec["diagnostics"]={"escaped_present":rp.escaped_mass,"escaped_absent":rq.escaped_mass,"transition":trans.diagnostics(max_c=min(k,b))}
        rec["warnings"].append("escaped/truncated mass exceeds certification tolerance"); return finish(rec,"NUMERICAL_INVALID")
    p,ep=_safe_prob(rp.mass,mass_tol); q,eq=_safe_prob(rq.mass,mass_tol); cf=chi2_mass(p,q); cr=chi2_mass(q,p); tr=transcript_chi2_exact(M,b,k,mu)
    eps_grid=cfg.get("epsilon_grid",[0.0,0.1,0.25,0.5,1.0,2.0]); deltas=cfg.get("delta_targets",[1e-3,1e-5])
    profile=privacy_profile_mass(p,q,eps_grid)
    # Treat all escaped/truncated and latent-bookkeeping mass as a TV defect
    # between the finite-grid mechanism and the intended unbounded discrete kernel.
    tau_p=min(1.0,float(rp.escaped_mass)+abs(float(rp.latent_mass_error)))
    tau_q=min(1.0,float(rq.escaped_mass)+abs(float(rq.latent_mass_error)))
    profile_upper=privacy_profile_with_tv_defect(p,q,eps_grid,tau_p,tau_q)
    eps_at={str(d):epsilon_for_delta(p,q,float(d)) for d in deltas}
    eps_upper={str(d):epsilon_upper_for_delta_with_tv_defect(p,q,float(d),tau_p,tau_q) for d in deltas}
    tv=next((float(x["delta_symmetric"]) for x in profile if abs(float(x["epsilon"]))<1e-15), None)
    tv_upper=next((float(x["delta_symmetric_upper"]) for x in profile_upper if abs(float(x["epsilon"]))<1e-15), None)
    if tv is None:
        tv=float(0.5*np.sum(np.abs(p-q)))
    if tv_upper is None:
        tv_upper=min(1.0,tv+tau_p+tau_q)
    rec["metrics"]={
        "final_chi2_forward":cf,"final_chi2_reverse":cr,"final_chi2_symmetric_max":max(cf,cr),"final_d2_forward":d2_from_chi2(cf),"final_d2_reverse":d2_from_chi2(cr),
        "transcript_chi2":tr,"transcript_d2":d2_from_chi2(tr),"gap_chi2":tr/cf if cf>0 and np.isfinite(cf) else float("nan"),"gap_d2":d2_from_chi2(tr)/d2_from_chi2(cf) if cf>0 and np.isfinite(cf) else float("nan"),
        "mu":mu,"noise_multiplier":z,"privacy_profile":profile,"privacy_profile_upper":profile_upper,"tv_symmetric":tv,"tv_symmetric_upper":tv_upper,
        "epsilon_at_delta":eps_at,"epsilon_upper_at_delta":eps_upper,
        "epsilon_delta_1e3":eps_at.get("0.001",float("nan")),
        "epsilon_delta_1e5":eps_at.get("1e-05",float("nan")),
        "epsilon_upper_delta_1e3":eps_upper.get("0.001",float("nan")),
        "epsilon_upper_delta_1e5":eps_upper.get("1e-05",float("nan")),
        "tv_defect_present":tau_p,"tv_defect_absent":tau_q,
    }
    rec["diagnostics"]={"escaped_present":rp.escaped_mass,"escaped_absent":rq.escaped_mass,"present_mass_error":ep,"absent_mass_error":eq,
                        "present_edge_mass_5":edge_mass(p),"absent_edge_mass_5":edge_mass(q),"transition":trans.diagnostics(max_c=min(k,b)),
                        "reference_stationarity":ref.diagnostics(max_c=0)}
    # For the theorem-consistent SGLD bridge, chi^2 should remain numerically finite.
    # For vanilla DP-SGD extrapolation, tiny Gaussian tails can underflow in double
    # precision; the certified comparison therefore uses the symmetric privacy profile
    # and Feldman--Shenfeld epsilon bounds rather than forcing a tail-unstable chi^2.
    if mode=="sgld" and (not np.isfinite(cf) or not np.isfinite(cr)):
        return finish(rec,"NUMERICAL_INVALID")
    rec["diagnostics"]["chi2_finite"] = bool(np.isfinite(cf) and np.isfinite(cr))
    return finish(rec)


def run_job(job:dict)->dict:
    family=job["family"]; cfg=job["config"]
    if family=="gaussian": return run_gaussian(cfg)
    if family=="nonlinear": return run_nonlinear(cfg)
    if family=="memory": return run_memory(cfg)
    if family=="multiepoch": return run_multiepoch(cfg)
    if family=="bridge_sgld": return run_bridge(cfg,"sgld")
    if family=="bridge_dpsgd": return run_bridge(cfg,"dpsgd")
    raise ValueError(f"unknown family {family}")

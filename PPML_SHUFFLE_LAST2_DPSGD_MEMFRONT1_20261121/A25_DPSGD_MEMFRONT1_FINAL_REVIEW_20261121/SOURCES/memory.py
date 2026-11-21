from __future__ import annotations

import json, math, os, time, hashlib
from pathlib import Path
from typing import Any

import numpy as np
from scipy.sparse.linalg import eigs

from scalecert.cc_solver import Grid1D
from scalecert.discrete_kernel import DiscreteGaussianKernel1D, dpsgd_noise_std


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        f.flush(); os.fsync(f.fileno())


def canonical_id(cfg: dict) -> str:
    b=json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(b).hexdigest()[:20]


def make_kernel(cfg: dict):
    grid=Grid1D.make(float(cfg["x_max"]), int(cfg["nx"]))
    kernel=DiscreteGaussianKernel1D(
        grid, float(cfg["m"]), float(cfg["A"]), float(cfg["B"]),
        float(cfg["eta"]), dpsgd_noise_std(float(cfg["eta"]), float(cfg["B"]), float(cfg["z"])),
        drop_tol=float(cfg.get("drop_tol", 0.0)), tail_sigma=float(cfg["tail_sigma"]))
    return grid, kernel


def invariant_mass(kernel, tol=1e-12, max_iter=1_000_000):
    """Dominant normalized invariant/quasi-invariant mass of finite K0."""
    K, lost, _, _ = kernel.kernel(0)
    # Power iteration is deterministic and keeps the nonnegative Perron vector.
    q=np.full(K.shape[0], 1.0/K.shape[0], dtype=np.float64)
    for it in range(1, max_iter+1):
        q1=np.asarray(K @ q).ravel()
        mass=float(q1.sum())
        if not (mass > 0):
            raise RuntimeError("K0 invariant iteration lost all mass")
        q1 /= mass
        residual=float(np.abs(q1-q).sum())
        q=q1
        if residual <= tol:
            raw=np.asarray(K @ q).ravel()
            invariance=float(np.abs(raw-q).sum())
            escape=float(np.dot(lost,q))
            return q, {"iterations":it,"normalized_iteration_residual_l1":residual,
                       "qK0_minus_q_l1_unrenormalized":invariance,"stationary_escape":escape}
    raise RuntimeError(f"K0 invariant iteration did not converge in {max_iter} steps")


def memory_response(cfg: dict, out_dir: Path, progress_path: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    jid=canonical_id(cfg); out=out_dir/f"memory_{jid}.json"
    if out.exists():
        obj=json.loads(out.read_text()); append_jsonl(progress_path,{"event":"memory_job_skipped_existing","id":jid,"status":obj.get("status")}); return obj
    t0=time.time(); append_jsonl(progress_path,{"event":"memory_job_start","id":jid,"config":cfg,"pid":os.getpid()})
    grid,kernel=make_kernel(cfg)
    q,stat=invariant_mass(kernel, tol=float(cfg.get("stationary_tol",1e-12)))
    if stat["qK0_minus_q_l1_unrenormalized"] > float(cfg["stationarity_l1_tol"]):
        raise RuntimeError("discrete invariant stationarity residual exceeds frozen tolerance")
    K0=kernel.kernel(0)[0]; K1=kernel.kernel(1)[0]
    p=np.asarray(K1 @ q).ravel()
    a0=0.5*float(np.abs(p-q).sum())
    if a0 <= 0: raise RuntimeError("one-shock TV is zero")
    records=[]; prevL=None; prevR=None; closed=False; status="MEMORY_NOT_CLOSED"
    maxR=int(cfg["max_R"]); R=int(cfg["initial_R"])
    # Compute in chunks, persisting a checkpoint each horizon.
    while True:
        need=R+1-len(records)
        for _ in range(need):
            av=0.5*float(np.abs(p-q).sum())
            records.append(av)
            p=np.asarray(K0 @ p).ravel()
        a=np.asarray(records,dtype=np.float64)
        L=float(a.sum()**2 / np.dot(a,a))
        quarter=max(1,(len(a)+3)//4)
        tail=a[-quarter:]
        qsum=float(tail.sum()/a.sum())
        qsq=float(np.dot(tail,tail)/np.dot(a,a))
        endpoint=float(a[-1]/a0)
        relL=float("inf") if prevL is None else abs(L-prevL)/max(abs(L),1e-300)
        closed=bool(endpoint<=1e-4 and relL<=0.01 and qsum<=0.01 and qsq<=0.005)
        checkpoint={"id":jid,"config":cfg,"status":"MEMORY_CLOSED" if closed else "MEMORY_NOT_CLOSED",
                    "grid":{"x_max":grid.x_max,"nx":len(grid.x),"dx":grid.dx},"stationarity":stat,
                    "R":R,"a0":a0,"L_TV":L,"endpoint_ratio":endpoint,"relative_L_change_vs_previous_horizon":relL,
                    "last_quarter_sum_fraction":qsum,"last_quarter_sumsq_fraction":qsq,
                    "r_1e":next((i for i,x in enumerate(a) if x<=a0/math.e),None),
                    "r_1pct":next((i for i,x in enumerate(a) if x<=a0*.01),None),
                    "r_1e4":next((i for i,x in enumerate(a) if x<=a0*1e-4),None),
                    "response":a.tolist(),"elapsed_s":time.time()-t0}
        tmp=out.with_suffix(".tmp"); tmp.write_text(json.dumps(checkpoint,separators=(",",":"))); tmp.replace(out)
        append_jsonl(progress_path,{"event":"memory_horizon_checkpoint","id":jid,"R":R,"L_TV":L,"endpoint_ratio":endpoint,"tail_sum_frac":qsum,"tail_sumsq_frac":qsq,"closed":closed,"elapsed_s":time.time()-t0})
        if closed:
            status="MEMORY_CLOSED"; break
        if R>=maxR: break
        prevL=L; prevR=R; R=min(2*R,maxR)
    checkpoint["status"]=status
    checkpoint["response"]=[float(x) for x in records]
    tmp=out.with_suffix(".tmp"); tmp.write_text(json.dumps(checkpoint,separators=(",",":"))); tmp.replace(out)
    append_jsonl(progress_path,{"event":"memory_job_complete","id":jid,"status":status,"L_TV":checkpoint["L_TV"],"R":checkpoint["R"],"elapsed_s":time.time()-t0})
    return checkpoint


def primary_memory_configs():
    common={"m":.5,"A":.55,"B":.05,"eta":.1,"x_max":6.,"nx":1201,"drop_tol":0.,"tail_sigma":40.,
            "initial_R":256,"max_R":16384,"stationary_tol":1e-12,"stationarity_l1_tol":1e-10}
    return [{**common,"z":z,"line":f"P{'1' if z==2 else '2'}","grid_label":"BASE"} for z in (2.,4.)]


def all_memory_configs():
    configs=[]
    # One memory curve per distinct (eta,z); packet size k does not affect one shock.
    for eta,z,line in [(.1,2.,"P1"),(.1,4.,"P2"),(.1,8.,"R1"),(.05,2.,"R2"),(.05,4.,"R3"),(.2,2.,"R4"),(.2,4.,"R5")]:
        base={"m":.5,"A":.55,"B":.05,"eta":eta,"z":z,"x_max":6.,"nx":1201,"drop_tol":0.,"tail_sigma":40.,
              "initial_R":256,"max_R":16384,"stationary_tol":1e-12,"stationarity_l1_tol":1e-10,"line":line,"grid_label":"BASE"}
        configs.append(base)
        if line in ("P1","P2"):
            configs.extend([{**base,"nx":2401,"grid_label":"GRID"},{**base,"x_max":8.,"nx":1601,"grid_label":"DOMAIN"}])
    return configs

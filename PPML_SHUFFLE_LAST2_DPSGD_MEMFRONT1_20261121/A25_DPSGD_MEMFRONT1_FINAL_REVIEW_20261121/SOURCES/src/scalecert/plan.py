from __future__ import annotations
import copy


def _job(family,cfg,stage,tag=None):
    c=copy.deepcopy(cfg); c["stage"]=stage
    if tag is not None: c["tag"]=tag
    return {"family":family,"config":c}


def _resolve_batch_amplitude(case:dict) -> dict:
    c=copy.deepcopy(case)
    if "G" in c:
        b=int(c["b"]); c["B"]=float(c["G"])/b; c["collision_parameterization"]="fixed_G"
    else:
        c["collision_parameterization"]=c.get("collision_parameterization","fixed_B")
    return c


def build_plan(C:dict,stages=None):
    default=C.get("stages",["gaussian","nonlinear","convergence","memory","multiepoch","stress","usercap","bridges","bridge_convergence"])
    stages=set(stages or default); jobs=[]; common=C.get("common",{})

    if "gaussian" in stages:
        G=C["gaussian"]
        for M in G["M"]:
            for b in G["b"]:
                for case in G["cases"]:
                    if int(case["k"])<=M*b:
                        jobs.append(_job("gaussian",{"M":M,"b":b,**case,"series_tol":G.get("series_tol",1e-13)},"gaussian"))

    if "nonlinear" in stages:
        N=C["nonlinear"]
        for M in N["M"]:
            for raw in N["cases"]:
                case=_resolve_batch_amplitude(raw)
                jobs.append(_job("nonlinear",{**common,**N.get("solver",{}),"M":M,**case},"nonlinear"))

    if "convergence" in stages:
        V=C["convergence"]
        for anchor in V["anchors"]:
            for i,var in enumerate(V["variants"]):
                if var.get("M_values") is not None and int(anchor["M"]) not in set(int(x) for x in var["M_values"]):
                    continue
                label=var.get("variant_label",f"conv{i}")
                clean={k:v for k,v in var.items() if k!="M_values"}
                cfg={**common,**V.get("solver",{}),**anchor,**clean,"convergence_group":anchor.get("convergence_group","default"),"convergence_variant":i,"variant_label":label}
                jobs.append(_job("nonlinear",cfg,"convergence",tag=label))

    if "memory" in stages:
        R=C["memory"]
        for case in R["cases"]:
            jobs.append(_job("memory",{**common,**R.get("solver",{}),**case},"memory"))

    if "multiepoch" in stages:
        R=C["multiepoch"]
        for case in R["cases"]:
            jobs.append(_job("multiepoch",{**common,**R.get("solver",{}),**case,"E_values":R["E_values"]},"multiepoch"))

    if "stress" in stages:
        S=C["stress"]
        for raw in S.get("nonlinear_cases",[]):
            case=_resolve_batch_amplitude(raw)
            jobs.append(_job("nonlinear",{**common,**S.get("solver",{}),**case},"stress"))
        for case in S.get("gaussian_cases",[]): jobs.append(_job("gaussian",case,"stress"))

    if "usercap" in stages and "usercap" in C:
        U=C["usercap"]
        for M in U["M"]:
            for base in U["cases"]:
                cap=int(base["cap_k"])
                for actual in range(1,cap+1):
                    case={k:v for k,v in base.items() if k!="cap_k"}
                    case["k"]=actual; case["user_cap"]=cap; case["adjacency_sweep"]=True
                    case=_resolve_batch_amplitude(case)
                    jobs.append(_job("nonlinear",{**common,**U.get("solver",{}),"M":M,**case},"usercap",tag=f"cap{cap}_m{actual}"))

    if "bridges" in stages:
        B=C["bridges"]
        for M in B["M"]:
            for k in B["k"]:
                for eta in B["eta_sgld"]:
                    cfg={**common,**B.get("solver_sgld",B.get("solver",{})),"M":M,"b":B.get("b",1),"k":k,"m":B["m"],"A":B["A"],"B":B["B"],"tau":B["tau"],"eta":eta,"mass_loss_tol":B.get("mass_loss_tol",1e-8),"tail_sigma":B.get("tail_sigma",10.0)}
                    jobs.append(_job("bridge_sgld",cfg,"bridges"))
                for eta in B["eta_dpsgd"]:
                    for z in B["noise_multiplier"]:
                        cfg={**common,**B.get("solver_dpsgd",B.get("solver",{})),"M":M,"b":B.get("b",1),"k":k,"m":B["m"],"A":B["A"],"B":B["B"],"eta":eta,"noise_multiplier":z,"mass_loss_tol":B.get("mass_loss_tol",1e-8),"tail_sigma":B.get("tail_sigma",10.0)}
                        jobs.append(_job("bridge_dpsgd",cfg,"bridges"))

    if "bridge_convergence" in stages and "bridge_convergence" in C:
        R=C["bridge_convergence"]
        for anchor in R["anchors"]:
            for i,var in enumerate(R["variants"]):
                cfg={**common,**R.get("solver",{}),**anchor,**var,"discrete_convergence_variant":i,"discrete_convergence_group":anchor.get("discrete_convergence_group","default"),"mass_loss_tol":R.get("mass_loss_tol",1e-8),"tail_sigma":R.get("tail_sigma",10.0)}
                jobs.append(_job("bridge_dpsgd",cfg,"bridge_convergence",tag=var.get("variant_label",f"dconv{i}")))
    return jobs

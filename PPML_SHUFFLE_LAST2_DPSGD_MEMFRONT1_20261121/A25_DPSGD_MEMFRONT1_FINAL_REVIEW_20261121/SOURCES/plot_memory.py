import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
PKG=ROOT/"A25_DPSGD_MEMFRONT1_FINAL_REVIEW_20261121"
rows=list(csv.DictReader((PKG/"MEMORY_RESPONSE.csv").open()))
figdir=PKG/"figures"; figdir.mkdir(exist_ok=True)
colors={"P1":"#0072B2","P2":"#D55E00","R1":"#009E73","R2":"#CC79A7","R3":"#56B4E9","R4":"#E69F00","R5":"#000000"}
fig,ax=plt.subplots(figsize=(9,5.5))
for line in ("P1","P2","R1","R2","R3","R4","R5"):
    rr=[r for r in rows if r["line"]==line and r["grid"]=="BASE"]
    ax.plot([int(r["r"]) for r in rr],[float(r["a_r_over_a0"]) for r in rr],label=line,color=colors[line])
ax.set_yscale("log"); ax.set_xlabel("common absent-user steps r"); ax.set_ylabel(r"one-shock TV response $a_r/a_0$")
ax.set_title("MEMFRONT1 discrete privacy memory (new confirmatory measurements)"); ax.grid(True,which="both",alpha=.25); ax.legend(ncol=2)
fig.tight_layout(); fig.savefig(figdir/"01_discrete_memory_response.png",dpi=170); plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5.5))
for line in ("P1","P2","R1","R2","R3","R4","R5"):
    rr=[r for r in rows if r["line"]==line and r["grid"]=="BASE"]
    x=float(rr[0]["eta"]); y=float(rr[0]["z"])
    s=next(csv.DictReader((PKG/"MEMORY_SUMMARY.csv").open())) if False else None
    # Plot the measured horizon estimate at its actual truncation horizon; P2-family nonclosures are marked open.
    import json,glob
    d=next(json.loads(Path(p).read_text()) for p in glob.glob(str(ROOT/"memfront/run/memory_jobs/*.json")) if (lambda q:q["config"]["line"]==line and q["config"]["grid_label"]=="BASE")(json.loads(Path(p).read_text())))
    ax.scatter([y],[d["L_TV"]],label=f"{line} ({d['status']})",color=colors[line],marker="o" if d["status"]=="MEMORY_CLOSED" else "x",s=60)
ax.set_yscale("log"); ax.set_xlabel("noise multiplier z"); ax.set_ylabel(r"measured $L_{TV}$ (open / capped estimate)")
ax.set_title("Discrete memory-horizon estimates by line"); ax.grid(True,which="both",alpha=.25); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(figdir/"02_memory_horizon_summary.png",dpi=170); plt.close(fig)

from __future__ import annotations

import importlib, inspect, json, pkgutil
from typing import Any


def _find_symbol(name:str):
    root=importlib.import_module("PLD_accounting")
    if hasattr(root,name): return getattr(root,name), root.__name__
    if hasattr(root,"__path__"):
        for m in pkgutil.walk_packages(root.__path__, root.__name__+"."):
            try:
                mod=importlib.import_module(m.name)
            except Exception:
                continue
            if hasattr(mod,name): return getattr(mod,name),m.name
    raise AttributeError(f"Could not find PLD_accounting symbol {name}")


def parse_epsilon_range(x:Any)->dict:
    """Robust parser; exact raw repr is retained for audit."""
    out={"raw_repr":repr(x)}
    if isinstance(x,(int,float)):
        out.update(epsilon_lower=float(x),epsilon_upper=float(x)); return out
    if isinstance(x,(tuple,list)) and len(x)>=2:
        nums=[float(v) for v in x[:2]]
        out.update(epsilon_lower=min(nums),epsilon_upper=max(nums)); return out
    if isinstance(x,dict):
        keys={str(k).lower():v for k,v in x.items()}
        lower=next((float(v) for k,v in keys.items() if "lower" in k),None)
        upper=next((float(v) for k,v in keys.items() if "upper" in k),None)
        if upper is None and "epsilon" in keys: upper=float(keys["epsilon"])
        if lower is None and upper is not None: lower=upper
        if upper is not None:
            out.update(epsilon_lower=lower,epsilon_upper=upper); return out
    # namedtuple/dataclass style
    attrs={n:getattr(x,n) for n in dir(x) if not n.startswith("_")}
    lower=next((float(v) for n,v in attrs.items() if "lower" in n.lower() and isinstance(v,(int,float))),None)
    upper=next((float(v) for n,v in attrs.items() if "upper" in n.lower() and isinstance(v,(int,float))),None)
    if upper is not None:
        out.update(epsilon_lower=lower if lower is not None else upper,epsilon_upper=upper); return out
    raise TypeError(f"Unrecognized epsilon range result: {type(x)} {x!r}")


def gaussian_allocation_epsilon(delta:float,sigma:float,num_steps:int,num_selected:int,num_epochs:int=1,epsilon_accuracy:float=-1.0)->dict:
    fn,module=_find_symbol("gaussian_allocation_epsilon_range")
    sig=str(inspect.signature(fn))
    res=fn(delta=float(delta),sigma=float(sigma),num_steps=int(num_steps),num_selected=int(num_selected),num_epochs=int(num_epochs),epsilon_accuracy=float(epsilon_accuracy))
    out=parse_epsilon_range(res)
    out.update({"api_module":module,"api_signature":sig,"delta":float(delta),"sigma":float(sigma),"num_steps":int(num_steps),"num_selected":int(num_selected),"num_epochs":int(num_epochs),"epsilon_accuracy":float(epsilon_accuracy)})
    return out


def package_probe()->dict:
    root=importlib.import_module("PLD_accounting")
    fn,module=_find_symbol("gaussian_allocation_epsilon_range")
    return {"package_module":root.__name__,"version":getattr(root,"__version__",None),"api_module":module,"signature":str(inspect.signature(fn))}

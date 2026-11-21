from pathlib import Path
import yaml
from scalecert.plan import build_plan
from scalecert.experiments import canonical_id

ROOT=Path(__file__).resolve().parents[1]


def test_full_plan_freeze_and_required_user_lines():
    C=yaml.safe_load((ROOT/'configs/full_staged.yaml').read_text())
    jobs=build_plan(C)
    ids=[canonical_id(j['family'],j['config']) for j in jobs]
    assert len(jobs)==336
    assert len(ids)==len(set(ids))
    for z in (2.0,4.0):
        ms=sorted({j['config']['M'] for j in jobs
                   if j['family']=='bridge_dpsgd'
                   and j['config'].get('stage')=='bridges'
                   and j['config'].get('k')==2
                   and abs(j['config'].get('eta',0)-0.1)<1e-12
                   and j['config'].get('noise_multiplier')==z})
        assert ms==[16,32,64,128]
        conv=sorted({j['config']['M'] for j in jobs
                     if j['family']=='bridge_dpsgd'
                     and j['config'].get('stage')=='bridge_convergence'
                     and j['config'].get('discrete_convergence_group')==f'z{int(z)}'})
        assert conv==[32,64,128]

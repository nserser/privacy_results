from pathlib import Path
import sys,json
R=Path(__file__).resolve().parents[1]
expected='fde83c45bf12c4f7821832c7d5d2ae2841fc0910'
got=(R/'input/PLD_accounting_git/PINNED_COMMIT.txt').read_text().strip()
assert got==expected,(got,expected)
assert (R/'M0_VERIFICATION.json').exists()
assert (R/'FINMEM1_EXPECTED_JOB_MANIFEST.json').exists()
assert len(list((R/'final_state/jobs').glob('*.json')))==32
assert len([p for p in (R/'final_state/fs').glob('*.json') if p.name not in ('EXPECTED_JOBS.json','SUMMARY.json')])>=1
print(json.dumps({'status':'PORTABLE_INPUTS_PASS','fs_commit':got,'final_state_jobs':32,'fs_cached_results':len([p for p in (R/'final_state/fs').glob('*.json') if p.name not in ('EXPECTED_JOBS.json','SUMMARY.json')])}))

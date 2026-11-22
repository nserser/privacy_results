import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parent
def now():return datetime.now(timezone.utc).isoformat()
events=[
 {'stage':'M0','status':'PASS','details':'138/138 MEMFRONT2 and 38/38 FINMEM1 interim manifest entries verified; 22 core tests pass; 1-vs-4 worker equivalence pass; archived 14/14 P1 FS queries validated.'},
 {'stage':'M1-M2','status':'PASS','details':'P1 finite-horizon memory recalculated for 7 horizons across BASE/GRID/GRID2/DOMAIN; continuum and defect gates pass; source spectral reconstruction held-out validation passes.'},
 {'stage':'M3','status':'PASS','details':'Fresh CPU P1 final-state BASE curve complete 7/7; convergence anchors complete 9/9; 8 worker processes; no GPU in these jobs.'},
 {'stage':'M6-M7','status':'PASS','details':'P1 x-normalized scaling and archived FS comparison recomputed from fresh final-state upper bounds and reused exact FS lower bounds.'},
 {'stage':'M8','status':'PASS','details':'Fresh CPU k4 final-state 3/3; conditional k4 FS query stopped after M=1024 and 2048 both passed at both deltas.'},
 {'stage':'GPU parity','status':'PASS','slurm_job_id':'934','details':'Same-node RTX A5000 synthetic CPU/GPU kernel bitwise parity passed; immutable FINMEM1 report documents exact same-node full-accountant parity at M=256 and 1024; pinned backend commit verified.'},
 {'stage':'Independent audit','status':'PASS','details':'22/22 independent checks pass; exact expected-vs-observed P1 final state, memory, k4 query sets verified.'},
 {'stage':'Deferred','status':'NOT_RUN','details':'P2 FS and k1 FS not authorized/run; k4 M=4096 FS marked NOT_TRIGGERED after two consecutive wins.'},
 {'stage':'Superseded','status':'CANCELLED','slurm_job_id':'930','details':'Broad old GPU accountant sweep cancelled and excluded from canonical baseline.'},
 {'stage':'Superseded','status':'CANCELLED','slurm_job_id':'931','details':'No output; initial k4 CPU full-accountant parity smoke cancelled before any scientific query.'},
 {'stage':'Superseded','status':'CANCELLED','slurm_job_id':'932','details':'No output; CPU accountant smoke cancelled before any scientific query.'},
 {'stage':'Superseded','status':'CANCELLED','slurm_job_id':'933','details':'No output; redundant per-call CPU replay version cancelled before any scientific query.'},
]
with (ROOT/'FINMEM1A_JOB_LEDGER.jsonl').open('w') as f:
 for e in events:f.write(json.dumps({'timestamp_utc':now(),**e},sort_keys=True)+'\n')
print(f'WROTE {len(events)} ledger events')

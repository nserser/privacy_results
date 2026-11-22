import hashlib, json, os, platform, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def count_ok(p): return sum(line.rstrip().endswith(' OK') for line in Path(p).read_text().splitlines())
def run(cmd):
    return subprocess.check_output(cmd,text=True).strip()
data={
 'milestone':'PPML-SHUFFLE-LAST2-DPSGD-FINMEM1A',
 'timestamp_utc':run(['date','-u','+%Y-%m-%dT%H:%M:%SZ']),
 'historical_statuses_preserved':{'SCALECERT':'STOP_REALISTIC_DP_SGD','MEMFRONT1':'REPAIR_MEMFRONT','MEMFRONT2':'REPAIR_MEMFRONT2','FINMEM1_GPU_INTERIM':'NOT_A_SCIENTIFIC_DECISION'},
 'source_packages':{
   'MEMFRONT2_review_sha256_manifest_entries_verified':count_ok(ROOT/'M0/memfront2_sha_check.txt'),
   'MEMFRONT2_expected_entries':138,
   'FINMEM1_interim_sha256_manifest_entries_verified':count_ok(ROOT/'M0/interim_sha_check.txt'),
   'FINMEM1_interim_expected_entries':38,
   'FINMEM1_interim_snapshot_status':'19/40; immutable archived baseline source',
 },
 'implementation':{
   'memfront2_scientific_source_commit':'5a53dda02a56d03b29bbe477efc9c10258c0ba28',
   'feldman_shenfeld_commit':Path(ROOT/'M0/fs_commit.txt').read_text().strip(),
   'core_tests_complete_source':'PASS 22/22',
   'abbreviated_source_copy_test':'21 passed, 1 failed at collection because configs/full_staged.yaml is absent from abbreviated source subset; not used as canonical verification',
   'equivalence_1_worker_vs_4_workers':Path(ROOT/'M0/equiv_compare.log').read_text().strip(),
   'equivalence_records':8,
 },
 'archived_p1_fs':json.loads((ROOT/'P1_FS_VALIDATION.json').read_text()),
 'execution_policy':{'gpu_requested':False,'gpu_used_for_new_f1_jobs':False,'old_broad_slurm_job_930':'CANCELLED before FINMEM1A; partial records excluded from canonical P1 FS snapshot'},
 'result_reuse_policy':'Archived P1 FS values reused exactly; new P1 final-state and finite-memory calculations run in this FINMEM1A branch.',
 'status':'PASS' if count_ok(ROOT/'M0/memfront2_sha_check.txt')==138 and count_ok(ROOT/'M0/interim_sha_check.txt')==38 and Path(ROOT/'P1_FS_VALIDATION.json').exists() else 'FAIL'
}
(ROOT/'M0_VERIFICATION.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:v for k,v in data.items() if k not in ('archived_p1_fs',)},indent=2))

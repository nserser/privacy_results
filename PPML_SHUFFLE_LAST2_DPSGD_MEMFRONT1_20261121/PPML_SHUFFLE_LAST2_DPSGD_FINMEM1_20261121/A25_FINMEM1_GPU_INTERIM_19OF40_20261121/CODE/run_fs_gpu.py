#!/usr/bin/env python3
"""GPU-only rerun of the frozen FINMEM1 Feldman--Shenfeld query manifest.

This is an isolated backend override. It does not modify the pinned accountant or
the historical CPU outputs. One process is launched per GPU by Slurm.
"""
from __future__ import annotations

import argparse, fcntl, hashlib, json, os, platform, sys, tempfile, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG = Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_DPSGD_FINMEM1_UBUNTU_RUN_PACKAGE_20261121')
FS = Path('/users/anis.bkakria/privacy/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_INVESTIGATION_20261121/PPML_SHUFFLE_LAST2_EMP1R2_THEORYCOMPLY_FINAL_20261121/external/PLD_accounting')
PINNED = 'fde83c45bf12c4f7821832c7d5d2ae2841fc0910'
OUT = ROOT / 'gpu_investigation' / 'fs_gpu_results'
KERNEL_PATH = ROOT / 'gpu_investigation'


def expected_tasks():
    tasks = [(line, m, d, z, 2) for line, z, ms in (
        ('P1', 2.0, [256, 512, 1024, 2048, 4096, 8192, 16384]),
        ('P2', 4.0, [4096, 8192, 16384, 32768, 65536, 131072, 262144]),
    ) for m in ms for d in (1e-3, 1e-5)]
    tasks += [('P1', m, d, 2.0, k) for k in (1, 4) for m in (1024, 2048, 4096) for d in (1e-3, 1e-5)]
    return tasks


def task_id(task):
    line, m, delta, z, k = task
    payload = {'line': line, 'M': m, 'delta': delta, 'z': z, 'k': k, 'b': 1}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def append_progress(path, record):
    with path.open('a') as f:
        f.write(json.dumps({'time_utc_epoch': time.time(), **record}, sort_keys=True) + '\n')
        f.flush(); os.fsync(f.fileno())


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(obj, f, sort_keys=True, indent=2); f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, default=int(os.environ.get('SLURM_PROCID', '0')))
    ap.add_argument('--world', type=int, default=int(os.environ.get('SLURM_NTASKS', '1')))
    args = ap.parse_args()
    if args.world < 1 or not 0 <= args.rank < args.world: raise SystemExit('invalid rank/world')
    sys.path.insert(0, str(PKG / 'input/CORE_SOURCE/src'))
    sys.path.insert(0, str(FS))
    sys.path.insert(0, str(KERNEL_PATH))
    import cupy as cp
    import PLD_accounting.geometric_convolution as geom
    from scalecert.baseline_fs import gaussian_allocation_epsilon
    from deterministic_geometric_kernel import deterministic_geometric_kernel
    actual_commit = os.popen(f'git -C {FS} rev-parse HEAD').read().strip()
    if actual_commit != PINNED: raise SystemExit(f'PLD commit mismatch {actual_commit}')
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
    manifest_path = OUT / 'EXPECTED_JOBS.json'
    tasks = expected_tasks()
    manifest = {
        'milestone': 'FINMEM1_FROZEN_FS_GPU_BACKEND_RERUN',
        'operator_override': 'GPU explicitly enabled by user on 2026-11-21; historical CPU-only spec preserved as written',
        'accountant_commit': actual_commit,
        'kernel': 'deterministic output-bin gather with Kahan accumulation; fmad disabled',
        'gpu': gpu_name,
        'tasks': [dict(id=task_id(t), line=t[0], M=t[1], delta=t[2], z=t[3], k=t[4], b=1) for t in tasks],
    }
    manifest_hash = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if args.rank == 0 and not manifest_path.exists():
        atomic_json(manifest_path, manifest)
    for _ in range(300):
        if manifest_path.exists(): break
        time.sleep(0.1)
    if not manifest_path.exists(): raise SystemExit('expected manifest was not initialized')
    else:
        on_disk = json.loads(manifest_path.read_text())
        h_disk = hashlib.sha256(json.dumps(on_disk, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        if h_disk != manifest_hash: raise SystemExit('expected manifest mismatch')
    OUT.mkdir(parents=True, exist_ok=True)
    task_file = OUT / 'tasks.json'
    if args.rank == 0 and not task_file.exists():
        atomic_json(task_file, [{'id': task_id(t), 'task': t} for t in tasks])
    cursor = OUT / 'cursor.txt'
    ready = OUT / 'READY_JOB_ID.txt'
    if args.rank == 0:
        # Resume from the first absent output; existing results remain immutable.
        first_missing = next((i for i, t in enumerate(tasks) if not (OUT / 'outputs' / f'{task_id(t)}.json').exists()), len(tasks))
        fd, tmp = tempfile.mkstemp(prefix='cursor.', dir=OUT)
        with os.fdopen(fd, 'w') as f:
            f.write(f'{first_missing}\n'); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, cursor)
        ready.write_text(os.environ.get('SLURM_JOB_ID', 'LOCAL') + '\n')
    this_job = os.environ.get('SLURM_JOB_ID', 'LOCAL')
    for _ in range(300):
        if ready.exists() and ready.read_text().strip() == this_job: break
        time.sleep(0.1)
    if not ready.exists() or ready.read_text().strip() != this_job: raise SystemExit('work cursor was not initialized for this job')
    progress = OUT / f'worker_{args.rank}.jsonl'
    append_progress(progress, {'event': 'worker_start', 'rank': args.rank, 'world': args.world,
                              'gpu': gpu_name, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                              'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'manifest_sha256': manifest_hash})
    old_kernel = geom._geometric_kernel
    geom._geometric_kernel = deterministic_geometric_kernel
    while True:
        with cursor.open('r+') as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            i = int(f.read().strip() or 0)
            if i >= len(tasks):
                fcntl.flock(f, fcntl.LOCK_UN); break
            f.seek(0); f.truncate(); f.write(f'{i+1}\n'); f.flush(); os.fsync(f.fileno())
            fcntl.flock(f, fcntl.LOCK_UN)
        task = tasks[i]; ident = task_id(task)
        line, m, delta, z, k = task
        out_path = OUT / 'outputs' / f'{ident}.json'
        start = time.perf_counter()
        try:
            result = gaussian_allocation_epsilon(delta=delta, sigma=z, num_steps=m, num_selected=k, num_epochs=1)
            row = {'id': ident, 'line': line, 'M': m, 'delta': delta, 'sigma': z, 'k': k, 'b': 1,
                   'evidence_label': 'EXTERNAL_ACCOUNTANT', 'backend': 'DETERMINISTIC_GPU_GATHER',
                   'privacy_contract': 'FELDMAN_SHENFELD_RANDOM_ALLOCATION',
                   'operator_override': 'GPU enabled explicitly; historical CPU-only spec preserved',
                   'source_commit': actual_commit, 'gpu_model': gpu_name,
                   'slurm_job_id': os.environ.get('SLURM_JOB_ID'), 'rank': args.rank,
                   'wall_seconds': time.perf_counter() - start, **result}
            if not (0 <= row['epsilon_lower'] <= row['epsilon_upper'] and
                    all(map(__import__('math').isfinite, (row['epsilon_lower'], row['epsilon_upper'])))):
                raise ValueError('invalid epsilon interval')
            atomic_json(out_path, row)
            append_progress(progress, {'event': 'job_complete', 'id': ident, 'line': line, 'M': m, 'k': k,
                                      'delta': delta, 'wall_seconds': row['wall_seconds'],
                                      'epsilon_lower': row['epsilon_lower'], 'epsilon_upper': row['epsilon_upper']})
        except Exception as e:
            append_progress(progress, {'event': 'job_failed', 'id': ident, 'line': line, 'M': m, 'k': k,
                                      'delta': delta, 'error_type': type(e).__name__, 'error': str(e)})
            raise
    geom._geometric_kernel = old_kernel
    append_progress(progress, {'event': 'worker_complete', 'rank': args.rank, 'world': args.world})
    print(f'GPU_WORKER_COMPLETE rank={args.rank} world={args.world} device={gpu_name}', flush=True)


if __name__ == '__main__': main()

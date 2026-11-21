#!/usr/bin/env python3
"""Run independent Feldman--Shenfeld queries with deterministic aggregation.

Process-level parallelism changes runtime only: each (record, delta) query is
independent. Results are sorted by their frozen source identifiers before the
final JSON is written. Progress is emitted as counts, never partial values.
"""
import argparse
import concurrent.futures
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _query(task):
    from scalecert.baseline_fs import gaussian_allocation_epsilon

    row, delta, epsilon_accuracy = task
    z = float(row["config"]["noise_multiplier"])
    M = int(row["config"]["M"])
    k = int(row["config"]["k"])
    c = row["config"]
    base = {
        "source_experiment_id": row["experiment_id"],
        "source_family": "bridge_dpsgd",
        "M": M,
        "k": k,
        "E": 1,
        "noise_multiplier": z,
        "eta": float(c["eta"]),
        "delta": float(delta),
    }
    try:
        ans = gaussian_allocation_epsilon(
            delta=float(delta), sigma=z, num_steps=M, num_selected=k,
            num_epochs=1, epsilon_accuracy=epsilon_accuracy,
        )
        return {**base, **ans}
    except Exception as exc:  # retain failed queries as explicit records
        return {**base, "status": "FAILED", "error": repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--deltas", default="1e-3,1e-5")
    ap.add_argument("--epsilon-accuracy", type=float, default=-1.0)
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    if a.workers < 1:
        ap.error("--workers must be >= 1")

    from scalecert.baseline_fs import package_probe

    inp = pathlib.Path(a.input)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    deltas = [float(x) for x in a.deltas.split(",")]
    rows = []
    for path in sorted(inp.rglob("*.json")):
        try:
            record = json.loads(path.read_text())
        except Exception:
            continue
        config = record.get("config", {})
        if (record.get("status") != "OK"
                or record.get("family") != "bridge_dpsgd"
                or config.get("stage") != "bridges"
                or int(config.get("b", 1)) != 1):
            continue
        rows.append(record)

    tasks = [(r, delta, a.epsilon_accuracy) for r in rows for delta in deltas]
    if not tasks:
        raise SystemExit("no matched b=1 DP-SGD baseline rows")

    progress_path = out / "feldman_shenfeld_progress.jsonl"
    progress_path.write_text("")
    results = []
    start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures = [pool.submit(_query, task) for task in tasks]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            result = future.result()
            results.append(result)
            event = {
                "completed": completed,
                "expected": len(tasks),
                "workers": a.workers,
                "elapsed_seconds": round(time.time() - start, 3),
                "status": result.get("status", "OK"),
            }
            with progress_path.open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            print(f"baseline_progress {completed}/{len(tasks)} workers={a.workers}", flush=True)

    results.sort(key=lambda r: (
        r["source_experiment_id"], r["delta"], r["M"], r["k"],
        r["noise_multiplier"], r["eta"],
    ))
    lock_path = ROOT / "external/FELDMAN_SHENFELD_RUNTIME_LOCK.json"
    lock = json.loads(lock_path.read_text()) if lock_path.exists() else {}
    payload = {"probe": package_probe(), "runtime_lock": lock, "rows": results}
    # Canonical output filename expected by aggregation.
    (out / "feldman_shenfeld_baseline.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    if any(r.get("status") == "FAILED" for r in results):
        raise SystemExit(2)
    print(f"baseline rows={len(results)} workers={a.workers}", flush=True)


if __name__ == "__main__":
    main()

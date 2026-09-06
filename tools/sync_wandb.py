"""Import saved study metrics, including partial and failed runs, into W&B.

Safe to repeat from a timer while legacy jobs finish. It never changes their
frozen training configuration. No checkpoint or dataset is uploaded.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import wandb

from graded_resonators.telemetry import DEFAULTS, flatten, model_id, open_run


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def directories():
    for base, pattern in (
        ("/mnt/hot/execute/athena/work", "grf-*/attempt-*/source/results/grf-*"),
        ("/mnt/cold/research/execute/artifacts", "grf-*/attempt-*/results/grf-*"),
    ):
        yield from sorted(Path(base).glob(pattern))


def sync(directory, work, state):
    if read(directory / "telemetry.json", {}).get("mode") == "native":
        return  # The trainer owns these W&B IDs while it is running.
    execute_id = directory.name
    children = []
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        contract = read(child / "contract.json", {})
        result = read(child / "result.json", {})
        config = contract.get("config", result.get("config"))
        if not config:
            continue
        files = sorted(p for p in child.glob("*.json*") if p.is_file() and p.suffix in {".json", ".jsonl"})
        # Ignore a partially written last line in an actively growing log.
        contents = {p.name: p.read_bytes() for p in files}
        contract = json.loads(contents.get("contract.json", b"{}"))
        result = json.loads(contents.get("result.json", b"{}"))
        config = contract.get("config", result.get("config"))
        fingerprint = hashlib.sha256(b"".join(k.encode() + v for k, v in contents.items())).hexdigest()
        identity = model_id(execute_id, child.name)
        outcome = result.get("status", "in_progress_snapshot")
        record = {"name": child.name, "id": identity, "outcome": outcome,
                  "url": f"https://wandb.ai/{DEFAULTS['entity']}/{DEFAULTS['project']}/runs/{identity}",
                  "task": config.get("task"), "arm": config.get("arm"), "seed": config.get("seed"),
                  "stage": config.get("stage"), "best_validation_loss": result.get("best_validation_loss"),
                  "test_accuracy": result.get("test", {}).get("accuracy")}
        children.append(record)
        if state.get(identity) == fingerprint:
            continue
        run = open_run(identity, child.name, execute_id,
                       config | {"execute_run_id": execute_id,
                                 "training_source_commit": contract.get("source_commit", result.get("identity", {}).get("training_source")),
                                 "evidence_kind": "robustness" if "evaluations" in result else "training"},
                       work, job_type="imported-results")
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("validation/*", step_metric="epoch")
        uploaded = int(run.summary.get("uploaded_epochs", 0))
        rows = []
        for line in contents.get("metrics.jsonl", b"").splitlines(keepends=True):
            if not line.endswith(b"\n"):
                continue
            rows.append(json.loads(line))
        for row in rows:
            if row["epoch"] >= uploaded:
                run.log(flatten(row))
        if rows:
            run.summary["uploaded_epochs"] = rows[-1]["epoch"] + 1
        run.summary.update(flatten(result))
        run.summary.update({"outcome": outcome, "result": result, "contract": contract,
                            "imported_from_saved_metrics": True})
        failure = read(child / "failure.json")
        if failure:
            run.summary["failure"] = failure
        if "evaluations" in result:
            table = wandb.Table(columns=["kind", "strength", "loss", "accuracy", "event_fraction", "payload_rms"],
                data=[[r["kind"], r["strength"], *[r["metrics"].get(k) for k in ("loss", "accuracy", "event_fraction", "payload_rms")]]
                      for r in result["evaluations"]])
            run.log({"robustness": table})
            for row in result["evaluations"]:
                run.summary[f"robustness/{row['kind']}/{row['strength']}"] = row["metrics"]
        # Save an exact snapshot; don't symlink a growing legacy metrics file.
        snapshot = work / "snapshots" / identity
        snapshot.mkdir(parents=True, exist_ok=True)
        for name, data in contents.items():
            destination = snapshot / name
            destination.write_bytes(data)
            run.save(str(destination), base_path=str(snapshot), policy="now")
        run.finish(exit_code=0 if outcome in {"complete", "in_progress_snapshot"} else 2)
        state[identity] = fingerprint
        save_state(work, state)
        print(json.dumps(record), flush=True)
    if not children:
        return
    collection_hash = hashlib.sha256(json.dumps(children, sort_keys=True).encode()).hexdigest()
    if state.get(execute_id) != collection_hash:
        run = open_run(execute_id, execute_id, execute_id,
                       {"execute_run_id": execute_id, "evidence_kind": "collection"}, work,
                       job_type="collection")
        columns = list(children[0])
        run.log({"experiments": wandb.Table(columns=columns, data=[[r[k] for k in columns] for r in children])})
        run.summary.update({"models": len(children), "complete": sum(r["outcome"] == "complete" for r in children),
                            "failed": sum(r["outcome"] not in {"complete", "in_progress_snapshot"} for r in children),
                            "partial": sum(r["outcome"] == "in_progress_snapshot" for r in children),
                            "model_links": {r["name"]: r["url"] for r in children},
                            "phase": "imported_snapshot", "source_directory": str(directory)})
        run.finish()
        state[execute_id] = collection_hash
        save_state(work, state)


def save_state(work, state):
    path = work / "sync-state.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def sync_one(arguments):
    directory, work = arguments
    work = work / directory.name
    work.mkdir(parents=True, exist_ok=True)
    sync(directory, work, read(work / "sync-state.json", {}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="*", type=Path)
    parser.add_argument("--work", type=Path, default=Path.home() / ".local/state/graded-resonators-wandb")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(sync_one, [(p, args.work) for p in (args.directories or directories())]))


if __name__ == "__main__":
    main()

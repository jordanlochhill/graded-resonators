"""Seed-level primary summaries, with failed and missing seeds explicit."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ARMS = ("brf", "graded_brf", "graded_observation", "graded_static")


def primary_records(roots):
    records = {}
    for root in roots:
        for path in sorted(Path(root).rglob("result.json")):
            result = json.loads(path.read_text())
            config = result.get("config", {})
            if "best_epoch" not in result or config.get("stage") != "main":
                continue  # pilots, tuning and post-training perturbation results
            if config.get("neuron") or "parent" in config:
                continue  # ablations are not primary-seed replicates
            if any(key in config for key in ("train_limit", "validation_limit", "epoch_limit")):
                raise ValueError(f"Limited run labelled as primary: {path}")
            key = (config["task"], config["arm"], config["seed"])
            if key in records:
                raise ValueError(f"Duplicate primary seed {key}: {path} and {records[key]['path']}")
            if result["status"] == "complete" and result.get("test", {}).get("accuracy") is None:
                raise ValueError(f"Completed primary seed has no finite test accuracy: {path}")
            records[key] = {"result": result, "path": str(path)}
    return records


def statistics(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "mean_ci95": None}
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else None
    half = float(t.ppf(.975, n - 1) * sd / np.sqrt(n)) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd,
            "mean_ci95": [mean - half, mean + half] if half is not None else None}


def summarise(records, tasks, seeds=range(5)):
    summary = {"tasks": {}, "paired_differences_percentage_points": {},
               "interval_note": "Unadjusted two-sided Student t intervals across paired training seeds; not uncertainty over dataset sampling or corrected multiple tests."}
    for task in tasks:
        summary["tasks"][task] = {}
        summary["paired_differences_percentage_points"][task] = {}
        for arm in ARMS:
            complete, failed, missing = {}, {}, []
            for seed in seeds:
                record = records.get((task, arm, seed))
                if record is None:
                    missing.append(seed)
                elif record["result"]["status"] != "complete":
                    failed[str(seed)] = record["result"]["status"]
                else:
                    complete[seed] = record["result"]
            metrics = {}
            for metric in ("accuracy", "event_fraction", "payload_rms", "max_membrane_component"):
                metrics[metric] = statistics([value["test"][metric] for value in complete.values()])
            summary["tasks"][task][arm] = {"expected_seeds": list(seeds), "completed_seeds": sorted(complete),
                                         "failed_seeds": failed, "missing_seeds": missing,
                                         "conditional_on_completed_training": bool(failed or missing),
                                         "metrics": metrics}
            if arm != "brf":
                paired = []
                paired_seeds = []
                for seed, value in complete.items():
                    baseline = records.get((task, "brf", seed), {}).get("result")
                    if baseline and baseline["status"] == "complete":
                        paired.append(100 * (value["test"]["accuracy"] - baseline["test"]["accuracy"]))
                        paired_seeds.append(seed)
                summary["paired_differences_percentage_points"][task][arm] = statistics(paired) | {"seeds": paired_seeds}
    return summary

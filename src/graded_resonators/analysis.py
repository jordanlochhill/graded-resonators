"""Seed-level primary summaries, with failed and missing seeds explicit."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import t

ARMS = ("brf", "graded_brf", "graded_observation", "graded_static")


def select_learning_rates(results, rates=(.075, .025, .0075), seeds=(100, 101),
                          arms=ARMS, group_key="arm"):
    """Two complete validation-only seeds are required for an eligible rate."""
    indexed = {}
    for result in results:
        config = result["config"]
        if config["stage"] != "tune" or "test" in result:
            raise ValueError("Learning-rate selection accepts validation-only tuning results")
        key = config[group_key], config["lr"], config["seed"]
        if key[0] not in arms or key[1] not in rates or key[2] not in seeds:
            raise ValueError(f"Unexpected tuning cell: {key}")
        if key in indexed:
            raise ValueError(f"Duplicate tuning seed: {key}")
        indexed[key] = result
    decisions = {}
    for arm in arms:
        candidates = []
        for rate in rates:
            rows = []
            for seed in seeds:
                key = arm, rate, seed
                if key not in indexed:
                    raise ValueError(f"Missing tuning seed: {key}")
                result = indexed[key]
                rows.append({"seed": seed, "status": result["status"], "validation_loss": result["best_validation_loss"]})
            eligible = all(row["status"] == "complete" and row["validation_loss"] is not None
                           and np.isfinite(row["validation_loss"]) for row in rows)
            candidates.append({"rate": rate, "eligible": eligible, "seeds": rows,
                               "mean_validation_loss": float(np.mean([row["validation_loss"] for row in rows])) if eligible else None})
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        chosen = min(eligible, key=lambda row: (row["mean_validation_loss"], row["rate"])) if eligible else None
        decisions[arm] = {"candidates": candidates, "selected_rate": chosen["rate"] if chosen else None}
    return decisions


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
            if config.get("comparison_role") or config.get("validation_reduction", "sample_mean") != "sample_mean":
                continue  # protocol diagnostics cannot become additional primary seeds
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

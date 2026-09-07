"""Comparable per-model W&B runs and one Execute-linked collection run.

Local JSON remains the recovery source. IDs are deterministic, so the same
records can be imported again after an interrupted upload without duplicates.
"""

import hashlib
import json
import os
from pathlib import Path

import wandb

DEFAULTS = {"project": "graded-resonators", "entity": "jordanlochhill-nmtafe"}


def live_settings(manifest):
    """All new experiments are online; old manifests inherit the project."""
    settings = DEFAULTS | manifest.get("wandb", {})
    if settings.pop("mode", "online") != "online" or os.environ.get("WANDB_MODE", "online") != "online":
        raise ValueError("New study runs require live W&B; offline/disabled mode is not allowed")
    if not settings.get("project") or not settings.get("entity"):
        raise ValueError("Live W&B requires a project and entity")
    return settings


def model_id(execute_id, name):
    return "grf-" + hashlib.sha256(f"{execute_id}/{name}".encode()).hexdigest()[:24]


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = f"{prefix}/{key}" if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            result[name] = item
    return result


def experiment_name(config):
    return config.get("name", f"{config['task']}-{config['arm']}-s{config['seed']}")


def open_run(identity, name, group, config, directory, settings=None, job_type="training"):
    Path(directory).mkdir(parents=True, exist_ok=True)
    run = wandb.init(**live_settings({"wandb": settings or DEFAULTS}), mode="online",
                      id=identity, name=name, group=group,
                      job_type=job_type, config=config, resume="allow",
                      reinit="create_new", dir=str(directory),
                      settings=wandb.Settings(quiet=True, disable_git=True,
                                              x_disable_stats=True, init_timeout=60))
    if run is None or run.settings.mode != "online":
        raise RuntimeError("W&B did not establish an online run; refusing to run unobserved")
    print(f"Live W&B: {run.url}", flush=True)
    return run


class Telemetry:
    def __init__(self, manifest, output):
        self.settings = live_settings(manifest)
        self.output = Path(output)
        self.execute_id = self.output.name
        self.current = None
        self.current_name = None
        self.collection = None
        if self.settings:
            self.output.mkdir(parents=True, exist_ok=True)
            self.collection = open_run(self.execute_id, self.execute_id, self.execute_id,
                                       {"manifest": manifest}, self.output, self.settings,
                                       job_type="collection")
            self.collection.summary["phase"] = "training"
            (self.output / "telemetry.json").write_text(json.dumps({
                "mode": "native", **self.settings, "url": self.collection.url}) + "\n")

    def start(self, config):
        if not self.settings:
            return None
        name = experiment_name(config)
        if self.current_name != name:
            if self.current:
                self.current.finish()
            self.current = open_run(model_id(self.execute_id, name), name, self.execute_id,
                                    config | {"execute_run_id": self.execute_id},
                                    self.output, self.settings)
            self.current.define_metric("epoch")
            self.current.define_metric("*", step_metric="epoch")
            self.current_name = name
            self.collection.summary[name] = {"url": self.current.url, "outcome": "running"}
        return self.current

    def initialisation(self, config, audit):
        run = self.start(config)
        if run:
            run.summary["initialisation"] = audit

    def epoch(self, config, row):
        run = self.start(config)
        if run:
            run.log(flatten(row))
            run.summary["uploaded_epochs"] = row["epoch"] + 1

    def result(self, config, result):
        run = self.start(config)
        if run:
            run.summary.update(flatten(result))
            run.summary["outcome"] = result["status"]
            run.summary["result"] = result
            self.collection.summary[self.current_name] = {
                "url": run.url, "outcome": result["status"],
                "best_validation_loss": result.get("best_validation_loss"),
                "test_accuracy": result.get("test", {}).get("accuracy")}
            run.finish(exit_code=0 if result["status"] == "complete" else 2)
            self.current, self.current_name = None, None

    def evaluation(self, config, row, index):
        run = self.start(config)
        run.define_metric("condition_index")
        run.define_metric("evaluation/*", step_metric="condition_index")
        run.log({"condition_index": index, "condition": row["kind"],
                 **flatten(row, "evaluation")})

    def finish(self, exit_code=0):
        if self.current:
            self.current.finish(exit_code=exit_code)
        if self.collection:
            self.collection.summary["phase"] = "complete" if exit_code == 0 else "failed_or_incomplete"
            self.collection.finish(exit_code=exit_code)

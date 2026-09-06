"""Create bounded per-arm jobs from the original-task primary manifests."""

from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path

project = Path(__file__).resolve().parents[1]
root = project / "manifests"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--platform", choices=("athena", "kaya"), default="athena")
parser.add_argument("--tasks", nargs="+", choices=("smnist", "psmnist", "ecg"), default=("smnist", "psmnist", "ecg"))
args = parser.parse_args()
for task in args.tasks:
    source = root / f"main-{task}.json"
    manifest = json.loads(source.read_text())
    recovered = []
    for arm in dict.fromkeys(item["arm"] for item in manifest["experiments"]):
        shard = deepcopy(manifest)
        shard["experiments"] = [item for item in manifest["experiments"] if item["arm"] == arm]
        shard["description"] += f" Scheduler shard: {arm}, all five seeds."
        shard["parent_manifest"] = {"path": source.name, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        shard["cost"] = {"gpu_type": "rtx4090", "gpus": 1,
                         "time_limit": "03:00:00" if task == "ecg" else "10:00:00",
                         "estimated_seconds": manifest["cost"]["estimated_seconds"] / 4,
                         "estimate_basis": manifest["cost"]["estimate_basis"]}
        suffix = ""
        if args.platform == "kaya":
            timing = project / "measurements/timing/grf-kaya-qualification-20260906" / f"timing-{task}"
            rows = [json.loads(line) for line in (timing / "metrics.jsonl").read_text().splitlines()]
            contract = json.loads((timing / "contract.json").read_text())
            samples = contract["data"]["arrays"]["train_y.npy"]["shape"][0]
            train_samples = samples - int(samples * .1)
            updates = math.ceil(train_samples / manifest["defaults"]["batch"]) * manifest["defaults"]["epochs"]
            estimate = math.ceil(updates * len(shard["experiments"]) * rows[-1]["batch_seconds_median"] * 1.15)
            shard["cost"] = {"gpu_type": "h100", "gpus": 1, "time_limit": "03:00:00" if task == "ecg" else "16:00:00",
                             "estimated_seconds": estimate,
                             "estimate_basis": "Original update budget times second-epoch H100 qualification batch median, plus 15% for validation/setup.",
                             "qualification_metrics_sha256": hashlib.sha256((timing / "metrics.jsonl").read_bytes()).hexdigest()}
            shard["description"] += " Kaya H100 allocation; the experiment recipe is unchanged."
            suffix = "-h100"
        (root / f"main-{task}-{arm.replace('_', '-')}{suffix}.json").write_text(json.dumps(shard, indent=2) + "\n")
        recovered += shard["experiments"]
    assert sorted(recovered, key=lambda x: (x["arm"], x["seed"])) == sorted(manifest["experiments"], key=lambda x: (x["arm"], x["seed"]))

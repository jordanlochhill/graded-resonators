"""Create bounded per-arm jobs from the original-task primary manifests."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "manifests"
for task in ("smnist", "psmnist", "ecg"):
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
        (root / f"main-{task}-{arm.replace('_', '-')}.json").write_text(json.dumps(shard, indent=2) + "\n")
        recovered += shard["experiments"]
    assert sorted(recovered, key=lambda x: (x["arm"], x["seed"])) == sorted(manifest["experiments"], key=lambda x: (x["arm"], x["seed"]))

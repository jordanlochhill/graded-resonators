"""Select threshold-excess conditions using validation only; emit confirmations."""

import argparse
import hashlib
import json
from pathlib import Path

from graded_resonators.analysis import select_learning_rates

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("results", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
parent = json.loads((root / "manifests/exact-gradient-shd.json").read_text())
paths = sorted(args.results.rglob("result.json"))
results = [json.loads(p.read_text()) for p in paths]
for result in results:
    condition = result["config"]["gradient_condition"]
    if result["config"].get("neuron") != parent["conditions"][condition]:
        raise ValueError(f"Changed neuron definition: {condition}")
decisions = select_learning_rates(results, arms=tuple(parent["conditions"]),
                                  group_key="gradient_condition")
manifest = {**parent, "description": "Five-seed confirmation of validation-selected exact versus surrogate gradients for threshold-excess emission, with fixed or learned thresholds.",
            "defaults": parent["defaults"] | {"stage": "main"},
            "experiments": [], "validation_selection": decisions,
            "selection_sources": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths],
            "ineligible_conditions": [],
            "cost": parent["cost"] | {"estimated_seconds": 3200,
                    "estimate_basis": "At most twenty full SHD trainings, approximately 150 seconds each, plus setup."}}
for condition, decision in decisions.items():
    if decision["selected_rate"] is None:
        manifest["ineligible_conditions"].append(condition)
        continue
    for seed in range(5):
        manifest["experiments"].append({"name": f"confirmed-{condition}-s{seed}",
            "gradient_condition": condition, "seed": seed, "lr": decision["selected_rate"],
            "neuron": parent["conditions"][condition]})
# Re-run the small fixed-threshold SG arm here for all five seeds under this
# source pin; compare its overlapping seeds with the earlier excess ablation,
# and count them once within each explicitly separate comparison.
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
print(json.dumps({"new_trainings": len(manifest["experiments"]),
                  "ineligible_conditions": manifest["ineligible_conditions"]}))

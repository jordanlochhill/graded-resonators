"""Select equal-budget validation rates and author five-seed confirmation recipes."""

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
paths = sorted(args.results.rglob("result.json"))
results = [json.loads(path.read_text()) for path in paths]
decisions = select_learning_rates(results)
parent = json.loads((root / "manifests/main-shd.json").read_text())
manifest = {key: value for key, value in parent.items() if key not in {"description", "experiments", "cost", "decision_rule"}}
manifest.update(description="Five-seed confirmation of validation-selected SHD learning rates; original-rate selections reuse the primary runs.",
                decision_rule="Report separately from fixed-recipe primary comparisons. Rates require two complete selection seeds; failed rates are ineligible. Choose lowest mean validation loss, breaking exact ties by smaller rate. No test-based selection.",
                cost={"gpu_type": "rtx4090", "gpus": 1, "time_limit": "02:00:00", "estimated_seconds": 3000,
                      "estimate_basis": "At most twenty additional SHD trainings, approximately 150 seconds each."},
                validation_selection=decisions,
                selection_sources=[{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
                reused_primary=[], ineligible_arms=[], experiments=[])
for arm, decision in decisions.items():
    rate = decision["selected_rate"]
    if rate is None:
        manifest["ineligible_arms"].append(arm)
    elif rate == parent["defaults"]["lr"]:
        manifest["reused_primary"].append(arm)
    else:
        manifest["experiments"] += [{"arm": arm, "seed": seed, "lr": rate,
                                    "name": f"confirmed-{arm}-s{seed}"} for seed in range(5)]
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
print(json.dumps({"manifest": str(args.output), "new_trainings": len(manifest["experiments"]),
                  "reused_primary": manifest["reused_primary"], "ineligible_arms": manifest["ineligible_arms"]}, indent=2))

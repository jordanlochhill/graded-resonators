"""Recover the published SHD mean from the pinned release's validation-selected logs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

from graded_resonators.analysis import statistics

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("reference", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.reference, text=True).strip()
if commit != "1a42b8c8aceedb13cae3b2327774c2fcc04fd696":
    raise SystemExit("Use the BRF release pinned by the study")
rows = []
for seed in range(5):
    path = args.reference / f"experiments/SHD/csv_files/shd_best_model_run_{seed}.csv"
    with path.open() as stream:
        raw = list(csv.DictReader(stream))
    validation = {float(r["step"]): float(r["value"]) for r in raw if r["metric"] == "Loss/val"}
    accuracy = {float(r["step"]): float(r["value"]) for r in raw if r["metric"] == "Accuracy/test"}
    epoch = min(validation, key=lambda e: (validation[e], e))
    rows.append({"upstream_run": seed, "selected_epoch": int(epoch), "test_accuracy_percent": accuracy[epoch],
                 "path": str(path.relative_to(args.reference)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({"upstream_commit": commit,
    "selection": "Minimum validation loss in released CSV, ties earliest epoch; test accuracy is read at that epoch only.",
    "seeds": rows, "accuracy_percent": statistics([r["test_accuracy_percent"] for r in rows]),
}, indent=2, allow_nan=False) + "\n")
print(args.output)

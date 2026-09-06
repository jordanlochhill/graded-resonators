"""Plot every primary training trajectory and export seed-level diagnostics.

These are training observations, not an additional selection rule. Thin lines
retain individual seeds; thick lines show the mean of the available trajectories
at each epoch. Missing and failed seeds remain explicit in the accompanying JSON.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graded_resonators.analysis import ARMS, primary_records, summarise

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sources", type=Path, nargs="+")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--task", default="shd")
args = parser.parse_args()
records = primary_records(args.sources)
summary = summarise(records, [args.task])
groups = summary["tasks"][args.task]
if any(group["missing_seeds"] for group in groups.values()):
    raise SystemExit("Training diagnostics require all planned primary seed outcomes")

panels = [
    (("train", "loss"), "Training loss per supervised step", 1, "linear"),
    (("validation", "accuracy"), "Validation accuracy (%)", 100, "linear"),
    (("validation", "event_fraction"), "Validation event fraction (%)", 100, "linear"),
    (("train", "payload_rms"), "Training payload RMS", 1, "log"),
    (("train", "max_membrane_component"), "Peak training membrane component", 1, "log"),
    (("gradient_norm_max",), "Peak gradient norm (summed loss)", 1, "log"),
]
colours = ["#314d63", "#1c8276", "#b37135", "#806298"]
labels = ["Binary BRF", "Graded BRF", "Graded observation", "Graded static"]
fig, axes = plt.subplots(2, 3, figsize=(10, 5.8))
rows = []
for arm, colour, label in zip(ARMS, colours, labels):
    trajectories = []
    for seed in range(5):
        record = records[args.task, arm, seed]
        result = record["result"]
        path = Path(record["path"]).with_name("metrics.jsonl")
        epochs = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        if len({epoch["epoch"] for epoch in epochs}) != len(epochs):
            raise ValueError(f"Duplicate epoch in {path}")
        if result["status"] == "complete" and len(epochs) != result["config"]["epochs"]:
            raise ValueError(f"Incomplete trajectory for completed training: {path}")
        trajectories.append(epochs)
        rows.append({
            "task": args.task, "arm": arm, "seed": seed, "status": result["status"],
            "epochs_recorded": len(epochs), "selected_epoch": result["best_epoch"] + 1 if result["best_epoch"] >= 0 else None,
            "peak_membrane_component": max((e["train"]["max_membrane_component"] for e in epochs), default=None),
            "peak_gradient_norm": max((e["gradient_norm_max"] for e in epochs), default=None),
            "warm_batch_seconds_median": float(np.median([e["batch_seconds_median"] for e in epochs[1:]])) if len(epochs) > 1 else None,
            "source": str(path),
        })
    for ax, (keys, title, scale, yscale) in zip(axes.flat, panels):
        by_epoch = {}
        for epochs in trajectories:
            values = []
            for epoch in epochs:
                value = epoch
                for key in keys:
                    value = value[key]
                values.append(scale * value)
                by_epoch.setdefault(epoch["epoch"], []).append(scale * value)
            ax.plot([e["epoch"] + 1 for e in epochs], values, color=colour, alpha=.22, lw=.7)
        indices = sorted(by_epoch)
        ax.plot([i + 1 for i in indices], [np.mean(by_epoch[i]) for i in indices], color=colour, lw=1.7, label=label)
        ax.set(title=title, xlabel="Completed training epochs", yscale=yscale)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.title.set_fontsize(9)
        ax.grid(alpha=.12)

failed = sum(len(group["failed_seeds"]) for group in groups.values())
fig.suptitle(f"{args.task.upper()}: five seeds per variant; {failed} failed trainings", fontsize=11)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=4, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, .06, 1, .96))
args.output.mkdir(parents=True, exist_ok=True)
fig.savefig(args.output / "training.pdf", bbox_inches="tight")
fig.savefig(args.output / "training.png", dpi=170, bbox_inches="tight")
with (args.output / "training-seeds.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(args.output / "training-summary.json").write_text(json.dumps({
    "task": args.task, "groups": groups,
    "aggregation": "Thin lines are individual seeds; thick lines are arithmetic means of recorded values at that epoch. Failures truncate trajectories; no forward filling. Peak metrics are maxima within each epoch, not state trajectories. Gradients use the summed temporal training loss.",
    "seeds": rows,
}, indent=2, allow_nan=False) + "\n")
print(f"Wrote diagnostics for {len(rows)} primary outcomes to {args.output}")

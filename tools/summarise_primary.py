"""Build primary result tables and a figure from curated, unique seed records."""

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
parser.add_argument("--tasks", nargs="+", default=["smnist", "psmnist", "ecg", "shd"])
parser.add_argument("--allow-partial", action="store_true")
args = parser.parse_args()
records = primary_records(args.sources)
summary = summarise(records, args.tasks)
missing = [(task, arm, value["missing_seeds"]) for task, arms in summary["tasks"].items()
           for arm, value in arms.items() if value["missing_seeds"]]
if missing and not args.allow_partial:
    raise SystemExit(f"Missing planned primary seeds (use --allow-partial only for a labelled progress view): {missing}")
summary["partial"] = bool(missing)
args.output.mkdir(parents=True, exist_ok=True)
(args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
with (args.output / "seeds.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=["task", "arm", "seed", "status", "test_accuracy", "event_fraction", "payload_rms", "parameters", "source"])
    writer.writeheader()
    for key, record in sorted(records.items()):
        task, arm, seed = key
        if task not in args.tasks:
            continue
        result = record["result"]
        test = result.get("test", {})
        writer.writerow({"task": task, "arm": arm, "seed": seed, "status": result["status"],
                         "test_accuracy": test.get("accuracy"), "event_fraction": test.get("event_fraction"),
                         "payload_rms": test.get("payload_rms"), "parameters": result["parameters"], "source": record["path"]})
labels = ["Binary\nBRF", "Graded\nBRF", "Adaptive\nobservation", "Fixed\nobservation"]
colours = ["#314d63", "#1c8276", "#b37135", "#806298"]
rows = (len(args.tasks) + 1) // 2
fig, axes = plt.subplots(rows, min(2, len(args.tasks)), figsize=(7.2, 2.8 * rows), squeeze=False)
for ax, task in zip(axes.flat, args.tasks):
    for i, arm in enumerate(ARMS):
        group = summary["tasks"][task][arm]
        points = [100 * records[task, arm, seed]["result"]["test"]["accuracy"] for seed in group["completed_seeds"]]
        if points:
            ax.scatter(i + np.linspace(-.1, .1, len(points)), points, s=14, alpha=.65, color=colours[i])
            stats = group["metrics"]["accuracy"]
            ax.errorbar(i, 100 * stats["mean"], yerr=100 * stats["sd"] if stats["sd"] is not None else None,
                        fmt="_", ms=13, capsize=3, color=colours[i])
        suffix = f"{len(points)}/5"
        if group["failed_seeds"]:
            suffix += f"; {len(group['failed_seeds'])} failed"
        ax.text(i, 1.015, suffix, transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
    ax.set(title=task, ylabel="Test accuracy (%)", xticks=range(4), xticklabels=labels)
    ax.set_xlim(-.5, 3.5)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.15)
for ax in list(axes.flat)[len(args.tasks):]:
    ax.set_visible(False)
fig.suptitle("Partial results: planned seeds still missing" if missing else "Primary comparison: points are seeds; bars are mean ± sample SD", fontsize=10)
fig.tight_layout()
fig.savefig(args.output / "accuracy.pdf", bbox_inches="tight")
fig.savefig(args.output / "accuracy.png", dpi=160, bbox_inches="tight")

# Accuracy and event activity are distinct outcomes: show both for every seed.
fig, axes = plt.subplots(rows, min(2, len(args.tasks)), figsize=(7.2, 3.3 * rows), squeeze=False)
table_labels = ["Binary BRF", "Graded BRF", "Graded observation", "Graded static"]
for ax, task in zip(axes.flat, args.tasks):
    table = []
    for arm, colour, label in zip(ARMS, colours, table_labels):
        group = summary["tasks"][task][arm]
        tests = [records[task, arm, seed]["result"]["test"] for seed in group["completed_seeds"]]
        if not tests:
            table.append(f"{label} & --- & --- & 0/5 " + r"\\")
            continue
        accuracy = group["metrics"]["accuracy"]
        activity = group["metrics"]["event_fraction"]
        ax.scatter([100 * t["event_fraction"] for t in tests], [100 * t["accuracy"] for t in tests],
                   s=17, alpha=.55, color=colour)
        ax.errorbar(100 * activity["mean"], 100 * accuracy["mean"],
                    xerr=100 * activity["sd"] if activity["sd"] is not None else None,
                    yerr=100 * accuracy["sd"] if accuracy["sd"] is not None else None,
                    fmt="D", ms=5, capsize=3, color=colour, label=label)
        def cell(stats):
            return f"{100 * stats['mean']:.2f}" + (f" $\\pm$ {100 * stats['sd']:.2f}" if stats["sd"] is not None else "")
        table.append(f"{label} & {cell(accuracy)} & {cell(activity)} & {len(tests)}/5 " + r"\\")
    table_header = "\n".join([r"\begin{tabular}{lrrr}", r"\toprule",
                               r"Variant & Accuracy (\%) & Event fraction (\%) & Completed\\", r"\midrule"])
    (args.output / f"{task}-table.tex").write_text(table_header + "\n" + "\n".join(table)
                                                + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
    ax.set(xlabel="Test event fraction (%)", ylabel="Test accuracy (%)")
    if len(args.tasks) > 1:
        ax.set_title(task)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
for ax in list(axes.flat)[len(args.tasks):]:
    ax.set_visible(False)
if missing:
    fig.suptitle("Partial results: planned seeds still missing", fontsize=10)
fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, .13 / rows, 1, 1))
fig.savefig(args.output / "activity-accuracy.pdf", bbox_inches="tight")
fig.savefig(args.output / "activity-accuracy.png", dpi=170, bbox_inches="tight")
print(json.dumps({"missing_groups": missing, "output": str(args.output)}, indent=2))

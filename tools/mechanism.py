"""Regenerate isolated-neuron traces and the Euler stability region on CPU."""

import os
os.environ["JAX_PLATFORMS"] = "cpu"
from dataclasses import replace
import json
from pathlib import Path
import argparse

import jax.numpy as jnp
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from graded_resonators.model import ARMS, forward

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("measurements/mechanism"))
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans", "pdf.fonttype": 42})
p = {"input": jnp.ones((1, 1)), "recurrent": jnp.zeros((1, 1)), "readout": jnp.ones((1, 1)),
     "omega": jnp.array([15.]), "damping": jnp.array([1.]), "tau": jnp.array([20.])}
x = jnp.zeros((300, 1, 1)).at[0, 0, 0].set(230).at[100, 0, 0].set(140)
traces = {}
for name, neuron in ARMS.items():
    _, output = forward(p, x, replace(neuron, recurrent=False), trace=True)
    traces[name] = {key: np.asarray(value[:, 0, 0]) for key, value in zip(("readout", "event", "payload", "real", "imag", "refractory"), output)}
np.savez(args.output / "traces.npz", **{f"{name}_{key}": value for name, record in traces.items() for key, value in record.items()})
record = {"scope": "Isolated neuron, fixed external impulses, no recurrent synaptic coupling; not a trained task result.",
          "steps": 300, "dt": .01, "omega": 15., "damping_offset": 1., "impulses": {"0": 230, "100": 140},
          "counts": {name: int(t["event"].sum()) for name, t in traces.items()}}
(args.output / "contract.json").write_text(json.dumps(record, indent=2) + "\n")
colors = {"brf": "#294f73", "graded_brf": "#b65d3d", "graded_observation": "#39846e", "graded_static": "#795b92"}
labels = {"brf": "Binary BRF", "graded_brf": "Graded BRF", "graded_observation": "Graded, threshold adapts", "graded_static": "Graded, fixed threshold"}
fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True, layout="constrained")
for name in ("brf", "graded_observation"):
    axes[0].plot(traces[name]["real"], color=colors[name], label="Damping adapts" if name == "brf" else "Damping fixed")
axes[0].set(ylabel="Real membrane", title="Event-dependent damping changes the oscillation")
axes[0].legend(loc="upper right", frameon=False, ncols=2)
for name in ("brf", "graded_brf"):
    axes[1].plot(traces[name]["payload"], color=colors[name], label=labels[name], lw=1.3)
axes[1].set(ylabel="Transmitted value", title="Graded events carry membrane amplitude")
axes[1].legend(frameon=False, ncols=2)
for index, name in enumerate(ARMS):
    t = np.flatnonzero(traces[name]["event"])
    axes[2].vlines(t, index - .24, index + .24, color=colors[name])
axes[2].set(yticks=range(4), yticklabels=[labels[name] for name in ARMS], xlabel="Time step", ylim=(-.6, 3.6), title="Threshold adaptation changes event timing")
for ax in axes:
    ax.grid(alpha=.15)
fig.savefig(args.output / "mechanism.pdf")
fig.savefig(args.output / "mechanism.png", dpi=160)
plt.close(fig)
w = np.linspace(0, 1, 1001)
upper = 2 * np.sqrt(1 - w ** 2)
fig, ax = plt.subplots(figsize=(6.7, 3.1), layout="constrained")
ax.set_facecolor("#f7e7e4")
ax.fill_between(w, 0, upper, color="#dbeee5")
ax.plot(w, upper, color="#287766", lw=2)
ax.text(.16, .7, "Contracting isolated Euler step", color="#246452")
ax.text(.64, 1.9, "Magnitude exceeds one", color="#914b41")
ax.set(xlim=(0, 1), ylim=(0, 2.2), xlabel=r"Angular increment $\delta\omega$", ylabel=r"Damping offset $\delta(b' + q)$")
fig.savefig(args.output / "stability.pdf")
fig.savefig(args.output / "stability.png", dpi=160)
print(json.dumps(record))

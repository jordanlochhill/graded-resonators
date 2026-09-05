"""Frozen-checkpoint SHD perturbations; all calibration uses validation data.

Packets are deleted after the event is observed. Additive Gaussian input noise
is applied to the binned input, without clipping or a change of front end.
Neither experiment is an acoustic-noise or neuromorphic-energy measurement.
"""

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .data import batches, datasets, digest
from .model import Neuron, forward
from .train import evaluate_batch, metric_dict, write_json


def sample_random(indices, steps, width, seed, kind):
    """Common random numbers independent of model arm and evaluation batching."""
    values = []
    for index in indices:
        rng = np.random.default_rng(np.random.SeedSequence([seed, int(index), kind]))
        values.append(rng.random((steps, width), dtype=np.float32) if kind == 0
                      else rng.standard_normal((steps, width), dtype=np.float32))
    return np.stack(values, axis=1)


def calibrate(p, dataset, permutation, config, neuron, quantile):
    amplitudes = []
    for x, _, mask in batches(dataset, config["evaluation_batch"], config["task"], permutation):
        _, trace = forward(p, x, neuron, trace=True)
        sent = np.abs(np.asarray(trace[2])[:, mask.astype(bool)])
        if not np.isfinite(sent).all():
            raise ValueError("Non-finite validation payload cannot be calibrated")
        amplitudes.append(sent[sent > 0])
    values = np.concatenate(amplitudes)
    if not values.size:
        return {"clip": 1.0, "active_components": 0, "quantile": quantile,
                "note": "Silent validation payload; fixed unit range"}
    return {"clip": float(np.quantile(values, quantile)), "quantile": quantile,
            "active_components": int(values.size), "maximum": float(values.max())}


def perturbation_evaluation(p, dataset, permutation, config, neuron, kind, strength, seed):
    values, weights = [], []
    batch_size = config["evaluation_batch"]
    indices = dataset[2]
    for batch_index, (x, y, mask) in enumerate(batches(dataset, batch_size, config["task"], permutation)):
        real = indices[batch_index * batch_size:(batch_index + 1) * batch_size]
        padded = np.pad(real, (0, batch_size - len(real)), mode="edge")
        keep = None
        if kind == "packet_loss":
            keep = sample_random(padded, x.shape[0], config["hidden"], seed, 0) >= strength
        elif kind == "input_noise":
            x = x + strength * sample_random(padded, x.shape[0], x.shape[2], seed, 1)
        _, metrics = evaluate_batch(p, x, y, mask, neuron, config["task"], keep)
        values.append(np.asarray(metrics))
        weights.append(float(mask.sum()))
    values = np.stack(values)
    means = np.average(values, weights=weights, axis=0)
    means[3] = np.sqrt(np.average(values[:, 3] ** 2, weights=weights))
    means[4] = values[:, 4].max()
    return metric_dict(means) | {"samples": int(sum(weights))}


def latency(p, dataset, permutation, config, neuron, repetitions):
    x, _, _ = next(batches(dataset, config["evaluation_batch"], config["task"], permutation))
    x = jnp.asarray(x)
    for _ in range(3):
        jax.block_until_ready(forward(p, x, neuron))
    durations = []
    for _ in range(repetitions):
        start = time.perf_counter()
        jax.block_until_ready(forward(p, x, neuron))
        durations.append(time.perf_counter() - start)
    return {"batch": int(x.shape[1]), "steps": int(x.shape[0]),
            "repetitions": repetitions, "median_seconds": float(np.median(durations)),
            "seconds": durations, "scope": "Dense JAX forward including emission statistics; input already on device"}


def run(checkpoint_directory, protocol, data_root, output):
    directory, output = Path(checkpoint_directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    training = json.loads((directory / "result.json").read_text())
    contract = json.loads((directory / "contract.json").read_text())
    config = training["config"] | {"evaluation_batch": protocol["batch"]}
    if config["stage"] != "main" or config["task"] != "shd":
        raise ValueError("Robustness requires a full-budget SHD training configuration")
    if training["status"] != "complete":
        result = {"status": "ineligible_failed_training", "training_status": training["status"],
                  "config": training["config"], "protocol": protocol}
        write_json(output / "result.json", result)
        return result
    identity = {"checkpoint_sha256": digest(directory / "best.npz"),
                "training_source": contract["source_commit"],
                "training_config_sha256": contract["config_sha256"],
                "protocol": protocol,
                "evaluation_source": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()}
    if (output / "result.json").exists():
        old = json.loads((output / "result.json").read_text())
        if old["identity"] != identity:
            raise ValueError("Existing evaluation has different provenance")
        return old
    split, permutation, data = datasets(data_root, "shd", config["split_seed"])
    if data != contract["data"]:
        raise ValueError("Evaluation dataset differs from training provenance")
    neuron = Neuron(**contract["neuron"])
    with np.load(directory / "best.npz") as f:
        p = {key[2:]: jnp.asarray(f[key]) for key in f.files if key.startswith("p_")}
    calibration = calibrate(p, split["validation"], permutation, config, neuron, protocol["clip_quantile"])
    result = {"status": "complete", "identity": identity, "config": training["config"], "calibration": calibration,
              "devices": [str(d) for d in jax.devices()], "jax": jax.__version__,
              "latency": latency(p, split["validation"], permutation, config, neuron, protocol["latency_repetitions"]),
              "evaluations": []}
    conditions = [("clean", 0.0, neuron)]
    if neuron.payload != "binary":
        conditions += [("payload_bits", bits, replace(neuron, payload_bits=bits, payload_clip=calibration["clip"]))
                       for bits in protocol["payload_bits"]]
    conditions += [("packet_loss", p, neuron) for p in protocol["packet_loss"]]
    conditions += [("input_noise", sigma, neuron) for sigma in protocol["input_noise"]]
    for kind, strength, variant in conditions:
        metrics = perturbation_evaluation(p, split["test"], permutation, config, variant,
                                          kind, strength, protocol["perturbation_seed"])
        row = {"kind": kind, "strength": strength, "metrics": metrics}
        result["evaluations"].append(row)
        print(json.dumps({"checkpoint": str(directory), **row}), flush=True)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or jax.default_backend() != "gpu":
        raise SystemExit("Checkpoint evaluation must run in a Slurm GPU allocation")
    manifest = json.loads(args.manifest.read_text())
    results = []
    for entry in manifest["checkpoints"]:
        results.append(run(entry["path"], manifest["protocol"], args.data, args.output / entry["name"]))
    write_json(args.output / "summary.json", results)


if __name__ == "__main__":
    main()

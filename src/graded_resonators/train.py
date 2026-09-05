"""Manifest-driven, resumable training with held-out checkpoint selection."""

import argparse
from dataclasses import asdict, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .data import batches, datasets
from .model import ARMS, initialise, objective


def write_json(path, value):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_checkpoint(path, p, m, v, step, epoch, best_loss, best_epoch):
    values = {f"{group}_{key}": np.asarray(value) for group, tree in (("p", p), ("m", m), ("v", v)) for key, value in tree.items()}
    values.update(step=step, epoch=epoch, best_loss=best_loss, best_epoch=best_epoch)
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as f:
        np.savez(f, **values)
    tmp.replace(path)


@partial(jax.jit, static_argnames=("neuron", "task"))
def train_step(p, m, v, step, x, y, mask, lr, neuron, task):
    (loss, metrics), g = jax.value_and_grad(objective, has_aux=True)(p, x, y, mask, neuron, task)
    step = step + 1
    m = jax.tree.map(lambda a, b: .9 * a + .1 * b, m, g)
    v = jax.tree.map(lambda a, b: .999 * a + .001 * b ** 2, v, g)
    p = jax.tree.map(lambda p, m, v: p - lr * (m / (1 - .9 ** step)) / (jnp.sqrt(v / (1 - .999 ** step)) + 1e-8), p, m, v)
    norm = jnp.sqrt(sum((z ** 2).sum() for z in jax.tree.leaves(g)))
    finite = jnp.isfinite(loss) & jnp.isfinite(norm) & jnp.all(jnp.array([jnp.all(jnp.isfinite(z)) for z in jax.tree.leaves(p)]))
    return p, m, v, step, metrics, norm, finite


evaluate_batch = jax.jit(objective, static_argnames=("neuron", "task"))


def evaluate(p, dataset, permutation, config, neuron, limit=None):
    values, weights, elapsed = [], [], time.perf_counter()
    for x, y, mask in batches(dataset, config["evaluation_batch"], config["task"], permutation, limit=limit):
        _, metrics = evaluate_batch(p, x, y, mask, neuron, config["task"])
        values.append(np.asarray(metrics))
        weights.append(float(mask.sum()))
    values = np.stack(values)
    mean = np.average(values, weights=weights, axis=0)
    mean[3] = np.sqrt(np.average(values[:, 3] ** 2, weights=weights))
    mean[4] = values[:, 4].max()
    return metric_dict(mean) | {"seconds": time.perf_counter() - elapsed, "samples": int(sum(weights))}


def metric_dict(metrics):
    names = ("loss", "accuracy", "event_fraction", "payload_rms", "max_membrane_component")
    return {name: float(value) if np.isfinite(value) else None for name, value in zip(names, metrics)}


def run(config, output, data_root):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "result.json").exists():
        result = json.loads((output / "result.json").read_text())
        if result["config"] != config:
            raise ValueError("Completed result has a different configuration")
        return result
    neuron = replace(ARMS[config["arm"]], **config.get("neuron", {}))
    split, permutation, data_record = datasets(data_root, config["task"], config["split_seed"])
    p = initialise(config["seed"], config["inputs"], config["hidden"], config["classes"],
                   config["omega_range"], config["damping_range"], config["tau_std"], neuron)
    contract = {"config": config, "neuron": asdict(neuron), "data": data_record,
                "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
                "parameters": sum(z.size for z in p.values()), "started_unix": time.time()}
    contract["config_sha256"] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if (output / "contract.json").exists():
        old = json.loads((output / "contract.json").read_text())
        if old["config_sha256"] != contract["config_sha256"] or old["source_commit"] != contract["source_commit"]:
            raise ValueError("Resume source/config differs; use a new result directory")
    else:
        write_json(output / "contract.json", contract)
    m, v = [jax.tree.map(jnp.zeros_like, p) for _ in range(2)]
    step, start_epoch, best_loss, best_epoch = jnp.array(0), 0, float("inf"), -1
    if (output / "last.npz").exists():
        with np.load(output / "last.npz") as f:
            p, m, v = [{k: jnp.asarray(f[f"{prefix}_{k}"]) for k in p} for prefix in ("p", "m", "v")]
            step, start_epoch = jnp.asarray(f["step"]), int(f["epoch"]) + 1
            best_loss, best_epoch = float(f["best_loss"]), int(f["best_epoch"])
    run_start = time.perf_counter()
    status = "complete"
    limit = config.get("train_limit")
    val_limit = config.get("validation_limit")
    epochs = config.get("epoch_limit", config["epochs"])
    for epoch in range(start_epoch, epochs):
        tick = time.perf_counter()
        lr = config["lr"] * (1 - epoch / config["epochs"])
        measurements, weights, gradient_norms, batch_seconds = [], [], [], []
        for x, y, mask in batches(split["train"], config["batch"], config["task"], permutation,
                                   shuffle_seed=config["seed"] * 100000 + epoch, limit=limit):
            batch_start = time.perf_counter()
            p, m, v, step, metrics, norm, finite = train_step(p, m, v, step, x, y, mask, lr, neuron, config["task"])
            if not bool(finite):
                status = "nonfinite"
                break
            measurements.append(np.asarray(metrics))
            weights.append(float(mask.sum()))
            gradient_norms.append(float(norm))
            batch_seconds.append(time.perf_counter() - batch_start)
        if status != "complete":
            write_json(output / "failure.json", {"epoch": epoch, "step": int(step), "status": status,
                                                  "elapsed_seconds": time.perf_counter() - run_start})
            break
        training = np.average(measurements, weights=weights, axis=0)
        training[3] = np.sqrt(np.average(np.asarray(measurements)[:, 3] ** 2, weights=weights))
        training[4] = np.asarray(measurements)[:, 4].max()
        validation = evaluate(p, split["validation"], permutation, config, neuron, val_limit)
        if validation["loss"] is None:
            status = "nonfinite_validation"
            break
        if validation["loss"] < best_loss:
            best_loss, best_epoch = validation["loss"], epoch
            save_checkpoint(output / "best.npz", p, m, v, step, epoch, best_loss, best_epoch)
        save_checkpoint(output / "last.npz", p, m, v, step, epoch, best_loss, best_epoch)
        row = {"epoch": epoch, "step": int(step), "lr": lr, "train": metric_dict(training),
               "validation": validation, "gradient_norm_mean": float(np.mean(gradient_norms)),
               "gradient_norm_max": float(np.max(gradient_norms)), "epoch_seconds": time.perf_counter() - tick,
               "batch_seconds_median": float(np.median(batch_seconds)), "best_epoch": best_epoch}
        with (output / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps({"task": config["task"], "arm": config["arm"], "seed": config["seed"], **row}), flush=True)
    result = {"status": status, "config": config, "best_epoch": best_epoch, "best_validation_loss": best_loss if np.isfinite(best_loss) else None,
              "steps": int(step), "seconds_this_invocation": time.perf_counter() - run_start,
              "parameters": contract["parameters"], "memory_stats": jax.devices()[0].memory_stats()}
    # Pilot runs deliberately never evaluate test accuracy.
    if status == "complete" and config["stage"] != "pilot":
        with np.load(output / "best.npz") as f:
            selected = {k: jnp.asarray(f[f"p_{k}"]) for k in p}
        result["test"] = evaluate(selected, split["test"], permutation, config, neuron)
    write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if not args.cpu and not os.environ.get("SLURM_JOB_ID"):
        raise SystemExit("GPU experiments must run inside a Slurm allocation; use --cpu for local diagnostics")
    if args.cpu:
        jax.config.update("jax_platforms", "cpu")
    elif jax.default_backend() != "gpu":
        raise RuntimeError("GPU job has no usable JAX GPU backend")
    manifest = json.loads(args.manifest.read_text())
    results = []
    for experiment in manifest["experiments"]:
        config = manifest["defaults"] | experiment
        name = config.get("name", f"{config['task']}-{config['arm']}-s{config['seed']}")
        results.append(run(config, args.output / name, args.data))
    write_json(args.output / "summary.json", results)
    if any(r["status"] != "complete" for r in results):
        raise SystemExit(2)


if __name__ == "__main__":
    main()

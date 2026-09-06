"""Generate Execute explanation metadata directly from a committed manifest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--run-id", required=True)
parser.add_argument("--execute", type=Path, default=Path.home() / "Documents/execute")
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--platform", choices=("athena", "kaya"), default="athena")
parser.add_argument("--dry-run", action="store_true")
parser.add_argument("--depends-on", action="append", default=[], help="Execute run id and afterok/afterany condition")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest_path = args.manifest.resolve()
relative = str(manifest_path.relative_to(root))
manifest = json.loads(manifest_path.read_text())
platform = args.platform
if manifest["cost"]["gpu_type"] != ("h100" if platform == "kaya" else "rtx4090"):
    raise SystemExit("Manifest GPU type must match the selected platform")
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
wall = manifest["cost"]["time_limit"]
if " " in wall:
    raise SystemExit("Resolve and commit the compute envelope before submitting")
uv = shutil.which("uv")
if not uv:
    raise SystemExit("uv is required on the submission host")
module = manifest.get("module", "graded_resonators.train")
if module not in {"graded_resonators.train", "graded_resonators.robustness"}:
    raise SystemExit("Unsupported experiment module")
command = ["uv", "run", "--frozen", "--extra", "gpu", "python", "-m", module, relative,
           "--data", str(args.data.resolve()), "--output", f"results/{args.run_id}"]
proposal = {
    "queued_version": 1, "run_id": args.run_id,
    "at": datetime.now(timezone.utc).isoformat(), "by": "jordan",
    "tags": manifest.get("tags", ["graded-resonators", "kairos-audio"]),
    "design_link": manifest.get("planning_workspace"),
    "description": manifest["description"],
    "manifest": {"path": relative, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 "source_commit": commit, "resolved": manifest},
}

with tempfile.TemporaryDirectory(prefix="graded-resonators-proposal-") as directory:
    proposal_path = Path(directory) / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
    resources = {"athena": ("research", "gpu", "rtx4090", "8", "32G"),
                 "kaya": ("sh001", "data-inst", "h100", "16", "64G")}[platform]
    account, partition, gpu, cpus, memory = resources
    uv_directory = str(Path(uv).parent) if platform == "athena" else "/home/jhill/.local/bin"
    argv = [sys.executable, "-m", "execute_flow.author", "--desired-root", str(args.execute),
            "--platform", platform, "--run-id", args.run_id, "--project-root", str(root),
            "--account", account, "--partition", partition, "--nodes", "1", "--gpus-per-node", "1",
            "--gpu-type", gpu, "--cpus-per-task", cpus, "--mem", memory, "--time", wall,
            "--capability", "gpu=true",
            "--provenance-file", str(proposal_path), "--env", "PYTHONUNBUFFERED=1",
            "--env", "XLA_PYTHON_CLIENT_PREALLOCATE=false",
            "--env", f"PATH={uv_directory}:/usr/local/bin:/usr/bin:/bin",
            "--env", "UV_LINK_MODE=copy"]
    if manifest.get("wandb"):
        argv += ["--capability", "wandb=true", "--env", "WANDB_PROJECT=" + manifest["wandb"]["project"],
                 "--env", "WANDB_ENTITY=" + manifest["wandb"]["entity"]]
    if platform == "kaya":
        argv += ["--artifact-policy", "dependent", "--artifact-path", f"results/{args.run_id}",
                 "--artifact-destination", f"{args.run_id}/attempt-1",
                 "--env", "UV_CACHE_DIR=/scratch/sh001/jhill/uv-cache",
                 "--env", "UV_PYTHON_INSTALL_DIR=/scratch/sh001/jhill/uv-python",
                 "--env", "TMPDIR=/scratch/sh001/jhill/tmp",
                 "--prelude", "mkdir -p /scratch/sh001/jhill/tmp /scratch/sh001/jhill/uv-cache /scratch/sh001/jhill/uv-python"]
    else:
        argv += ["--artifact-policy", "none"]
    if args.dry_run:
        argv.append("--dry-run")
    for dependency in args.depends_on:
        argv += ["--depends-on", dependency]
    argv += ["--", *command]
    env = os.environ | {"PYTHONPATH": str(args.execute / "lib")}
    subprocess.run(argv, cwd=root, env=env, check=True)

"""Generate Execute explanation metadata directly from a committed manifest."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("manifest", type=Path)
parser.add_argument("--run-id", required=True)
parser.add_argument("--execute", type=Path, default=Path.home() / "Documents/execute")
parser.add_argument("--data", type=Path, required=True)
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
manifest_path = args.manifest.resolve()
relative = str(manifest_path.relative_to(root))
manifest = json.loads(manifest_path.read_text())
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
wall = manifest["cost"]["time_limit"]
if " " in wall:
    raise SystemExit("Resolve and commit the compute envelope before submitting")
command = ["uv", "run", "--frozen", "--extra", "gpu", "python", "-m", "graded_resonators.train", relative,
           "--data", str(args.data.resolve()), "--output", f"results/{args.run_id}"]
proposal = {
    "proposal_version": 1, "run_id": args.run_id, "platform": "athena",
    "created_at": datetime.now(timezone.utc).isoformat(), "proposed_by": "codex",
    "programme": "graded-resonators", "tier": "experiment", "title": manifest["description"].split(".")[0],
    "rationale": manifest["description"], "decision_rule": manifest["decision_rule"],
    "manifest": {"path": relative, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 "source_commit": commit, "description": manifest["description"],
                 "planning_workspace": manifest.get("planning_workspace"), "resolved": manifest},
    "baseline": {"reference": "Higuchi et al. ICML 2024 BRF", "upstream_commit": "1a42b8c8aceedb13cae3b2327774c2fcc04fd696"},
    "resolved_args": {"command": command}, "overrides": [], "cost": manifest["cost"],
}
with tempfile.TemporaryDirectory(prefix="graded-resonators-proposal-") as directory:
    proposal_path = Path(directory) / "proposal.json"
    proposal_path.write_text(json.dumps(proposal, indent=2) + "\n")
    argv = [sys.executable, "-m", "execute_flow.author", "--desired-root", str(args.execute),
            "--platform", "athena", "--run-id", args.run_id, "--project-root", str(root),
            "--account", "research", "--partition", "gpu", "--nodes", "1", "--gpus-per-node", "1",
            "--gpu-type", "rtx4090", "--cpus-per-task", "8", "--mem", "32G", "--time", wall,
            "--capability", "gpu=true", "--artifact-policy", "none", "--policy-programme", "graded-resonators",
            "--proposal-file", str(proposal_path), "--env", "PYTHONUNBUFFERED=1",
            "--env", "XLA_PYTHON_CLIENT_PREALLOCATE=false"]
    if args.dry_run:
        argv.append("--dry-run")
    argv += ["--", *command]
    env = os.environ | {"PYTHONPATH": str(args.execute / "lib")}
    subprocess.run(argv, cwd=root, env=env, check=True)

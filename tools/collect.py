"""Copy completed result records, preserving bytes and a SHA-256 inventory.

Usage: python tools/collect.py --output measurements/primary RESULTS_ROOT ...
Checkpoints and datasets remain outside Git. A missing summary means a run is
unfinished or interrupted and needs inspection before it can be curated here.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("sources", nargs="+", type=Path)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
for source in args.sources:
    if not (source / "summary.json").is_file():
        raise SystemExit(f"No terminal summary: {source}; inspect interruption before collection")
    destination = args.output / source.name
    inventory = {}
    paths = [source / "summary.json"]
    for name in ("contract.json", "metrics.jsonl", "result.json", "failure.json"):
        paths += sorted(source.glob(f"*/{name}"))
    for path in paths:
        relative = path.relative_to(source)
        target = destination / relative
        contents = path.read_bytes()
        if target.exists() and target.read_bytes() != contents:
            raise SystemExit(f"Refusing to replace different evidence: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        inventory[str(relative)] = {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}
    record = {"run": source.name, "files": inventory}
    (destination / "inventory.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"Collected {len(paths)} records: {destination}")

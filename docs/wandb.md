# Viewing the study in W&B

Project: https://wandb.ai/jordanlochhill-nmtafe/graded-resonators

Each model/seed has its own run. Filter by `config.task`, `config.arm`,
`config.gradient_condition`, `config.lr`, `config.seed` and `config.stage`.
`train/loss`, `validation/loss`, accuracy, event fraction, payload RMS,
membrane maximum and gradient norms use common metric names. Learned-threshold
curves are under `threshold_summary/`. A collection run with the Execute ID
links the models belonging to each submitted workload; the runs site links
to this collection. Robustness results use a table of perturbation kind,
strength and metrics, plus per-condition summary values.

`summary.outcome` is the scientific outcome. `complete`, numerical failure
and `in_progress_snapshot` are distinct. A completed upload of a partial
snapshot does not mean training finished. Pilots and validation-only tuning
remain labelled as such; no missing test accuracy is invented.

New training manifests enable the `wandb` project/entity settings. The trainer
streams per-epoch metrics and initialisation diagnostics. Local JSON files
remain the recovery source. The Execute submission wrapper declares telemetry
and supplies the same project/entity to the runs-site link resolver.

Older accepted jobs had telemetry disabled. `tools/sync_wandb.py` imports their
actual saved metrics and returned Kaya artifacts without changing their frozen
workloads. It uploads JSON provenance and metrics, excluding datasets and
checkpoints. Deterministic model IDs and uploaded-epoch markers permit repeated
syncs without duplicating models or completed epochs. Four importer processes
handle independent Execute jobs; state is isolated per Execute ID under
`~/.local/state/graded-resonators-wandb/`. Native live telemetry is excluded
from this legacy importer to avoid concurrent writers.

The Athena user timer `graded-resonators-wandb.timer` refreshes legacy records
every five minutes after a sync completes. Kaya curves appear when artifacts
return; this bridge cannot stream files that have not left Kaya. Imported
links live in `runs/RUN_ID/telemetry.json` beside the frozen Execute spec.

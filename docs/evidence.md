# Evidence ledger

## Implementation qualification, 6 September 2026

The independent BRF recurrence, network output, spike count, summed temporal
cross-entropy and gradients with respect to inputs and all trainable parameters
match the authors' pinned PyTorch implementation within the declared float32
tolerances (`tests/test_model.py`). Causal prefix, streaming, analytic polar
impulse response, masked final batch and upstream SHD bin edges are tested.
These are implementation checks, not learned benchmark results.

## Isolated-neuron illustration

`tools/mechanism.py` generates `measurements/mechanism/`. Two fixed input impulses,
no recurrent synapses. Event counts are 3/3/4/21 for binary BRF, graded BRF,
observation-only adaptation and fixed observation. The binary and graded BRF
states coincide here because payload is not fed back. This result does not
predict task accuracy or show the network has linear dynamics.

## SHD pilot

Execute `grf-shd-pilot-20260906a` failed before training: the scheduler's PATH did
not include uv. No model evidence. Preserved scheduler record: job 718, exit 127.

Execute `grf-shd-pilot-20260906b`, source `b197809`, job 719, completed four arms,
one seed, two epochs on 1,024 training / 256 validation samples. No test examples
were evaluated. Curated complete metrics and contracts are under
`measurements/pilot-shd-20260906/`; full checkpoints remain in the Execute work
directory. All arms stayed finite and training loss decreased. Median update
time was approximately 28 ms on Athena's RTX 4090.

The observation-only arm briefly reached membrane component magnitude 3,267 and
gradient norm 11,470, recovering by the next epoch. This is a warning about
unbounded recurrent amplitude feedback, not proof that observation adaptation
is worse. Its full-budget behaviour and equal-budget learning-rate controls
must be examined before interpreting it. Pilot validation accuracy must not be
substituted for a benchmark result.

## Qualification decision (completed)

`manifests/qualification.json` ran the full 20-epoch BRF SHD recipe at seed zero
and times the other original task shapes. This is the first learned baseline
qualification. Its seed-zero BRF result should be included once, not retrained
and counted twice when the five-seed primary comparison is assembled.

## First full SHD baseline

Execute `grf-qualification-20260906`, source `bd2950b`, job 720, completed
20 full SHD epochs (4,600 updates). Seed zero's lowest-validation-loss checkpoint
was epoch 11 (zero based). Held-out test accuracy: 90.5477% on all 2,264
examples; event fraction 10.4007%, approximately 3,328.2 events per sequence.
The published BRF result is 91.7 ± 0.8%, five seeds. The earlier 90.4% figure
belongs to the ALIF comparator. One fresh seed is 1.15 percentage points below
the published mean; this qualifies the multi-seed comparison but does not yet
establish agreement of the means. No recipe was changed using this test result.

Curated contracts and metrics: `measurements/primary/grf-qualification-20260906/`. Full
checkpoints are retained in the corresponding Execute work directory. Seed zero
is excluded from the next 19-run SHD manifest and included once when aggregating.

Measured median batch times at original shapes: SHD 27.8 ms, sMNIST 83.6 ms,
psMNIST 83.9 ms and ECG 76.6 ms. Image and ECG measurements are tiny timing
pilots, not trained benchmark results. Extrapolation to the 80 primary trainings
is approximately 66 hours plus validation/setup overhead on Athena, rounded to
about 70 GPU-hours. Kaya H100 qualification is queued before assigning those
longer tasks to that hardware.

## Committed secondary design

The SHD learning-rate control uses three rates (0.075, 0.025, 0.0075), two
selection seeds (100, 101) and all four arms: 24 full trainings, validation only.
The explicit tune stage cannot evaluate the test set. Any selected recipe must
be confirmed on the five main seeds; original-recipe results remain separate.

Fourteen targeted SHD contrasts, three paired seeds each, cover integration,
signed/complex payload, near-matched parameter count, excess, reset, surrogate,
smooth transmission, recovery decay and interaction controls. The main matrix,
tuning and ablations are distinct evidence groups; do not pool their seeds.

The nineteen remaining primary SHD trainings are running as
`grf-shd-main-20260906` (source `dc839cd`). The 42-run ablation and 24-run
validation-only learning-rate manifests are authored as
`grf-shd-ablations-20260906` and `grf-shd-tune-20260906`, respectively. Scheduler
completion and final model eligibility must be checked independently.

The other original tasks now have measured compute envelopes and twelve
derived, per-arm scheduler manifests, each retaining all five seeds. Athena is
the verified execution platform. Kaya has now acknowledged its timing request
as Slurm job 1175552, queued. Its throughput remains unverified and is not
included in any completion-time promise.

Post-training robustness is committed before evaluation: active-component
99.9th-percentile clipping from validation, payload widths 2/4/8 bits, packet
deletion 1/5/10/20%, and additive Gaussian binned-input noise with standard
deviation 0.01/0.05/0.1. Test perturbations use sample-indexed common random
numbers. Packet deletion preserves the sender's local event/refractory update.
All conditions are reported; no test-condition selection is permitted. Dense
forward latency includes emission statistics, uses resident device inputs, and
is not a neuromorphic-energy claim.

## H100 qualification and allocation decision

Kaya job 1175552 completed the three original-shape timing pilots. Returned
files were checked against their SHA-256 sidecars and curated under
`measurements/timing/grf-kaya-qualification-20260906/`. Warm batch medians were
118.0 ms (sMNIST), 118.4 ms (psMNIST) and 97.3 ms (ECG), slower per batch than
the RTX 4090 for these small sequential recurrences. The advantage available
here is concurrent independent jobs, not faster individual training.

The planned allocation therefore keeps SHD and sMNIST on Athena and moves all
psMNIST and ECG variants to Kaya, with at most two H100 jobs concurrently.
Per-task paired comparisons stay on one hardware type. Sliding afterany
dependencies enforce the Kaya cap. Superseded pending Athena jobs must be
cancelled through the Execute store and replacements recorded in the inventory;
no duplicate model is an extra seed. This reduces estimated elapsed compute
from roughly three days to roughly a day and a half if scheduler slots remain
available; it does not reduce aggregate GPU time.

Artifact returns initially failed with HTTP 404 because Athena's receiver
accepted only timestamp-style identifiers. The receiver now uses Execute's
canonical run-ID validator (Athena commit `88f85b4`, three HTTP regression
tests). Requests 1 and 2 remain failed transport records. Request 3 returned
the same completed qualification after the fix; no training was repeated.

The eight H100 primary replacements have been authored and their finalisation
returns requested. The corresponding eight Athena jobs were cancelled while
pending after checking that every default and experiment row matched its
replacement. `docs/execution.json` marks those scheduler records superseded.
The cancellation CLI's pull-after-edit ordering initially refused its own
changes; the verified eight monotonic controls were committed and pushed, and
the CLI was repaired upstream with three real-Git regression tests.

## Released-checkpoint selection audit

`tools/upstream_summary.py` reconstructs the SHD result from the five released
BRF CSV logs at the pinned upstream commit. Selecting each minimum-validation-loss
epoch gives test accuracies 92.2703, 90.6802, 91.9611, 90.9011 and 92.5353%:
mean 91.6696%, sample SD 0.8314 percentage points. These reproduce the published
91.7 ± 0.8 figure. The derived record and CSV hashes are in
`measurements/upstream/shd-released-summary.json`; these are the authors' runs,
not five extra independent replications by this study.

The released SHD trainer averages validation batch means equally. With 815
validation examples and batches of 256, the final 47 examples receive 25% of
the checkpoint-selection objective rather than their 5.77% sample share. Our
committed primary protocol uses sample-weighted validation loss. This is an
explicit protocol difference, alongside independent random draws and a fresh
fixed split; it must not be described as exact execution of the released
training script. Model forward/gradient conformance does not erase it.

`manifests/checkpoint-selection-shd.json` adds a bounded five-seed BRF diagnostic
that changes only this selection reduction. Training is replayed because the
primary run retains only the selected and final checkpoints, not every epoch.
Compare the replayed training trajectories before attributing a test difference
to selection. Keep these results under `measurements/selection-control`, outside
the primary and learning-rate confirmation groups. This diagnostic does not
authorise selecting the better validation rule using test accuracy. The sample
mean remains the primary protocol.

## Full five-seed SHD primary comparison

Jobs 720 (the qualification's BRF seed zero) and 721 (the other nineteen models)
now provide all twenty primary outcomes. Job 721 completed with exit 0 in
46:13. Every model completed 20 epochs, 4,600 updates and all 2,264 test examples;
all have 108,820 parameters and the same validation-index digest. No numerical
failure or missing seed is hidden. Source changes between the qualification and
main batch only admitted the validation-only tune stage; the main training
calculation is unchanged. The qualification's seed zero is counted once.

| Variant | Test accuracy, mean ± sample SD (%) | Event fraction, mean ± sample SD (%) |
| --- | --- | --- |
| BRF | 91.57 ± 0.90 | 10.32 ± 0.42 |
| Graded BRF | 90.40 ± 0.60 | 5.17 ± 0.27 |
| Graded observation | 89.60 ± 1.06 | 4.57 ± 0.26 |
| Graded static | 89.81 ± 0.61 | 6.41 ± 0.36 |

The BRF mean is close to the reported result. Relative to that fresh baseline,
graded variants lower mean accuracy by 1.17, 1.97 and 1.77 percentage points,
while reducing event activity by 49.9, 55.7 and 37.8%. These describe the
fixed-original-rate condition; learning-rate controls are still required.
Unadjusted paired t intervals are in `measurements/shd-primary-summary/summary.json`;
their upper bounds are close to zero and there are only five pairs. They are
not multiple-comparison-adjusted or dataset/speaker-resampling intervals.
Do not claim a universal design ranking or energy saving.

The full trajectories show similar declining training losses, with no failure
to fit. The maximum recorded gradient norms across the five trainings are
68.1, 879.2, 377.9 and 234.6, respectively, using the summed temporal loss.
Maximum membrane component magnitudes are 451.8, 845.5, 564.1 and 313.7. Larger
gradients are an observation, not proof of an optimisation cause for the test
gap. Median warm update times remain approximately 28 ms for every variant.
The allocator peak is cumulative within a batch process and cannot compare
per-model peak memory.

Exact result, metric and contract bytes plus inventories are in
`measurements/primary/grf-shd-main-20260906/`. `tools/summarise_primary.py`
regenerates seed CSV, statistics, accuracy and activity figures and table rows;
`tools/training_diagnostics.py` regenerates all seed trajectories and diagnostics.
The generated figures were visually inspected. All 19 implementation checks,
including the upstream forward/gradient oracle and unequal-batch loss reduction,
passed after the selection-control addition.

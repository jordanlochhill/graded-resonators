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

Curated contracts and metrics: `measurements/qualification-20260906/`. Full
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
the verified execution platform. The Kaya timing request has no published
receipt yet, so its capacity is not included in any completion-time promise.

Post-training robustness is committed before evaluation: active-component
99.9th-percentile clipping from validation, payload widths 2/4/8 bits, packet
deletion 1/5/10/20%, and additive Gaussian binned-input noise with standard
deviation 0.01/0.05/0.1. Test perturbations use sample-indexed common random
numbers. Packet deletion preserves the sender's local event/refractory update.
All conditions are reported; no test-condition selection is permitted. Dense
forward latency includes emission statistics, uses resident device inputs, and
is not a neuromorphic-energy claim.

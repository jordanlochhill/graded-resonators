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

## Next decision

`manifests/qualification.json` runs the full 20-epoch BRF SHD recipe at seed zero
and times the other original task shapes. This is the first learned baseline
qualification. Its seed-zero BRF result should be included once, not retrained
and counted twice when the five-seed primary comparison is assembled.

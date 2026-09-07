# Playground comparison, 7 September 2026

The earlier graded runs changed the BRF emission without importing the audio
playground's normalization and initialisation. Their failures cannot establish
that the playground architecture is unstable. The successful playground recipe
is a useful next control, but its reconstruction results do not predict SHD
classification accuracy.

## What the playground actually used

`kairos-audio/core/train_core.js` is the executable reference. Its `makeNet`
initialises each threshold independently from U(0.1, 0.3). At every time step,
each layer reads its complex membranes through

```
r = sqrt(mean_j(u_j**2 + v_j**2) + 1e-6)
(read_u_j, read_v_j) = gain_j * (u_j, v_j) / r
```

The gain is learned, initially one. The mean is over neurons within one sample
at one time, without centering or bias. Gating and emission use this read; the
stored oscillator membrane is not overwritten by normalization. Normalizing
after a completely silent gate would not repair its zero threshold gradient.

The event playground passes both signed complex components gated by a hard
threshold and uses a sigmoid surrogate. Its final audio output reads membranes
directly. The no-reset runs have no BRF adaptive threshold/damping state. They
also use polar oscillator dynamics, audio-specific frequency/bandwidth ranges,
and full inter-layer kicks. These differences remain separate from the read
normalization now implemented in the independent BRF-derived model.

The two archived 8–8–8 runs, `20260902-211630-c117` and
`20260903-080743-1f69`, both start at step zero with random low thresholds,
normalization enabled, real-component signed gating, full kicks and no reset.
After 3,000 updates their final training-window SNRs are 39.04 and 34.98 dB.
The extracted records and original file hashes are in
`measurements/playground/reference.json`. These are training-window measures,
not held-out results or a paired normalization ablation. The old service did
not record the source commit at execution time.

## Implemented comparison options

`Neuron(read_normalization="complex_rms")` adds this per-sample read operation
before the gate and payload. The existing study has one recurrent hidden layer:
its emitted values feed its readout and recurrent projection. There is no stack
of hidden layers yet. Future graded stacks should use the same read boundary at
every layer; this option does not itself evaluate depth or reproduce the whole
playground. A normalized model with a raw-unit subtractive reset is rejected
until that conversion has an explicit tested definition.

For learned thresholds, `threshold=0.2, threshold_spread=0.5` gives U(0.1, 0.3).
Random threshold draws occur after all shared parameter draws, preserving paired
weights. `threshold_spread=0` supplies the constant-0.2 control. Quantile
calibration and uniform random initialisation are separate alternatives and
cannot silently overwrite one another. Thresholds remain positive through
softplus. The original BRF defaults and frozen run definitions are unchanged.

For threshold-excess emission, an active unit has a true derivative of -1 with
respect to its threshold, and an inactive unit has zero. Randomness around a
threshold that is too high does not ensure activity. Low thresholds in meaningful
read units, accompanied by initial activity and gradient checks, address that
problem. RMS normalization can couple membrane gradients across units, but an
inactive unit's own threshold still has no true gradient through its emission.

## Next comparisons to discuss before queueing

First compare the existing calibrated, unnormalised excess control with a
normalized constant-low-threshold control and a normalized random-low-threshold
control. Pair exact and surrogate derivatives within each forward definition,
retain learned thresholds, use the same seeds and retune initial learning rates
on validation under the same decay schedule. This distinguishes normalization
from randomness without attributing either effect to the derivative choice.

Then isolate refractory utility with a matched threshold/damping 2-by-2 ablation
under one selected emission and initialization recipe. The old gated-membrane
SHD ablation already has all four combinations; the no-refractory arm learns,
and it does not establish a necessity for refractory feedback. The calibrated
true-gradient excess runs likewise learn without refractory feedback.

Hard event-driven refractory updates introduce a discontinuous state path;
adding them to the exact-gradient control changes the question. A recovery state
driven by graded excess is a distinct, piecewise differentiable alternative,
not an already tested BRF refractory rule. No new training has been queued for
these options. These implementation checks are not evidence of improved task
performance or stability on the longer sequences.

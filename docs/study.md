# Graded emissions and refractory feedback in resonator networks

## Question

Does transmitting membrane amplitude improve temporal learning, and should an
event suppress the subsequent observation, the membrane itself, or both?

The primary comparison has four arms, each with the same complex state,
input/recurrent/readout matrices, initial parameters, minibatch order and number
of optimiser updates. BRF is reproduced from Higuchi et al. (ICML 2024), checked
against upstream commit `1a42b8c8aceedb13cae3b2327774c2fcc04fd696`.

| Arm | Transmitted value | Threshold adaptation | Damping adaptation |
| --- | --- | --- | --- |
| Binary BRF | binary event | yes | yes |
| Graded BRF | event × real membrane | yes | yes |
| Graded observation adaptation | event × real membrane | yes | no |
| Graded fixed observation | event × real membrane | no | no |

The refractory accumulator is always driven by the binary event, never by its
amplitude. The main comparison retains BRF's Euler integration and positive-real
threshold. It therefore isolates the emission change and the two feedback paths
without confounding them with a new integrator, wider messages or a new gate.
These are controlled reductions of the audio design, not the complete audio model.

## Current audio design consulted

Canvas `50b74ebefe`, viewed 6 September 2026, separates a linear complex membrane,
a nonlinear transmission to the next resonator, and an event observation feeding
persistent memory. The proposed membrane has no threshold-triggered edit.
Continuous waveform synthesis reads membranes; a slower core updates the
decoder's instruction. Jordan's annotation leaves the audio-embedding fusion
unsettled. This paper studies the resonator primitive, not that fusion or a
working full-duplex language model.

The older audio playground transmits full complex gated membranes and has
optional reset policies. Core transmits signed threshold excess and subtracts a
real reset. Neither should silently stand in for the newer no-reset proposal.

## Secondary contrasts

After the main comparison, vary one factor at a time on SHD: a polar homogeneous
update with the same explicit input kick; signed real versus phase-carrying complex emission; full membrane
versus threshold excess; subtractive reset; surrogate family; and refractory
decay. Include a smooth, continuously graded transmission control to distinguish
amplitude coding from learning without a threshold surrogate. Complex emission
requires a parameter-matched width control because it doubles the recurrent and
readout input dimensions. Any claimed efficiency improvement must include that
cost.

An extension to waveform sound recognition must retain a common front end and
speaker-disjoint splits. SHD is already speech classification, not raw-waveform
ASR. Do not describe a synthetic tone experiment as speech recognition. SSC or
Speech Commands is an extension only after its data/protocol and compute budget
are explicitly committed. Full speech transcription is outside the initial
primitive study.

## Reproduction and evaluation

### Later addition: exact gradients and threshold learning

Jordan raised this comparison after the first SHD results on 6 September 2026.
The primary graded arms emit `u H(u-theta)` with a fixed base threshold. They
do not test whether threshold-excess emission needs a surrogate. For
`max(0, u-theta)`, the ordinary threshold derivative is -1 above threshold and
zero below it, with a convention at equality. A surrogate changes this
derivative while leaving the forward emission unchanged.

The added SHD control crosses ordinary versus BRF surrogate derivatives with
fixed versus learned per-neuron thresholds. All four conditions emit positive
real threshold excess, use Euler integration, and omit event-dependent
threshold, damping and reset feedback. Finite differences through recurrence
check the exact derivative. Hard event feedback is excluded: removing its
surrogate would introduce another gradient change without making that
discontinuous state path smooth.

Learned thresholds are `softplus(raw)` and begin at one. Adding these 128
parameters preserves all other random draws. Derivative comparisons have equal
parameter counts within each threshold condition. Inactive units' zero ordinary
gradient and the surrogate's extra active-region term are part of the question.

`manifests/exact-gradient-shd.json` defines an equal validation-only search of
initial rates .075, .025 and .0075, with the same linear decay, on seeds 100 and
101 for all four conditions (24 trainings). Select by mean best validation loss;
both seeds must complete finitely and ties favour the smaller rate.
`tools/select_exact_gradient_rates.py` authors at most twenty confirmation
trainings on seeds 0-4. Keep them separate from the primary table. The overlapping
original-rate fixed-threshold surrogate condition checks the earlier excess
ablation; it must not become extra primary seeds.

### Proposed audio extensions

Jordan also proposed mixing continuous membrane reads with event connections,
and treating the waveform itself as the first membrane. These remain design
alternatives, without new raw-waveform training admitted by this note. Compare
event connections, continuous membrane connections, and continuous early layers
followed by event layers at fixed depth and projection sizes. State whether reads
expose the real component or full complex state, retain the same event-to-memory
branch where applicable, and count continuous transfers alongside events. A
learned blend adds routing and scale choices; it is a separate experiment.

Interpret waveform-as-membrane initially as `z0[t] = x[t]`: an externally supplied
state without independent recurrence. This differs from a driven resonator
`z0[t] = A z0[t-1] + B x[t]`, which filters sound and can ring after silence,
and from a waveform plus learned recurrent residual. Raw-sample observation
requires explicit scaling, signed thresholds and crossing/rearming semantics.
Share sample rate, causal preprocessing and speaker split. SHD's already encoded
cochlear events cannot evaluate these raw-waveform alternatives.
For a waveform bounded by [-1, 1], a strict threshold of one produces no events.
Do not inherit BRF's threshold scale for the externally supplied waveform state:
ordinary threshold-excess gradients cannot train that threshold while every
sample is inactive. Initialise for some activity using training data only.

The live analysis is https://analysis.prosodylabs.com.au/graded-resonators/contents/ .

### Original-task protocol

Reproduce BRF's four published tasks: sMNIST, psMNIST, QTDB ECG and SHD. Recipes
preserve their widths, sequence lengths, batch sizes, Adam learning rates and
epoch budgets (300/300/400/20). The independent implementation uses JAX scans;
PyTorch conformance checks compare forward trajectories and surrogate gradients
with the authors' implementation before training. No framework speed comparison
is attributed to the neuron design.

Use five paired seeds for the main study. Use the same fixed 90/10 training
split and psMNIST permutation for all arms, recording their hashes. Select the
lowest sample-weighted validation loss checkpoint. Evaluate each selected model
once on the test set. Report means, sample standard deviations and paired
differences, including failed seeds. Preserve upstream pre-processing quirks in
the reproduction recipe and identify them in the paper; separately test any
correction rather than silently repairing the baseline.

The released SHD trainer uses an equal mean of validation batch means, whereas
the primary protocol weights examples equally. A separately labelled five-seed
BRF replay restores that selection reduction. Independent random draws and the
fresh fixed split also prevent a claim of exact released training trajectories.
See the evidence ledger's selection audit; do not choose between reductions
using test accuracy.

Metrics: classification accuracy (ECG per-time-step accuracy), validation
learning curves against epochs and elapsed training time, event fraction, payload RMS,
gradient norm, maximum membrane magnitude, non-finite failures, training time,
inference latency, parameter count and peak device memory when available.
Event counts are not energy measurements. Graded transmission multiplies a
payload by a weight; binary transmission can use conditional additions.
No scalar time-to-target threshold was fixed before the primary curves became
available; report the curves rather than select a favourable target afterwards.
The recorded JAX allocator peak is cumulative within a multi-model process,
so it cannot establish per-variant peak memory without separate profiling.

Post-training perturbations use validation-calibrated payload quantisation at
2/4/8 bits plus float32, and matched event deletion/input noise. Refit nothing on
test data. Compare accuracy against event rate and payload bit volume, never
claiming a hardware energy saving from software operation counts alone.

## Stages and decision rules

1. CPU conformance, causal/chunk equivalence, gate gradients and data contracts.
2. A bounded GPU timing/learning pilot on SHD; pilot data are not main results.
   Failures block larger jobs until explained. Measured throughput determines
   the full compute estimate; do not represent a guessed estimate as measured.
3. Full-budget BRF reproduction, followed by the paired four-arm comparison.
   If the baseline cannot approach the published task result, diagnose before
   interpreting a relative gain as an improvement over BRF.
4. Secondary ablations, robustness, held-out evaluation, figures and manuscript.
   Changes to this scope or recipe are committed before the affected run.

The paper is publishable as a useful null result if it establishes which
feedbacks matter. It is not complete merely because the implementation runs or
a small pilot improves training loss. Until the empirical stages finish, any
archive draft must state that its experiments are pending.

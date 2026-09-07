# Active initialisation for the exact-gradient comparison

The threshold-one sweep completed all 24 tuning runs, but all twelve exact
gradient runs remained silent: event fraction and gradient norm were zero,
and validation loss stayed at log(20). This establishes a dead initial state,
not a performance limit of exact-gradient learning. Jordan authorised a paired
repeat after correcting initial activity.

`manifests/active-init-gradient-shd.json` keeps the same Xavier weights, 128
units, 20 epochs, linear learning-rate decay, three rates (0.075, 0.025,
0.0075) and two selection seeds (100, 101). It crosses exact and double-Gaussian
surrogate gradients with fixed and learned positive thresholds. The emitted
value is threshold excess; there is no event-dependent refractory feedback or
reset in this control. Recurrent connections remain enabled during training.

For each seed, measure real membrane states without recurrent message input
on the first 64 examples of the training split. Set one network-wide threshold
to the 95th percentile of positive states. Labels do not determine this value;
neither validation nor test inputs enter calibration. Learned per-neuron
thresholds start at this same scalar, represented through softplus. All other
parameters are untouched. The quantile is a specified activity heuristic,
not an optimised hyperparameter. This is an explicit change of initialisation
protocol, separate from the original reproduction and silent sweep.

The CPU audit uses the full recurrent model after calibration. For seeds
100/101, thresholds were 0.0353584/0.0339264, event fractions were
0.0250161/0.0252476 and active-unit fractions were 0.945313/0.992188. Exact
learned-threshold input-gradient norms were 1.55704/1.59563, and raw-threshold
gradient norms were 0.0710523/0.0791099. The four conditions have the same
initial forward outputs within numerical tolerance for each seed. These are
initialisation diagnostics, not evidence of successful training or test accuracy.

Regenerate with CUDA hidden and JAX on CPU:

```
python tools/check_active_initialisation.py --data DATA --output measurements/initialisation/active-init-audit.json
```

The committed audit includes the manifest, dataset provenance, input/index
hashes and every condition. Runtime guards reject absent or excessive firing,
fewer than half the units activating, nonfinite diagnostics, or zero input or
learned-threshold gradients. They do not search for another threshold.

Choose rates on validation loss using the existing paired-seed rule, then
confirm on five seeds. The old silent sweep remains a separate negative
diagnostic and does not receive blind performance confirmations. Activity can
still disappear during training; the epoch curves must be inspected.

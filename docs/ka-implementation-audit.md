# KA implementation audit — 7 September 2026

The current KA implementation is evidence of what was implemented, not an
authority for Jordan's intended neuron. Jordan explicitly rejected the assumed
`tanh` activations on 7 September. Do not queue the replacement unit experiments
by copying that implementation. The replacement training code in this worktree
is unfinished and has not been queued.

## Reviewed sources

- KA implementation branch `feature/run-infrastructure`, commit
  `ea35431`, particularly `ka.py`, `config.py`, `docs/architecture.md` and
  `docs/run-infrastructure.md`. Canonical KA `main` still contains the earlier
  architecture-only checkout; reviewing it alone misses the implemented model.
- Its exact Core dependency is
  `64b2446d2465f6eefdd93d792106a3b17447cb98`, particularly
  `src/kairos/audio/resonator.py` and `src/kairos/surrogate.py`.
- Playground `core/train_core.js` at checkout
  `339258a48b1ac63798e3ba395bed1654375a1157`, with archived run snapshots.
- KLM's serial layer boundaries use RMS normalisation. Its conventional Core
  neuron emits signed real threshold excess; this differs from the playground's
  full complex gated payload.

## Confirmed departures

KA applies componentwise `tanh` to the real/imaginary membrane features before
the next encoder or decoder resonator. This entered in implementation commit
`a4f4e4be1818e162d792d22b58fc6e90b0fec7c8`. The architecture notes discuss a
nonlinear transmission and alternatives; they do not establish Jordan's
selection of `tanh`. Jordan has now explicitly said he did not request it.

There is no normalisation at those resonator boundaries. `membrane_features`
only concatenates real and imaginary coordinates; the pinned `Resonator` does
not normalise them either. Normalisation elsewhere in the text core or event
memory does not implement normalisation between resonator layers. This is a
departure from Jordan's reiterated KLM/playground finding.

Componentwise `tanh` is not equivalent to complex RMS normalisation. It saturates
coordinates independently, can change a lane's complex argument, and supplies
its own amplitude-dependent derivative. A positive common scale for each lane's
real and imaginary coordinates preserves that lane's argument. Normalisation
itself is nonlinear; a further activation is not required merely to introduce
nonlinearity between layers. Neither operation guarantees task-level retention
of phase after learned mixing, gating and readout.

## Further differences requiring an explicit variant definition

These are verified implementation differences, not established defects:

| Choice | Latest KA implementation | Successful playground reference |
| --- | --- | --- |
| Event rule | Upward crossing of absolute real membrane; explicit previous-above state | Above-threshold transmission on each sample, subject to selected reset |
| Threshold units | Raw real membrane | Complex-RMS-normalised real read |
| Initial thresholds | All lanes start at 0.04; learned positive thereafter | Independent U(0.1, 0.3); learned |
| Input integration | Exact held-drive ODE integration at every layer | First-layer leak-scaled waveform drive and full complex interlayer kicks in the cited event runs |
| Interlayer payload | Concatenated real/imaginary values after componentwise tanh, optionally gated | Full complex normalised membrane, optionally gated |
| Default reset | None | Both no-reset and subtractive-reset records trained; preserve the distinction |
| Waveform readout | Linear combination of final real pre-reset membrane | Complex linear read of final stored membrane with the documented normalisation convention |

An upward-crossing detector and a repeated above-threshold transmission are
different temporal mechanisms, especially without reset. They must not share
one unexplained label, "firing". A binary admission decision can carry a graded
payload; calling the detector binary does not make the transmitted amplitude
binary.

The KA decoder is correctly serial at the composition level: the held core
residual enters the first resonator, successive resonators consume the previous
layer's transmission, and the waveform head reads the last layer. This does not
validate the constituent neuron choices.

## Requirements for the replacement study

Use Jordan's requirements as the governing contract: normalisation between
layers, explicit membrane-versus-firing transmission variants, a strictly serial
decoder with an initial residual-vector input, and no assumed tanh activation.
Keep event observation, transmitted amplitude and edits to carried membrane
state distinct in the definition, whether their paths are shared or separate.
Specify the threshold units, initialisation, event rule and reset explicitly.

Retire the earlier BRF-derived graded results from evidence for this intended
unit. Preserve their files and provenance as historical diagnostics. A fresh
binary BRF reproduction remains a separately defined comparison.

The small five-layer, six-lane playground speech reconstruction records around
4–6 dB clip SNR matter alongside the earlier tone/no-reset examples. They are
not held-out generalisation evidence. Browser-local models personally saved by
Jordan have not yet been recovered; do not imply that reviewing service run
archives also reviewed those saved browser models.

No new training was queued during this audit. Resume the replacement queue only
after the neuron contract and implementation checks have been reconciled with
this correction. Every new run must connect to live W&B before training.

## Paper motivation retained

BRF's future-work discussion of raw audio and multiple temporal resolutions is
the starting point. WavJEPA is a recent example of concern about phase loss,
alongside earlier phase and waveform work, not the origin or sole authority for
that concern. Distinguish magnitude-only spectral representations from complex
spectral representations that retain phase. Phase retention and behaviour over
different elapsed times are empirical questions for the unit study.

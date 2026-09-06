# Graded resonators

Independent source for **Graded Emissions and Refractory Feedback in Resonator
Networks**, a reproduction and extension of Higuchi et al., *Balanced
Resonate-and-Fire Neurons*, ICML 2024.

**Status: SHD primary comparison complete; other experiments underway.** Five-seed
test accuracies are 91.57 ± 0.90% (BRF), 90.40 ± 0.60% (graded BRF),
89.60 ± 1.06% (graded observation) and 89.81 ± 0.61% (graded static), with no
numerical failures. The published BRF result is 91.7 ± 0.8%. Graded variants
reduce event activity by 38–56% but have lower mean accuracy under the original
learning rate. Rate controls, ablations, robustness and the other three tasks
remain pending; this does not establish an energy advantage. The protocol is in
[docs/study.md](docs/study.md). The manuscript is maintained in Jordan Hill's
[papers archive source](https://github.com/jordanlochhill/papers).

Four primary arms keep the architecture and trainable parameter count fixed:
binary BRF; graded BRF; graded emission with adaptive observation only; graded
emission with fixed observation. Secondary controls examine the audio design's
integration, signed/complex payloads, reset and surrogate choices. Current audio
architecture decisions are documented separately from this primitive study.

## Install and verify

Python 3.11 or 3.12 and [uv](https://docs.astral.sh/uv/) are supported.

```sh
uv sync --frozen --extra test
JAX_PLATFORMS=cpu uv run --frozen --extra test pytest
```

The external conformance check compares complete BRF forward outputs, spike
counts, loss and BPTT gradients with the authors' PyTorch implementation. Clone
`https://github.com/AdaptiveAILab/brf-neurons`, check out commit
`1a42b8c8aceedb13cae3b2327774c2fcc04fd696`, and set
`BRF_REFERENCE_ROOT=/path/to/brf-neurons` when running the tests. That oracle is
optional for users; it was used before admitting the initial training pilot.

The model is implemented with JAX scans. It neither imports nor requires
Kairos Core or the authors' training package. PyTorch is a test dependency only.
The surrogate matches the released BRF code's normalised Gaussian densities and
gain of 0.5. The hard forward threshold is strictly greater than zero.

## Data and runs

```sh
uv run python -m graded_resonators.data all --root data
uv sync --frozen --extra gpu
# Run inside a scheduler allocation on a CUDA 12 GPU:
uv run --frozen --extra gpu python -m graded_resonators.train \
  manifests/pilot-shd.json --data data --output results/pilot-shd
```

CPU diagnostics can use `--cpu` and `JAX_PLATFORMS=cpu`. Athena GPU jobs are
submitted through Execute using `tools/submit.py`; local GPU execution requires
a Slurm allocation. Downloads use upstream checksums where supplied; every
prepared dataset receives a SHA-256 provenance record. Dataset files are not
redistributed in this repository. SHD/SSC are CC BY 4.0 (Cramer et al.); ECG is
the supplied Yin et al. preprocessing, obtained from the pinned BRF release.

Each experiment writes its full contract, data/split digests, source commit,
epoch metrics, resumable last state and lowest-validation-loss checkpoint.
Pilot manifests never evaluate the test set. Full experiments evaluate the
selected checkpoint once. A failed seed remains a failed seed in the summary.
Generated result directories are ignored by git; curated measurements for the
paper will live under `measurements/` with an evidence ledger.

`tools/split_primary.py` regenerates the twelve bounded original-task scheduler
manifests from their three complete manifests. `manifests/robustness-shd.json`
specifies post-training quantisation, packet deletion and binned-input Gaussian
noise. Replace its checkpoint paths with your own selected-checkpoint directories
when reproducing the evaluation. Its clipping range uses the 99.9th percentile
of nonzero absolute validation payloads. A packet is lost after the local event
updates refractory state; the same loss affects recurrent and readout delivery.
Noise is added without clipping, in units where a binned input event has value
one. These are numerical perturbations, not an acoustic-noise benchmark.

To curate a completed run and regenerate the primary comparison:

```sh
python tools/collect.py --output measurements/primary /path/to/results/RUN
uv run python tools/summarise_primary.py measurements/primary --output measurements/primary-summary
# The completed SHD comparison and its training trajectories:
uv run python tools/summarise_primary.py measurements/primary --tasks shd --output measurements/shd-primary-summary
uv run python tools/training_diagnostics.py measurements/primary --task shd --output measurements/shd-primary-summary
```

The collector preserves record bytes and writes checksums. The summary refuses
duplicate seeds and missing planned seeds. `--tasks shd --allow-partial` produces
an explicitly labelled progress view. Failed seeds remain visible and means
over successful seeds are marked as conditional. Tuning, ablation and robustness
results belong in separate measurement groups.

The released trainer uses an equal mean of validation batch means; the primary
study weights examples equally. A five-seed BRF selection control isolates that
protocol difference. See [the evidence ledger](docs/evidence.md) for the audit,
released-log verification, source pins and current findings. Selection controls
must not be pooled as additional primary seeds.

## Attribution

Higuchi, Kairat, Bohté and Otte (2024), [Balanced Resonate-and-Fire
Neurons](https://proceedings.mlr.press/v235/higuchi24a.html), PMLR 235,
18305–18323. The upstream implementation is MIT licensed; see
[THIRD_PARTY.md](THIRD_PARTY.md). Graded resonate-and-fire signalling predates
this study (Frady et al., 2022); the research question concerns controlled
learning comparisons and the separation of refractory feedback.

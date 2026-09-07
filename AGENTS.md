# Graded resonator study

This is an independent, publishable implementation for a paper comparing binary
BRF with graded output and separated refractory feedback. It does not import
Kairos Core, the audio playground, or the authors' training code.

The scientific contract is `docs/study.md`; the executable recipes are under
`manifests/`. Preserve original-paper reproduction separately from changed
protocols. Never promote pilot results to the main table or choose checkpoints,
hyperparameters, quantisation scales, or stopping rules using test performance.

Use `uv run --extra test pytest` with `JAX_PLATFORMS=cpu` for local conformance
tests. Every GPU experiment goes through Slurm/Execute, using committed source
and a manifest. Failed/non-finite runs are evidence and must remain recorded.

Every new training or evaluation run must log live to W&B in the
`jordanlochhill-nmtafe/graded-resonators` project. Open the collection and model
run before computation; connection failure must abort startup. Historical
imports are for recovery only, never a substitute for new live telemetry.

Graded layer boundaries must account for amplitude scale. The playground
reference uses complex RMS membrane reads before thresholding/transmission,
with raw carried state and random initial thresholds in [0.1, 0.3]. The study's
optional `read_normalization=complex_rms` preserves that placement. Keep the
original unnormalised BRF reproduction explicit and separate; see
`docs/playground-comparison.md` before designing the next graded comparison.

The manuscript lives in the parent papers repository. Data and trained weights
are downloaded/generated artefacts; checksums and provenance belong in the
source. Do not commit datasets or opaque upstream checkpoints.

Jordan develops the manuscript interactively. Background continuations collect
evidence and prepare the analysis page; they do not write, revise or publish
papers. The current working draft is already in his private papers archive.

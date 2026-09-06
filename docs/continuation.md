# Prepare BRF evidence for interactive paper development

Jordan authorised the committed BRF reproduction, graded alternatives, rate
controls and analysis. On 6 September he clarified that papers are developed
interactively with him: no autoresearch manuscripts. This worker may collect
evidence, run the already admitted confirmations, update the analysis and
prepare findings for discussion. It must not write, revise, merge or publish
a manuscript, modify the papers archive, or intake a paper into the library.

The nine-page working draft is already available at
https://papers.prosodylabs.com.au/2026-graded-resonators/ (private archive).
PR #10 merged as 1b70ee7; archive availability and study completion are separate.
The ongoing paper is the compact core resonator unit study. Broader KLM and
Kairos Audio reports, and an independent steering experiment, are separate
interactive writing projects. Complete the available evidence work and leave
one verified successor for admitted experiments that are still running.

## Workspaces and discovery

- Your source worktree is `/home/jordan/Documents/worktrees/graded-resonators-shd-review`.
- Papers are read-only for this worker. The canonical manuscript is on papers main
  under `2026-graded-resonators/`; no manuscript worktree is needed.
- The original source checkout is inside
  `/home/jordan/Documents/worktrees/papers-brf-graded/2026-graded-resonators/code`.
  It holds the prepared datasets. Treat that checkout as read-only.
- The first manuscript PR, https://github.com/jordanlochhill/papers/pull/10,
  is merged. Do not push to it or open a manuscript PR from this worker.
- Source repository: https://github.com/jordanlochhill/graded-resonators.
- Jordan additionally requested a live analysis page. Maintain the existing
  https://analysis.prosodylabs.com.au/graded-resonators/contents/ at
  `/home/jordan/Documents/analysis` on main, following its instructions and the
  analysis skill. This is an authorised write scope; update this same page.
- `docs/execution.json` is the current run inventory, including source pins,
  manifests, roles and result locations. Read it afresh; it is not live status.

Read the source worktree's AGENTS.md instructions, `docs/study.md`, `docs/evidence.md`,
the executable manifests, and the paper's `review.md` and current TeX. Use the
analysis, glossary, runs, gpu-train, agents, when and worktrees skills for
their respective work. The paper skill records the interactive-only authorship rule. Read the current canvas if interpreting implications
for audio; its embedding fusion remains an open question. The own study card is
item `297c5457f1f7` in workspace `50b74ebefe`, at x=17470, y=24691, width=900.
Other items and Jordan's ink are outside your write scope.

On arrival, inspect status and fetch the source remote. Fast-forward your clean
source branch to origin/main if the original coordinator added newer evidence
or platform decisions. Never reset, stash or change the branch of either
canonical checkout. The canonical papers directory has other work in progress.
The source worktree should use its own uv environment, not a symlink to another
checkout's editable environment. The data path is given in the run inventory.

## First continuation: complete the SHD evidence

The coordinator already curated the complete primary SHD comparison and updated
the methods draft: BRF 91.57 ± 0.90%, graded BRF 90.40 ± 0.60%, observation
89.60 ± 1.06%, static 89.81 ± 0.61%, all five seeds complete. Treat these as the
fixed-rate primary result; review them with the controls, rather than rerunning
the primary jobs. The pending evidence can change the interpretation.

1. Read fresh Execute states from
   `/home/jordan/.local/state/athena-executor/execute-store/runs`. Check scheduler
   outcome as well as `state`; finalised can still mean FAILED. The trigger opts
   into failure so you can diagnose it. Numerical failure of one model does not
   erase other models in a manifest: the trainer records each outcome and then
   exits 2 if any model failed. A scheduler interruption without a summary needs
   diagnosis and a faithful resume, not silent omission.
2. Curate terminal result bytes with `tools/collect.py`, into distinct
   `measurements/primary`, `measurements/ablations`, `measurements/tuning` and
   `measurements/robustness` groups. The qualification's BRF SHD seed zero is
   already in primary and must be counted once. Its other three models are
   timing pilots, not primary results. Preserve inventories, source contracts,
   epoch metrics, failures and all seeds. Do not commit datasets or checkpoints.
3. Generate the SHD primary summary with `tools/summarise_primary.py --tasks shd`.
   Generate trajectories with `tools/training_diagnostics.py --task shd`.
   Inspect trajectories, event rates, payload scale, gradients and failed seeds.
   The published BRF reference is 91.7 ± 0.8%, not 90.4% (the latter is ALIF).
   Our first full-budget seed was 90.5477%. Explain a multi-seed reproduction gap
   before interpreting an alternative as an improvement over published BRF.
   The coordinator also added a five-seed checkpoint-selection diagnostic,
   `grf-shd-selection-20260906`. Read the ledger's selection audit. Curate it under
   `measurements/selection-control`, check replayed training trajectories against
   primary BRF, and report changes of selected epoch and test accuracy. Its
   equal-batch validation reduction reproduces one detail of the released script;
   the primary sample-mean reduction stays fixed. Do not pool these extra runs
   as new primary seeds or select a preferred validation rule from test accuracy.
4. Run `tools/select_rates.py` on the complete validation-only tuning group.
   Each rate needs both selection seeds complete; failed rates are ineligible.
   Choose lowest mean best validation loss, breaking exact ties by smaller rate.
   This rule was committed before tuning results arrived. Commit the generated
   confirmation manifest before submission. Train its changed-rate arms on seeds
   0–4 through Execute; reuse original primary results when the selected rate
   is unchanged. Do not submit an empty manifest. Keep the selected-rate evidence
   separate from the original-recipe primary table.
5. Summarise the fourteen ablation contrasts against their manifest's stated
   parents. Do not turn the test set into a search criterion or hide failures.
   Native surrogate gains differ, so a shape-only explanation needs a matched
   gain control; otherwise retain the explicit limitation. Complex width 112
   gives 108,212 parameters versus 108,820 for the primary SHD model; width 128
   is a separate, larger control.
6. Build the precision, packet-loss and binned-input-noise figures from the
   frozen-checkpoint results. Pair seeds and perturbations. Quantisation scales
   come from validation; no refitting or test-condition selection. A locally
   observed event still updates refractory state if its packet is lost. Dense
   forward latency is a software measurement, not neuromorphic energy. A payload
   bit proxy must state its encoding assumption and exclude or explicitly count
   address/timestamp overhead; event counts alone are not energy savings.
7. Update the evidence ledger, README, analysis page and own canvas card with
   actual results and limits. Record implications and questions for Jordan in
   the evidence ledger, including informative negatives. Do not edit the paper.

## Complete the remaining study

The later exact-gradient request is part of this study. Read the added section
in docs/study.md and `manifests/exact-gradient-shd.json`; its Execute run is
`grf-shd-exact-gradient-tune-20260906`. It crosses exact versus BRF surrogate
gradients for identical threshold-excess emission with fixed versus learned
positive thresholds, without event feedback or reset. Curate the 24 validation
selections separately. Run `tools/select_exact_gradient_rates.py RESULTS --output
MANIFEST` to author up to twenty confirmations. Commit before submitting to
Execute; never submit an empty manifest. Report all four conditions, the 128
extra learned-threshold parameters, inactivity, threshold/gradient scale,
failures and paired test differences. This control was added after the primary
SHD results. The older softplus control alone does not answer this question.

After source records are curated and committed, ingest primary evidence in the
analysis repo with `python3 tools/ingest_graded_resonators.py --source
YOUR_SOURCE_WORKTREE`, then `python3 tools/build_graded_resonators_data.py`.
Extend ingest and builder for the secondary groups without pooling them into
primary evidence. Keep prose hand-authored, exact tables beside each chart,
source revisions and hashes, and visibly missing results. Update the header,
coverage, gallery and interpretation. Follow the desktop/mobile browser audit,
deterministic rebuild and commit/push requirements. At first publication the
six targeted tests passed; `make test` encountered an unrelated absent historical
single-token log under the retired auto programme. Never shrink other reports
to hide that absence.

The original four-task primary design is 80 trainings: four arms, five paired
seeds, original 300/300/400/20 epoch budgets for sMNIST/psMNIST/ECG/SHD. The
Athena manifests are split into twelve per-arm jobs for the other tasks. The
first estimate was roughly 70 GPU-hours on Athena, before secondary controls.
These full budgets must not be replaced by short pilots and called replicated.

Kaya H100 qualification is complete and its returned metrics are curated.
The allocation now keeps SHD/sMNIST on Athena and all psMNIST/ECG arms on Kaya,
with an explicit two-job H100 dependency chain. The eight old Athena psMNIST/ECG
jobs were cancelled while pending and are marked superseded in the inventory.
Do not count them as missing or failed models or launch those comparisons again.
Kaya has public-source
HTTPS access and prepares MNIST/ECG directly in
`/scratch/sh001/jhill/graded-resonators/qualification-data`. Never dial the VPN
unattended or use the prohibited reverse tunnel. Status and artifact returns
use Execute without a VPN. The `execute-artifacts` and `execute-cancel` entrypoints
are under `/home/jordan/Documents/execute/bin/`, not necessarily on PATH.
The qualification's requests 1 and 2 failed because of a receiver run-ID mismatch;
Athena `88f85b4` fixed the receiver and request 3 returned the files with verified
hashes. The eight H100 primary runs each have request 1 queued for finalisation.
Inspect their ledgers rather than submitting duplicates. If allocation changes,
record replacements in the inventory and cancel only superseded pending jobs
through Execute. Do not count a cancelled scheduler attempt as a failed model.
Preserve complete per-task comparisons and report any hardware differences.

New training remains within the committed design, changed-rate confirmations,
and bounded diagnostics needed to explain failures or a reproduction mismatch.
Do not expand into an unbounded architecture search. A raw-waveform/SSC extension
is optional and has not been admitted; SHD already tests spoken-digit event
classification, and neither transcription nor synthesis is established here.

Jordan's mixed membrane/event layers and waveform-as-first-state ideas remain
recorded proposals. Do not claim they were evaluated or turn them into an
unbounded search. The exact-gradient study above is the bounded admitted addition.

If jobs are still running when the bounded session ends, author and arm one
successor trigger on their actual terminal conditions, with `on_failure: true`.
Give it a fresh source worktree and this objective, current inventory,
evidence and interactive-authorship boundary. It may in turn arm one further bounded
continuation when a newly required job is pending. No parallel worker fan-out.
Use the existing global policy and launch window; do not change or bypass them.
Validate and test the trigger and record its exact name. A promise to follow up
is insufficient. Do useful available analysis before ending; do not spend an
agent session polling training for hours.

## Evidence handoff for interactive paper review

Before declaring the study complete, account for every planned primary seed,
the ablation and robustness groups, and selected-rate confirmations. Distinguish
numerical failures, scheduler failures, missing runs and ineligible comparisons.
The primary summary refuses missing planned seeds and duplicate baselines.
An unexplained failure cannot disappear into an average of survivors. Finish
the required diagnoses or report their consequence explicitly in the results.

Prepare a concise evidence handoff in the source ledger: which findings changed,
which proposed mechanisms remain untested, which numerical results are ready
for discussion, and where the current PDF has become stale. Keep the paper's
claims separate from the latest findings until Jordan revises it interactively.
Do not generate a replacement paper or a second manuscript in the source repo.

Run meaningful source checks (CPU, including the pinned upstream oracle when
available), the relevant analysis checks and a desktop/mobile browser audit.
Commit and push coherent source and analysis updates without force, integrating
intervening work first. Keep the analysis linked to the existing private archive
PDF and mark new evidence that has not yet entered that draft. Archive and
library publication are outside this worker's scope.

## Completion contract for this first worker

Work notes may be written while you proceed. Write
`docs/shd-review-complete.json` in your source worktree **last**, only after the
available SHD review is complete, changes are committed/pushed, and any necessary
successor is actually armed and tested. Include findings without presupposing a
winner, failed/missing run IDs, new confirmation run IDs, next trigger name,
source and analysis commits, the current working-paper URL, and the questions
ready for interactive review. Do not describe an unchanged draft as newly revised. If an external blocker prevents this contract,
report the exact blocker and fail visibly rather than emitting a false completion.

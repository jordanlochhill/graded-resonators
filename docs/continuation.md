# Continue the BRF reproduction through the final paper

Jordan explicitly requested an independent, publishable replication of Balanced
Resonate-and-Fire Neurons, with two or three graded alternatives and a broad
controlled design study, informed by the current Kairos Audio canvas. He also
explicitly requested that the final draft be pushed to the existing papers
archive. He subsequently said “continue”. This authorises the planned training,
analysis, code publication and completed-paper publication. Do not ask for the
same permission again. It does not authorise invented findings, changes to the
audio system, or publishing an unfinished experiment as a completed result.

This brief is for a bounded continuation worker, not a request to stop at a
plan. Complete the available work and leave an armed, verifiable continuation
for work awaiting long jobs. The final paper is the intended outcome.

## Workspaces and discovery

- Your source worktree is `/home/jordan/Documents/worktrees/graded-resonators-shd-review`.
- Your manuscript worktree is `/home/jordan/Documents/worktrees/papers-graded-shd-review`.
- The original source checkout is inside
  `/home/jordan/Documents/worktrees/papers-brf-graded/2026-graded-resonators/code`.
  It holds the prepared datasets. Treat that checkout as read-only.
- The existing manuscript branch and draft PR are `paper/brf-graded` and
  https://github.com/jordanlochhill/papers/pull/10.
- Source repository: https://github.com/jordanlochhill/graded-resonators.
- `docs/execution.json` is the current run inventory, including source pins,
  manifests, roles and result locations. Read it afresh; it is not live status.

Read both worktrees' AGENTS.md instructions, `docs/study.md`, `docs/evidence.md`,
the executable manifests, and the paper's `review.md` and current TeX. Use the
paper, style, glossary, runs, gpu-train, agents, when and worktrees skills for
their respective work. Read the current canvas if interpreting implications
for audio; its embedding fusion remains an open question. The own study card is
item `297c5457f1f7` in workspace `50b74ebefe`, at x=17470, y=24691, width=900.
Other items and Jordan's ink are outside your write scope.

On arrival, inspect status and fetch the source remote. Fast-forward your clean
source branch to origin/main if the original coordinator added newer evidence
or platform decisions. For the paper, integrate the latest origin/paper/brf-graded
into your own clean branch. Never reset, stash or change the branch of either
canonical checkout. The canonical papers directory has other work in progress.
The source worktree should use its own uv environment, not a symlink to another
checkout's editable environment. The data path is given in the run inventory.

## First continuation: complete the SHD evidence

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
7. Update the evidence ledger, README, manuscript and own canvas card with the
   actual results and limits. Keep the paper's pending sections explicit until
   the remaining original-task comparisons finish. Its main argument must
   follow the findings, including an informative negative if that is the result.

## Complete the remaining study

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

If jobs are still running when the bounded session ends, author and arm one
successor trigger on their actual terminal conditions, with `on_failure: true`.
Give it fresh source and paper worktrees and this objective, current inventory,
evidence and publication instructions. It may in turn arm one further bounded
continuation when a newly required job is pending. No parallel worker fan-out.
Use the existing global policy and launch window; do not change or bypass them.
Validate and test the trigger and record its exact name. A promise to follow up
is insufficient. Do useful available analysis before ending; do not spend an
agent session polling training for hours.

## Final manuscript and publication

Before declaring the study complete, account for every planned primary seed,
the ablation and robustness groups, and selected-rate confirmations. Distinguish
numerical failures, scheduler failures, missing runs and ineligible comparisons.
The primary summary refuses missing planned seeds and duplicate baselines.
An unexplained failure cannot disappear into an average of survivors. Finish
the required diagnoses or report their consequence explicitly in the results.

Replace the methods-draft abstract and outstanding-work section with the actual
research result. The paper must stand alone: mechanisms, numbers, uncertainty,
related work, interpretation and limits; no internal run IDs, code paths or
functions in its prose. The source ledger carries provenance. Read the nearest
papers in the central library, including BRF, its convergence analysis, Frady's
graded resonator work and S5-RF. Do not claim graded resonators were invented
here or claim an exhaustive novelty search. State clearly what the study licenses
for the latest audio primitive and what remains untested in the full system.

Run the source's meaningful checks (CPU, with the pinned upstream oracle when
available), the paper's `make check`, and the archive validation/tests. Render
and visually inspect every PDF page and result figure; clean LaTeX logs missed
clipped plot titles once. Perform the paper skill's hostile-review and reader
passes, fix or acknowledge the three weakest points, and reconcile every result
against committed data. Regenerating figures must reproduce their numbers.

Publish coherent source waypoints on GitHub and update PR #10's title/body to
describe the completed manuscript. Source changes can be pushed from the source
worktree to main after fetching and integrating any intervening work, without
force. Paper changes can be pushed to the existing paper/brf-graded PR branch
after integrating its current tip, also without force. Jordan's explicit final
archive-publication request authorises the necessary final publication; do not
leave it at an unmerged PR with a request to approve the same action again.
The manuscript may remain scholarly status `draft` when complete; do not imply
venue acceptance. Use an accurate kind such as “Research draft”.

Publish to the existing `papers.prosodylabs.com.au` archive only. It is served
from `/home/jordan/Documents/papers/_site` with editable source in the canonical
papers directory. Inspect its current branch, dirty files and service before
deployment. Preserve others' work. In particular, `scripts/papers.py build`
deletes its output tree: never point it blindly at the live tree from an old
worktree. Build and validate a staging tree first, then publish the new paper's
page/PDF and a correctly regenerated archive index without replacing unrelated
live papers. Use the existing render functions and record metadata; no duplicate
browser version of the paper and no new hosting platform. Ensure the source link
resolves to the final manuscript on GitHub main. Sync only this final paper into
the central reference library and commit/push that intake through `ref`.

Verify the final archive landing page and PDF over HTTPS, compare the downloaded
PDF hash with the built final file, and verify its source and code links. Keep
the full source/result provenance available. Update the own canvas card and
write the final completion record with paper URL, PDF URL, code URL, commits,
validation results, findings and remaining scientific limitations.

## Completion contract for this first worker

Work notes may be written while you proceed. Write
`docs/shd-review-complete.json` in your source worktree **last**, only after the
available SHD review is complete, changes are committed/pushed, and any necessary
successor is actually armed and tested. Include findings without presupposing a
winner, failed/missing run IDs, new confirmation run IDs, next trigger name,
source/paper commits and whether final archive publication is complete. A methods
draft is not the final artifact. If an external blocker prevents this contract,
report the exact blocker and fail visibly rather than emitting a false completion.

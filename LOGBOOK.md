# Experiment logbook

[2026-08-27 20:21 EDT] AUTHORITY qwen3-8b-confirmatory-24h

Context: Run the complete 45-arm Qwen3-8B confirmatory grid, one baseline, and
four robustness profiles on PACE ICE, targeting 24 hours and reporting exact
state by 48 hours.

Action: Resolved authority before implementation and submission.

Result:

- repo_publish: allowed on topic branch `glenn/ice-confirmatory-24h`; default and
  protected branches are prohibited.
- slurm_submit: allowed for the prepared ICE smoke and confirmatory dependency
  chain, up to 46 simultaneous GPUs, `coe-ice`, one `HX00` GPU per task.
- deploy: exact-SHA releases through `pace-code-release` only.
- delete/cleanup: prohibited outside receipt-bound temporary staging; partial
  submission cleanup may cancel only job IDs created by that failed submission.
- decisions_consumed: full 45-arm grid; operational-only scale gate; four
  robustness profiles; account `gmatlin3`; artifact root
  `/storage/ice-shared/cs7634/staff/econml`; three attempts maximum; no automatic
  cancellation at 48 hours.
- morning_list_only: new scientific scope, destructive work outside the run
  contract, protected-branch operations, or methodology changes.
- done means: exact terminal receipts for baseline, 45 arms, analysis, and four
  robustness profiles, plus allocation-only accounting and a checksummed export.

[2026-08-27 20:21 EDT] Initial hardening slice

Context: The original single-file experiment used lower-bound dependencies,
count-only GPUs, halfway checkpoints, and code-execution opt-in without a real
security boundary.

Action: Added immutable run contracts, result fingerprints, ten-percent
checkpoints, bubblewrap containment, robustness isolation, receipt/accounting
utilities, a frozen dependency specification, and an ICE-specific Slurm chain.

Result: Thirteen focused offline tests and shell syntax checks passed before the
full dependency environment was installed. Deployment and live ICE gates remain
pending.

[2026-08-27 22:38 EDT] Pre-release two-axis review

Context: The ICE implementation needed both standards review against the run
contract and specification review against the 24-hour experiment charter before
an exact SHA could be deployed.

Action: Reviewed every terminal receipt and Slurm edge, exercised the scientific
self-test, resolved the full dependency lock, loaded custom lm-eval task
overrides, and inspected bubblewrap device and filesystem bindings.

Result:

- Fixed robustness preparation to warm and pin MATH-500 and HumanEval before
  offline GPU work.
- Froze Qwen3-8B plus all six training, primary-evaluation, and robustness
  dataset commits in `design.json` and `run_contract.json` before download.
- Replaced the whole `/dev` bind with minimal `/dev` plus NVIDIA device nodes;
  the live compute-node doctor remains the acceptance gate.
- Completed release lineage for environment, preparation, smoke, baseline,
  arms, endpoint report, robustness, analysis, status snapshots, and finalizer.
- Added 4, 12, 20, 24, and 48-hour scheduler/receipt snapshots and expanded the
  student export to include raw metrics and predictions, provenance, status,
  accounting, interpretation, limitations, and exact reproduction instructions.
- Verification: 18 pytest tests passed; Ruff lint and formatting passed; Pyright
  reported zero errors; the 118-package lock resolved under `uv lock --check`;
  shell syntax and `git diff --check` passed; the experiment self-test passed.

Deployment, the allocated-node sandbox/GPU doctor, same-model smoke, and every
scientific receipt remain pending live ICE evidence.

[2026-08-27 22:17 EDT] Correction and v3 deployment gate repair

Context: The preceding `22:38 EDT` timestamp was recorded incorrectly; this
entry supplies the correct ordering without rewriting the append-only record.

Action: The first deployment used ICE's legacy v2 helper and the submitter
rejected it before `sbatch`. After explicit approval, a tested helper-only
dotfiles topic release activated the v3-capable stable wrapper. The unchanged
experiment SHA then received a verified v3 receipt in the new release root.

Result: A traced submission still stopped before run-directory creation or
`sbatch` because v3's read-only executable sealing appeared as file-mode-only
Git changes. Updated all remote cleanliness gates to use
`core.fileMode=false` while retaining exact SHA, byte, untracked-file, contract,
and v3 receipt validation. Nineteen tests, Ruff, Pyright, shell syntax, and diff
checks pass. No experiment jobs have been submitted yet.

[2026-08-27 22:28 EDT] First live chain and ICE bubblewrap compatibility failure

Context: Release `b2db6c24f8e4a375c95389d1188f9216c62f8132` was sealed with a
v3 receipt and submitted as run
`qwen3-8b-confirmatory-20260828T022045Z-b2db6c24f8e4`. The submission receipt
binds jobs `5594275` through `5594293` to that release.

Action: Bootstrap and both preparation jobs completed, warming the shared model
and dataset cache. The allocated-node GPU doctor ran on one H100 as job
`5594278` and failed before containment with exit `1:0` because ICE provides
bubblewrap 0.4.1, which supports `--unsetenv` but not `--clearenv`.

Result: No smoke baseline, smoke arm, confirmatory baseline, confirmatory arm,
robustness, or analysis workload started. The failed allocation row is retained.
Replaced the unsupported flag with an explicit unset of every inherited
environment key before restoring only the fixed HOME/PATH and allowlisted GPU
and offline-cache variables. A regression test proves inherited values do not
cross the command boundary. Nineteen tests, Ruff, Pyright, and Slurm syntax
checks pass. A new executor release and a new run contract are required; the old
chain will not be reused.

[2026-08-27 22:37 EDT] GPU sandbox driver-metadata diagnosis

Context: Replacement release `51cfc37c267bc6d63b7b9947e768225c12a0b302`
was submitted as run
`qwen3-8b-confirmatory-20260828T023038Z-51cfc37c267b`. Bootstrap and both
preparation jobs completed, but GPU doctor `5594300` failed because contained
NVML returned exit 17 without the host NVIDIA driver metadata mounts.

Action: Preserved the failed doctor and bounded diagnostic allocations
`5594316` through `5594322`, including launcher failure `5594317`. The probes
showed that read-only `/sys` and `/proc/driver/nvidia` views are required in
addition to NVIDIA device nodes and libraries. With those views, Slurm's device
cgroup exposed exactly one allocated H100 inside bubblewrap. No diagnostic
wrote an experiment artifact.

Result: Added read-only driver metadata mounts to GPU-enabled bubblewrap and
strengthened the doctor to require the contained `nvidia-smi` count to equal
`CUDA_VISIBLE_DEVICES`. The network namespace, empty inherited environment,
read-only inputs, and sole writable candidate/output directory remain intact.
Nineteen tests, Ruff, Pyright, and Slurm syntax checks pass. The replacement
chain's downstream work did not start and must be superseded by another exact
release and fresh run contract.

[2026-08-27 22:41 EDT] GPU doctor preparation dependency race

Context: Release `5f3a665179fc5f64ba5789b2fe7c5dda0a4df79d` was
submitted as run `qwen3-8b-confirmatory-20260828T023910Z-5f3a665179fc`.
Bootstrap completed, but doctor `5594326` started before full preparation had
created `provenance.json` and failed at the run-contract gate. Full preparation
later completed successfully in 1:10.

Action: Added the missing `afterok:${FULL_PREP_JOB}` edge to the GPU doctor and
a structural test that locks this producer-consumer dependency into the sealed
submitter.

Result: The failed job did not reach bubblewrap, and no smoke or confirmatory
work started. Nineteen tests, Ruff, Pyright, and Slurm syntax checks pass. This
chain will be retained for accounting and superseded by a fresh exact release.

[2026-08-27 22:54 EDT] Accepted smoke and 46-way confirmatory fan-out

Context: Active release `5ba692c42de46fe3d0f856d5bc47117d99bb2613` is
bound to run `qwen3-8b-confirmatory-20260828T024303Z-5ba692c42de4` and jobs
`5594350` through `5594368`.

Action: Full preparation completed, and allocated-node doctor `5594353`
produced a receipt proving network isolation, nonexistent sandbox home, and
exactly one Slurm-authorized H100 inside bubblewrap. Smoke preparation
`5594351` aborted during Python finalization after writing its cache and
provenance but before `prepare.json`; a diagnosed single requeue under the same
job ID and release short-circuited the prepared cache, warmed the model, and
completed with `Restarts=1`. The two original smoke dependency records were
refreshed to the same successful preparation and doctor IDs.

Result: The baseline, all three smoke arms, and smoke gate completed with exit
zero and four reconciled DONE receipts. The full baseline plus all 45 unique
confirmatory cells then launched together; allocation-only `sacct` and `squeue`
both reported 46 RUNNING jobs, and all 45 arms created checkpoint-contract
fingerprints. This is launch evidence only. Robustness, analysis, finalization,
and accepted export remain pending terminal receipts.

[2026-08-27 23:27 EDT] Four confirmatory arms stopped at quantized-model preparation

Context: Allocation-only `sacct -X` rows reported exit `1:0` for array tasks
`5594359_2`, `_3`, `_5`, and `_8` after 3:08-3:20. The corresponding frozen
cells are `(z=0, seed=47)`, `(z=0, seed=71)`, `(z=0.125, seed=11)`, and
`(z=0.125, seed=71)`. Three tasks ran on `atl1-1-03-011-13-0` and one on
`atl1-1-03-011-18-0`; other allocations on both nodes continued running.

Action: Read the four task logs, the frozen environment receipt, and the exact
dependency source without resubmitting an arm. Every task loaded all five Qwen
shards, constructed the PEFT trainer, and then raised the same
`torch.device(None)` exception from the 4-bit Accelerate device-map path before
the first optimizer step. The frozen environment uses Accelerate 1.8.1. The
signature matches upstream issue `huggingface/accelerate#3758`; merged fix
`huggingface/accelerate#3895` changes the indexless device comparison that calls
`torch.device(None)`.

Result: This is a diagnosed runtime/dependency exception, not a Slurm
`PREEMPTED`, `NODE_FAIL`, or time-limit exit, so no blind retry or requeue was
performed. At the snapshot, 42 allocations remained RUNNING, four downstream
stages remained dependency-pending, and the active run had no baseline or arm
DONE receipt yet. Recovery requires an explicit release/run-contract decision;
the sealed `5ba692c42de46fe3d0f856d5bc47117d99bb2613` executor and the frozen 45-cell
design remain unchanged.

[2026-08-28 00:07 EDT] Full baseline receipt completed

Observation: Allocation `5594357` completed with exit `0:0` after 1:01:37 and
wrote `baseline/DONE.json`. The receipt binds executor and coordinator release
`5ba692c42de46fe3d0f856d5bc47117d99bb2613`, design digest
`1ae2afcfcd1dee7c67bb8fdd5532c037fd762d9dc025ebc08a80ca769db16303`, and
the active run contract. The metrics artifact checksum is
`8d741d6ce24f65c6de69d564fac45261570e112b574a32ee5aae3a24a5f50cad`.

Inference: The baseline stage is receipt-complete, but it does not make the run
acceptable because no confirmatory arm has a DONE receipt and four frozen cells
remain failed. Forty-one arm allocations were still RUNNING, and robustness,
analysis, and finalization remained dependency-pending. No scheduler mutation
was performed.

[2026-08-28 01:40 EDT] Four robustness profiles failed at model construction

Observation: The three seed-11 endpoint arms and baseline reconciled, and
endpoint-report job `5594360` completed with exit `0:0` and exact-release receipt
`endpoint_seed_0011.json`. Robustness array tasks `5594361_0` through `_3` then
each failed with exit `1:0` after 46-47 seconds. Every profile passed the
bubblewrap launch boundary, selected `cuda:0`, and stopped before checkpoint
loading with `Qwen3ForCausalLM.__init__() got an unexpected keyword argument
'dtype'`. The sealed command supplied `dtype=bfloat16` through lm-eval 0.4.10.

Inference: This is a deterministic runtime/interface exception in the frozen
robustness launcher, not PREEMPTED, NODE_FAIL, a time-limit exit, OOM, or a
sandbox failure. No robustness retry was submitted. At the snapshot, 38 of 45
arm receipts existed, three arms were still RUNNING, the four previously
diagnosed arm failures remained unchanged, and analysis/finalization were still
dependency-blocked.

[2026-08-28 02:12 EDT] Confirmatory grid reached terminal partial state

Observation: All baseline and confirmatory allocations are terminal. Forty-one
unique arm DONE receipts match the sealed executor/coordinator release and the
frozen design, provenance, and run-contract digests. The four missing cells are
`(z=0, seed=47)`, `(z=0, seed=71)`, `(z=0.125, seed=11)`, and
`(z=0.125, seed=71)`, matching the four diagnosed exit-1 allocations. Across
the baseline, 45 arm allocations, and four failed robustness allocations, the
50 allocation-only rows total 85.135833 GPU-hours.

Inference: The active run cannot satisfy its 45-arm receipt-completeness gate,
so main analysis and finalization remain dependency-blocked. The existing 41
arm receipts are valid partial artifacts but do not constitute accepted results.
No retry, cancellation, design reduction, or downstream bypass was performed.

[2026-08-28 11:06 EDT] AUTHORITY correction for corrected replacement run

Context: The active sealed run exposed two deterministic executor defects and
cannot progress beyond 41 of 45 arm receipts or produce robustness receipts.

Action: The owner explicitly granted standing, persistent authorization for
this experiment task to push the topic branch, including detailed run metadata
and append-only ledgers, to
`https://github.com/Evan-eng/econml_model_routing`; create and push corrected
topic-branch releases; seal exact-SHA deployments and run contracts; and submit
and operate the complete replacement 45-arm chain. This decision is consumed
and must not be requested again after compaction or in later sessions.

Result:

- repo_publish: authorized for this task on topic branch
  `glenn/ice-confirmatory-24h` to the named remote, including run metadata.
- slurm_submit: authorized for complete corrected replacement chains within the
  existing Qwen3-8B baseline, 45-arm, and four-profile robustness scope.
- deploy: exact-SHA releases through `pace-code-release` only.
- protected/default branches, force operations, merges, repository
  administration, out-of-contract deletion, and credentials disclosure remain
  prohibited.

[2026-08-28 11:13 EDT] Corrected deterministic executor failures

Observation: The failed arm traces showed Accelerate receiving an unindexed
device map during 4-bit single-GPU Trainer setup. The failed robustness traces
showed lm-eval 0.4.10 passing `dtype=` into Transformers 4.53.2, whose
`PreTrainedModel.from_pretrained` consumes only `torch_dtype=` and forwarded the
unknown key into the Qwen3 constructor.

Action: Quantized model loads now fail closed without CUDA and otherwise bind
the whole model to `torch.cuda.current_device()`. Robustness retains lm-eval's
public `dtype` interface but executes through a sealed, exact-version
compatibility entrypoint that translates the keyword only at the Transformers
call boundary. The executor release is mounted read-only inside the networkless
bubblewrap boundary so the entrypoint is available without expanding writable
access.

Verification: The changes were developed with failing regression tests. The
final local checks passed: Ruff, Pyright, 24 Pytest tests, shell syntax for all
Slurm scripts, `uv lock --check`, `git diff --check`, and a live import probe
confirming that the compatibility method is inherited by Qwen3 under the
frozen lm-eval 0.4.10 and Transformers 4.53.2 tuple. The requested clean
reviewer worker was unavailable, so the primary agent completed the exact-diff
standards/spec review and caught the missing read-only release mount before
this checkpoint.

Inference: These changes address both deterministic failure signatures without
changing the frozen design, model or dataset revisions, attempt policy, or
sandbox write boundary. A same-model compute-node smoke remains the required
operational proof before the complete replacement fan-out.

[2026-08-28 11:15 EDT] Corrected release deployed and replacement chain submitted

Observation: Topic-branch SHA
`2843dc350de2a5eb154af1e833687d8437587796` was pushed to the approved origin,
sealed under
`/storage/ice-shared/cs7634/staff/econml/releases/econml_model_routing_v3`,
and verified by its `pace-code-release-v3` receipt. The fresh run is
`qwen3-8b-confirmatory-20260828T151445Z-2843dc350de2`. Its submission receipt
binds bootstrap `5594908`, smoke/full preparation `5594909`/`5594910`, GPU
doctor `5594911`, smoke baseline/array/gate `5594912`/`5594913`/`5594914`, full
baseline `5594915`, endpoint/remaining arrays `5594916`/`5594917`, endpoint
report `5594918`, robustness `5594919`, analysis `5594920`, finalizer `5594921`,
and status jobs `5594922` through `5594926` to the same executor and coordinator
release.

Action: Submitted the complete smoke-gated baseline, 45-arm, four-profile
robustness, analysis, finalization, and scheduled-status dependency chain with
`MAX_PARALLEL=45`. Restored a 30-minute exact-run heartbeat carrying the
persistent authority, failure-recovery policy, accounting rules, and
superseded-run exclusions.

Result: At the first queue snapshot, bootstrap `5594908` was RUNNING and the
scientific jobs were PENDING on their declared dependencies. This is verified
submission and queue evidence only; preparation, GPU doctor, smoke, full-grid,
robustness, analysis, and export acceptance remain pending receipts.

[2026-08-28 11:20 EDT] Smoke preparation cache race and serialized replacement

Observation: In release `2843dc350de2a5eb154af1e833687d8437587796`, full
preparation `5594910` and GPU doctor `5594911` completed with exit zero, but
concurrent smoke preparation `5594909` exited `134:0` during CPython
finalization after atomically writing its cache marker and provenance. The
trace was the same `PyGILState_Release` finalization abort previously observed
when smoke and full preparation shared the dataset/cache stack concurrently.
Smoke jobs `5594912` and `5594913` therefore became
`DependencyNeverSatisfied`; no smoke or confirmatory GPU work started.

Action: A proposed same-job requeue was rejected because an application abort
is outside the run contract's automatic recovery list, so no retry or
dependency rewrite was performed. Instead, reordered the sealed submitter so
full preparation completes before smoke preparation starts. GPU doctor may
still overlap the later smoke preparation because it only consumes the fully
receipted full run. Added a structural regression test for the new dependency.

Verification: Ruff, Pyright, 24 Pytest tests, all Slurm shell syntax,
`uv lock --check`, and `git diff --check` pass. The change preserves the exact
design and all receipt gates while removing concurrent preparation against the
shared cache.

Inference: Run `qwen3-8b-confirmatory-20260828T151445Z-2843dc350de2` is
terminally incomplete and becomes accounting-only. A fresh exact-SHA release
and run contract are required; its completed full-preparation and doctor
receipts are not reused across the release boundary.

[2026-08-28 11:24 EDT] Smoke finalization abort reproduced without concurrency

Observation: Serialized run
`qwen3-8b-confirmatory-20260828T152037Z-b55d30cea848` completed bootstrap
`5594938`, full preparation `5594939`, and GPU doctor `5594941` with exit zero.
Smoke preparation `5594940` then reproduced the same exit `134:0` after 44
seconds, despite running after full preparation. This falsifies the shared
concurrency hypothesis and localizes the failure to CPython extension cleanup
after the smaller streaming preparation returns successfully. Smoke jobs
`5594942` and `5594943` became `DependencyNeverSatisfied`; no scientific GPU
work started.

Action: Added an ICE-only `--skip-python-finalization` option. It calls
`os._exit(0)` only after `prepare_data` returns, stdout/stderr are flushed, and
all cache/provenance writes have completed through the existing atomic,
fsynced writers. Any exception before successful return still follows the
normal nonzero fail-closed path. The Slurm preparation worker opts into this
boundary; model warming and the preparation receipt remain a separate fresh
Python process and therefore continue to validate the sealed artifacts before
downstream work.

Verification: Two red-green regression tests prove that only successful
preparation takes the hard-exit path and that the sealed Slurm worker requests
it before the separately receipted warm-cache stage. Ruff, Pyright, 26 Pytest
tests, Slurm shell syntax, `uv lock --check`, and `git diff --check` pass.

Inference: Run `qwen3-8b-confirmatory-20260828T152037Z-b55d30cea848` is
terminally incomplete and becomes accounting-only. A fresh exact-SHA run is
required; no application-abort requeue or dependency rewrite is authorized or
performed.

[2026-08-28 11:30 EDT] Preparation rate limit and offline-cache hardening

Observation: In run `qwen3-8b-confirmatory-20260828T152609Z-ab988a9ea7ed`,
bootstrap `5594985` completed, but full preparation `5594986` exited `1:0`
after Hugging Face returned HTTP 429 while refetching the pinned code-training
parquet. The run directory contains only partial pre-marker cache files and no
provenance or preparation receipt. Downstream preparation, doctor, smoke, and
scientific jobs remained dependency-blocked.

Action: No blind retry was submitted. The shared cache already contains the
same exact model and dataset revisions from earlier receipted preparation, so
the sealed preparation worker now enables Hugging Face, Datasets, and
Transformers offline modes before loading. Missing cache content therefore
fails closed instead of contacting an external mutable service or receiving a
rate limit.

Verification: Added a red-green structural regression test that requires
offline preparation and prohibits the prior environment unsetting. Ruff,
Pyright, 27 Pytest tests, Slurm shell syntax, `uv lock --check`, and
`git diff --check` pass.

Inference: Run `qwen3-8b-confirmatory-20260828T152609Z-ab988a9ea7ed` is
terminally incomplete and becomes accounting-only. A fresh exact-SHA run is
required to combine offline preparation with the successful-finalization
boundary; no incomplete cache files are reused as run artifacts.

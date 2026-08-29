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

[2026-08-28 11:35 EDT] Streaming-cache boundary and bounded rate-limit recovery

Observation: Offline run
`qwen3-8b-confirmatory-20260828T153021Z-31b02693c37e` completed bootstrap
`5595010`, then full preparation `5595011` failed in 19 seconds because the
materialized evaluation datasets were cached but the pinned
`open-r1/OpenR1-Math-220k` streaming source could not be resolved offline. No
provenance, preparation receipt, doctor, smoke, or scientific artifact was
produced.

Action: Restored network access only inside the preparation stage. Added a
three-attempt application-level retry that recognizes HTTP 429 plus `Too Many
Requests`, waits 15 then 30 seconds, and restarts the deterministic atomic
preparation. All other exceptions fail immediately. The success-only
finalization boundary remains, and all GPU stages remain offline.

Verification: Red-green tests cover one successful 429 recovery, immediate
propagation of non-rate-limit failures, the finalization boundary, and the
preparation-only network scope. Ruff, Pyright, 29 Pytest tests, Slurm shell
syntax, `uv lock --check`, and `git diff --check` pass.

Inference: Run `qwen3-8b-confirmatory-20260828T153021Z-31b02693c37e` is
terminally incomplete and becomes accounting-only. A fresh exact-SHA run is
required; no failed preparation job is requeued.

[2026-08-28 11:38 EDT] Active bounded-recovery run reached same-model smoke

Observation: Release `c229e3c901dfa9f708f51f5fa882eee0a0f11710` is sealed
and bound to run `qwen3-8b-confirmatory-20260828T153458Z-c229e3c901df`.
Submission receipt jobs are bootstrap `5595031`, full/smoke preparation
`5595032`/`5595033`, doctor `5595034`, smoke baseline/array/gate
`5595035`/`5595036`/`5595037`, full baseline `5595038`, endpoint/remaining
arrays `5595039`/`5595040`, endpoint report `5595041`, robustness `5595042`,
analysis `5595043`, finalizer `5595044`, and status jobs `5595045` through
`5595049`.

Action: Full preparation encountered an explicit HTTP 429, performed its
sealed 15-second first retry, then completed with exit zero and a preparation
receipt. Serialized smoke preparation completed with exit zero under the
success-only finalization boundary. GPU doctor completed with exit zero.

Result: Allocation-only scheduler rows show smoke baseline `5595035` and all
three tasks of smoke array `5595036` RUNNING concurrently on allocated H100s.
This proves the preparation repairs operationally, but it is launch evidence
only: the smoke gate, full grid, robustness, analysis, final manifest, and
export remain pending terminal receipts.

[2026-08-28 12:37 EDT] CUDA allocation failure recovery hardening

Observation: The accepted smoke gate released full baseline `5595038`, endpoint
array `5595039`, and remaining array `5595040` at 11:42 EDT. At 12:36 EDT the
baseline and 39 unique arms remained RUNNING after about 54 minutes, while six
remaining-array tasks had failed after 30 seconds with exit `1:0`. Their stderr
reports the exact infrastructure error `CUDA-capable device(s) is/are busy or
unavailable`; the failed cells are indices 6, 10, 14, 15, 17, and 21. No
baseline or arm DONE receipt existed at this observation.

Action: Added a pre-scientific CUDA allocation probe to baseline, arm, and
robustness GPU workers. Only that exact unavailable-device error requeues the
same receipted Slurm job or array task, and only while
`SLURM_RESTART_COUNT + 1 < MAX_ATTEMPTS`. OOM, scientific exceptions, all
other probe failures, and exhausted attempts retain their original nonzero
exit without automatic recovery. Existing jobs were neither cancelled nor
altered.

Verification: Red-green tests cover same-job requeue, the sealed attempt
limit, non-retry of OOM, and probe ordering before scientific work. Ruff,
Pyright, all 33 Pytest tests, Slurm shell syntax, `uv lock --check`, and
`git diff --check` pass. The requested clean reviewer agent was unavailable;
a local standards/spec review found no additional issue.

Inference: The 39 running arms may still yield valid accounting evidence and
scientific artifacts, but this run cannot satisfy the frozen 45-arm acceptance
gate because the failed jobs have left controller state and cannot be requeued
in place. A fresh exact-SHA replacement chain is required to validate the
bounded recovery on live allocations and produce a complete accepted run.

[2026-08-28 12:40 EDT] Sealed CUDA-recovery replacement submitted

Observation: Tested commit `c3cb569cf51c3f53a3c7492388c7183c9f91c6df`
was pushed on `glenn/ice-confirmatory-24h` and sealed under the v3 release root.
Its release receipt verifies the approved Git origin, full SHA, deployment
path, and tracked `uv.lock`, `pyproject.toml`, and `slurm` identities.

Action: Submitted run
`qwen3-8b-confirmatory-20260828T163748Z-c3cb569cf51c` from that exact release.
The sealed submission receipt binds bootstrap `5595161`, full/smoke preparation
`5595162`/`5595163`, doctor `5595164`, smoke baseline/array/gate
`5595165`/`5595166`/`5595167`, full baseline `5595168`, endpoint/remaining
arrays `5595169`/`5595170`, endpoint report `5595171`, robustness `5595172`,
analysis `5595173`, finalizer `5595174`, and status jobs `5595175` through
`5595179`. The prior c229 run remains untouched and becomes accounting-only.

Result: At 12:39 EDT bootstrap and full preparation had completed with exit
zero; smoke preparation and the GPU doctor were RUNNING. This is verified
startup evidence, not a smoke or scientific result. The heartbeat now follows
the c3cb replacement identity and lists the c229 run as accounting-only.

[2026-08-28 12:54 EDT] Bounded CUDA recovery exhausted; bad nodes excluded

Observation: Run `qwen3-8b-confirmatory-20260828T163748Z-c3cb569cf51c`
completed bootstrap, both preparations, the GPU doctor, and smoke baseline.
The smoke baseline emitted a valid DONE receipt with math `0.875` and code
`0.000`. All three smoke arms failed the CUDA allocation probe on the exact
busy-or-unavailable condition. Allocation-only `sacct` rows record
`Restarts=2` for every task, proving that each task consumed all three sealed
attempts before its final `FAILED 1:0` state; no blind fourth retry occurred.

Observation: Across the c229 full grid and c3cb smoke attempts, unavailable
allocations were observed on nodes `atl1-1-03-010-30-0`,
`atl1-1-03-011-8-0`, `atl1-1-03-011-13-0`, and
`atl1-1-03-011-18-0`. The c229 baseline has a valid DONE receipt, while its
three endpoint arms remain RUNNING. Its four already-submitted MATH-500 and
HumanEval robustness profiles remain dependency-bound only to the baseline and
endpoint receipts, not to the other 42 grid cells.

Action: Submitted replacement
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c` from the same sealed
executor/coordinator release. Its submission receipt binds jobs `5595230`
through `5595248`, with finalizer `5595243`. Before any GPU stage became
eligible, applied the exact four-node exclusion to doctor, smoke baseline and
array, full baseline and arrays, and robustness jobs listed in that receipt;
`scontrol show job` verified the exclusion on scalar and array tasks. No job
outside the active receipt was changed, and the 16:37 run is accounting-only.

Inference: Robustness can overlap the rest of the grid as designed, but the
three adapted profiles cannot start before their corresponding endpoint
adapters have valid DONE receipts. The already-complete baseline receipt is
not sufficient for those adapted profiles.

[2026-08-28 13:25 EDT] Excluded-node replacement passed smoke and launched full grid

Observation: Active run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c` completed bootstrap
`5595230`, full/smoke preparation `5595231`/`5595232`, GPU doctor `5595233`,
smoke baseline `5595234`, all three smoke arms `5595235`, and smoke gate
`5595236` with exit zero. Smoke arm allocation rows show no restarts and ran
only on nodes outside the four-node exclusion.

Result: The smoke gate released full baseline `5595237`, endpoint array
`5595238`, and all 42 remaining-array tasks `5595239`. At 13:24 EDT all 46
full GPU allocations had been RUNNING for about 27 minutes with zero recorded
restarts or failures. Endpoint report `5595240`, four robustness profiles
`5595241`, analysis `5595242`, and finalizer `5595243` remain correctly
dependency-bound. No full baseline, arm, robustness, analysis, or final DONE
receipt existed at this observation.

Inference: Excluding the four empirically bad nodes removed the immediate CUDA
allocation failure mode through smoke and initial full-grid launch. Continued
RUNNING state is not a scientific result; acceptance still requires all
terminal receipts and the final manifest/export.

[2026-08-28 13:56 EDT] Full grid completed training checkpoints

Observation: Full baseline and all 45 arms remained RUNNING after 58 minutes,
with zero allocation restarts or failed rows. Every arm now contains both
retained training checkpoints, `checkpoint-117` and `checkpoint-128`, so all
45 arms have crossed the sealed final optimizer checkpoint and are in
post-training evaluation or artifact finalization. No baseline or arm DONE
receipt existed yet.

Inference: Checkpoint completeness is strong progress evidence but not terminal
acceptance. Endpoint report and robustness remain correctly blocked until the
baseline and three endpoint evaluations emit reconciled DONE receipts.

[2026-08-28 14:18 EDT] AUTHORITY all-profile exploratory robustness extension

goal: Evaluate MATH-500 and HumanEval for the 42 nonconfirmatory adapters and
join them with the four preregistered robustness profiles without altering or
blocking the confirmatory run.

repo_publish: Allowed on topic branch `glenn/ice-confirmatory-24h`; default and
protected branches remain prohibited.

slurm_submit: Allowed for one sealed 42-task sparse one-HX00-GPU array, its
environment bootstrap, compute-node CUDA sandbox doctor, receipt-bound
finalizer, and bounded infrastructure recovery with at most three attempts.

deploy: Exact-SHA release through `pace-code-release` only, with
executor_release equal to coordinator_release. The source input producer is
active confirmatory run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c`; its executor remains
`c3cb569cf51c3f53a3c7492388c7183c9f91c6df`.

decisions_consumed: The user explicitly authorized the 42-profile robustness
extension on top of the existing work. The four original profiles remain
confirmatory; the other 42 are exploratory. Matching-task `aftercorr`
dependencies and maximum 42-way evaluation concurrency are authorized.

boundaries: The extension uses a separate run contract, artifacts, manifest,
ledger, and export. It must not overwrite, cancel, delay, reinterpret, or
become a dependency of the confirmatory run. Jobs outside the source and
extension submission receipts remain untouched.

[2026-08-28 14:37 EDT] All-profile extension implementation verified

Observation: The extension selects exactly sparse grid indices
`1-19,21-39,41-44`, preserving the original baseline and seed-11 endpoint
profiles as the four confirmatory robustness results. Each extension task
waits on its matching source array task with `aftercorr`, validates the sealed
source and extension role tuple, and writes only to the extension run root.
The finalizer requires 42 exploratory receipts and four confirmatory receipts,
records allocation-only `sacct -X` rows, and emits a separate read-only
46-profile manifest and export.

Verification: `uv run ruff check .`, `uv run pyright`, `uv run pytest -q`
(44 passed), `uv lock --check`, `git diff --check`, `bash -n slurm/*.sbatch
slurm/*.sh`, and ShellCheck with only the expected dynamic-source `SC1091`
suppression all exited zero. Native independent reviewer dispatch was
unavailable in this session, so the fixed-base standards/spec review was
performed locally against `a7be3e8f512df92bf3ad3b5ff318d0cbb486c094`.

Inference: The implementation is ready for topic-branch publication, exact-SHA
sealing, and submission. No cluster execution or scientific result is implied
by these local gates.

[2026-08-28 14:41 EDT] Exploratory robustness extension submitted and old runs audited

Observation: Topic-branch release
`d172354539dce0025befbbe9d76f4d6eab31d3ad` was pushed, verified against the
remote ref, and sealed under
`/storage/ice-shared/cs7634/staff/econml/releases/econml_model_routing_v3`.
Extension run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-all-20260828T183840Z-d172354539dc`
has submission receipt jobs bootstrap `5595431`, GPU doctor `5595432`,
42-profile sparse array `5595433`, and finalizer `5595434`. The receipt
binds the new executor/coordinator SHA to source run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c` and source release
`c3cb569cf51c3f53a3c7492388c7183c9f91c6df`.

Observation: Slurm shows the sparse array indices
`1-19,21-39,41-44%42`, dependency
`aftercorr:5595239,afterok:5595432`, one HX00 GPU per task, 15:30 time limit,
and the four recorded bad CUDA nodes excluded. Bootstrap and the compute-node
GPU sandbox doctor completed with exit zero. Finalizer `5595434` waits for
both array `5595433` and confirmatory robustness job `5595241`.

Observation: The accepted source run now has the baseline and 2 of 45 arm DONE
receipts; its other 43 arm allocations remain RUNNING after about 1:42. It has
no robustness, analysis, final, or export acceptance receipt yet. The bounded
audit of all explicitly superseded runs found no finalized run. Superseded
run ending `c229e3c901df` still had two stray remaining-array allocations,
`5595040_27` and `5595040_33`, running for 10,687 seconds each. Both exact
receipt-listed tasks were cancelled so the run is accounting-only; their
allocation rows remain preserved as `CANCELLED by 3039927`. No accepted-run
job or unrelated account job was altered.

Inference: Submission and its operational gates are verified, but there is no
new exploratory scientific result yet. Extension tasks will release
individually only after their matching source arm exits successfully; the
accepted source run remains the critical path.

[2026-08-28 15:24 EDT] AUTHORITY receipt-bound robustness recovery goal

goal: Repair the deterministic whole-process sandbox failure, reuse the
accepted source baseline and all 45 adapters without retraining, and evaluate
MATH-500 and HumanEval for all 46 profiles through an accepted combined export.
The baseline and seed-11 endpoint profiles remain confirmatory; the other 42
profiles remain exploratory.

authority: The owner's standing authorization covers topic-branch publication,
exact-SHA deployment, replacement robustness submission, source-aligned
one-GPU fan-out, and bounded operational recovery inside this experiment
contract. It does not cover protected branches, force operations, unrelated
jobs, design reduction, or blind application-failure retries. This authority
has been consumed and must not be requested again for this chain.

[2026-08-28 15:24 EDT] Deterministic robustness sandbox failure contained

Observation: Adapter robustness tasks in superseded extension
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-all-20260828T183840Z-d172354539dc`
failed across independent array elements when the PEFT/bitsandbytes/Triton
import path attempted absolute `/sbin/ldconfig` inside bubblewrap. The exact
exception was `FileNotFoundError` for `/sbin/ldconfig`; this was an application
sandbox defect, not an OOM, node failure, preemption, or scientific result.

Action: Receipt-listed extension array `5595433` and finalizer `5595434` were
cancelled after the repeated deterministic failures. The source release used
the same sandbox construction, so its not-yet-allocated robustness job
`5595241` and dependent finalizer `5595243` were also cancelled. Source
training, endpoint reporting, and analysis were not altered. All failed and
cancelled allocation rows remain retained for accounting.

Inference: Retrying either sealed robustness stage would reproduce the same
defect. A new exact-SHA recovery release is required; accepted source adapters
remain valid and do not require retraining.

[2026-08-28 15:24 EDT] Recovery implementation passed local release gates

Observation: The corrected GPU sandbox exposes only `/sbin/ldconfig` and
`/etc/ld.so.cache` as read-only inputs while retaining the networkless
whole-process boundary. A new recovery contract binds the accepted source run,
failed extension, new GPU doctor, bounded adapter smoke, exact source profile
release, and each terminal output. The source-aligned fan-out contains one
baseline, endpoint indices `0,20,40`, and remaining indices
`1-19,21-39,41-44`; its finalizer requires all 46 robustness receipts and the
source analysis, exports required raw predictions and interpretation files,
and accounts allocation-only rows from the source, failed extension, and
recovery jobs.

Verification: `uv run ruff check .`, `uv run pyright`, `uv run pytest -q`
(54 passed), `uv lock --check`, `bash -n slurm/*.sbatch slurm/*.sh`, ShellCheck
with only the expected dynamic-source `SC1091` suppression, and
`git diff --check` all exited zero. Native reviewer dispatch was unavailable,
so the fixed-base standards/spec review was performed locally against
`218d8dd4329292e49eda2eb6834be18d0a25eab0`.

Observation: At 15:24 EDT the accepted source run had a baseline and 41 of 45
arm DONE receipts. Four arm allocations remained RUNNING, analysis `5595242`
remained dependency-pending, and no source robustness or final acceptance
receipt existed. These are progress observations, not scientific results.

Inference: The recovery smoke can start as soon as its already-completed source
endpoint adapter and the new GPU doctor validate. Most profile evaluations can
then run immediately while the last four source arms remain the only per-profile
dependencies and source analysis remains the combined-export dependency.

[2026-08-28 15:32 EDT] First recovery submission failed closed on historical dependencies

Observation: Exact release `b10f1b6c716702ebacbb4cebd4f4e6463127f44c`
was pushed, remotely verified, and sealed. Recovery initialization created run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T192701Z-b10f1b6c7167`,
but ICE rejected new `afterok` and `aftercorr` dependencies on historical
source array jobs with `Batch job submission failed: Job dependency problem`.
No submission receipt or robustness result was created. Partial setup jobs
`5595518` (bootstrap) and `5595519` (GPU doctor) completed successfully; their
allocation rows and the incomplete run remain accounting-only.

Observation: The partial-chain trap did not stop at the first failed `sbatch`
because command-substitution errexit semantics allowed `submit_job` to return
the status of its final `printf`. Blank partial-ID lines also made its cleanup
call unreliable. No recovery training or evaluation GPU job was submitted.

Action: The launcher now returns immediately when `sbatch` fails and filters
blank IDs before exact partial-chain cancellation. Dependencies on historical
source baseline, arrays, and analysis were removed. New receipt-producing CPU
gates wait for and validate each exact baseline/arm DONE receipt and the source
analysis DONE receipt. New recovery arrays use `aftercorr` only against their
new source-gate arrays, so 41 ready adapters can release immediately while the
four unfinished adapters wait independently without occupying GPUs.

Verification: The corrected gate implementation and per-profile provenance
lineage passed `uv run ruff check .`, `uv run pyright`, `uv run pytest -q`
(54 passed), `uv lock --check`, shell syntax, ShellCheck, and `git diff --check`.

Inference: The dependency rejection was a launcher/scheduler integration
failure, not an experiment or model result. The accepted source artifacts and
the failed extension accounting remain valid inputs to a new exact-SHA
replacement recovery run.

[2026-08-28 15:40 EDT] Adapter smoke exposed linker-alias sandbox dependency

Observation: Receipt-gated recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T193323Z-7674d9170f53`
submitted successfully under exact release
`7674d9170f5311884e59748148fcdad3562cfcb7`. Bootstrap and the new GPU doctor
completed, 42 source-profile gates emitted exact readiness receipts, four
source-profile gates waited on the four unfinished source arms, and adapter
smoke `5595549` loaded all five Qwen3-8B checkpoint shards and selected CUDA.

Observation: During PEFT adapter injection, Triton compiled its CUDA utility
with `/usr/bin/gcc`. GCC then failed with `collect2: fatal error: cannot find
'ld'`. On ICE, `/usr/bin/ld` is a symlink to `/etc/alternatives/ld`, which is
itself a symlink to the already-visible `/usr/bin/ld.bfd`. The sandbox exposed
`/usr` but not the single `/etc/alternatives/ld` link. Smoke exited 1 without a
DONE receipt. Dependent evaluation jobs never ran. The four waiting source
gates, analysis gate, baseline/array evaluations, and finalizer were cancelled
by exact receipt-listed IDs; completed gates and failed smoke remain retained
for accounting.

Action: The GPU sandbox now exposes only `/etc/alternatives/ld` read-only in
addition to the previously added read-only `/sbin/ldconfig` and
`/etc/ld.so.cache`. The next recovery plan and source receipt also bind the
failed `7674d9170f53` recovery submission, job map, failure class, and receipt
digest, and its allocation rows are included in combined accounting.

Verification: The new linker regression test failed before the correction and
passed afterward. The complete release gate then passed `uv run ruff check .`,
`uv run pyright`, `uv run pytest -q` (55 passed), `uv lock --check`, shell
syntax, ShellCheck, and `git diff --check`.

Inference: This was a second deterministic sandbox integration failure, not a
model, benchmark, OOM, or node result. Reusing release `7674d9170f53` would
reproduce it; the accepted source adapters still do not require retraining.

[2026-08-28 15:48 EDT] Smoke passed linker and exposed read-only dataset lock

Observation: Recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T194113Z-58aea50cde31`
under exact release `58aea50cde311a2776e1d3ac902ad98dc0322af8`
bound the prior failed recovery in its source and submission receipts. Smoke
`5595624` loaded the adapter, completed Triton CUDA utility compilation through
the corrected linker alias, and reached offline MATH-500 dataset loading.

Observation: Hugging Face Datasets then attempted to create a cache lock file
under the deliberately read-only shared cache and failed with `OSError: [Errno
30] Read-only file system`. No smoke DONE receipt or benchmark result was
created. Dependent evaluation jobs never ran. The four waiting source gates,
analysis gate, evaluations, and finalizer were cancelled by exact receipt-listed
IDs and remain in accounting.

Action: Robustness jobs now copy only the pinned
`HuggingFaceH4___math-500` and `openai___openai_humaneval` dataset cache trees,
about 586 KiB total, into a unique writable node-local `$TMPDIR` directory.
Bubblewrap binds that directory writable, points `HF_DATASETS_CACHE` to it, and
continues to mount the shared Hugging Face cache read-only. The disposable copy
is removed after evaluation. Model weights, adapters, the sealed release, and
the shared experiment tree remain read-only inside the whole-process sandbox.

Verification: New tests prove that only the two pinned datasets are staged and
that the node-local cache is the only added writable bind. The full release
gate passed `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` (57
passed), `uv lock --check`, shell syntax, ShellCheck, and `git diff --check`.

Inference: This is another operational containment mismatch, not a scientific
result. Node-local staging preserves the intended storage policy and avoids
shared-cache lock contention when up to 46 evaluations fan out concurrently.

[2026-08-28 15:58 EDT] Dataset staging passed and exposed missing offline code_eval metric

Observation: Recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T194913Z-69b14c85a84b`
under exact release `69b14c85a84b948164b329903f746eb6399d9bcd`
successfully staged both pinned datasets in node-local scratch. Smoke `5595700`
loaded Qwen3-8B and its adapter, compiled the Triton CUDA utility, evaluated the
bounded MATH-500 item, and reached HumanEval metric setup. Evaluate then tried
to create or download `code_eval` under the read-only shared Hugging Face cache
and failed with `OSError: [Errno 30] Read-only file system`. No smoke DONE
receipt or benchmark result was created. Exact receipt-listed dependent and
waiting jobs `5595698`, `5595699`, and `5595701` through `5595704` were
cancelled; all completed, failed, and cancelled allocation rows remain in the
append-only accounting chain.

Action: The recovery now prepares the official `evaluate-metric/code_eval`
module at pinned revision
`262b7e74cf29a715d74f8b02ba1d6ef74e432333` in a dedicated bounded-network CPU
stage after the frozen environment is built. Its receipt binds the recovery
source release, environment release, exact revision, and SHA-256 digests of
`code_eval.py` and `execute.py`. Each GPU evaluation copies the prepared
`evaluate` and `modules` trees, together with only the two pinned dataset
caches, into a unique writable node-local directory. The whole-process sandbox
continues to expose the shared model cache, release, source experiment, and
adapters read-only with networking disabled.

Verification: A local offline probe loaded `code_eval` solely from the prepared
module cache. New unit tests cover the pinned revision and receipt, runtime
cache staging, sandbox environment allowlist, and Slurm dependency edge. The
targeted suite passed 26 tests, Ruff passed, Pyright reported no errors, shell
syntax passed, and the complete release gate remains pending below.

Inference: The failure was a deterministic missing-input preparation defect,
not a model, benchmark, OOM, or node result. The accepted source adapters remain
valid and do not require retraining; a new exact-SHA recovery must bind the
failed `69b14c85a84b` submission and may reuse all source DONE receipts.

[2026-08-28 16:01 EDT] Offline metric recovery release passed its complete gate

Observation: An exact-environment local probe showed that Evaluate keeps metric
instance state under `HF_METRICS_CACHE`, separately from its download and module
caches. With `HF_HOME` pointed at an unwritable location and the three explicit
cache variables pointed at disposable storage, `evaluate.load("code_eval")`
loaded the pinned cached module and instantiated `CodeEval` without network or
writes outside the disposable root. The runtime sandbox now allowlists and sets
all three paths.

Verification: `uv run ruff check .`, `uv run pyright`, `uv run pytest -q` (59
passed), `uv lock --check`, `bash -n slurm/*.sbatch slurm/*.sh`, ShellCheck with
only the expected dynamic-source `SC1091` exclusion, and `git diff --check` all
exited zero. This verifies the source release locally; it does not yet prove an
ICE compute-node smoke or a scientific result.

[2026-08-28 16:08 EDT] Pinned metric loaded and explicit code-evaluation consent was missing

Observation: Recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T200236Z-bca43db9685b`
was submitted under exact release
`bca43db9685b53c64e14e9d2e6bd1a47b9b42b41`. Its frozen environment, pinned
metric preparation, and CUDA sandbox doctor completed with receipts. Smoke
`5595791` loaded Qwen3-8B and its adapter, loaded both datasets offline, and
loaded the exact cached `code_eval` module from node-local scratch. It then
failed before candidate execution because the whole-process sandbox removed
the host's `HF_ALLOW_CODE_EVAL` variable and did not set it internally. No smoke
DONE receipt or robustness result was created.

Action: Exact receipt-listed waiting and dependent recovery jobs `5595788`,
`5595789`, `5595790`, `5595792`, `5595793`, `5595794`, and `5595807` were
cancelled. Completed preparation and doctor jobs plus failed smoke remain in the
append-only accounting chain. The sandbox allowlist now admits
`HF_ALLOW_CODE_EVAL`, and the runner sets it to `1` only after the existing
explicit `--allow-generated-code-execution` and `--code-sandbox=bwrap` gates
have passed. It is not inherited from the host and remains absent for any run
without the explicit generated-code opt-in.

Verification: The sandbox command regression test now proves the consent
variable crosses only the filtered boundary. The complete release gate passed
Ruff, Pyright, all 59 tests, lock verification, shell syntax, ShellCheck, and
`git diff --check`.

Inference: This was a fail-closed application-consent integration defect, not
candidate execution, a scientific result, OOM, or node failure. The successful
offline dataset and metric loads validate the preceding recovery changes; the
next exact-SHA recovery can reuse the same accepted source adapters and shared
pinned cache without retraining or downloading new inputs.

[2026-08-28 16:15 EDT] Whole-process smoke completed both tasks but result filename differed

Observation: Recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T201011Z-f3190c6051a9`
under exact release `f3190c6051a97dee7d5b00ebffd504ea868bca6c`
passed pinned cache loading and explicit code-evaluation consent. Smoke
`5595863` built both task contexts, generated both bounded responses, executed
the HumanEval candidate inside bubblewrap, printed the two smoke metrics, and
exited lm-eval successfully. lm-eval 0.4.10 wrote one timestamped
`results_2026-08-28T20-14-06.859295.json` plus both per-task sample files. The
wrapper then failed closed because it expected the requested canonical
`results.json`; therefore no smoke DONE receipt was issued and the displayed
limit-1 metrics are operational smoke output, not accepted scientific results.

Action: Exact receipt-listed waiting and dependent recovery jobs `5595860`,
`5595861`, `5595862`, and `5595864` through `5595867` were cancelled and
retained for accounting. The runner now accepts only one unambiguous
timestamped `results_*.json`, atomically renames it to the canonical
`results.json`, parses it, and requires a task-results mapping before writing a
DONE receipt. Zero or multiple candidates fail closed.

Verification: Two new tests cover successful timestamp normalization and the
ambiguous-artifact failure. The complete release gate passed Ruff, Pyright, all
61 tests, lock verification, shell syntax, ShellCheck, and `git diff --check`.

Inference: The generated-code containment and both benchmark executions now
work end to end on an ICE H100. The remaining defect is artifact naming across
the pinned lm-eval release, not a scientific, model, sandbox, OOM, or node
failure. Accepted source adapters remain reusable without retraining.

[2026-08-28 16:21 EDT] Receipt-backed smoke passed and 43 production profiles fanned out

Observation: Recovery
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T201706Z-aae927901acf`
under exact release `aae927901acf251c93710fbb0a6d1b20cfa55102`
produced a reconciled smoke DONE receipt after `5595932` completed with exit 0
in 1:55 on an H100. The receipt binds the source adapter release, pinned metric
preparation release, CUDA sandbox-doctor release, source-ready release, exact
run fingerprints, and canonical result checksum. This is an operational smoke
receipt; its limit-1 scores are not scientific results.

Observation: The successful smoke released the production fan-out. At the
receipt snapshot, 43 one-GPU production profile allocations were RUNNING, zero
production profile DONE receipts existed, and the finalizer remained
dependency-pending. The source run had one baseline plus 42 of 45 arm DONE
receipts; three source arms remained RUNNING, and source analysis remained
dependency-pending. No unrelated job was modified.

Inference: The recovery is no longer blocked on implementation or launch
infrastructure. Forty-three profiles can evaluate immediately while the last
three profile-specific gates wait for their accepted source adapters. Full
completion still requires all 46 production DONE receipts, the source's 45-arm
analysis receipt, the combined final manifest, allocation-only accounting, and
the read-only export.

[2026-08-28 16:38 EDT] Production robustness fan-out is making forward progress

Observation: Allocation-only `sacct -X` showed all 43 released production
robustness profiles RUNNING on receipt-listed H100 or H200 allocations, with no
failed production allocation row. The recovery finalizer remained
dependency-pending. The source run still had baseline plus 42 of 45 arm DONE
receipts; source allocations `5595271`, `5595282`, and `5595283` remained
RUNNING, and source analysis `5595242` remained dependency-pending.

Observation: Live `sstat` telemetry showed each unfinished source arm using
about 5.3 GiB RSS with more than 5.5 GB of disk writes and about 3:35 CPU time.
Their stderr files received fresh writes between 16:36:45 and 16:37:30 EDT.
Representative production baseline job `5595933` reached 243 of 664 generation
requests (37 percent) at about 6.1 seconds per request, with the progress meter
estimating about 43 minutes remaining in that generation phase. No production
robustness DONE receipt existed at this snapshot.

Observation: The ICE SSH endpoint remained intermittently unavailable during
banner exchange, but bounded retries recovered live visibility. This was an
observation-path interruption only; scheduler allocations continued running
and writing logs independently.

Inference: The current evidence supports continued unattended execution rather
than intervention. Completion remains unproven until the three source arm
receipts, source analysis receipt, all 46 production robustness receipts,
recovery `FINALIZED.json`, allocation-only accounting, checksummed manifest,
and read-only export reconcile.

[2026-08-28 16:40 EDT] Forty-third source arm released its robustness profile

Observation: Source allocation `5595283` completed with exit code 0 after
3:40:30 and emitted the exact source receipt
`arms/z_0p6250/seed_0071/DONE.json`. The source run now has 43 of 45 unique arm
DONE receipts plus its baseline receipt. Source allocations `5595271` and
`5595282` remain RUNNING, while source analysis remains dependency-pending.

Observation: The corresponding previously gated production robustness
allocation `5596010` started automatically and remained RUNNING at the next
allocation-only accounting snapshot. This raised the production robustness
fan-out from 43 to 44 active profiles without retraining or manual dependency
repair. The other production allocations remained RUNNING with no terminal
failure row and no production DONE receipt yet.

Inference: The per-profile source-gate dependency behaves as designed. The
remaining two robustness profiles should release independently when their two
source DONE receipts appear; no scheduler intervention is currently warranted.

[2026-08-28 16:45 EDT] Forty-fourth source arm released profile 45

Observation: Source allocation `5595282` completed with exit code 0 after
3:45:18 and emitted `arms/z_0p6250/seed_0047/DONE.json`. The source run now has
44 of 45 unique arm DONE receipts plus its baseline receipt. Its corresponding
production robustness allocation `5596009` started automatically at 16:43:07
EDT, raising the production fan-out to 45 of 46 profiles.

Observation: The final source allocation `5595271`, corresponding to the only
missing `z_0p3750/seed_0011` receipt, remained RUNNING after 3:48. Its stderr
file received a fresh write at 16:45:13 EDT. Source analysis and the last
profile-specific robustness gate remained dependency-pending. All 45 released
production robustness allocations remained RUNNING with no terminal failure
row and no production DONE receipt at this snapshot.

Inference: The last source arm is slow but not stalled according to current
telemetry. The correct action remains to let it finish and allow the sealed
dependency chain to release source analysis and production profile 46.

[2026-08-28 16:53 EDT] Final source arm entered code evaluation; robustness passed 60 percent

Observation: The final source arm `z_0p3750/seed_0011` wrote its complete
755,419-byte `predictions_math.jsonl` at 16:52:34 EDT and remained RUNNING in
its final code-evaluation phase. A completed neighboring arm took about 22
minutes between its math prediction artifact and its code predictions,
metrics, and DONE receipt. The source remains at 44 of 45 arm DONE receipts
until the final receipt reconciles.

Observation: Representative production baseline allocation `5595933` reached
420 of 664 generation requests (63 percent) after 32:56, at about 6.03 seconds
per request. Its progress meter estimated about 24:30 remaining in generation.
All 45 released production robustness allocations remained RUNNING, no
production DONE receipt existed, and allocation-only accounting contained no
terminal production row at this snapshot.

Inference: The source critical path is expected near 17:15 EDT based on the
neighboring code-evaluation duration. The first robustness wave is expected to
reach generation completion near 17:18 EDT before final scoring and receipt
writing. Both estimates are operational projections, not completion evidence.

[2026-08-28 18:11 EDT] Source accepted; robustness at 42 of 46; corrected MATH scorer prepared

Observation: Source allocation `5595271` completed at 17:21:45 EDT after
4:23:59 and emitted the final missing `z_0p3750/seed_0011` terminal receipt.
The source run now has its baseline, all 45 unique arm DONE receipts, and an
analysis DONE receipt from allocation `5595242`. This is receipt-level source
and analysis completeness; it is not yet a combined export acceptance claim.

Observation: At 18:01:13 EDT the active exact-SHA robustness recovery had 42
of 46 profile DONE receipts. Allocations `5595935_1`, `5595935_4`,
`5595935_15`, and `5595935_28` remained RUNNING, finalizer `5595936` remained
dependency-pending, and allocation-only `sacct -X` showed no failed or
cancelled production row. No recovery FINALIZED receipt or export existed at
that snapshot.

Observation: Inspection of accepted raw MATH-500 samples proved that the
lm-eval 0.4.10 legacy `routing_math500` string metric rejects correct
explanatory responses without extracting their final mathematical answers.
The original field is therefore retained with status `invalid`; HumanEval and
the raw MATH generations remain valid inputs and no GPU work was cancelled.

Observation: A CPU-only exact-SHA extension now freezes `math-verify==0.9.0`,
validates all 500 source rows and full receipt lineage, uses list-based
`any_match` extraction, records unparseable predictions as audited incorrect,
and records verification timeouts as unresolved. The local frozen-sample probe
found 174 confirmed correct, 499 resolved, a resolved-only rate of
0.3486973948, and a full-sample sensitivity interval of 0.348 to 0.350; this is
implementation evidence only, not a published experiment result.

Observation: The rescore extension uses a 46-way CPU array, consumes the
accepted parent adapters and raw evaluation artifacts without retraining, and
publishes its terminal receipt only after an atomic read-only export passes
manifest reconciliation. Local gates passed: Ruff, Pyright, 92 tests, Bash
syntax, and diff checks. Independent re-review remains the final publication
gate before commit, push, exact-SHA deployment, and submission.

Inference: The four remaining GPU profiles and the CPU rescore release are
independent critical paths. Continuing both in parallel minimizes wall time;
neither current scheduler state nor the local probe proves final experiment
acceptance.

[2026-08-28 18:27 EDT] Exact-SHA MATH rescore extension submitted

Observation: Independent standards and specification reviewers returned PASS
after the rescore extension fixed full extraction-list matching, unresolved
verification timeouts, HumanEval byte binding, exact parent and environment
lineage, submission receipt races, idempotent append-only accounting, and
terminal-receipt publication ordering. The final local gate passed Ruff,
Pyright, 95 tests, Bash syntax, diff checks, and ShellCheck with only expected
dynamic-source notices.

Observation: Topic-branch SHA
`dc549e0ad61b7d3bcfc1afaf6eef23be9703b44e` was pushed and sealed under
`/storage/ice-shared/cs7634/staff/econml/releases/econml_model_routing_v3`.
The release receipt binds `uv.lock`, `pyproject.toml`, `math500_rescore.py`,
and `slurm/submit_math500_rescore.sh` to the approved origin and exact SHA.

Observation: The new rescore run is
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T201706Z-aae927901acf-math500-rescore-20260828T222636Z-dc549e0ad61b`.
Its sealed submission receipt binds bootstrap `5596188`, 46-way parent gate
`5596189`, 46-way CPU rescore `5596190`, and finalizer `5596191` to release
`dc549e0ad61b7d3bcfc1afaf6eef23be9703b44e` and the accepted parent source,
submission, and rescore-source releases. The first direct launcher invocation
failed before any side effect because the tracked script lacked its executable
mode; invoking the identical sealed bytes through Bash submitted the chain.

Observation: At the first allocation-only snapshot bootstrap `5596188` was
RUNNING and the other three stages were dependency-pending. The parent GPU
recovery had 45 of 46 DONE receipts; allocation `5595998` remained RUNNING and
parent finalizer `5595936` remained dependency-pending. No failure row or final
export existed at that snapshot.

Inference: Submission is complete, but result acceptance is not. The next
gates are all 46 corrected profile receipts, the parent recovery final receipt
and manifest, the corrected exact manifest, allocation ledgers, and verified
read-only permissions.

[2026-08-28 19:11 EDT] MBPP failure isolated and receipt-bound CPU recovery prepared

Observation: Reconciliation across all 46 accepted source prediction files
found 11,822 frozen MBPP rows. Of these, 11,671 admitted candidates failed
before candidate execution with bubblewrap reporting `Creating new namespace
failed: Resource temporarily unavailable`; the other 151 were static rejects.
The scorer installed `RLIMIT_NPROC=(1,1)` in `preexec_fn` before bubblewrap
created its namespaces, while the generic doctor launched bubblewrap without
that outer limiter. The legacy MBPP values and every analysis result depending
on them are therefore retained but labeled invalid.

Observation: The recovery keeps the accepted raw generations, adapters, and
the frozen 257-row evaluation cache unchanged. It launches bubblewrap before
installing and verifying hard CPU, file-size, file-descriptor, process-count,
and address-space limits inside a sealed runner. The exact scoring-path doctor
proves network, home, source-tree, and fork denial. Candidate setup failures
are operational and profile-fatal; candidate exceptions, deaths, and timeouts
after setup are scientific incorrect outcomes. No GPU work or retraining is
required.

Observation: The new CPU-only chain freezes the exact source receipt roster,
all source DONE and prediction digests, the evaluation-cache digest, and the
accepted corrected-MATH final export. It runs one doctor, 46 independent source
gates, 46 independent rescores, corrected analysis with 2,000 bootstraps, and
one fail-closed read-only export finalizer. It preserves legacy outputs and
publishes corrected MBPP artifacts separately with nested manifests and full
producer lineage.

Observation: Two independent rereviews found no blocking specification or
provenance issue after repair. Local verification passed Ruff, Pyright, all
105 tests, Bash syntax, and `git diff --check`. Standing authority covers the
topic-branch commit and push, exact-SHA deployment, and submission of this
bounded recovery chain without another approval boundary.

Inference: The implementation is ready for the final containment rereview,
then exact-SHA publication and ICE submission. Completion remains unproven
until all 46 corrected profile receipts, corrected analysis, allocation-only
accounting, final manifest, and read-only export reconcile.

[2026-08-28 19:15 EDT] First MBPP submission failed closed on stale scheduler dependency

Observation: Exact-SHA release
`1cfb81d7a329eb51752e1cb04b0e5fca9fa9a603` was pushed and sealed under the
v3 ICE release root. Its first launcher invocation created accounting-only run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-mbpp-rescore-20260828T231414Z-1cfb81d7a329`
but Slurm rejected the finalizer's dependency on the already-completed MATH
finalizer because that terminal allocation had aged out of the scheduler's
dependency window. No submission receipt or experiment output was published.

Observation: The launcher's error trap cancelled all four newly submitted
partial job IDs, `5596331` through `5596334`. Allocation-only `sacct -X`
reported all four as CANCELLED with zero elapsed time; the run retains its
source receipt and partial-job roster for append-only failed/debug accounting.
No unrelated allocation was altered.

Observation: The corrected launcher removes the unnecessary scheduler edge to
the historical MATH finalizer and instead requires its FINALIZED receipt and
manifest before submission. The MBPP finalizer still independently validates
the exact parent run, release tuple, all 46 profile receipts, read-only tree,
nested manifest, and parent output release before producing any final output.
The job-submission helper now propagates any `sbatch` failure immediately so a
partial chain cannot continue with an empty job ID. Ruff, Pyright, 105 tests,
Bash syntax, and diff checks pass after the correction.

Inference: This was a coordinator submission defect, not an evaluation or
sandbox failure. It requires a new exact-SHA release; the accounting-only run
must not be resumed.

[2026-08-28 19:19 EDT] Forty-six-way MBPP scorer fan-out exceeded node namespace capacity

Observation: Corrected release
`47d5a4270e8035a485430a81109348d41b0288a9` submitted run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-mbpp-rescore-20260828T231652Z-47d5a4270e80`
with bootstrap `5596348`, doctor `5596349`, source-gate array `5596350`,
rescore array `5596351`, and finalizer `5596352`. The sealed receipt binds
equal executor and coordinator releases and the exact accepted source inputs.

Observation: Bootstrap, the exact scoring-path sandbox doctor, and all 46
source gates completed with exit code 0. With 46 scorer tasks released
together on ICE, 10 profiles completed and 36 failed closed after about 20
seconds. Every failure occurred before candidate execution with bubblewrap
reporting user-namespace creation `No space left on device`; there was no
false pass, partial DONE receipt, or containment downgrade. The permanently
dependency-blocked finalizer `5596352` was cancelled, and no unrelated job was
altered.

Inference: The exact doctor proves a single scorer path, while the observed
10-success/36-failure split on the same compute wave shows that 46 concurrent
bubblewrap namespaces exceed the node's user-namespace capacity. This is an
operational concurrency defect, not a scientific outcome and not evidence of
storage exhaustion.

Observation: The replacement launcher keeps all 46 frozen profiles but caps
only the CPU rescore array at eight concurrent tasks, below the 10-task live
success boundary. Source gates remain fully parallel. The resulting six short
waves preserve the design and should add only minutes of wall time. Ruff,
Pyright, 105 tests, Bash syntax, and diff checks pass after the change.

Inference: The `47d5a427...` run is accounting-only and must not be resumed.
A clean exact-SHA complete replacement is required so every accepted profile
shares one coherent release and run contract.

[2026-08-28 19:33 EDT] Corrected MBPP results complete; allocation export found incomplete

Observation: Release `7b6fcc4a3c4686988fe72dea3d2c701ca4ed2b34`
completed clean replacement run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-mbpp-rescore-20260828T232208Z-7b6fcc4a3c46`.
Bootstrap, doctor, all 46 source gates, all 46 eight-at-once rescores, and
finalizer `5596449` completed with exit code 0. The run produced 46 unique
profile DONE receipts, corrected 2,000-bootstrap analysis, a terminal receipt,
manifest digest `f0f4babad7057e9e051bce02132afa857e5c7c59b46f8a836eef415968b71ed8`,
and a read-only export with zero writable paths.

Observation: Independent acceptance found that the exported allocation ledger
contained only bootstrap `5596445`. The launcher put comma-separated job IDs
inside Slurm's comma-delimited `--export` value, so the finalizer received only
the first ID. The scientific profile artifacts and receipts are complete, but
the bundle fails the experiment's full-attempt accounting requirement and is
therefore not the accepted final export.

Observation: A live allocation-only query over the two failed chains and the
scientifically complete `7b6fcc4...` chain returned 194 rows: 153 COMPLETED,
36 FAILED, and 5 CANCELLED. This proves that the required history remains
available and can be captured without modifying any historical artifact.

Observation: The repair exports job IDs with colon delimiters, reconstructs
commas inside the finalizer, and validates every expected active and
superseded root plus terminal state before accounting or export. The next clean
replacement's ledger includes the failed submission chain, namespace-capacity
chain, scientifically complete but accounting-incomplete chain, and its own
active allocations. Ruff, Pyright, 106 tests, Bash syntax, and diff checks
pass.

Inference: One final short CPU replacement is required for a coherent receipt,
analysis, manifest, read-only export, and complete append-only allocation
ledger. Corrected MBPP point estimates are already scientifically available
but remain provisional until that final accounting gate passes.

[2026-08-28 19:51 EDT] Confirmatory and all-profile robustness experiment accepted

Observation: Final release
`510fd8f85a9b21950dbb6ea9e5f45a94b6e39133` completed run
`qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-mbpp-rescore-20260828T233734Z-510fd8f85a9b`.
All 46 profile receipts and finalizer `5596556` completed with exit code 0.
Independent reconstruction validated the source, environment, submission,
exact scoring-path doctor, every profile, corrected-MATH parent, final receipt,
and read-only export. The accepted manifest digest is
`f31232b36ef4f70000e0614a966138a9b095722854bcac454022fa7ca8519e26`;
the nested corrected-MATH manifest digest is
`ec93fb4c06d0580db915e0d3580a0c8cbc129747d15dfbda818de48cdf8e3f90`.

Observation: The final allocation-only ledger contains 288 rows across the
active and all superseded MBPP attempts: 247 COMPLETED, 36 FAILED, and 5
CANCELLED. It records zero GPU hours for the CPU-only correction. The accepted
export and every descendant have zero writable permission bits.

Observation: A second complete execution reproduced all 46 per-sample files
byte-for-byte and all 46 corrected metric payloads after excluding only the
run-specific source-ready release field. Every analysis table was byte-for-byte
identical; `analysis_summary.json` differed only in `analyzed_at_utc`.

Observation: The preregistered baseline and seed-11 endpoints, reported as
GSM8K / corrected MBPP / corrected MATH-500 sensitivity interval / HumanEval,
are: baseline 89.386 / 6.226 / 22.2-22.4 / 84.756 percent; z=0 32.904 /
4.669 / 14.6-14.8 / 70.732; z=0.5 72.782 / 4.669 / 29.0-29.2 / 71.951;
and z=1 70.584 / 5.837 / 29.4 / 78.049. Across all five seeds, the largest
mean in each adapted grid was z=0.125 for GSM8K at 72.631 percent, z=0.875 for
MBPP at 5.603 percent, and z=1 for corrected MATH-500 and HumanEval at 29.877
and 76.829 percent respectively.

Observation: The locked 2,000-bootstrap analysis estimated the mathematical
frontier stationary point at z=0.624 with percentile interval [0.591, 0.660].
The math curvature parameter was positive with interval [0.790, 1.536], while
the code curvature interval [-0.119, 0.059] included zero. Math endpoint gain
was positive in all five seeds; corrected MBPP endpoint gain was positive in
two of five.

Inference: Controlled post-training improves the harder MATH-500 frontier but
does not improve the code frontier and reduces the unadapted model's GSM8K,
MBPP, and HumanEval performance. These results characterize the frozen
post-training intervention only; they do not identify provider R&D response or
equilibrium behavior.

Observation: A compact 19 MB, 440-file copy was preserved off ICE under
`artifacts/accepted/qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-mbpp-rescore-20260828T233734Z-510fd8f85a9b`.
Its read-only compact manifest verifies and has SHA-256
`067baa4575df0f3b2b4a70249c94a46e48ec6a1c82a38069709c9d604bd707e0`.
The full 196 MB accepted export remains on ICE.

[2026-08-28 19:53 EDT] Compact bundle augmented with terminal receipt

Observation: The off-ICE compact bundle now also contains the terminal
`FINALIZED` receipt. It has 441 files, zero writable paths, and a verified
compact-manifest SHA-256 of
`8cf8109607a4c099fb34f749f5df49e8f44bab23fd8606fb1e9329c499080b2c`.
This entry supersedes only the compact-copy count and compact-manifest digest
reported in the preceding entry; the accepted ICE manifest is unchanged.

[2026-08-28 20:29 EDT] AUTHORITY artifact-publication-completion

repo_publish: allowed on topic branch `glenn/ice-confirmatory-24h` for every
accepted experiment artifact, including trained adapters, retained
checkpoints, raw predictions, receipts, ledgers, analyses, and archive
manifests; default/protected branches remain prohibited.

cluster_read_and_copy: allowed for receipt-bound inventory and off-cluster
transfer from the accepted run and its named robustness/correction descendants;
no new compute or job mutation is authorized or required.

artifact_storage: Git LFS is authorized for large or binary artifacts. Preserve
exact bytes and checksums, exclude credentials, caches, disposable staging,
third-party base-model downloads, and byte-identical duplicate working copies.

decisions_consumed: the owner explicitly requested that all accepted artifacts,
including weights and LoRAs, be committed and pushed to the project repository.

morning_list_only: protected-branch operations, force pushes, history rewrites,
repository administration, secrets, or publication of third-party cached model
weights whose pinned upstream revision is already reproducible.

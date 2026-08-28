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

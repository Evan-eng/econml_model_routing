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

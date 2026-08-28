from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IceSlurmScriptTests(unittest.TestCase):
    def test_math500_rescore_uses_46_way_cpu_array_and_parent_finalizer(self) -> None:
        submitter = (ROOT / "slurm" / "submit_math500_rescore.sh").read_text(encoding="utf-8")
        parent_gate = (ROOT / "slurm" / "math500_rescore_parent_gate.sbatch").read_text(
            encoding="utf-8"
        )
        scorer = (ROOT / "slurm" / "math500_rescore.sbatch").read_text(encoding="utf-8")
        finalizer = (ROOT / "slurm" / "math500_rescore_finalize.sbatch").read_text(encoding="utf-8")

        self.assertIn('PROFILE_INDICES="0-45"', submitter)
        self.assertIn('--array="${PROFILE_INDICES}%46"', submitter)
        self.assertIn("aftercorr:${PARENT_GATE_JOB}", submitter)
        self.assertIn("afterok:${PARENT_FINALIZE_JOB}", submitter)
        self.assertIn('HF_HOME="${ARTIFACT_ROOT}/cache/huggingface"', submitter)
        self.assertIn("HF_HOME=${HF_HOME}", submitter)
        self.assertIn("${VENV_DIR}/bin/python", submitter)
        self.assertIn("robustness-recovery-20260828T201706Z-aae927901acf", submitter)
        self.assertIn('[[ "${PARENT_RUN_ID}" == "${APPROVED_PARENT_RUN_ID}" ]]', submitter)
        self.assertIn('--input-producer-release "${RESCORE_SOURCE_RELEASE}"', submitter)
        self.assertIn("wait-parent", parent_gate)
        self.assertIn("--profile-index", parent_gate)
        self.assertIn("--profile-index", scorer)
        self.assertIn("sacct -X", finalizer)
        self.assertIn('if [[ ! -f "${RAW_ACCOUNTING}" ]]', finalizer)
        self.assertIn('ln "${raw_temp}" "${RAW_ACCOUNTING}"', finalizer)
        self.assertNotIn('> "${EXPERIMENT_DIR}/compute/sacct_raw.psv"', finalizer)
        for text in (parent_gate, scorer, finalizer):
            self.assertNotIn("#SBATCH --gres", text)
            self.assertNotIn("#SBATCH --account", text)
            self.assertNotIn("#SBATCH --partition", text)

    def test_robustness_recovery_uses_adapter_smoke_and_source_aligned_arrays(self) -> None:
        submitter = (ROOT / "slurm" / "submit_robustness_recovery.sh").read_text(encoding="utf-8")
        worker = (ROOT / "slurm" / "robustness_recovery.sbatch").read_text(encoding="utf-8")
        smoke = (ROOT / "slurm" / "robustness_recovery_smoke.sbatch").read_text(encoding="utf-8")
        finalizer = (ROOT / "slurm" / "robustness_recovery_finalize.sbatch").read_text(
            encoding="utf-8"
        )

        self.assertIn('ENDPOINT_INDICES="0,20,40"', submitter)
        self.assertIn('REMAINING_INDICES="1-19,21-39,41-44"', submitter)
        self.assertIn("aftercorr:${ENDPOINT_GATE_JOB}", submitter)
        self.assertIn("aftercorr:${REMAINING_GATE_JOB}", submitter)
        self.assertIn("afterok:${SMOKE_JOB}", submitter)
        self.assertIn("METRIC_PREP_JOB", submitter)
        self.assertIn(
            "afterok:${METRIC_PREP_JOB}:${DOCTOR_JOB}:${ENDPOINT_GATE_JOB}",
            submitter,
        )
        self.assertNotIn("aftercorr:${SOURCE_ENDPOINT_JOB}", submitter)
        self.assertNotIn("aftercorr:${SOURCE_REMAINING_JOB}", submitter)
        self.assertNotIn(':${SOURCE_ANALYSIS_JOB}"', submitter)
        self.assertIn("%42", submitter)
        self.assertIn("SOURCE_ACCOUNT_JOB_IDS", submitter)
        self.assertIn("SUPERSEDED_ACCOUNT_JOB_IDS", submitter)
        self.assertIn("PRIOR_RECOVERY_RUN_ID", submitter)
        self.assertIn("PRIOR_RECOVERY_ACCOUNT_JOB_IDS", submitter)
        source_gate = (ROOT / "slurm" / "robustness_recovery_source_gate.sbatch").read_text(
            encoding="utf-8"
        )
        analysis_gate = (ROOT / "slurm" / "robustness_recovery_analysis_gate.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("wait-source", source_gate)
        self.assertIn("wait-analysis", analysis_gate)
        self.assertIn("--recovery-profile", worker)
        self.assertIn("--recovery-profile 40", smoke)
        self.assertIn("--recovery-smoke", smoke)
        self.assertIn("--limit 1", smoke)
        self.assertIn("sacct -X", finalizer)
        metric_prep = (ROOT / "slurm" / "robustness_recovery_metric_prep.sbatch").read_text(
            encoding="utf-8"
        )
        self.assertIn("prepare-code-eval-metric", metric_prep)
        for text in (worker, smoke, finalizer, source_gate, analysis_gate, metric_prep):
            self.assertNotIn("#SBATCH --account", text)
            self.assertNotIn("#SBATCH --partition", text)

    def test_exploratory_launcher_uses_sparse_aftercorr_array(self) -> None:
        submitter = (ROOT / "slurm" / "submit_exploratory_robustness.sh").read_text(
            encoding="utf-8"
        )
        worker = (ROOT / "slurm" / "exploratory_robustness.sbatch").read_text(encoding="utf-8")
        finalizer = (ROOT / "slurm" / "exploratory_finalize.sbatch").read_text(encoding="utf-8")

        self.assertIn('PROFILE_INDICES="1-19,21-39,41-44"', submitter)
        self.assertIn("aftercorr:${SOURCE_ARRAY_JOB}", submitter)
        self.assertIn('--array="${PROFILE_INDICES}%42"', submitter)
        self.assertIn("afterok:${PROFILES_JOB}:${SOURCE_ROBUSTNESS_JOB}", submitter)
        self.assertIn("EXCLUDE_NODES", submitter)
        self.assertIn("--source-experiment-dir", worker)
        self.assertIn("--profile-index", worker)
        self.assertIn("ensure_cuda_allocation_or_requeue", worker)
        self.assertIn("sacct -X", finalizer)
        for text in (worker, finalizer):
            self.assertNotIn("#SBATCH --account", text)
            self.assertNotIn("#SBATCH --partition", text)

    def test_busy_cuda_allocation_requeues_same_receipted_job(self) -> None:
        script = f"""
source {ROOT / "slurm" / "common.sh"}
srun() {{ printf '%s\\n' 'CUDA-capable device(s) is/are busy or unavailable' >&2; return 1; }}
scontrol() {{ printf 'scontrol:%s\\n' "$*"; }}
export SLURM_JOB_ID=12345
export SLURM_RESTART_COUNT=0
export MAX_ATTEMPTS=3
ensure_cuda_allocation_or_requeue
"""

        completed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("scontrol:requeue 12345", completed.stdout)

    def test_cuda_probe_does_not_requeue_scientific_failures(self) -> None:
        script = f"""
source {ROOT / "slurm" / "common.sh"}
srun() {{ printf '%s\\n' 'CUDA out of memory' >&2; return 7; }}
scontrol() {{ printf 'unexpected-requeue:%s\\n' "$*"; }}
export SLURM_JOB_ID=12345
export SLURM_RESTART_COUNT=0
export MAX_ATTEMPTS=3
ensure_cuda_allocation_or_requeue
"""

        completed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 7)
        self.assertNotIn("unexpected-requeue", completed.stdout)

    def test_cuda_probe_honors_attempt_limit(self) -> None:
        script = f"""
source {ROOT / "slurm" / "common.sh"}
srun() {{ printf '%s\\n' 'CUDA-capable device(s) is/are busy or unavailable' >&2; return 1; }}
scontrol() {{ printf 'unexpected-requeue:%s\\n' "$*"; }}
export SLURM_JOB_ID=12345
export SLURM_RESTART_COUNT=2
export MAX_ATTEMPTS=3
ensure_cuda_allocation_or_requeue
"""

        completed = subprocess.run(
            ["bash", "-c", script],
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertNotIn("unexpected-requeue", completed.stdout)

    def test_gpu_jobs_use_ice_hx00_resources_and_bwrap(self) -> None:
        for name in ("baseline.sbatch", "run_array.sbatch", "robustness.sbatch"):
            text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --qos=coe-ice", text, name)
            self.assertIn("#SBATCH --gres=gpu:1", text, name)
            self.assertIn("#SBATCH --constraint=HX00", text, name)
            self.assertIn("#SBATCH --mem-per-gpu=224G", text, name)
            self.assertNotIn("#SBATCH --account", text, name)
            self.assertNotIn("#SBATCH --partition", text, name)
            self.assertIn("srun", text, name)
            self.assertIn("--code-sandbox bwrap", text, name)

    def test_gpu_jobs_probe_cuda_before_scientific_work(self) -> None:
        for name in ("baseline.sbatch", "run_array.sbatch", "robustness.sbatch"):
            text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
            probe = text.index("ensure_cuda_allocation_or_requeue")
            validation = text.index("validate_stage_release")
            scientific_work = text.rindex("srun python")
            self.assertLess(validation, probe, name)
            self.assertLess(probe, scientific_work, name)

    def test_submitter_requests_full_parallel_grid_and_all_terminal_dependencies(self) -> None:
        text = (ROOT / "slurm" / "submit_ice_confirmatory.sh").read_text(encoding="utf-8")
        self.assertIn('MAX_PARALLEL="${MAX_PARALLEL:-45}"', text)
        self.assertIn('ENDPOINT_INDICES="0,20,40"', text)
        self.assertIn('REMAINING_INDICES="1-19,21-39,41-44"', text)
        self.assertIn("robustness", text)
        self.assertIn("finalize", text)
        for offset in (4, 12, 20, 24, 48):
            self.assertIn(f'--begin="now+{offset}hours"', text)
        self.assertIn("status_snapshot.sbatch", text)
        self.assertIn(
            'GPU_DOCTOR_JOB="$(submit_job --dependency="afterok:${FULL_PREP_JOB}"',
            text,
        )
        self.assertIn(
            'SMOKE_PREP_JOB="$(submit_job --dependency="afterok:${FULL_PREP_JOB}"',
            text,
        )

    def test_release_cleanliness_checks_ignore_v3_sealing_modes_only(self) -> None:
        for name in (
            "common.sh",
            "bootstrap.sbatch",
            "status_snapshot.sbatch",
            "submit_ice_confirmatory.sh",
        ):
            text = (ROOT / "slurm" / name).read_text(encoding="utf-8")
            self.assertIn("git -c core.fileMode=false -C", text, name)
            self.assertIn("status --porcelain --untracked-files=all", text, name)

    def test_prepare_worker_skips_only_successful_python_finalization(self) -> None:
        text = (ROOT / "slurm" / "prepare.sbatch").read_text(encoding="utf-8")

        self.assertIn("prepare \\", text)
        self.assertIn("--skip-python-finalization", text)
        self.assertIn("warm-cache", text)

    def test_prepare_enables_network_only_for_the_pinned_input_stage(self) -> None:
        text = (ROOT / "slurm" / "prepare.sbatch").read_text(encoding="utf-8")

        self.assertIn(
            "unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE",
            text,
        )


if __name__ == "__main__":
    unittest.main()

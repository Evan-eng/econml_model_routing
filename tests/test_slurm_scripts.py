from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class IceSlurmScriptTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

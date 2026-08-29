from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from ice_pipeline import (
    parse_sacct_rows,
    validated_terminal_release,
    write_accounting,
    write_submission_receipt,
)


class ComputeAccountingTests(unittest.TestCase):
    def test_accounting_retry_validates_existing_append_only_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw = root / "sacct_raw.psv"
            output = root / "ledger"
            raw.write_text(
                "123|rescore|COMPLETED|0:0|60|cpu=2|2026-08-28T00:00:00|2026-08-28T00:01:00\n",
                encoding="utf-8",
            )

            write_accounting(raw, output)
            first_csv = (output / "gpu_hours_ledger.csv").read_bytes()
            first_summary = (output / "compute_summary.json").read_bytes()
            write_accounting(raw, output)

            self.assertEqual((output / "gpu_hours_ledger.csv").read_bytes(), first_csv)
            self.assertEqual((output / "compute_summary.json").read_bytes(), first_summary)
            raw.write_text(
                "123|rescore|FAILED|1:0|60|cpu=2|2026-08-28T00:00:00|2026-08-28T00:01:00\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "existing accounting"):
                write_accounting(raw, output)

    def test_extension_submission_receipt_names_source_inputs_and_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract = root / "run_contract.json"
            output = root / "submission.json"
            contract.write_text(
                json.dumps(
                    {
                        "run_id": "extension",
                        "executor_release": "a" * 40,
                        "coordinator_release": "a" * 40,
                        "release_receipt_sha256": "release-receipt-sha",
                    }
                ),
                encoding="utf-8",
            )

            write_submission_receipt(
                contract_path=contract,
                output_path=output,
                jobs=["bootstrap=123", "profiles=456"],
                input_producer_releases=["source-submission-release"],
                pre_side_effect_release_gate="submit_exploratory_robustness.sh",
            )

            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["input_producer_releases"], ["source-submission-release"])
            self.assertEqual(
                receipt["pre_side_effect_release_gate"],
                "submit_exploratory_robustness.sh",
            )

    def test_allocation_rows_exclude_steps_and_compute_gpu_hours(self) -> None:
        raw = "\n".join(
            [
                "123|frontier-arm|COMPLETED|0:0|3600|billing=8,cpu=8,gres/gpu=1,gres/gpu:h100=1|2026-08-27T00:00:00|2026-08-27T01:00:00",
                "123.batch|batch|COMPLETED|0:0|3590|cpu=8,gres/gpu=1|2026-08-27T00:00:01|2026-08-27T01:00:00",
                "124|frontier-analysis|COMPLETED|0:0|120|billing=8,cpu=8|2026-08-27T01:00:00|2026-08-27T01:02:00",
            ]
        )

        rows = parse_sacct_rows(raw)

        self.assertEqual([row["job_id"] for row in rows], ["123", "124"])
        self.assertEqual(rows[0]["allocated_gpus"], 1)
        self.assertEqual(rows[0]["gpu_hours"], 1.0)
        self.assertEqual(rows[1]["gpu_hours"], 0.0)

    def test_terminal_receipt_must_match_fingerprints_and_name_output_release(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "DONE.json"
            path.write_text(
                json.dumps(
                    {
                        "design_sha256": "design",
                        "executor_release": "a" * 40,
                        "output_release": "analysis-release",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                validated_terminal_release(
                    path,
                    {"design_sha256": "design", "executor_release": "a" * 40},
                ),
                "analysis-release",
            )
            with self.assertRaisesRegex(RuntimeError, "receipt mismatch"):
                validated_terminal_release(
                    path,
                    {"design_sha256": "other", "executor_release": "a" * 40},
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import robustness_recovery as recovery


class RobustnessRecoveryTests(unittest.TestCase):
    def test_cli_exposes_init_source_receipt_and_finalize(self) -> None:
        parser = recovery.build_parser()

        self.assertEqual(
            parser.parse_args(
                [
                    "init",
                    "--source-experiment-dir",
                    "/source",
                    "--superseded-experiment-dir",
                    "/failed",
                    "--experiment-dir",
                    "/recovered",
                ]
            ).command,
            "init",
        )
        self.assertEqual(
            parser.parse_args(
                ["source-receipt", "--experiment-dir", "/recovered"]
            ).command,
            "source-receipt",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "wait-source",
                    "--experiment-dir",
                    "/recovered",
                    "--recovery-profile",
                    "40",
                ]
            ).command,
            "wait-source",
        )
        self.assertEqual(
            parser.parse_args(
                ["wait-analysis", "--experiment-dir", "/recovered"]
            ).command,
            "wait-analysis",
        )
        self.assertEqual(
            parser.parse_args(
                [
                    "finalize",
                    "--experiment-dir",
                    "/recovered",
                    "--export-dir",
                    "/export",
                ]
            ).command,
            "finalize",
        )

    def test_roster_contains_baseline_and_all_45_arms_with_frozen_labels(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        roster = recovery.recovery_profile_roster(design)

        self.assertEqual(len(roster), 46)
        self.assertEqual(roster[0]["profile"], "baseline")
        self.assertIsNone(roster[0]["grid_index"])
        self.assertEqual(
            {int(str(item["grid_index"])) for item in roster[1:]},
            set(range(45)),
        )
        confirmatory = [item["profile"] for item in roster if item["classification"] == "confirmatory"]
        exploratory = [item["profile"] for item in roster if item["classification"] == "exploratory"]
        self.assertEqual(
            confirmatory,
            [
                "baseline",
                "z_0p0000_seed_0011",
                "z_0p5000_seed_0011",
                "z_1p0000_seed_0011",
            ],
        )
        self.assertEqual(len(exploratory), 42)

    def test_profile_selector_accepts_baseline_and_every_source_grid_index(self) -> None:
        design = {
            "mixtures": [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
            "seeds": [11, 29, 47, 73, 101],
        }

        self.assertEqual(recovery.recovery_profile_selector(design, "baseline"), "baseline")
        self.assertEqual(recovery.recovery_profile_selector(design, "0"), "0.0:11")
        self.assertEqual(recovery.recovery_profile_selector(design, "20"), "0.5:11")
        self.assertEqual(recovery.recovery_profile_selector(design, "44"), "1.0:101")
        with self.assertRaisesRegex(IndexError, "outside"):
            recovery.recovery_profile_selector(design, "45")


if __name__ == "__main__":
    unittest.main()

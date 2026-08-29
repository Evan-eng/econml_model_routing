from __future__ import annotations

import unittest

import lm_eval_transformers_compat as compat


class FakePreTrainedModel:
    @classmethod
    def from_pretrained(cls, model: str, **kwargs: object) -> dict[str, object]:
        return {"model": model, **kwargs}


class LmEvalTransformersCompatTests(unittest.TestCase):
    def test_patch_translates_dtype_at_transformers_call_boundary(self) -> None:
        compat.install_dtype_compatibility(
            pretrained_model=FakePreTrainedModel,
            lm_eval_version="0.4.10",
            transformers_version="4.53.2",
        )

        result = FakePreTrainedModel.from_pretrained("Qwen/Qwen3-8B", dtype="bfloat16")

        self.assertEqual(result["torch_dtype"], "bfloat16")
        self.assertNotIn("dtype", result)

    def test_patch_fails_closed_for_unreviewed_dependency_versions(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "unsupported compatibility tuple"):
            compat.install_dtype_compatibility(
                pretrained_model=FakePreTrainedModel,
                lm_eval_version="0.4.11",
                transformers_version="4.53.2",
            )


if __name__ == "__main__":
    unittest.main()

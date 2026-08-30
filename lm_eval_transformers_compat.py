#!/usr/bin/env python3
"""Run lm-eval with the reviewed dtype bridge for the frozen cluster stack."""

from __future__ import annotations

from typing import Any


SUPPORTED_COMPATIBILITY = ("0.4.10", "4.53.2")


def install_dtype_compatibility(
    *,
    pretrained_model: type[Any],
    lm_eval_version: str,
    transformers_version: str,
) -> None:
    """Translate lm-eval's dtype keyword for the pinned Transformers release."""
    observed = (lm_eval_version, transformers_version)
    if observed != SUPPORTED_COMPATIBILITY:
        raise RuntimeError(
            "unsupported compatibility tuple: "
            f"lm-eval={lm_eval_version}, transformers={transformers_version}"
        )
    original = pretrained_model.from_pretrained.__func__

    @classmethod
    def compatible_from_pretrained(
        cls: type[Any], model: str, *model_args: object, **kwargs: object
    ) -> Any:
        dtype = kwargs.pop("dtype", None)
        if dtype is not None:
            if "torch_dtype" in kwargs:
                raise RuntimeError("both dtype and torch_dtype were supplied")
            kwargs["torch_dtype"] = dtype
        return original(cls, model, *model_args, **kwargs)

    pretrained_model.from_pretrained = compatible_from_pretrained


def main() -> None:
    import lm_eval
    import transformers
    from lm_eval.__main__ import cli_evaluate
    from transformers import PreTrainedModel

    install_dtype_compatibility(
        pretrained_model=PreTrainedModel,
        lm_eval_version=lm_eval.__version__,
        transformers_version=transformers.__version__,
    )
    cli_evaluate()


if __name__ == "__main__":
    main()

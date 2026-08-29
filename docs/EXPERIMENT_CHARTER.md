# Confirmatory experiment charter

## Scientific object

Estimate the controlled Qwen3-8B post-training capability frontier induced by a
fixed 1,048,576-token adaptation budget and randomized math/code training share.
The confirmatory grid has nine mixture shares and five paired seeds, for 45
independent one-GPU arms. An untouched baseline and standardized MATH-500 and
HumanEval profiles provide reference and robustness measurements.

This intervention does not identify provider R&D response, investment response,
or equilibrium behavior. That limitation remains binding even if the measured
frontier matches the paper's maintained assumptions.

## Frozen design and decisions

- Model: `Qwen/Qwen3-8B` at
  `b968826d9c46dd6066d109eabc6255188de91218`.
- Mixtures: `0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1`.
- Paired seeds: `11, 29, 47, 71, 101`.
- Evaluation: full GSM8K test and sanitized MBPP test sets, with all training,
  evaluation, MATH-500, and HumanEval repository commits frozen in the initial
  run contract before remote download.
- Uncertainty: 2,000 paired two-way bootstrap replications.
- Robustness profiles: baseline and seed-11 `z=0`, `z=0.5`, and `z=1` on
  MATH-500 and HumanEval Instruct.
- Scale gate: operational only. Weak, null, reversed, or assumption-violating
  effects are results and do not reduce or stop the grid.

## Blocking gates

The chain may proceed only when exact release and dependency identities validate,
the prepared model/data provenance is complete, the same-model smoke reconciles,
and bubblewrap proves network, filesystem, and allocated-GPU containment on an
ICE compute node. Provenance, sandbox, corrupt-artifact, scientific-exception,
and OOM failures are not blindly retried.

## Acceptance

Acceptance requires one matching baseline receipt, 45 unique matching arm
receipts, a matching analysis receipt, four matching robustness receipts,
allocation-only compute accounting including overhead, and a checksummed
read-only export. Submission, running state, or partial files are not completion.

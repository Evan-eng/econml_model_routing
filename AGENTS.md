# Experiment operating rules

This repository runs an instructional PACE ICE experiment. Pair the PACE ICE
site rules with portable Slurm guidance before changing or submitting jobs.

- Persistent datasets, Hugging Face caches, checkpoints, predictions, receipts,
  and exports belong under `/storage/ice-shared/cs7634/staff/econml`.
- Node-local `$TMPDIR` is disposable staging only. Home and `~/scratch` hold no
  experiment data, model cache, or run outputs.
- ICE scripts use no account and no partition. The authorized run uses
  `--qos=coe-ice`, one `HX00` GPU per independent arm, and at most 15:30 per
  GPU attempt.
- Every remote stage validates the sealed `pace-code-release-v3` receipt, exact
  executor/coordinator SHA, run contract, design digest, and dependency lock
  before a side effect. Dual-release execution is prohibited.
- Generated MBPP and HumanEval code runs only with both the explicit opt-in and
  the verified bubblewrap boundary. A failed sandbox doctor blocks GPU work.
- Do not cancel or alter jobs outside the job IDs in this run's submission
  receipt. Preserve failed, retry, debug, and cancelled allocation rows.
- `LOGBOOK.md` is append-only. Record corrections as new entries.

Definition of done for code changes:

```bash
uv run ruff check .
uv run pyright
uv run pytest -q
bash -n slurm/*.sbatch slurm/*.sh
```

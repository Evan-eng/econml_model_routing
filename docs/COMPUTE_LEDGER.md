# Compute ledger

GPU accounting is derived from allocation-only `sacct -X` rows. GPU-hours equal
`ElapsedRaw * allocated_gpus / 3600`; `.batch` and `.extern` steps are excluded.
Failed, retried, debug, and cancelled allocation rows remain in the detailed
ledger rather than being charged only to successful scientific outputs.

The active confirmatory run has not yet produced terminal accounting. Its remote
finalizer writes `compute/ledger/gpu_hours_ledger.csv` and
`compute/ledger/compute_summary.json` into the run tree and read-only export.
After reconciliation, append those allocation rows to
`docs/compute/gpu_hours_ledger.csv` and add a dated rollup below.

## Rollups

- Pending: Qwen3-8B confirmatory run, exact run ID and terminal GPU-hours not yet
  available.

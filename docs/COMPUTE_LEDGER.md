# Compute ledger

GPU accounting is derived from allocation-only `sacct -X` rows. GPU-hours equal
`ElapsedRaw * allocated_gpus / 3600`; `.batch` and `.extern` steps are excluded.
Failed, retried, debug, and cancelled allocation rows remain in the detailed
ledger rather than being charged only to successful scientific outputs.

Terminal accounting for the accepted confirmatory run and its correction and
robustness stages is complete. The canonical records are stored in
`artifacts/accepted/accepted-full-export-qwen3-8b-confirmatory-20260828.zip`,
including the source, robustness-recovery, MATH-500 correction, and MBPP rescore
ledgers and summaries. See `artifacts/accepted/README.md` for the receipt-bound
archive digest and verification commands.

`docs/compute/gpu_hours_ledger.csv` is retained as a schema-only development
ledger. It is not the accepted accounting record and must not be used to infer
the paper's aggregate compute. In particular, partial totals recorded during
execution do not replace the terminal ledgers in the accepted export.

## Accepted resource contract

- Cluster: Georgia Tech PACE ICE.
- GPU allocation: one `HX00` GPU per independent GPU task; the primary run used
  H100 allocations and robustness recovery used H100 or H200 allocations.
- Per-task request: eight CPU cores, 224 GiB host memory, 50 GiB node-local
  scratch, and a 15:30 attempt limit.
- Aggregate compute: use the terminal `compute_summary.json` files and detailed
  allocation-only ledgers inside the accepted full export. They retain failed,
  retried, diagnostic, and cancelled allocation rows.

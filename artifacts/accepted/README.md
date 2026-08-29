# Accepted experiment artifacts

This directory publishes the terminal, receipt-bound export for the Qwen3-8B
confirmatory experiment and its robustness extension. The ZIP archives are
stored with Git LFS; install Git LFS before cloning if you need their contents.

The complete training run is published separately under
`artifacts/ice-run/qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c/`.
That LFS-backed tree contains all 2,101 source files and 57,247,182,883 logical
bytes from the accepted run: all 45 final LoRA adapters, both retained
checkpoints for every arm, optimizer and scheduler state, tokenizer and
configuration files, raw predictions, metrics, receipts, status history, and
analysis outputs. Its `SOURCE_ARTIFACT_MANIFEST.json` has SHA-256
`71e80fb84069b5d3940e7553f761404c2491273506b8e09559751c1b2c1d84ac`.

## Canonical archives

- `accepted-full-export-qwen3-8b-confirmatory-20260828.zip` is the complete
  accepted export. It contains all 1,232 files from the terminal MBPP-corrected
  result tree, including the corrected MATH-500 overlay, raw predictions,
  receipts, manifests, analyses, ledgers, tables, and figures. Its SHA-256 is
  `c272f14c672050db2ffa3b5caee2146cfe288064634fec366df04636be7f33bb`.
- `accepted-paper-packet-qwen3-8b-confirmatory-20260828.zip` is the 23-file
  writing subset mirrored into the Overleaf project. Its SHA-256 is
  `330941d128e9e75ed03f78ec5863c1ddd1c13fcd2ef5affd8d1174f48fb58b33`.
- `ARCHIVES.sha256` records both archive digests. The complete export's internal
  `MANIFEST.json` has SHA-256
  `f31232b36ef4f70000e0614a966138a9b095722854bcac454022fa7ca8519e26`.
- `GITHUB_PUBLICATION_RECEIPT.json` binds the exact artifact commit, source
  manifest, LFS object count and bytes, remote-presence audit, and fresh-clone
  checksum verification.

The unpacked local export, its duplicate working copy, and the compact staging
bundle remain ignored because their distinct contents are already present in
the canonical full-export ZIP. This avoids storing the same accepted bytes
several times while keeping every artifact available from the Git commit.

The public Qwen3-8B base model is not duplicated in this repository. Its exact
upstream model and revision are pinned in the run contract and provenance
receipts; the repository stores every experiment-produced weight and state
needed to reproduce or resume the accepted run.

## Verification

From the repository root:

```bash
git lfs pull
shasum -a 256 -c artifacts/accepted/ARCHIVES.sha256
unzip -tq artifacts/accepted/accepted-full-export-qwen3-8b-confirmatory-20260828.zip
unzip -tq artifacts/accepted/accepted-paper-packet-qwen3-8b-confirmatory-20260828.zip
(cd artifacts/ice-run && jq -r '.entries[] | "\(.sha256)  \(.path)"' \
  SOURCE_ARTIFACT_MANIFEST.json | shasum -a 256 -c -)
```

Read the paper packet's `README.md`, `COAUTHOR_GUIDE.md`, and
`raw/METRIC_STATUS.json` before interpreting results. Corrected MATH-500 and
MBPP metrics supersede the legacy string-scored outputs retained for audit.

# Receipt-bound robustness recovery

## Purpose

This recovery evaluates MATH-500 and HumanEval for the baseline and all 45
trained Qwen3-8B adapters without retraining. It replaces only the failed
robustness stages. The accepted source run remains the immutable producer of
the baseline, adapters, primary metrics, predictions, and confirmatory
analysis.

The baseline and seed-11 endpoint adapters at mixture shares 0, 0.5, and 1
remain the four confirmatory robustness profiles. The other 42 adapter
profiles are an exploratory extension. This classification is frozen in the
recovery plan and repeated in the final profile manifest.

## Failure and correction

The superseded all-profile extension reached its compute-node sandbox gate but
adapter evaluation failed when PEFT imported bitsandbytes and Triton attempted
to execute `/sbin/ldconfig`. The whole-process bubblewrap boundary had not
exposed that executable. Multiple independent array tasks failed with the same
application exception, so the remaining extension tasks and the known-bad
four-profile robustness stage were cancelled and retained for accounting.

The corrected sandbox exposes only `/sbin/ldconfig` and `/etc/ld.so.cache` as
read-only inputs for GPU evaluation. Network access remains disabled, system
directories remain read-only or absent, and each generated-code evaluation
still runs in a disposable node-local working directory with the existing
resource limits.

The first corrected smoke then exposed a second missing system input: the
`/usr/bin/ld` symlink resolves through `/etc/alternatives/ld`. The next sealed
recovery therefore exposes that single linker alias read-only as well. It does
not expose the rest of `/etc/alternatives` or make any system path writable.

## Execution contract

The recovery has a new exact-SHA run contract and output tree. Its source
receipt binds the accepted confirmatory preparation, submission, and GPU
sandbox receipts plus the superseded extension source and submission receipts.
Before the 46-profile fan-out, a new compute-node GPU sandbox doctor must pass
and one adapter must complete a bounded MATH-500 and HumanEval smoke in a
separate output namespace.

After that smoke, the baseline and all 45 source-aligned profiles may evaluate
concurrently. Each terminal receipt consumes the recovery source gate, new GPU
doctor, adapter smoke, and exact source baseline or adapter DONE release. The
finalizer requires the source analysis receipt and all 46 recovery receipts,
preserves allocation-only accounting for the source, failed extension, and
recovery jobs, and emits a checksummed read-only export.

## Interpretation boundary

The full mixture-by-seed robustness surface can support benchmark-specific
frontier plots, mixture-by-benchmark interactions, seed variation, and paired
task-bootstrap intervals. Only the four preregistered profiles are
confirmatory. The additional 42 profiles remain exploratory, including any
profile-wise tests. Null, adverse, or reversed findings remain in the record.

This controlled post-training intervention does not identify provider R&D
response, investment response, or equilibrium behavior.

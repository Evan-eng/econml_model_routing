# All-profile exploratory robustness extension

This extension evaluates MATH-500 and HumanEval for the 42 trained adapters
outside the prespecified baseline and seed-11 endpoint set. Joined with the
four confirmatory robustness profiles, it covers all 46 model profiles.

The original four-profile result remains the confirmatory robustness test. The
additional 42 profiles are exploratory: they support benchmark-specific
frontier plots, paired profile contrasts, mixture-by-benchmark interactions,
training-seed sensitivity, and task-bootstrap intervals, but they do not add
new confirmatory hypotheses or profile-wise significance claims.

The extension has its own exact-SHA run contract, output tree, submission
receipt, allocation-only compute ledger, final receipt, and read-only export.
Each sparse array task uses `aftercorr` to wait for the matching confirmatory
training task, then validates the source run contract, preparation receipt,
GPU sandbox receipt, exact adapter DONE receipt, and extension source receipt
before loading a model or writing an output. The original analysis, robustness
jobs, finalizer, and export are unchanged and do not depend on this extension.

HumanEval remains fail-closed behind the networkless bubblewrap boundary. All
jobs run offline, use one HX00 GPU, retain the three-attempt infrastructure
ceiling, and preserve failed or retried allocation rows in the extension
ledger.

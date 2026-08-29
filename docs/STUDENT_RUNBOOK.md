# Student experiment runbook

The instructor submits and monitors the ICE jobs. The student does not need
cluster credentials and should work from the read-only export bundle.

Before launch, review `docs/EXPERIMENT_CHARTER.md`, `design.json`, the hypotheses,
and the claim boundary. Flag scientific concerns before the design is sealed;
runtime observations do not authorize changing the confirmatory grid.

During the run, use status snapshots as operational evidence only. A queued or
running job is not a result. A `COMPLETED` Slurm state is also insufficient until
its DONE receipt, release tuple, and artifact checksums reconcile.

After completion:

1. Start with `MANIFEST.json`, `design.json`, `provenance.json`,
   `status_history/`, and the compute summary so the evidence base is clear.
2. Interpret the complete frontier, including null, adverse, and
   assumption-violating outcomes. Do not select only favorable mixture shares.
3. Report uncertainty intervals together with bootstrap identification rates and
   failure reasons.
4. Keep the provider-response limitation explicit in every summary and paper
   revision.
5. Complete `INTERPRETATION_WORKSHEET.md`, then use `REPRODUCE.md` and the exact
   release SHA when asking another reviewer to check the result.

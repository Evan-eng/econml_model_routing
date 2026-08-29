# GPU-ready capability-frontier experiment

This artifact runs the empirical experiment proposed in *Model Routing as
Innovation Policy* on a workstation or Slurm cluster. It uses no Modal
services. The main design trains independent QLoRA adapters at different
math/code data allocations, measures each adapter's two-domain capabilities,
then compares the provider equilibrium with the quality-maximizing pair under
the same softmax router.

## PACE ICE confirmatory run

The frozen ICE workflow runs the complete 45-arm confirmatory design, one
untouched baseline, and four robustness profiles. It deploys an exact Git SHA,
builds the locked environment on a CPU node, prepares same-model smoke and full
designs concurrently, proves bubblewrap containment on an allocated GPU, and
then fans the full grid out to 46 GPUs at most.

```bash
export RELEASE_SHA=<full-pushed-40-character-sha>
export MAX_PARALLEL=45
bash slurm/submit_ice_confirmatory.sh
```

The submitter uses `coe-ice`, requests no account or partition, keeps persistent
artifacts under `/storage/ice-shared/cs7634/staff/econml`, and writes a submission
receipt before the chain runs. Run it only from the exact remote release named
by `pace-code-release`; it refuses a dirty or mismatched deployment, reduced
grid, or duplicate run directory.

## Default overnight experiment

The default is sized for an approximately 12-hour overnight run on four strong
GPUs. It uses `Qwen/Qwen3-8B` with five fixed-budget math shares

`z = 0, 0.25, 0.5, 0.75, 1`

and five paired training seeds. This is 25 independent one-GPU adapter arms,
plus one untouched baseline evaluation. Every arm
uses the same base weights, adapter architecture, optimizer, number of token
slots, evaluation questions, and decoding rule. Only the math/code allocation
changes.

- Math training: `open-r1/OpenR1-Math-220k`
- Code training: `bigcode/self-oss-instruct-sc2-exec-filter-50k`
- Math evaluation: 256 locked GSM8K test questions
- Code evaluation: the full 257-item sanitized MBPP test split
- Adaptation: 4-bit QLoRA, rank 16, 262,144 token slots per arm
- Primary environment: equally likely math and code queries (`mu = 0.5`)
- Uncertainty: 1,000 paired bootstrap replications over training seeds and
  held-out tasks, reported as exploratory because five seed clusters cannot
  guarantee nominal percentile-bootstrap coverage

The primary outcomes are the estimated capability functions
`q_math(z)` and `q_code(z)`, the empirical specialization threshold `beta_E`,
the pure- and mixed-Nash portfolios on the observed action grid, the
planner-optimal pair, and the routed-quality gap between them. The mixed
equilibrium is solved as a constant-sum game, so the comparison remains defined
even when sampling noise removes every pure equilibrium. The design is
falsifiable: if the
frontiers are not increasing and locally concave, or if planner differentiation
does not improve routed quality, the paper's maintained technology conditions
are not supported in this setting.

The generalized threshold is evaluated at the fitted interior common portfolio
`z_star`, the stationary point of average quality, rather than being assumed to
occur at `z = 0.5`. Bootstrap draws with a boundary stationary point or failed
monotonicity/curvature conditions are marked unidentified, and the identification
rate is reported alongside the conditional interval.

The overnight grid is the recommended workshop run. The small-model smoke test
below verifies plumbing only; it does not diagnose Qwen3-8B treatment power.
Before launching the full grid, run the actual-model endpoint preflight:

```bash
export EXPERIMENT_DIR=$PWD/experiments/qwen3_8b_overnight
export GPUS=0,1,2,3
export ALLOW_CODE_EXECUTION=1
bash run_endpoint_preflight.sh
```

This runs the untouched baseline and the `z = 0, 0.5, 1` arms for seed 11,
writes a paired item-level endpoint report, and leaves those completed jobs in
the registered overnight directory for reuse by the full launcher. The report
is conditional on one training seed and must not be presented as total
uncertainty. With many GPUs,
parallelize across arms rather than within an arm; this preserves independence
and usually gives the best wall-clock scaling. An 8B QLoRA arm is intended for
one modern NVIDIA GPU; 48--80 GB is comfortable, while smaller cards may require
a lower evaluation batch size.

```bash
python capability_mixture_experiment.py init \
  --experiment-dir experiments/qwen3_8b_overnight
```

Because `overnight` and `Qwen/Qwen3-8B` are the CLI defaults, no preset or
model flags are needed. The one-command local workflow is:

```bash
export EXPERIMENT_DIR=$PWD/experiments/qwen3_8b_overnight
export GPUS=0,1,2,3
export ALLOW_CODE_EXECUTION=1
bash run_overnight_local.sh
```

After a successful preflight, `run_overnight_local.sh` skips its completed jobs. On
four GPUs the complete design requires seven scheduling waves at most: the
untouched baseline and 25 adapters. To finish within 12 hours, average job time
must be roughly 1.7 hours or less. Runtime depends on GPU type, storage, and
generation speed; use the real-model preflight timings rather than the
small-model smoke test to assess the overnight target.

## Smaller six-hour option

With roughly four GPUs and a six-hour wall-clock limit, use the `six-hour`
preset. It schedules 15 adapters plus one unadapted baseline: five mixture
shares, three paired seeds, 262,144 token slots per arm, 256 GSM8K questions,
and 128 MBPP questions. The baseline runs concurrently with the adapter grid.
This is a credible workshop experiment, but it is a powered pilot rather than
the 45-arm confirmatory design.

```bash
export EXPERIMENT_DIR=$PWD/experiments/qwen3_8b_six_hour
export GPUS=0,1,2,3
export ALLOW_CODE_EXECUTION=1
bash run_six_hour_local.sh
```

The local launcher schedules the unadapted baseline automatically. The sealed
ICE chain submits the baseline beside the GPU arrays and waits for every receipt
before analysis. Runtime depends heavily on GPU, storage, and generation speed;
use the same-model smoke receipt rather than assuming a target from another
cluster.

## Installation

The PACE release is locked for Python 3.10 and CUDA-capable Linux nodes.

```bash
uv sync --frozen --extra robustness
uv run python capability_mixture_experiment.py self-test
```

Set a shared Hugging Face cache on clusters:

```bash
export HF_HOME=/shared/$USER/huggingface
```

Qwen3 thinking mode is disabled by the code so all arms use the same visible
generation budget.

## Stage 1: smoke test

This verifies downloads, training, evaluation, and analysis with three small
arms. It is not a treatment-power diagnostic. Generated MBPP code must only be
evaluated in a disposable, network-disabled container or VM.

```bash
python capability_mixture_experiment.py init \
  --experiment-dir experiments/smoke \
  --preset smoke \
  --model Qwen/Qwen3-0.6B
python capability_mixture_experiment.py prepare --experiment-dir experiments/smoke
python capability_mixture_experiment.py warm-cache --experiment-dir experiments/smoke
python capability_mixture_experiment.py evaluate-baseline \
  --experiment-dir experiments/smoke \
  --allow-generated-code-execution \
  --code-sandbox bwrap
python capability_mixture_experiment.py run-grid \
  --experiment-dir experiments/smoke \
  --allow-generated-code-execution \
  --code-sandbox bwrap
python capability_mixture_experiment.py analyze --experiment-dir experiments/smoke
```

## Stage 2: optional pilot

The preregistered pilot uses five mixture shares and three seeds (15 arms):

```bash
python capability_mixture_experiment.py init \
  --experiment-dir experiments/qwen3_8b_pilot \
  --preset pilot \
  --model Qwen/Qwen3-8B
python capability_mixture_experiment.py prepare \
  --experiment-dir experiments/qwen3_8b_pilot
python capability_mixture_experiment.py warm-cache \
  --experiment-dir experiments/qwen3_8b_pilot
```

On a multi-GPU workstation, fill GPUs 0--7 with the pending arms:

```bash
python launch_local_gpu_pool.py \
  --experiment-dir experiments/qwen3_8b_pilot \
  --gpus 0,1,2,3,4,5,6,7 \
  --allow-generated-code-execution \
  --code-sandbox bwrap
python capability_mixture_experiment.py analyze \
  --experiment-dir experiments/qwen3_8b_pilot
```

The launcher is resumable and skips completed arms unless `--force` is used.
It also skips an already completed baseline; pass `--skip-baseline` only when
the baseline is being run separately.

## Stage 3: optional confirmatory Slurm run

Initialize the 45-arm design before submitting the pipeline:

```bash
python capability_mixture_experiment.py init \
  --experiment-dir /shared/$USER/routing_qwen3_8b_confirmatory \
  --preset confirmatory \
  --model Qwen/Qwen3-8B
```

For `--eval-math-n` and `--eval-code-n`, a value of `0` means the entire locked
test split. The confirmatory preset therefore uses all 1,319 GSM8K and all 257
sanitized MBPP test items.

Deploy the pushed exact SHA and submit the complete dependency chain:

```bash
~/dotfiles/scripts/pace-code-release \
  --host pace-ice-agent \
  --repo-url https://github.com/Evan-eng/econml_model_routing \
  --revision "$RELEASE_SHA" \
  --release-root /storage/ice-shared/cs7634/staff/econml/releases/econml_model_routing \
  --tracked-identity uv.lock \
  --tracked-identity pyproject.toml \
  --tracked-identity slurm
ssh pace-ice-agent \
  "RELEASE_SHA=$RELEASE_SHA MAX_PARALLEL=45 bash /storage/ice-shared/cs7634/staff/econml/releases/econml_model_routing/$RELEASE_SHA/slurm/submit_ice_confirmatory.sh"
```

The sealed ICE submitter is the only supported Slurm path for the confirmatory
run; local launchers remain available for workstation development.

`ALLOW_CODE_EXECUTION=1` is only the human opt-in. The evaluator also requires
`--code-sandbox=bwrap`; it fails closed unless bubblewrap provides the
network-disabled, read-only host boundary, and retains static restrictions,
isolated Python mode, timeouts, and resource limits inside that boundary.

Check progress or safely resubmit failed array indices:

```bash
python capability_mixture_experiment.py grid-status \
  --experiment-dir "$EXPERIMENT_DIR"
```

Completed arms are fingerprinted against `design.json` and `provenance.json`.
They are skipped on resubmission; mismatched results are rejected instead of
being silently combined.

## Outputs

Each arm writes its adapter, allocation record, raw predictions, metrics, and a
completion marker under `EXPERIMENT_DIR/arms/`. Final analysis is written to
`EXPERIMENT_DIR/analysis/`:

- `mean_capability_frontier.csv`
- `baseline_metrics.csv`
- `routing_game_sweep.csv`
- `bootstrap_thresholds.csv`
- `seed_sensitivity.csv`
- `analysis_summary.json`
- `capability_routing_results.png`

`analysis_summary.json` reports the balanced and best specialized adapters'
changes relative to the untouched checkpoint. This distinguishes capability
reallocation from adaptation that merely degrades both domains. It also reports
the fitted `z_star`, bootstrap identification reasons and rate, and ranges from
single-seed and leave-one-seed-out sensitivity checks. The paired two-way
bootstrap reflects both task and seed resampling, but its nominal coverage is
not guaranteed with only five training seeds; item-only uncertainty is not
substituted for total uncertainty. Its generalized `beta_E` interval is
conditional on identification because unidentified bootstrap draws are omitted;
always report the identification rate and failure-reason counts beside it.

Each arm's `allocation.json`, and the preparation provenance, record domain-wise
training-example truncation rates, retained-token fractions, and original token
length quantiles. The fixed allocation remains internally comparable when
truncation differs by domain, but the estimated frontier then partly reflects
that preprocessing choice and should not be interpreted as an absolute
leaderboard capability. Inspect these diagnostics before changing `max_length`.

## Optional harder-benchmark check

The default overnight run deliberately prioritizes training-seed replication
over expensive additional generation. The locked `robustness` extra supplies
the standardized layer for the baseline plus endpoint and balanced adapters:

```bash
uv run python run_benchmark_robustness.py \
  --experiment-dir experiments/qwen3_8b_overnight \
  --allow-generated-code-execution \
  --code-sandbox bwrap
```

This uses lm-eval's `hendrycks_math500` and `humaneval_instruct` tasks and
writes results under `robustness_benchmarks/`. It is evaluation-only: no new
adapters are trained. Use `--limit 20 --dry-run` to inspect the commands, but
never report a limited debug run as a benchmark result. To parallelize the four
default profiles, launch four processes on separate GPUs with `--profiles`
set respectively to `baseline`, `0:11`, `0.5:11`, and `1:11`.

## High-value robustness runs

After the confirmatory run, the most informative additions are:

1. Reinitialize a new experiment with half and double the token budget. This
   tests whether the frontier and routing wedge depend on adaptation scale.
2. Reinitialize with `--completion-only-loss`. This tests the full-sequence
   loss normalization used in the primary design.
3. Repeat the same preregistered grid on a second open model family. Do not pool
   the model families; report the frontiers and thresholds separately.
4. Calibrate router responsiveness with randomized assignments near quality
   ties and pass the resulting audit file to `analyze --router-audit FILE`.

The experiment identifies a controlled post-training capability frontier. It
does not by itself show that frontier model providers change R&D in response to
routed traffic; that remains a separate provider-response estimand.

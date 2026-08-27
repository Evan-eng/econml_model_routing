#!/usr/bin/env bash
#SBATCH --job-name=frontier-analysis
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to the unzipped GPU artifact directory}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

python "${ARTIFACT_DIR}/capability_mixture_experiment.py" grid-status \
  --experiment-dir "${EXPERIMENT_DIR}"
python "${ARTIFACT_DIR}/capability_mixture_experiment.py" analyze \
  --experiment-dir "${EXPERIMENT_DIR}"

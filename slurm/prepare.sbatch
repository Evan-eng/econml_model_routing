#!/usr/bin/env bash
#SBATCH --job-name=frontier-prep
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to the unzipped GPU artifact directory}"

if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

export HF_HOME="${HF_HOME:-${EXPERIMENT_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

python "${ARTIFACT_DIR}/capability_mixture_experiment.py" prepare \
  --experiment-dir "${EXPERIMENT_DIR}"
python "${ARTIFACT_DIR}/capability_mixture_experiment.py" warm-cache \
  --experiment-dir "${EXPERIMENT_DIR}"

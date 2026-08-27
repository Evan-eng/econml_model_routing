#!/usr/bin/env bash
#SBATCH --job-name=frontier-arm
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${ARTIFACT_DIR:?Set ARTIFACT_DIR to the unzipped GPU artifact directory}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

if [[ "${ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
  echo "Set ALLOW_CODE_EXECUTION=1 only inside a disposable, network-isolated environment." >&2
  exit 2
fi
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  source "${VENV_ACTIVATE}"
fi

export HF_HOME="${HF_HOME:-${EXPERIMENT_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python "${ARTIFACT_DIR}/capability_mixture_experiment.py" run-array-arm \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --allow-generated-code-execution

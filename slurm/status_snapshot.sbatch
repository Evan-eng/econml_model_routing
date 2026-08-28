#!/usr/bin/env bash
#SBATCH --job-name=routing-status
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00

set -euo pipefail

: "${RELEASE_DIR:?Set RELEASE_DIR}"
: "${RELEASE_SHA:?Set RELEASE_SHA}"
: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
: "${RUN_CONTRACT_PATH:?Set RUN_CONTRACT_PATH}"
: "${STATUS_LABEL:?Set STATUS_LABEL}"
module purge
module load anaconda3/2023.03
export EXECUTOR_RELEASE="${RELEASE_SHA}"
export PYTHONPATH="${RELEASE_DIR}"
[[ "$(git -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]
srun python "${RELEASE_DIR}/ice_pipeline.py" status-snapshot \
  --experiment-dir "${EXPERIMENT_DIR}" --label "${STATUS_LABEL}"

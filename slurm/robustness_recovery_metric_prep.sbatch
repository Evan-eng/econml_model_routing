#!/usr/bin/env bash
#SBATCH --job-name=routing-code-eval-metric
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
srun python "${RELEASE_DIR}/robustness_recovery.py" prepare-code-eval-metric \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --hf-home "${HF_HOME:?Set HF_HOME}"

#!/usr/bin/env bash
#SBATCH --job-name=routing-mbpp-rescore-source-gate
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/mbpp_rescore.py" wait-source --experiment-dir "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}" --profile-index "${SLURM_ARRAY_TASK_ID:?Set SLURM_ARRAY_TASK_ID}"

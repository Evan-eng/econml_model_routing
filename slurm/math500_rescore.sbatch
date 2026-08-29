#!/usr/bin/env bash
#SBATCH --job-name=routing-math500-rescore
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/math500_rescore.py" rescore \
  --experiment-dir "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}" \
  --profile-index "${SLURM_ARRAY_TASK_ID:?Submit as an array}"

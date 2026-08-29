#!/usr/bin/env bash
#SBATCH --job-name=routing-math500-rescore-parent-gate
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=15:30:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/math500_rescore.py" wait-parent \
  --experiment-dir "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}" \
  --profile-index "${SLURM_ARRAY_TASK_ID:?Submit as an array}" \
  --timeout-seconds 54000 --poll-seconds 30

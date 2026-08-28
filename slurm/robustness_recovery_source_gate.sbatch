#!/usr/bin/env bash
#SBATCH --job-name=routing-robustness-source-gate
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=15:30:00

set -euo pipefail
profile="${RECOVERY_PROFILE:-${SLURM_ARRAY_TASK_ID:?Set RECOVERY_PROFILE or submit an array}}"
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/robustness_recovery.py" wait-source \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --recovery-profile "${profile}"

#!/usr/bin/env bash
#SBATCH --job-name=routing-smoke-gate
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" grid-status \
  --experiment-dir "${EXPERIMENT_DIR}" --json
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" endpoint-report \
  --experiment-dir "${EXPERIMENT_DIR}" --seed 11
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" accept-smoke \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --gpu-doctor-receipt "${GPU_DOCTOR_RECEIPT:?Set GPU_DOCTOR_RECEIPT}"

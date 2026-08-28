#!/usr/bin/env bash
#SBATCH --job-name=routing-endpoints
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
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" endpoint-report \
  --experiment-dir "${EXPERIMENT_DIR}" --seed 11

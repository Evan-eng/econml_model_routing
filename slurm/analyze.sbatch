#!/usr/bin/env bash
#SBATCH --job-name=frontier-analysis
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" grid-status \
  --experiment-dir "${EXPERIMENT_DIR}"
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" analyze \
  --experiment-dir "${EXPERIMENT_DIR}"

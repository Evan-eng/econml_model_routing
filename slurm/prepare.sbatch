#!/usr/bin/env bash
#SBATCH --job-name=frontier-prep
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${RELEASE_DIR:?Set RELEASE_DIR}"
source "${RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
enable_offline_model_loading

srun python "${RELEASE_DIR}/capability_mixture_experiment.py" prepare \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --skip-python-finalization
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" warm-cache \
  --experiment-dir "${EXPERIMENT_DIR}"

#!/usr/bin/env bash
#SBATCH --job-name=routing-gpu-doctor
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --tmp=50G
#SBATCH --time=00:20:00

set -euo pipefail

source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
enable_offline_model_loading
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" sandbox-doctor \
  --cuda --experiment-dir "${EXPERIMENT_DIR}" \
  --output "${EXPERIMENT_DIR}/stage_receipts/gpu_sandbox.json"
srun python "${RELEASE_DIR}/capability_mixture_experiment.py" doctor \
  --experiment-dir "${EXPERIMENT_DIR}"

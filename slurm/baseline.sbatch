#!/usr/bin/env bash
#SBATCH --job-name=frontier-baseline
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=HX00
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=224G
#SBATCH --tmp=50G
#SBATCH --time=15:30:00
#SBATCH --requeue

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${RELEASE_DIR:?Set RELEASE_DIR}"

if [[ "${ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
  echo "Set ALLOW_CODE_EXECUTION=1 only inside a disposable, network-isolated environment." >&2
  exit 2
fi
source "${RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
enable_offline_model_loading

srun python "${RELEASE_DIR}/capability_mixture_experiment.py" evaluate-baseline \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --allow-generated-code-execution \
  --code-sandbox bwrap

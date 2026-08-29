#!/usr/bin/env bash
#SBATCH --job-name=routing-robustness-all
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
: "${SLURM_ARRAY_TASK_ID:?Submit as the sparse 42-profile array}"
: "${SOURCE_EXPERIMENT_DIR:?Set SOURCE_EXPERIMENT_DIR}"
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
enable_offline_model_loading
ensure_cuda_allocation_or_requeue
srun python "${RELEASE_DIR}/run_benchmark_robustness.py" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --source-experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
  --profile-index "${SLURM_ARRAY_TASK_ID}" \
  --allow-generated-code-execution \
  --code-sandbox bwrap

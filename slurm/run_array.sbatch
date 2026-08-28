#!/usr/bin/env bash
#SBATCH --job-name=frontier-arm
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
#SBATCH --signal=B:USR1@600

set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
: "${RELEASE_DIR:?Set RELEASE_DIR}"
: "${SLURM_ARRAY_TASK_ID:?This script must be submitted as a Slurm array}"

if [[ "${ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
  echo "Set ALLOW_CODE_EXECUTION=1 only inside a disposable, network-isolated environment." >&2
  exit 2
fi
source "${RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
enable_offline_model_loading

child_pid=""
handle_time_limit() {
  if [[ -n "${child_pid}" ]]; then
    kill -TERM "${child_pid}" 2>/dev/null || true
    wait "${child_pid}" 2>/dev/null || true
  fi
  restart_count="${SLURM_RESTART_COUNT:-0}"
  if (( restart_count + 1 < ${MAX_ATTEMPTS:-3} )); then
    scontrol requeue "${SLURM_JOB_ID}"
    exit 0
  fi
  exit 124
}
trap handle_time_limit USR1

srun python "${RELEASE_DIR}/capability_mixture_experiment.py" run-array-arm \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --allow-generated-code-execution \
  --code-sandbox bwrap &
child_pid=$!
wait "${child_pid}"

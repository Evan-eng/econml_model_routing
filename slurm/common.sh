#!/usr/bin/env bash

activate_ice_environment() {
  : "${VENV_DIR:?Set VENV_DIR}"
  module purge
  module load anaconda3/2023.03
  module load cuda/12.6.1
  source "${VENV_DIR}/bin/activate"
  export HF_HOME="${HF_HOME:?Set HF_HOME}"
  export HF_DATASETS_CACHE="${HF_HOME}/datasets"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
}

enable_offline_model_loading() {
  export HF_HUB_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
}

ensure_cuda_allocation_or_requeue() {
  local output
  local status
  local restart_count

  if output="$(srun python -c 'import torch; torch.empty(1, device="cuda"); torch.cuda.synchronize()' 2>&1)"; then
    return 0
  else
    status=$?
  fi

  printf '%s\n' "${output}" >&2
  if [[ "${output}" == *"CUDA-capable device(s) is/are busy or unavailable"* ]]; then
    restart_count="${SLURM_RESTART_COUNT:-0}"
    if (( restart_count + 1 < ${MAX_ATTEMPTS:-3} )); then
      printf 'CUDA allocation unavailable; requeueing job %s for attempt %d\n' \
        "${SLURM_JOB_ID:?Set SLURM_JOB_ID}" "$((restart_count + 2))" >&2
      scontrol requeue "${SLURM_JOB_ID}"
      exit 0
    fi
  fi
  return "${status}"
}

validate_stage_release() {
  : "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
  : "${RELEASE_SHA:?Set RELEASE_SHA}"
  : "${RELEASE_DIR:?Set RELEASE_DIR}"
  : "${RUN_CONTRACT_PATH:?Set RUN_CONTRACT_PATH}"
  export EXECUTOR_RELEASE="${RELEASE_SHA}"
  [[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
  [[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]
  PYTHONPATH="${RELEASE_DIR}" python -c 'from pathlib import Path; import os; from ice_contract import validate_run_contract; validate_run_contract(Path(os.environ["RUN_CONTRACT_PATH"]), expected_release=os.environ["RELEASE_SHA"], expected_experiment_dir=Path(os.environ["EXPERIMENT_DIR"]))'
}

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

validate_stage_release() {
  : "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}"
  : "${RELEASE_SHA:?Set RELEASE_SHA}"
  : "${RELEASE_DIR:?Set RELEASE_DIR}"
  : "${RUN_CONTRACT_PATH:?Set RUN_CONTRACT_PATH}"
  export EXECUTOR_RELEASE="${RELEASE_SHA}"
  [[ "$(git -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
  [[ -z "$(git -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]
  PYTHONPATH="${RELEASE_DIR}" python -c 'from pathlib import Path; import os; from ice_contract import validate_run_contract; validate_run_contract(Path(os.environ["RUN_CONTRACT_PATH"]), expected_release=os.environ["RELEASE_SHA"], expected_experiment_dir=Path(os.environ["EXPERIMENT_DIR"]))'
}

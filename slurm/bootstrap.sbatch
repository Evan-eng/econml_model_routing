#!/usr/bin/env bash
#SBATCH --job-name=routing-bootstrap
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

: "${ARTIFACT_ROOT:?Set ARTIFACT_ROOT}"
: "${RELEASE_DIR:?Set RELEASE_DIR}"
: "${RELEASE_SHA:?Set RELEASE_SHA}"
: "${VENV_DIR:?Set VENV_DIR}"
: "${UV_BIN:?Set UV_BIN}"

module purge
module load anaconda3/2023.03
[[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]
export EXECUTOR_RELEASE="${RELEASE_SHA}"
export PYTHONPATH="${RELEASE_DIR}"
srun python -c 'from pathlib import Path; import os; from ice_contract import validate_run_contract; validate_run_contract(Path(os.environ["RUN_CONTRACT_PATH"]), expected_release=os.environ["RELEASE_SHA"], expected_experiment_dir=Path(os.environ["EXPERIMENT_DIR"]))'

export UV_CACHE_DIR="${ARTIFACT_ROOT}/cache/uv"
export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"
srun "${UV_BIN}" sync --frozen --extra robustness \
  --project "${RELEASE_DIR}" --python "$(command -v python)"
srun "${VENV_DIR}/bin/python" -c \
  'import platform, sys; print(sys.version); print(platform.platform())'
srun "${VENV_DIR}/bin/python" "${RELEASE_DIR}/capability_mixture_experiment.py" \
  receipt-environment --experiment-dir "${EXPERIMENT_DIR}"

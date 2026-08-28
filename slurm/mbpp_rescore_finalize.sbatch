#!/usr/bin/env bash
#SBATCH --job-name=routing-mbpp-rescore-finalize
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
mkdir -p "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR}/compute"
RAW_ACCOUNTING="${EXPERIMENT_DIR}/compute/sacct_raw.psv"
[[ "${ACCOUNT_JOB_IDS:?Set ACCOUNT_JOB_IDS}" =~ ^[0-9]+(:[0-9]+)*$ ]]
SACCT_JOB_IDS="${ACCOUNT_JOB_IDS//:/,}"
if [[ ! -f "${RAW_ACCOUNTING}" ]]; then
  raw_temp="$(mktemp "${EXPERIMENT_DIR}/compute/.sacct_raw.XXXXXX")"
  trap 'rm -f "${raw_temp:-}"' EXIT
  sacct -X -j "${SACCT_JOB_IDS}" --noheader --parsable2 \
    --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Start,End > "${raw_temp}"
  if ! ln "${raw_temp}" "${RAW_ACCOUNTING}" 2>/dev/null; then [[ -f "${RAW_ACCOUNTING}" ]]; fi
  chmod 0444 "${RAW_ACCOUNTING}"; rm -f "${raw_temp}"; trap - EXIT
fi
srun python "${RELEASE_DIR}/mbpp_rescore.py" validate-accounting \
  --raw "${RAW_ACCOUNTING}" --job-ids "${ACCOUNT_JOB_IDS}"
srun python "${RELEASE_DIR}/ice_pipeline.py" account --raw "${RAW_ACCOUNTING}" --output-dir "${EXPERIMENT_DIR}/compute/ledger"
srun python "${RELEASE_DIR}/mbpp_rescore.py" finalize --experiment-dir "${EXPERIMENT_DIR}" --export-dir "${EXPORT_DIR:?Set EXPORT_DIR}" --parent-export-dir "${PARENT_EXPORT_DIR:?Set PARENT_EXPORT_DIR}"

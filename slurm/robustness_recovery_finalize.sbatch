#!/usr/bin/env bash
#SBATCH --job-name=routing-robustness-recovery-finalize
#SBATCH --qos=coe-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail
source "${RELEASE_DIR:?Set RELEASE_DIR}/slurm/common.sh"
activate_ice_environment
validate_stage_release
mkdir -p "${EXPERIMENT_DIR}/compute"
sacct -X -j "${ACCOUNT_JOB_IDS:?Set ACCOUNT_JOB_IDS}" --noheader --parsable2 \
  --format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES,Start,End \
  > "${EXPERIMENT_DIR}/compute/sacct_raw.psv"
srun python "${RELEASE_DIR}/ice_pipeline.py" account \
  --raw "${EXPERIMENT_DIR}/compute/sacct_raw.psv" \
  --output-dir "${EXPERIMENT_DIR}/compute/ledger"
srun python "${RELEASE_DIR}/robustness_recovery.py" finalize \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --export-dir "${EXPORT_DIR:?Set EXPORT_DIR}"

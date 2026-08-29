#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/storage/ice-shared/cs7634/staff/econml}"
REPO_URL="${REPO_URL:-https://github.com/Evan-eng/econml_model_routing}"
RELEASE_SHA="${RELEASE_SHA:?Set RELEASE_SHA to the exact deployed 40-character SHA}"
RELEASE_ROOT="${RELEASE_ROOT:-${ARTIFACT_ROOT}/releases/econml_model_routing}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_SHA}"
RELEASE_RECEIPT="${RELEASE_ROOT}/.receipts/${RELEASE_SHA}.json"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?Set SOURCE_RUN_ID to the sealed confirmatory run}"
SOURCE_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${SOURCE_RUN_ID}"
PROFILE_INDICES="1-19,21-39,41-44"
RUN_ID="${RUN_ID:-${SOURCE_RUN_ID}-robustness-all-$(date -u +%Y%m%dT%H%M%SZ)-${RELEASE_SHA:0:12}}"
EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${RUN_ID}"
EXPORT_DIR="${ARTIFACT_ROOT}/exports/${RUN_ID}"
LOG_DIR="${ARTIFACT_ROOT}/logs/${RUN_ID}"
HF_HOME="${ARTIFACT_ROOT}/cache/huggingface"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

[[ "${RELEASE_SHA}" =~ ^[0-9a-f]{40}$ ]]
[[ -x "${UV_BIN}" ]]
[[ -f "${RELEASE_RECEIPT}" ]]
[[ -f "${SOURCE_EXPERIMENT_DIR}/stage_receipts/submission.json" ]]
python3 -c 'import json,sys; p,sha,deploy,origin=sys.argv[1:]; r=json.load(open(p)); assert r.get("contract")=="pace-code-release-v3"; assert r.get("revision")==sha; assert r.get("deployment_path")==deploy; assert r.get("origin")==origin' \
  "${RELEASE_RECEIPT}" "${RELEASE_SHA}" "${RELEASE_DIR}" "${REPO_URL}"
[[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]

LOCK_SHA="$(sha256sum "${RELEASE_DIR}/uv.lock" | cut -d' ' -f1)"
VENV_DIR="${ARTIFACT_ROOT}/envs/${LOCK_SHA}"
umask 077
mkdir -p "${ARTIFACT_ROOT}/runs" "${ARTIFACT_ROOT}/envs" \
  "${ARTIFACT_ROOT}/cache" "${ARTIFACT_ROOT}/exports" "${LOG_DIR}"

python3 "${RELEASE_DIR}/exploratory_robustness.py" init \
  --source-experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
  --experiment-dir "${EXPERIMENT_DIR}"
python3 "${RELEASE_DIR}/capability_mixture_experiment.py" create-run-contract \
  --experiment-dir "${EXPERIMENT_DIR}" --run-id "${RUN_ID}" \
  --artifact-root "${ARTIFACT_ROOT}" --release-receipt "${RELEASE_RECEIPT}" \
  --expected-release "${RELEASE_SHA}" --lock-path "${RELEASE_DIR}/uv.lock" \
  --max-attempts 3
export EXECUTOR_RELEASE="${RELEASE_SHA}"
export RUN_CONTRACT_PATH="${EXPERIMENT_DIR}/run_contract.json"
python3 "${RELEASE_DIR}/exploratory_robustness.py" source-receipt \
  --source-experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
  --experiment-dir "${EXPERIMENT_DIR}"

read -r SOURCE_ARRAY_JOB SOURCE_ROBUSTNESS_JOB SOURCE_SUBMISSION_RELEASE SOURCE_RECEIPT_RELEASE < <(
  python3 - "${SOURCE_EXPERIMENT_DIR}" "${EXPERIMENT_DIR}" <<'PY'
import json
import pathlib
import sys

source = pathlib.Path(sys.argv[1])
extension = pathlib.Path(sys.argv[2])
submission = json.load((source / "stage_receipts" / "submission.json").open())
source_receipt = json.load((extension / "stage_receipts" / "source.json").open())
print(
    submission["jobs"]["remaining_array"],
    submission["jobs"]["robustness"],
    submission["output_release"],
    source_receipt["output_release"],
)
PY
)
[[ "${SOURCE_ARRAY_JOB}" =~ ^[0-9]+$ ]]
[[ "${SOURCE_ROBUSTNESS_JOB}" =~ ^[0-9]+$ ]]

ENVIRONMENT_RECEIPT="${EXPERIMENT_DIR}/stage_receipts/environment.json"
common_exports="ALL,ARTIFACT_ROOT=${ARTIFACT_ROOT},RELEASE_DIR=${RELEASE_DIR},RELEASE_SHA=${RELEASE_SHA},VENV_DIR=${VENV_DIR},HF_HOME=${HF_HOME},UV_BIN=${UV_BIN},MAX_ATTEMPTS=3,EXPERIMENT_DIR=${EXPERIMENT_DIR},RUN_CONTRACT_PATH=${RUN_CONTRACT_PATH},SOURCE_EXPERIMENT_DIR=${SOURCE_EXPERIMENT_DIR},EXPORT_DIR=${EXPORT_DIR},ENVIRONMENT_RECEIPT=${ENVIRONMENT_RECEIPT}"
SUBMITTED_IDS_PATH="${EXPERIMENT_DIR}/stage_receipts/submitted_job_ids.partial"
: > "${SUBMITTED_IDS_PATH}"
cancel_partial_chain() {
  if [[ -s "${SUBMITTED_IDS_PATH}" ]]; then
    mapfile -t submitted_ids < "${SUBMITTED_IDS_PATH}"
    scancel "${submitted_ids[@]}" 2>/dev/null || true
  fi
}
trap cancel_partial_chain ERR
submit_job() {
  local result
  result="$(sbatch --parsable "$@")"
  result="${result%%;*}"
  [[ "${result}" =~ ^[0-9]+$ ]]
  printf '%s\n' "${result}" >> "${SUBMITTED_IDS_PATH}"
  printf '%s' "${result}"
}
gpu_options=()
if [[ -n "${EXCLUDE_NODES}" ]]; then
  gpu_options+=(--exclude="${EXCLUDE_NODES}")
fi

BOOTSTRAP_JOB="$(submit_job --export="${common_exports}" \
  --output="${LOG_DIR}/bootstrap_%j.out" --error="${LOG_DIR}/bootstrap_%j.err" \
  "${RELEASE_DIR}/slurm/bootstrap.sbatch")"
DOCTOR_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/gpu_doctor_%j.out" --error="${LOG_DIR}/gpu_doctor_%j.err" \
  "${RELEASE_DIR}/slurm/exploratory_gpu_doctor.sbatch")"
PROFILES_JOB="$(submit_job \
  --dependency="aftercorr:${SOURCE_ARRAY_JOB},afterok:${DOCTOR_JOB}" \
  --array="${PROFILE_INDICES}%42" "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/profile_%A_%a.out" --error="${LOG_DIR}/profile_%A_%a.err" \
  "${RELEASE_DIR}/slurm/exploratory_robustness.sbatch")"
final_exports="${common_exports},DOCTOR_JOB_ID=${DOCTOR_JOB},PROFILES_JOB_ID=${PROFILES_JOB}"
FINALIZE_JOB="$(submit_job \
  --dependency="afterok:${PROFILES_JOB}:${SOURCE_ROBUSTNESS_JOB}" \
  --export="${final_exports}" --output="${LOG_DIR}/finalize_%j.out" \
  --error="${LOG_DIR}/finalize_%j.err" \
  "${RELEASE_DIR}/slurm/exploratory_finalize.sbatch")"

python3 "${RELEASE_DIR}/ice_pipeline.py" submission-receipt \
  --contract "${RUN_CONTRACT_PATH}" \
  --output "${EXPERIMENT_DIR}/stage_receipts/submission.json" \
  --input-producer-release "pace-code-release:$(sha256sum "${RELEASE_RECEIPT}" | cut -d' ' -f1)" \
  --input-producer-release "${SOURCE_SUBMISSION_RELEASE}" \
  --input-producer-release "${SOURCE_RECEIPT_RELEASE}" \
  --pre-side-effect-release-gate submit_exploratory_robustness.sh \
  --job "bootstrap=${BOOTSTRAP_JOB}" --job "gpu_doctor=${DOCTOR_JOB}" \
  --job "profiles=${PROFILES_JOB}" --job "finalize=${FINALIZE_JOB}"

trap - ERR
printf 'run_id=%s\nexperiment_dir=%s\nexport_dir=%s\nprofiles_job=%s\nfinalize_job=%s\n' \
  "${RUN_ID}" "${EXPERIMENT_DIR}" "${EXPORT_DIR}" "${PROFILES_JOB}" "${FINALIZE_JOB}"

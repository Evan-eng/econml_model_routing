#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/storage/ice-shared/cs7634/staff/econml}"
REPO_URL="${REPO_URL:-https://github.com/Evan-eng/econml_model_routing}"
RELEASE_SHA="${RELEASE_SHA:?Set RELEASE_SHA to the exact deployed 40-character SHA}"
RELEASE_ROOT="${RELEASE_ROOT:-${ARTIFACT_ROOT}/releases/econml_model_routing}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_SHA}"
RELEASE_RECEIPT="${RELEASE_ROOT}/.receipts/${RELEASE_SHA}.json"
MAX_PARALLEL="${MAX_PARALLEL:-45}"
ENDPOINT_INDICES="0,20,40"
REMAINING_INDICES="1-19,21-39,41-44"
RUN_ID="${RUN_ID:-qwen3-8b-confirmatory-$(date -u +%Y%m%dT%H%M%SZ)-${RELEASE_SHA:0:12}}"
SMOKE_RUN_ID="${RUN_ID}-smoke"
EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${RUN_ID}"
SMOKE_DIR="${ARTIFACT_ROOT}/runs/${SMOKE_RUN_ID}"
EXPORT_DIR="${ARTIFACT_ROOT}/exports/${RUN_ID}"
LOG_DIR="${ARTIFACT_ROOT}/logs/${RUN_ID}"
HF_HOME="${ARTIFACT_ROOT}/cache/huggingface"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

[[ "${RELEASE_SHA}" =~ ^[0-9a-f]{40}$ ]]
[[ "${MAX_PARALLEL}" =~ ^[0-9]+$ ]] && (( MAX_PARALLEL == 45 ))
[[ -x "${UV_BIN}" ]]
[[ -f "${RELEASE_RECEIPT}" ]]
[[ -f "${RELEASE_DIR}/uv.lock" ]]
python3 -c 'import json,sys; p,sha,deploy,origin=sys.argv[1:]; r=json.load(open(p)); assert r.get("contract")=="pace-code-release-v3"; assert r.get("revision")==sha; assert r.get("deployment_path")==deploy; assert r.get("origin")==origin' \
  "${RELEASE_RECEIPT}" "${RELEASE_SHA}" "${RELEASE_DIR}" "${REPO_URL}"
[[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]

LOCK_SHA="$(sha256sum "${RELEASE_DIR}/uv.lock" | cut -d' ' -f1)"
VENV_DIR="${ARTIFACT_ROOT}/envs/${LOCK_SHA}"
umask 077
mkdir -p "${ARTIFACT_ROOT}/runs" "${ARTIFACT_ROOT}/envs" \
  "${ARTIFACT_ROOT}/cache" "${ARTIFACT_ROOT}/exports" "${LOG_DIR}"
chmod 0755 "${ARTIFACT_ROOT}/exports"

python3 "${RELEASE_DIR}/capability_mixture_experiment.py" init \
  --experiment-dir "${SMOKE_DIR}" --preset smoke --model Qwen/Qwen3-8B
python3 "${RELEASE_DIR}/capability_mixture_experiment.py" init \
  --experiment-dir "${EXPERIMENT_DIR}" --preset confirmatory --model Qwen/Qwen3-8B
python3 "${RELEASE_DIR}/capability_mixture_experiment.py" create-run-contract \
  --experiment-dir "${SMOKE_DIR}" --run-id "${SMOKE_RUN_ID}" \
  --artifact-root "${ARTIFACT_ROOT}" --release-receipt "${RELEASE_RECEIPT}" \
  --expected-release "${RELEASE_SHA}" --lock-path "${RELEASE_DIR}/uv.lock" \
  --max-attempts 3
python3 "${RELEASE_DIR}/capability_mixture_experiment.py" create-run-contract \
  --experiment-dir "${EXPERIMENT_DIR}" --run-id "${RUN_ID}" \
  --artifact-root "${ARTIFACT_ROOT}" --release-receipt "${RELEASE_RECEIPT}" \
  --expected-release "${RELEASE_SHA}" --lock-path "${RELEASE_DIR}/uv.lock" \
  --max-attempts 3

ENVIRONMENT_RECEIPT="${EXPERIMENT_DIR}/stage_receipts/environment.json"
GPU_DOCTOR_RECEIPT="${EXPERIMENT_DIR}/stage_receipts/gpu_sandbox.json"
common_exports="ALL,ARTIFACT_ROOT=${ARTIFACT_ROOT},RELEASE_DIR=${RELEASE_DIR},RELEASE_SHA=${RELEASE_SHA},VENV_DIR=${VENV_DIR},HF_HOME=${HF_HOME},UV_BIN=${UV_BIN},MAX_ATTEMPTS=3,ENVIRONMENT_RECEIPT=${ENVIRONMENT_RECEIPT}"
full_exports="${common_exports},EXPERIMENT_DIR=${EXPERIMENT_DIR},RUN_CONTRACT_PATH=${EXPERIMENT_DIR}/run_contract.json,EXPORT_DIR=${EXPORT_DIR},ALLOW_CODE_EXECUTION=1"
smoke_exports="${common_exports},EXPERIMENT_DIR=${SMOKE_DIR},RUN_CONTRACT_PATH=${SMOKE_DIR}/run_contract.json,ALLOW_CODE_EXECUTION=1,GPU_DOCTOR_RECEIPT=${GPU_DOCTOR_RECEIPT}"
SUBMITTED_IDS_PATH="${EXPERIMENT_DIR}/stage_receipts/submitted_job_ids.partial"
mkdir -p "$(dirname "${SUBMITTED_IDS_PATH}")"
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
  if ! result="$(sbatch --parsable "$@")"; then
    return 1
  fi
  result="${result%%;*}"
  [[ "${result}" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "${result}" >> "${SUBMITTED_IDS_PATH}"
  printf '%s' "${result}"
}

BOOTSTRAP_JOB="$(submit_job --export="${full_exports}" \
  --output="${LOG_DIR}/bootstrap_%j.out" --error="${LOG_DIR}/bootstrap_%j.err" \
  "${RELEASE_DIR}/slurm/bootstrap.sbatch")"
FULL_PREP_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --export="${full_exports}" --output="${LOG_DIR}/full_prep_%j.out" \
  --error="${LOG_DIR}/full_prep_%j.err" "${RELEASE_DIR}/slurm/prepare.sbatch")"
SMOKE_PREP_JOB="$(submit_job --dependency="afterok:${FULL_PREP_JOB}" \
  --export="${smoke_exports}" --output="${LOG_DIR}/smoke_prep_%j.out" \
  --error="${LOG_DIR}/smoke_prep_%j.err" "${RELEASE_DIR}/slurm/prepare.sbatch")"
GPU_DOCTOR_JOB="$(submit_job --dependency="afterok:${FULL_PREP_JOB}" \
  --export="${full_exports}" --output="${LOG_DIR}/gpu_doctor_%j.out" \
  --error="${LOG_DIR}/gpu_doctor_%j.err" "${RELEASE_DIR}/slurm/gpu_doctor.sbatch")"

smoke_dependency="afterok:${SMOKE_PREP_JOB}:${GPU_DOCTOR_JOB}"
SMOKE_BASELINE_JOB="$(submit_job --dependency="${smoke_dependency}" \
  --export="${smoke_exports}" --output="${LOG_DIR}/smoke_baseline_%j.out" \
  --error="${LOG_DIR}/smoke_baseline_%j.err" "${RELEASE_DIR}/slurm/baseline.sbatch")"
SMOKE_ARRAY_JOB="$(submit_job --dependency="${smoke_dependency}" --array="0-2%3" \
  --export="${smoke_exports}" --output="${LOG_DIR}/smoke_arm_%A_%a.out" \
  --error="${LOG_DIR}/smoke_arm_%A_%a.err" "${RELEASE_DIR}/slurm/run_array.sbatch")"
SMOKE_GATE_JOB="$(submit_job --dependency="afterok:${SMOKE_BASELINE_JOB}:${SMOKE_ARRAY_JOB}" \
  --export="${smoke_exports}" --output="${LOG_DIR}/smoke_gate_%j.out" \
  --error="${LOG_DIR}/smoke_gate_%j.err" "${RELEASE_DIR}/slurm/smoke_gate.sbatch")"

full_dependency="afterok:${SMOKE_GATE_JOB}:${FULL_PREP_JOB}"
FULL_BASELINE_JOB="$(submit_job --dependency="${full_dependency}" \
  --export="${full_exports}" --output="${LOG_DIR}/baseline_%j.out" \
  --error="${LOG_DIR}/baseline_%j.err" "${RELEASE_DIR}/slurm/baseline.sbatch")"
ENDPOINT_ARRAY_JOB="$(submit_job --dependency="${full_dependency}" \
  --array="${ENDPOINT_INDICES}%3" --export="${full_exports}" \
  --output="${LOG_DIR}/endpoint_arm_%A_%a.out" --error="${LOG_DIR}/endpoint_arm_%A_%a.err" \
  "${RELEASE_DIR}/slurm/run_array.sbatch")"
REMAINING_PARALLEL=$((MAX_PARALLEL - 3))
REMAINING_ARRAY_JOB="$(submit_job --dependency="${full_dependency}" \
  --array="${REMAINING_INDICES}%${REMAINING_PARALLEL}" --export="${full_exports}" \
  --output="${LOG_DIR}/arm_%A_%a.out" --error="${LOG_DIR}/arm_%A_%a.err" \
  "${RELEASE_DIR}/slurm/run_array.sbatch")"

ENDPOINT_REPORT_JOB="$(submit_job --dependency="afterok:${FULL_BASELINE_JOB}:${ENDPOINT_ARRAY_JOB}" \
  --export="${full_exports}" --output="${LOG_DIR}/endpoint_report_%j.out" \
  --error="${LOG_DIR}/endpoint_report_%j.err" "${RELEASE_DIR}/slurm/endpoint_report.sbatch")"
ROBUSTNESS_JOB="$(submit_job --dependency="afterok:${ENDPOINT_REPORT_JOB}" --array="0-3%4" \
  --export="${full_exports}" --output="${LOG_DIR}/robustness_%A_%a.out" \
  --error="${LOG_DIR}/robustness_%A_%a.err" "${RELEASE_DIR}/slurm/robustness.sbatch")"
ANALYSIS_JOB="$(submit_job --dependency="afterok:${FULL_BASELINE_JOB}:${ENDPOINT_ARRAY_JOB}:${REMAINING_ARRAY_JOB}" \
  --export="${full_exports}" --output="${LOG_DIR}/analysis_%j.out" \
  --error="${LOG_DIR}/analysis_%j.err" "${RELEASE_DIR}/slurm/analyze.sbatch")"

ACCOUNT_JOB_IDS="${GPU_DOCTOR_JOB},${SMOKE_BASELINE_JOB},${SMOKE_ARRAY_JOB},${FULL_BASELINE_JOB},${ENDPOINT_ARRAY_JOB},${REMAINING_ARRAY_JOB},${ROBUSTNESS_JOB}"
final_exports="${full_exports},ACCOUNT_JOB_IDS=${ACCOUNT_JOB_IDS}"
FINALIZE_JOB="$(submit_job --dependency="afterok:${ANALYSIS_JOB}:${ROBUSTNESS_JOB}" \
  --export="${final_exports}" --output="${LOG_DIR}/finalize_%j.out" \
  --error="${LOG_DIR}/finalize_%j.err" "${RELEASE_DIR}/slurm/finalize.sbatch")"

STATUS_4H_JOB="$(submit_job --begin="now+4hours" \
  --export="${full_exports},STATUS_LABEL=t04h" \
  --output="${LOG_DIR}/status_t04h_%j.out" --error="${LOG_DIR}/status_t04h_%j.err" \
  "${RELEASE_DIR}/slurm/status_snapshot.sbatch")"
STATUS_12H_JOB="$(submit_job --begin="now+12hours" \
  --export="${full_exports},STATUS_LABEL=t12h" \
  --output="${LOG_DIR}/status_t12h_%j.out" --error="${LOG_DIR}/status_t12h_%j.err" \
  "${RELEASE_DIR}/slurm/status_snapshot.sbatch")"
STATUS_20H_JOB="$(submit_job --begin="now+20hours" \
  --export="${full_exports},STATUS_LABEL=t20h" \
  --output="${LOG_DIR}/status_t20h_%j.out" --error="${LOG_DIR}/status_t20h_%j.err" \
  "${RELEASE_DIR}/slurm/status_snapshot.sbatch")"
STATUS_24H_JOB="$(submit_job --begin="now+24hours" \
  --export="${full_exports},STATUS_LABEL=t24h" \
  --output="${LOG_DIR}/status_t24h_%j.out" --error="${LOG_DIR}/status_t24h_%j.err" \
  "${RELEASE_DIR}/slurm/status_snapshot.sbatch")"
STATUS_48H_JOB="$(submit_job --begin="now+48hours" \
  --export="${full_exports},STATUS_LABEL=t48h" \
  --output="${LOG_DIR}/status_t48h_%j.out" --error="${LOG_DIR}/status_t48h_%j.err" \
  "${RELEASE_DIR}/slurm/status_snapshot.sbatch")"

python3 "${RELEASE_DIR}/ice_pipeline.py" submission-receipt \
  --contract "${EXPERIMENT_DIR}/run_contract.json" \
  --output "${EXPERIMENT_DIR}/stage_receipts/submission.json" \
  --job "bootstrap=${BOOTSTRAP_JOB}" --job "smoke_prepare=${SMOKE_PREP_JOB}" \
  --job "full_prepare=${FULL_PREP_JOB}" --job "gpu_doctor=${GPU_DOCTOR_JOB}" \
  --job "smoke_baseline=${SMOKE_BASELINE_JOB}" --job "smoke_array=${SMOKE_ARRAY_JOB}" \
  --job "smoke_gate=${SMOKE_GATE_JOB}" --job "full_baseline=${FULL_BASELINE_JOB}" \
  --job "endpoint_array=${ENDPOINT_ARRAY_JOB}" --job "remaining_array=${REMAINING_ARRAY_JOB}" \
  --job "endpoint_report=${ENDPOINT_REPORT_JOB}" --job "robustness=${ROBUSTNESS_JOB}" \
  --job "analysis=${ANALYSIS_JOB}" --job "finalize=${FINALIZE_JOB}" \
  --job "status_4h=${STATUS_4H_JOB}" --job "status_12h=${STATUS_12H_JOB}" \
  --job "status_20h=${STATUS_20H_JOB}" --job "status_24h=${STATUS_24H_JOB}" \
  --job "status_48h=${STATUS_48H_JOB}"

trap - ERR
printf 'run_id=%s\nexperiment_dir=%s\nexport_dir=%s\nfinalize_job=%s\n' \
  "${RUN_ID}" "${EXPERIMENT_DIR}" "${EXPORT_DIR}" "${FINALIZE_JOB}"

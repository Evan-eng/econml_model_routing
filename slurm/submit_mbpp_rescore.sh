#!/usr/bin/env bash
# Submit the CPU-only, 46-profile MBPP re-execution from frozen source outputs.
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/storage/ice-shared/cs7634/staff/econml}"
REPO_URL="${REPO_URL:-https://github.com/Evan-eng/econml_model_routing}"
RELEASE_SHA="${RELEASE_SHA:?Set RELEASE_SHA to the exact deployed 40-character SHA}"
RELEASE_ROOT="${RELEASE_ROOT:-${ARTIFACT_ROOT}/releases/econml_model_routing_v3}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_SHA}"
RELEASE_RECEIPT="${RELEASE_ROOT}/.receipts/${RELEASE_SHA}.json"
SOURCE_RUN_ID="qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c"
SOURCE_RELEASE="c3cb569cf51c3f53a3c7492388c7183c9f91c6df"
SOURCE_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${SOURCE_RUN_ID}"
PARENT_RUN_ID="qwen3-8b-confirmatory-20260828T165149Z-c3cb569cf51c-robustness-recovery-20260828T201706Z-aae927901acf-math500-rescore-20260828T222636Z-dc549e0ad61b"
PARENT_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${PARENT_RUN_ID}"
PARENT_EXPORT_DIR="${ARTIFACT_ROOT}/exports/${PARENT_RUN_ID}"
PROFILE_INDICES="0-45"
RESCORE_MAX_PARALLEL=8
RUN_ID="${RUN_ID:-${SOURCE_RUN_ID}-mbpp-rescore-$(date -u +%Y%m%dT%H%M%SZ)-${RELEASE_SHA:0:12}}"
EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${RUN_ID}"
EXPORT_DIR="${ARTIFACT_ROOT}/exports/${RUN_ID}"
LOG_DIR="${ARTIFACT_ROOT}/logs/${RUN_ID}"
HF_HOME="${ARTIFACT_ROOT}/cache/huggingface"
UV_BIN="${UV_BIN:-${HOME}/.local/bin/uv}"

[[ "${RELEASE_SHA}" =~ ^[0-9a-f]{40}$ ]]
[[ -x "${UV_BIN}" ]]
[[ -f "${RELEASE_RECEIPT}" ]]
[[ -f "${SOURCE_EXPERIMENT_DIR}/stage_receipts/submission.json" ]]
[[ -f "${PARENT_EXPERIMENT_DIR}/stage_receipts/submission.json" ]]
[[ -f "${PARENT_EXPERIMENT_DIR}/stage_receipts/FINALIZED.json" ]]
[[ -f "${PARENT_EXPORT_DIR}/MANIFEST.json" ]]
python3 -c 'import json,sys; p,sha,deploy,origin=sys.argv[1:]; r=json.load(open(p)); assert r.get("contract")=="pace-code-release-v3"; assert r.get("revision")==sha; assert r.get("deployment_path")==deploy; assert r.get("origin")==origin' \
  "${RELEASE_RECEIPT}" "${RELEASE_SHA}" "${RELEASE_DIR}" "${REPO_URL}"
[[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]

LOCK_SHA="$(sha256sum "${RELEASE_DIR}/uv.lock" | cut -d' ' -f1)"
VENV_DIR="${ARTIFACT_ROOT}/envs/${LOCK_SHA}"
VENV_PYTHON="${VENV_DIR}/bin/python"
[[ -x "${VENV_PYTHON}" ]]
umask 077
mkdir -p "${ARTIFACT_ROOT}/runs" "${ARTIFACT_ROOT}/exports" "${LOG_DIR}"
"${VENV_PYTHON}" "${RELEASE_DIR}/mbpp_rescore.py" init \
  --source-experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
  --experiment-dir "${EXPERIMENT_DIR}"
"${VENV_PYTHON}" "${RELEASE_DIR}/capability_mixture_experiment.py" create-run-contract \
  --experiment-dir "${EXPERIMENT_DIR}" --run-id "${RUN_ID}" \
  --artifact-root "${ARTIFACT_ROOT}" --release-receipt "${RELEASE_RECEIPT}" \
  --expected-release "${RELEASE_SHA}" --lock-path "${RELEASE_DIR}/uv.lock" \
  --max-attempts 1
export EXECUTOR_RELEASE="${RELEASE_SHA}"
export RUN_CONTRACT_PATH="${EXPERIMENT_DIR}/run_contract.json"
"${VENV_PYTHON}" "${RELEASE_DIR}/mbpp_rescore.py" source-receipt \
  --experiment-dir "${EXPERIMENT_DIR}"

ENVIRONMENT_RECEIPT="${EXPERIMENT_DIR}/stage_receipts/environment.json"
common_exports="ALL,ARTIFACT_ROOT=${ARTIFACT_ROOT},RELEASE_DIR=${RELEASE_DIR},RELEASE_SHA=${RELEASE_SHA},VENV_DIR=${VENV_DIR},HF_HOME=${HF_HOME},UV_BIN=${UV_BIN},EXPERIMENT_DIR=${EXPERIMENT_DIR},RUN_CONTRACT_PATH=${RUN_CONTRACT_PATH},EXPORT_DIR=${EXPORT_DIR},PARENT_EXPORT_DIR=${PARENT_EXPORT_DIR},ENVIRONMENT_RECEIPT=${ENVIRONMENT_RECEIPT}"
SUBMITTED_IDS_PATH="${EXPERIMENT_DIR}/stage_receipts/submitted_job_ids.partial"
: > "${SUBMITTED_IDS_PATH}"
cancel_partial_chain() {
  if [[ -s "${SUBMITTED_IDS_PATH}" ]]; then
    mapfile -t submitted_ids < <(sed '/^$/d' "${SUBMITTED_IDS_PATH}")
    if (( ${#submitted_ids[@]} > 0 )); then
      scancel "${submitted_ids[@]}" 2>/dev/null || true
    fi
  fi
}
trap cancel_partial_chain ERR
submit_job() {
  local result
  if ! result="$(sbatch --parsable "$@")"; then
    return 1
  fi
  result="${result%%;*}"
  if [[ ! "${result}" =~ ^[0-9]+$ ]]; then
    return 1
  fi
  printf '%s\n' "${result}" >> "${SUBMITTED_IDS_PATH}"
  printf '%s' "${result}"
}

BOOTSTRAP_JOB="$(submit_job --export="${common_exports}" \
  --output="${LOG_DIR}/bootstrap_%j.out" --error="${LOG_DIR}/bootstrap_%j.err" \
  "${RELEASE_DIR}/slurm/bootstrap.sbatch")"
DOCTOR_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --export="${common_exports}" --output="${LOG_DIR}/doctor_%j.out" \
  --error="${LOG_DIR}/doctor_%j.err" "${RELEASE_DIR}/slurm/mbpp_rescore_doctor.sbatch")"
SOURCE_GATE_JOB="$(submit_job --dependency="afterok:${DOCTOR_JOB}" \
  --array="${PROFILE_INDICES}%46" --export="${common_exports}" \
  --output="${LOG_DIR}/source_gate_%A_%a.out" \
  --error="${LOG_DIR}/source_gate_%A_%a.err" \
  "${RELEASE_DIR}/slurm/mbpp_rescore_source_gate.sbatch")"
RESCORE_JOB="$(submit_job --dependency="aftercorr:${SOURCE_GATE_JOB}" \
  --array="${PROFILE_INDICES}%${RESCORE_MAX_PARALLEL}" --export="${common_exports}" \
  --output="${LOG_DIR}/rescore_%A_%a.out" \
  --error="${LOG_DIR}/rescore_%A_%a.err" \
  "${RELEASE_DIR}/slurm/mbpp_rescore.sbatch")"
ACCOUNT_JOB_IDS="${BOOTSTRAP_JOB},${DOCTOR_JOB},${SOURCE_GATE_JOB},${RESCORE_JOB}"
FINALIZE_JOB="$(submit_job \
  --dependency="afterok:${RESCORE_JOB}" \
  --export="${common_exports},ACCOUNT_JOB_IDS=${ACCOUNT_JOB_IDS}" \
  --output="${LOG_DIR}/finalize_%j.out" --error="${LOG_DIR}/finalize_%j.err" \
  "${RELEASE_DIR}/slurm/mbpp_rescore_finalize.sbatch")"

"${VENV_PYTHON}" "${RELEASE_DIR}/ice_pipeline.py" submission-receipt \
  --contract "${RUN_CONTRACT_PATH}" \
  --output "${EXPERIMENT_DIR}/stage_receipts/submission.json" \
  --input-producer-release "pace-code-release:$(sha256sum "${RELEASE_RECEIPT}" | cut -d' ' -f1)" \
  --input-producer-release "${SOURCE_RELEASE}:${SOURCE_RUN_ID}:submission:chain" \
  --input-producer-release "$(${VENV_PYTHON} -c 'import json,sys; print(json.load(open(sys.argv[1]))["output_release"])' "${EXPERIMENT_DIR}/stage_receipts/source.json")" \
  --pre-side-effect-release-gate submit_mbpp_rescore.sh \
  --job "bootstrap=${BOOTSTRAP_JOB}" --job "doctor=${DOCTOR_JOB}" \
  --job "source_gate=${SOURCE_GATE_JOB}" --job "rescore=${RESCORE_JOB}" \
  --job "finalize=${FINALIZE_JOB}"

trap - ERR
printf 'run_id=%s\nexperiment_dir=%s\nexport_dir=%s\nrescore_job=%s\nfinalize_job=%s\n' \
  "${RUN_ID}" "${EXPERIMENT_DIR}" "${EXPORT_DIR}" "${RESCORE_JOB}" "${FINALIZE_JOB}"

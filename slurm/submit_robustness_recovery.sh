#!/usr/bin/env bash
set -euo pipefail

ARTIFACT_ROOT="${ARTIFACT_ROOT:-/storage/ice-shared/cs7634/staff/econml}"
REPO_URL="${REPO_URL:-https://github.com/Evan-eng/econml_model_routing}"
RELEASE_SHA="${RELEASE_SHA:?Set RELEASE_SHA to the exact deployed 40-character SHA}"
RELEASE_ROOT="${RELEASE_ROOT:-${ARTIFACT_ROOT}/releases/econml_model_routing_v3}"
RELEASE_DIR="${RELEASE_ROOT}/${RELEASE_SHA}"
RELEASE_RECEIPT="${RELEASE_ROOT}/.receipts/${RELEASE_SHA}.json"
SOURCE_RUN_ID="${SOURCE_RUN_ID:?Set SOURCE_RUN_ID to the accepted confirmatory run}"
SUPERSEDED_RUN_ID="${SUPERSEDED_RUN_ID:?Set SUPERSEDED_RUN_ID to the failed extension}"
PRIOR_RECOVERY_RUN_ID="${PRIOR_RECOVERY_RUN_ID:-}"
SOURCE_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${SOURCE_RUN_ID}"
SUPERSEDED_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${SUPERSEDED_RUN_ID}"
PRIOR_RECOVERY_EXPERIMENT_DIR="${ARTIFACT_ROOT}/runs/${PRIOR_RECOVERY_RUN_ID}"
ENDPOINT_INDICES="0,20,40"
REMAINING_INDICES="1-19,21-39,41-44"
RUN_ID="${RUN_ID:-${SOURCE_RUN_ID}-robustness-recovery-$(date -u +%Y%m%dT%H%M%SZ)-${RELEASE_SHA:0:12}}"
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
[[ -f "${SUPERSEDED_EXPERIMENT_DIR}/stage_receipts/submission.json" ]]
if [[ -n "${PRIOR_RECOVERY_RUN_ID}" ]]; then
  [[ -f "${PRIOR_RECOVERY_EXPERIMENT_DIR}/stage_receipts/submission.json" ]]
fi
python3 -c 'import json,sys; p,sha,deploy,origin=sys.argv[1:]; r=json.load(open(p)); assert r.get("contract")=="pace-code-release-v3"; assert r.get("revision")==sha; assert r.get("deployment_path")==deploy; assert r.get("origin")==origin' \
  "${RELEASE_RECEIPT}" "${RELEASE_SHA}" "${RELEASE_DIR}" "${REPO_URL}"
[[ "$(git -c core.fileMode=false -C "${RELEASE_DIR}" rev-parse HEAD)" == "${RELEASE_SHA}" ]]
[[ -z "$(git -c core.fileMode=false -C "${RELEASE_DIR}" status --porcelain --untracked-files=all)" ]]

LOCK_SHA="$(sha256sum "${RELEASE_DIR}/uv.lock" | cut -d' ' -f1)"
VENV_DIR="${ARTIFACT_ROOT}/envs/${LOCK_SHA}"
umask 077
mkdir -p "${ARTIFACT_ROOT}/runs" "${ARTIFACT_ROOT}/envs" \
  "${ARTIFACT_ROOT}/cache" "${ARTIFACT_ROOT}/exports" "${LOG_DIR}"

prior_recovery_args=()
if [[ -n "${PRIOR_RECOVERY_RUN_ID}" ]]; then
  prior_recovery_args+=(
    --prior-recovery-experiment-dir "${PRIOR_RECOVERY_EXPERIMENT_DIR}"
  )
fi
python3 "${RELEASE_DIR}/robustness_recovery.py" init \
  --source-experiment-dir "${SOURCE_EXPERIMENT_DIR}" \
  --superseded-experiment-dir "${SUPERSEDED_EXPERIMENT_DIR}" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  "${prior_recovery_args[@]}"
python3 "${RELEASE_DIR}/capability_mixture_experiment.py" create-run-contract \
  --experiment-dir "${EXPERIMENT_DIR}" --run-id "${RUN_ID}" \
  --artifact-root "${ARTIFACT_ROOT}" --release-receipt "${RELEASE_RECEIPT}" \
  --expected-release "${RELEASE_SHA}" --lock-path "${RELEASE_DIR}/uv.lock" \
  --max-attempts 3
export EXECUTOR_RELEASE="${RELEASE_SHA}"
export RUN_CONTRACT_PATH="${EXPERIMENT_DIR}/run_contract.json"
python3 "${RELEASE_DIR}/robustness_recovery.py" source-receipt \
  --experiment-dir "${EXPERIMENT_DIR}"

read -r SOURCE_BASELINE_JOB SOURCE_ENDPOINT_JOB SOURCE_REMAINING_JOB SOURCE_ANALYSIS_JOB SOURCE_ACCOUNT_JOB_IDS SUPERSEDED_ACCOUNT_JOB_IDS PRIOR_RECOVERY_ACCOUNT_JOB_IDS SOURCE_SUBMISSION_RELEASE RECOVERY_SOURCE_RELEASE SUPERSEDED_SUBMISSION_RELEASE PRIOR_RECOVERY_SUBMISSION_RELEASE < <(
  python3 - "${EXPERIMENT_DIR}" "${SOURCE_EXPERIMENT_DIR}" <<'PY'
import json
import pathlib
import sys

recovery = pathlib.Path(sys.argv[1])
source = pathlib.Path(sys.argv[2])
plan = json.load((recovery / "recovery_plan.json").open())
source_submission = json.load((source / "stage_receipts" / "submission.json").open())
source_receipt = json.load((recovery / "stage_receipts" / "source.json").open())
source_account_jobs = ",".join(dict.fromkeys(plan["source_jobs_all"].values()))
superseded_account_jobs = ",".join(
    dict.fromkeys(plan["supersedes_jobs"].values())
)
prior = plan.get("prior_recovery")
prior_account_jobs = (
    ",".join(dict.fromkeys(prior["jobs"].values())) if prior else "none"
)
prior_submission_release = prior["submission_release"] if prior else "none"
print(
    plan["source_jobs"]["full_baseline"],
    plan["source_jobs"]["endpoint_array"],
    plan["source_jobs"]["remaining_array"],
    plan["source_jobs"]["analysis"],
    source_account_jobs,
    superseded_account_jobs,
    prior_account_jobs,
    source_submission["output_release"],
    source_receipt["output_release"],
    plan["supersedes_submission_release"],
    prior_submission_release,
)
PY
)
for job_id in "${SOURCE_BASELINE_JOB}" "${SOURCE_ENDPOINT_JOB}" \
  "${SOURCE_REMAINING_JOB}" "${SOURCE_ANALYSIS_JOB}"; do
  [[ "${job_id}" =~ ^[0-9]+$ ]]
done

ENVIRONMENT_RECEIPT="${EXPERIMENT_DIR}/stage_receipts/environment.json"
common_exports="ALL,ARTIFACT_ROOT=${ARTIFACT_ROOT},RELEASE_DIR=${RELEASE_DIR},RELEASE_SHA=${RELEASE_SHA},VENV_DIR=${VENV_DIR},HF_HOME=${HF_HOME},UV_BIN=${UV_BIN},MAX_ATTEMPTS=3,EXPERIMENT_DIR=${EXPERIMENT_DIR},RUN_CONTRACT_PATH=${RUN_CONTRACT_PATH},SOURCE_EXPERIMENT_DIR=${SOURCE_EXPERIMENT_DIR},EXPORT_DIR=${EXPORT_DIR},ENVIRONMENT_RECEIPT=${ENVIRONMENT_RECEIPT}"
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
METRIC_PREP_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --export="${common_exports}" --output="${LOG_DIR}/metric_prep_%j.out" \
  --error="${LOG_DIR}/metric_prep_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_metric_prep.sbatch")"
DOCTOR_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/gpu_doctor_%j.out" --error="${LOG_DIR}/gpu_doctor_%j.err" \
  "${RELEASE_DIR}/slurm/exploratory_gpu_doctor.sbatch")"
baseline_gate_exports="${common_exports},RECOVERY_PROFILE=baseline"
BASELINE_GATE_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --export="${baseline_gate_exports}" --output="${LOG_DIR}/source_baseline_gate_%j.out" \
  --error="${LOG_DIR}/source_baseline_gate_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_source_gate.sbatch")"
ENDPOINT_GATE_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --array="${ENDPOINT_INDICES}%3" --export="${common_exports}" \
  --output="${LOG_DIR}/source_endpoint_gate_%A_%a.out" \
  --error="${LOG_DIR}/source_endpoint_gate_%A_%a.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_source_gate.sbatch")"
REMAINING_GATE_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --array="${REMAINING_INDICES}%42" --export="${common_exports}" \
  --output="${LOG_DIR}/source_remaining_gate_%A_%a.out" \
  --error="${LOG_DIR}/source_remaining_gate_%A_%a.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_source_gate.sbatch")"
ANALYSIS_GATE_JOB="$(submit_job --dependency="afterok:${BOOTSTRAP_JOB}" \
  --export="${common_exports}" --output="${LOG_DIR}/source_analysis_gate_%j.out" \
  --error="${LOG_DIR}/source_analysis_gate_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_analysis_gate.sbatch")"
SMOKE_JOB="$(submit_job \
  --dependency="afterok:${METRIC_PREP_JOB}:${DOCTOR_JOB}:${ENDPOINT_GATE_JOB}" \
  "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/smoke_%j.out" --error="${LOG_DIR}/smoke_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_smoke.sbatch")"
baseline_exports="${common_exports},RECOVERY_PROFILE=baseline"
BASELINE_JOB="$(submit_job --dependency="afterok:${SMOKE_JOB}:${BASELINE_GATE_JOB}" \
  "${gpu_options[@]}" --export="${baseline_exports}" \
  --output="${LOG_DIR}/baseline_%j.out" --error="${LOG_DIR}/baseline_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery.sbatch")"
ENDPOINT_JOB="$(submit_job \
  --dependency="aftercorr:${ENDPOINT_GATE_JOB},afterok:${SMOKE_JOB}" \
  --array="${ENDPOINT_INDICES}%3" "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/endpoint_%A_%a.out" --error="${LOG_DIR}/endpoint_%A_%a.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery.sbatch")"
REMAINING_JOB="$(submit_job \
  --dependency="aftercorr:${REMAINING_GATE_JOB},afterok:${SMOKE_JOB}" \
  --array="${REMAINING_INDICES}%42" "${gpu_options[@]}" --export="${common_exports}" \
  --output="${LOG_DIR}/remaining_%A_%a.out" --error="${LOG_DIR}/remaining_%A_%a.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery.sbatch")"
account_job_ids=(
  "${SOURCE_ACCOUNT_JOB_IDS}"
  "${SUPERSEDED_ACCOUNT_JOB_IDS}"
)
if [[ "${PRIOR_RECOVERY_ACCOUNT_JOB_IDS}" != "none" ]]; then
  account_job_ids+=("${PRIOR_RECOVERY_ACCOUNT_JOB_IDS}")
fi
account_job_ids+=(
  "${BOOTSTRAP_JOB}" "${METRIC_PREP_JOB}" "${DOCTOR_JOB}" "${BASELINE_GATE_JOB}"
  "${ENDPOINT_GATE_JOB}" "${REMAINING_GATE_JOB}" "${ANALYSIS_GATE_JOB}"
  "${SMOKE_JOB}" "${BASELINE_JOB}" "${ENDPOINT_JOB}" "${REMAINING_JOB}"
)
ACCOUNT_JOB_IDS="$(IFS=,; printf '%s' "${account_job_ids[*]}")"
final_exports="${common_exports},ACCOUNT_JOB_IDS=${ACCOUNT_JOB_IDS}"
FINALIZE_JOB="$(submit_job \
  --dependency="afterok:${BASELINE_JOB}:${ENDPOINT_JOB}:${REMAINING_JOB}:${ANALYSIS_GATE_JOB}" \
  --export="${final_exports}" --output="${LOG_DIR}/finalize_%j.out" \
  --error="${LOG_DIR}/finalize_%j.err" \
  "${RELEASE_DIR}/slurm/robustness_recovery_finalize.sbatch")"

submission_inputs=(
  --input-producer-release "pace-code-release:$(sha256sum "${RELEASE_RECEIPT}" | cut -d' ' -f1)"
  --input-producer-release "${SOURCE_SUBMISSION_RELEASE}"
  --input-producer-release "${RECOVERY_SOURCE_RELEASE}"
  --input-producer-release "${SUPERSEDED_SUBMISSION_RELEASE}"
)
if [[ "${PRIOR_RECOVERY_SUBMISSION_RELEASE}" != "none" ]]; then
  submission_inputs+=(
    --input-producer-release "${PRIOR_RECOVERY_SUBMISSION_RELEASE}"
  )
fi
python3 "${RELEASE_DIR}/ice_pipeline.py" submission-receipt \
  --contract "${RUN_CONTRACT_PATH}" \
  --output "${EXPERIMENT_DIR}/stage_receipts/submission.json" \
  "${submission_inputs[@]}" \
  --pre-side-effect-release-gate submit_robustness_recovery.sh \
  --job "bootstrap=${BOOTSTRAP_JOB}" \
  --job "code_eval_metric_prep=${METRIC_PREP_JOB}" \
  --job "gpu_doctor=${DOCTOR_JOB}" \
  --job "baseline_source_gate=${BASELINE_GATE_JOB}" \
  --job "endpoint_source_gate=${ENDPOINT_GATE_JOB}" \
  --job "remaining_source_gate=${REMAINING_GATE_JOB}" \
  --job "analysis_source_gate=${ANALYSIS_GATE_JOB}" \
  --job "smoke=${SMOKE_JOB}" --job "baseline=${BASELINE_JOB}" \
  --job "endpoint_profiles=${ENDPOINT_JOB}" \
  --job "remaining_profiles=${REMAINING_JOB}" --job "finalize=${FINALIZE_JOB}"

trap - ERR
printf 'run_id=%s\nexperiment_dir=%s\nexport_dir=%s\nsmoke_job=%s\nfinalize_job=%s\n' \
  "${RUN_ID}" "${EXPERIMENT_DIR}" "${EXPORT_DIR}" "${SMOKE_JOB}" "${FINALIZE_JOB}"

#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the initialized experiment directory}"
if [[ "${ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
  echo "Set ALLOW_CODE_EXECUTION=1 only after arranging disposable, network-isolated execution." >&2
  exit 2
fi

ARTIFACT_DIR="${ARTIFACT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MAX_PARALLEL="${MAX_PARALLEL:-16}"
DESIGN_PATH="${EXPERIMENT_DIR}/design.json"

if [[ ! -f "${DESIGN_PATH}" ]]; then
  echo "Missing ${DESIGN_PATH}; run the init command first." >&2
  exit 2
fi

ARM_COUNT=$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d["mixtures"])*len(d["seeds"]))' "${DESIGN_PATH}")
LAST_INDEX=$((ARM_COUNT - 1))
EXPORTS="ALL,EXPERIMENT_DIR=${EXPERIMENT_DIR},ARTIFACT_DIR=${ARTIFACT_DIR},ALLOW_CODE_EXECUTION=1"

PREP_JOB=$(sbatch --parsable --export="${EXPORTS}" "${ARTIFACT_DIR}/slurm/prepare.sbatch")
BASELINE_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --export="${EXPORTS}" "${ARTIFACT_DIR}/slurm/baseline.sbatch")
ARRAY_JOB=$(sbatch --parsable --dependency="afterok:${PREP_JOB}" \
  --array="0-${LAST_INDEX}%${MAX_PARALLEL}" --export="${EXPORTS}" \
  "${ARTIFACT_DIR}/slurm/run_array.sbatch")
ANALYSIS_JOB=$(sbatch --parsable --dependency="afterok:${ARRAY_JOB}:${BASELINE_JOB}" \
  --export="${EXPORTS}" "${ARTIFACT_DIR}/slurm/analyze.sbatch")

echo "prepare job:  ${PREP_JOB}"
echo "baseline job: ${BASELINE_JOB}"
echo "array job:    ${ARRAY_JOB} (${ARM_COUNT} arms, at most ${MAX_PARALLEL} concurrent)"
echo "analysis job: ${ANALYSIS_JOB}"

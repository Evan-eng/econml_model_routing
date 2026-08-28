#!/usr/bin/env bash
set -euo pipefail

: "${EXPERIMENT_DIR:?Set EXPERIMENT_DIR to the experiment output directory}"
: "${GPUS:?Set GPUS to comma-separated CUDA IDs, for example 0,1,2,3}"
if [[ "${ALLOW_CODE_EXECUTION:-0}" != "1" ]]; then
  echo "Set ALLOW_CODE_EXECUTION=1 only inside a disposable, network-isolated environment." >&2
  exit 2
fi

ARTIFACT_DIR="${ARTIFACT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
export HF_HOME="${HF_HOME:-${EXPERIMENT_DIR}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TOKENIZERS_PARALLELISM=false

if [[ ! -f "${EXPERIMENT_DIR}/design.json" ]]; then
  python "${ARTIFACT_DIR}/capability_mixture_experiment.py" init \
    --experiment-dir "${EXPERIMENT_DIR}" \
    --preset overnight \
    --model "${MODEL}"
fi
python "${ARTIFACT_DIR}/capability_mixture_experiment.py" prepare \
  --experiment-dir "${EXPERIMENT_DIR}"
python "${ARTIFACT_DIR}/capability_mixture_experiment.py" warm-cache \
  --experiment-dir "${EXPERIMENT_DIR}"
python "${ARTIFACT_DIR}/launch_local_gpu_pool.py" \
  --experiment-dir "${EXPERIMENT_DIR}" \
  --gpus "${GPUS}" \
  --allow-generated-code-execution \
  --code-sandbox bwrap
python "${ARTIFACT_DIR}/capability_mixture_experiment.py" analyze \
  --experiment-dir "${EXPERIMENT_DIR}"

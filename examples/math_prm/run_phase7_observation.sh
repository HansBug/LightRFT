#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-25}"
PHASE7_LOG_DIR="${PHASE7_LOG_DIR:-${ROOT_DIR}/tmp/ursa_stage3/phase7_observation}"
mkdir -p "${PHASE7_LOG_DIR}"

export PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/home/ubuntu/URSA-MATH/checkpoints/URSA-8B}"
export PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B}"
export PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl}"
export EXPECTED_REWARD_LABEL="${EXPECTED_REWARD_LABEL:-math_psgrpo}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-stage3-phase7-observation}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-localhost}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-20292}"
export ENGINE_TYPE="${ENGINE_TYPE:-hf}"
if [[ "${ENGINE_TYPE}" == "hf" ]]; then
    export ENGINE_TP="${ENGINE_TP:-1}"
else
    export ENGINE_TP="${ENGINE_TP:-2}"
fi

# Phase 7 observation run:
# - full-data manifest
# - Stage 3 reward path (math_psgrpo)
# - keep Stage 3 reward semantics on full-data manifest
# - keep 8-GPU layout because the current colocated FSDP reward path is exercised in 8-way shards
# - intentionally shrink sample count and decode cap so one bounded run can still finish
export N_SAMPLES="${N_SAMPLES:-4}"
export EPISODE="${EPISODE:-1}"
export WARMUP="${WARMUP:-0.03}"
export RBS="${RBS:-8}"
export TBS="${TBS:-8}"
export MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
export KL="${KL:-0.003}"
export LR="${LR:-2e-6}"
export PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-4096}"
export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-1.0}"
export TOP_K="${TOP_K:--1}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
export MAX_SAMPLES="${MAX_SAMPLES:-8}"
export SAVE_STEPS="${SAVE_STEPS:-1}"
export MAX_CKPT_NUM="${MAX_CKPT_NUM:-1}"
export NUM_TRAJECTORIES_TO_SAVE="${NUM_TRAJECTORIES_TO_SAVE:-16}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="${PHASE7_LOG_DIR}/phase7_observation_${RUN_TAG}.log"
GPU_BEFORE="${PHASE7_LOG_DIR}/gpu_before_${RUN_TAG}.txt"
GPU_AFTER="${PHASE7_LOG_DIR}/gpu_after_${RUN_TAG}.txt"
SUMMARY_JSON="${PHASE7_LOG_DIR}/phase7_summary_${RUN_TAG}.json"
RESULT_DIR_RECORD="${PHASE7_LOG_DIR}/phase7_result_dir_${RUN_TAG}.txt"

cleanup() {
    local status=$?
    if [[ -n "${RUN_PGID:-}" ]]; then
        kill -TERM -- "-${RUN_PGID}" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-${RUN_PGID}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "${GPU_AFTER}" 2>/dev/null || true
    echo "[run_phase7_observation.sh] cleanup done, status=${status}, log=${RUN_LOG}"
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "${GPU_BEFORE}"

echo "[run_phase7_observation.sh] timeout=${TIMEOUT_MINUTES}m"
echo "[run_phase7_observation.sh] log=${RUN_LOG}"
echo "[run_phase7_observation.sh] dataset=${PATH_TO_YOUR_MATH_DATASET}"
echo "[run_phase7_observation.sh] reward_label=${EXPECTED_REWARD_LABEL}"
echo "[run_phase7_observation.sh] experiment=${EXPERIMENT_NAME}"

if [[ ! -f "${PATH_TO_YOUR_MATH_DATASET}" ]]; then
    if [[ "${PATH_TO_YOUR_MATH_DATASET}" == "/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl" ]]; then
        echo "[run_phase7_observation.sh] default math_psgrpo manifest missing, generating it now"
        python examples/math_prm/prepare_ursa_stage3_manifest.py \
            --output-path "${PATH_TO_YOUR_MATH_DATASET}" \
            --summary-path "/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.summary.json"
    else
        echo "[run_phase7_observation.sh] dataset not found: ${PATH_TO_YOUR_MATH_DATASET}" >&2
        exit 1
    fi
fi

setsid bash -lc "cd '${ROOT_DIR}' && timeout --signal=TERM --kill-after=30s ${TIMEOUT_MINUTES}m bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh" > "${RUN_LOG}" 2>&1 &
RUN_PID=$!
RUN_PGID=$RUN_PID

if wait "${RUN_PID}"; then
    RUN_STATUS=0
else
    RUN_STATUS=$?
fi

if [[ "${RUN_STATUS}" -ne 0 && "${RUN_STATUS}" -ne 124 && "${RUN_STATUS}" -ne 137 ]]; then
    echo "[run_phase7_observation.sh] training run failed, status=${RUN_STATUS}" >&2
    exit "${RUN_STATUS}"
fi

LATEST_RESULT_DIR="$(find "results/${EXPERIMENT_NAME}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sort | tail -n 1)"
if [[ -z "${LATEST_RESULT_DIR}" ]]; then
    echo "[run_phase7_observation.sh] could not find result directory under results/${EXPERIMENT_NAME}" >&2
    exit 1
fi

echo "${LATEST_RESULT_DIR}" > "${RESULT_DIR_RECORD}"
echo "[run_phase7_observation.sh] analyzing ${LATEST_RESULT_DIR}"

python examples/math_prm/analyze_phase7_observation.py \
    --results-dir "${LATEST_RESULT_DIR}" \
    --log-path "${RUN_LOG}" \
    --dataset-path "${PATH_TO_YOUR_MATH_DATASET}" \
    --rm-path "${PATH_TO_URSA_RM}" \
    --image-scan-limit 128 \
    --multimodal-check-samples 2 \
    --output-json "${SUMMARY_JSON}"

echo "[run_phase7_observation.sh] summary_json=${SUMMARY_JSON}"

if [[ "${RUN_STATUS}" -eq 124 || "${RUN_STATUS}" -eq 137 ]]; then
    echo "[run_phase7_observation.sh] timeout reached after ${TIMEOUT_MINUTES} minutes"
    exit 0
fi

exit 0

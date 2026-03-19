#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

TIMEOUT_MINUTES="${TIMEOUT_MINUTES:-20}"
SMOKE_LOG_DIR="${SMOKE_LOG_DIR:-${ROOT_DIR}/tmp/ursa_stage3/phase3_smoke}"
mkdir -p "${SMOKE_LOG_DIR}"

export PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/home/ubuntu/URSA-MATH/checkpoints/URSA-8B}"
export PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B}"
export PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl}"

export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-math-prm-phase3-smoke}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-localhost}"
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-20192}"
export ENGINE_TYPE="${ENGINE_TYPE:-hf}"
if [[ "${ENGINE_TYPE}" == "hf" ]]; then
    export ENGINE_TP="${ENGINE_TP:-1}"
else
    export ENGINE_TP="${ENGINE_TP:-2}"
fi
export EVAL_SPLIT="${EVAL_SPLIT:-}"

export N_SAMPLES="${N_SAMPLES:-2}"
export EPISODE="${EPISODE:-1}"
export WARMUP="${WARMUP:-0.0}"
export RBS="${RBS:-8}"
export TBS="${TBS:-8}"
export MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-1}"
export MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-1}"
export KL="${KL:-0.003}"
export LR="${LR:-2e-6}"
export PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-4096}"
export GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-1024}"
export MAX_SAMPLES="${MAX_SAMPLES:-16}"
export SAVE_STEPS="${SAVE_STEPS:-999999}"
export MAX_CKPT_NUM="${MAX_CKPT_NUM:-1}"

RUN_LOG="${SMOKE_LOG_DIR}/phase3_smoke_$(date +%Y%m%d_%H%M%S).log"
GPU_BEFORE="${SMOKE_LOG_DIR}/gpu_before_$(date +%Y%m%d_%H%M%S).txt"
GPU_AFTER="${SMOKE_LOG_DIR}/gpu_after_$(date +%Y%m%d_%H%M%S).txt"

cleanup() {
    local status=$?
    if [[ -n "${RUN_PGID:-}" ]]; then
        kill -TERM -- "-${RUN_PGID}" 2>/dev/null || true
        sleep 5
        kill -KILL -- "-${RUN_PGID}" 2>/dev/null || true
    fi
    nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "${GPU_AFTER}" 2>/dev/null || true
    echo "[run_phase3_smoke.sh] cleanup done, status=${status}, log=${RUN_LOG}"
}
trap cleanup EXIT INT TERM

nvidia-smi --query-gpu=index,name,memory.used,memory.free --format=csv,noheader > "${GPU_BEFORE}"

echo "[run_phase3_smoke.sh] timeout=${TIMEOUT_MINUTES}m"
echo "[run_phase3_smoke.sh] log=${RUN_LOG}"
echo "[run_phase3_smoke.sh] dataset=${PATH_TO_YOUR_MATH_DATASET}"
echo "[run_phase3_smoke.sh] engine_type=${ENGINE_TYPE}"
echo "[run_phase3_smoke.sh] eval_split=${EVAL_SPLIT:-<disabled>}"

setsid bash -lc "cd '${ROOT_DIR}' && timeout --signal=TERM --kill-after=30s ${TIMEOUT_MINUTES}m bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh" > "${RUN_LOG}" 2>&1 &
RUN_PID=$!
RUN_PGID=$RUN_PID

wait "${RUN_PID}"
RUN_STATUS=$?

if [[ "${RUN_STATUS}" -eq 124 || "${RUN_STATUS}" -eq 137 ]]; then
    echo "[run_phase3_smoke.sh] timeout reached after ${TIMEOUT_MINUTES} minutes"
    exit 0
fi

exit "${RUN_STATUS}"

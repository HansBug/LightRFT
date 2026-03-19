#!/bin/bash
#
# LightRFT GRPO Training Script – URSA-8B with URSA-8B-RM (Math PRM)
#
# Trains URSA-8B (multimodal math VLM) with URSA-8B-RM as the Process Reward Model.
# This is the stage3 training from URSA-MATH, migrated to LightRFT framework.
#
# Key features:
#   - Actor: URSA-8B (hybrid vision tower + Qwen2.5-Math-Instruct)
#   - Reward: URSA-8B-RM (process reward model for step-level scoring)
#   - Algorithm: Phase 4 GRPO with PS-GRPO reward via math_psgrpo label
#   - Dataset: converted MMathCoT-1M Stage 3 manifest
#
# Step-scoring protocol (see MathPRMReward in reward_models.py):
#   1. The actor generates a chain-of-thought response.
#   2. The response is formatted with "Step N:" headings and "†Answer:" prefix.
#   3. Each step boundary is marked with Cyrillic ' и' (U+0438) token.
#   4. A single forward pass through URSA-8B-RM yields per-step probabilities.
#   5. In Phase 4, MathPRMReward maps step scores + correctness to PS-GRPO reward.
#

################################################################################
#                         Part 1: User Configuration                           #
# Update paths and keys to match your environment before running.              #
################################################################################

# --- Actor (policy) model ---
# URSA-8B: A multimodal math VLM with hybrid vision tower (SAM-B + SigLIP-L) + Qwen2.5-Math-Instruct
# This is the output from URSA-MATH stage1 training.
PATH_TO_YOUR_BASE_MODEL="${PATH_TO_YOUR_BASE_MODEL:-/home/ubuntu/URSA-MATH/checkpoints/URSA-8B}"
# Example HuggingFace name (verify the exact repo name before use):
# PATH_TO_YOUR_BASE_MODEL="AI-MO/URSA-8B"

# --- Reward model ---
# URSA-8B-RM: a step-level Process Reward Model for mathematical reasoning.
# Set to your local copy or a HuggingFace model name.
PATH_TO_URSA_RM="${PATH_TO_URSA_RM:-/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B}"
# Example HuggingFace name (verify the exact repo name before use):
# PATH_TO_URSA_RM="AI-MO/URSA-8B-RM"

# --- Dataset ---
# MMathCoT-1M subset (15K samples) for stage3 training.
# Dataset format:
#   "prompt"  : the math question (string, may include images)
#   "images"  : list of image paths (optional, for multimodal problems)
#   "label"   : "math_psgrpo" → triggers Phase 4 PS-GRPO reward
#               "math_prm"  →  Phase 3 baseline PRM-only reward
#               "math_prm_combined" → PRM + rule-based accuracy
#   "reference": ground-truth answer string (optional, for rule-based component)
# See examples/data_preprocess/ for preprocessing helpers.
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl}"
EXPECTED_REWARD_LABEL="${EXPECTED_REWARD_LABEL:-math_psgrpo}"

# --- Experiment metadata ---
EXPERIMENT_NAME="${EXPERIMENT_NAME:-lightrft-ursa8b-math-prm-grpo-phase4}"

# --- W&B ---
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-LightRFT-URSA8B-Stage3}"


################################################################################
#                       Part 2: Training Hyperparameters                       #
################################################################################

# --- GRPO (Phase 4: reward = PS-GRPO over PRM step scores + correctness) ---
N_SAMPLES="${N_SAMPLES:-8}"           # Responses per prompt (must be > 1 for group_norm).
EPISODE="${EPISODE:-20}"              # Total training episodes.
WARMUP="${WARMUP:-0.03}"              # LR warmup ratio.

# --- Batch sizes ---
RBS="${RBS:-128}"                     # Rollout batch size (total across all GPUs).
TBS="${TBS:-128}"                     # Training batch size.
MICRO_TRAIN_BATCH_SIZE="${MICRO_TRAIN_BATCH_SIZE:-4}"
MICRO_ROLLOUT_BATCH_SIZE="${MICRO_ROLLOUT_BATCH_SIZE:-8}"

# --- Optimisation ---
KL="${KL:-0.01}"                      # KL divergence coefficient (higher than text-only).
LR="${LR:-1e-6}"                      # Actor learning rate.
MAX_LENGTH="${MAX_LENGTH:-4096}"      # Max total sequence length (prompt + generation).
PROMPT_MAX_LEN="${PROMPT_MAX_LEN:-1024}"   # Max prompt length.
GENERATE_MAX_LEN="${GENERATE_MAX_LEN:-3072}" # Max generation length (leave room for CoT).
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
TEMPERATURE="${TEMPERATURE:-1.0}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
NO_REPEAT_NGRAM_SIZE="${NO_REPEAT_NGRAM_SIZE:-0}"
MAX_SAMPLES="${MAX_SAMPLES:-1000000}"
SAVE_STEPS="${SAVE_STEPS:-20}"
MAX_CKPT_NUM="${MAX_CKPT_NUM:-2}"

# --- Multi-modal Settings ---
limit_mm_image_per_prompt="${limit_mm_image_per_prompt:-10}"  # Max number of images per prompt.


################################################################################
#                    Part 3: Distributed Training Setup                        #
################################################################################

export MLP_WORKER_NUM="${MLP_WORKER_NUM:-1}"               # Number of nodes.
export MLP_WORKER_GPU="${MLP_WORKER_GPU:-8}"               # GPUs per node.
export MLP_ROLE_INDEX="${MLP_ROLE_INDEX:-0}"               # Rank of this node.
export MLP_WORKER_0_HOST="${MLP_WORKER_0_HOST:-localhost}"  # Master node IP.
export MLP_WORKER_0_PORT="${MLP_WORKER_0_PORT:-20092}"        # Master node port.

export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# vLLM/SGLang tensor-parallelism for the *actor* inference engine.
# URSA-8B (8B params + vision towers) requires TP for efficient inference.
# URSA-8B-RM (8B params) runs on a single GPU; this controls the actor engine.
ENGINE_TYPE="${ENGINE_TYPE:-hf}"
if [[ "${ENGINE_TYPE}" == "hf" ]]; then
    ENGINE_TP="${ENGINE_TP:-1}"
else
    ENGINE_TP="${ENGINE_TP:-2}"
fi
EVAL_SPLIT="${EVAL_SPLIT:-}"
USE_URSA_ENGINE_WRAPPER="${USE_URSA_ENGINE_WRAPPER:-1}"
URSA_ENGINE_CHECKPOINT_DIR="${URSA_ENGINE_CHECKPOINT_DIR:-/data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with \"Step N:\" (e.g. \"Step 1:\", \"Step 2:\") on its own line. After all steps, output exactly one final answer line prefixed with \"†Answer:\" (e.g. \"†Answer: 42\"). Stop immediately after the \"†Answer:\" line and do not output any extra text, repeated answer markers, or additional steps.}"


################################################################################
#                      Part 4: Execution and Logging                           #
################################################################################

current_time=$(date +"%Y%m%d_%H%M%S")
SAVE_MODEL_NAME="${EXPERIMENT_NAME}-ep${EPISODE}-kl${KL}-lr${LR}-${current_time}"
WANDB_RUN_NAME="${EXPERIMENT_NAME}-${current_time}"

mkdir -p "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}"
mkdir -p "rft_logs/${EXPERIMENT_NAME}"

export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export NCCL_DEBUG="WARN"
export IGNORE_EOS=0
export WANDB_MODE="${WANDB_MODE:-offline}"   # Set to "online" for real-time W&B logging.

python - <<'PY'
import json
import os
from pathlib import Path

dataset_path = Path(os.environ["PATH_TO_YOUR_MATH_DATASET"])
expected_label = os.environ["EXPECTED_REWARD_LABEL"]
if not dataset_path.exists():
    raise SystemExit(f"[run_grpo_math_prm_ursa_8b.sh] Dataset not found: {dataset_path}")

seen = set()
with dataset_path.open("r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if idx >= 128:
            break
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        seen.add(record.get("label"))

if seen != {expected_label}:
    raise SystemExit(
        "[run_grpo_math_prm_ursa_8b.sh] Expected dataset label "
        f"{expected_label!r}, but sampled labels were {sorted(seen)!r}. "
        "Rebuild the manifest with examples/math_prm/prepare_ursa_stage3_manifest.py "
        "or override EXPECTED_REWARD_LABEL if you intentionally want another reward path."
    )
print(
    "[run_grpo_math_prm_ursa_8b.sh] Dataset label check passed: "
    f"{expected_label!r} from {dataset_path}"
)
PY

# JSON config passed to --reward_pretrain.
# Format: '{"<type>": "<path>"}' where <type> must match a RewardModelType value.
# URSA-8B-RM is a text-only HF model → engine mode NOT recommended for PRM
# (requires logit access).  The builder in reward_models_utils.py ignores
# use_engine for math_prm/math_psgrpo and loads via HF directly.
REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

WANDB_ARGS=()
if [[ -n "${WANDB_API_KEY}" && "${WANDB_API_KEY}" != "YOUR_WANDB_API_KEY" ]]; then
    WANDB_ARGS=(
        --use_wandb "${WANDB_API_KEY}"
        --wandb_project "${WANDB_PROJECT}"
        --wandb_run_name "${WANDB_RUN_NAME}"
    )
else
    echo "[run_grpo_math_prm_ursa_8b.sh] WANDB disabled for this run."
fi

EVAL_ARGS=()
if [[ -n "${EVAL_SPLIT}" ]]; then
    EVAL_ARGS=(
        --eval_split "${EVAL_SPLIT}"
    )
else
    echo "[run_grpo_math_prm_ursa_8b.sh] Eval split disabled for this run."
fi

if [[ "${ENGINE_TYPE}" != "hf" && "${USE_URSA_ENGINE_WRAPPER}" == "1" && -d "${PATH_TO_YOUR_BASE_MODEL}" ]]; then
    echo "[run_grpo_math_prm_ursa_8b.sh] Preparing URSA engine wrapper checkpoint at ${URSA_ENGINE_CHECKPOINT_DIR}"
    PATH_TO_YOUR_BASE_MODEL="$(
        python examples/math_prm/prepare_ursa_engine_checkpoint.py \
            --source-model-path "${PATH_TO_YOUR_BASE_MODEL}" \
            --output-path "${URSA_ENGINE_CHECKPOINT_DIR}"
    )"
    echo "[run_grpo_math_prm_ursa_8b.sh] Using wrapped URSA checkpoint: ${PATH_TO_YOUR_BASE_MODEL}"
fi

set -x


################################################################################
#                         Part 5: Main Training Command                        #
################################################################################

torchrun \
    --nnodes $NNODES \
    --nproc-per-node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master-port $MASTER_PORT \
    --master-addr $MASTER_ADDR \
    examples/math_prm/train_colocate.py \
    --pretrain "${PATH_TO_YOUR_BASE_MODEL}" \
    --mixed_mm_data \
    --save_trajectories \
    --num_trajectories_to_save 16 \
    --print_replay_buffer_stats \
    --loss_agg_mode "seq-mean-token-mean" \
    --fsdp \
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size ${MICRO_TRAIN_BATCH_SIZE} \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size ${MICRO_ROLLOUT_BATCH_SIZE} \
    --rollout_batch_size ${RBS} \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --top_k $TOP_K \
    --repetition_penalty $REPETITION_PENALTY \
    --no_repeat_ngram_size $NO_REPEAT_NGRAM_SIZE \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --use_kl_loss \
    --init_kl_coef $KL \
    --kl_estimator "k3" \
    --prompt_data "${PATH_TO_YOUR_MATH_DATASET}" \
    --max_samples ${MAX_SAMPLES} \
    --input_key "prompt" \
    --images_key "images" \
    --label_key "label" \
    --apply_chat_template \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps ${SAVE_STEPS} \
    --max_ckpt_num ${MAX_CKPT_NUM} \
    --rm_use_engine \
    --engine_type "${ENGINE_TYPE}" \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --enable_engine_sleep \
    --system_prompt "${SYSTEM_PROMPT}" \
    --l2 1.0e-2 \
    --freeze_prefix \
    --adam_offload \
    --limit_mm_image_per_prompt $limit_mm_image_per_prompt \
    "${EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# This script migrates URSA-MATH stage3 training to LightRFT framework.       #
#                                                                              #
# Step 1: Prepare URSA-8B model                                                #
#   - Download or train URSA-8B (stage1 output from URSA-MATH)                #
#   - Model structure: Hybrid vision tower (SAM-B + SigLIP-L) + Qwen2.5-Math  #
#   - Set PATH_TO_YOUR_BASE_MODEL to the model directory                       #
#                                                                              #
# Step 2: Prepare URSA-8B-RM reward model                                      #
#   - Download or train URSA-8B-RM (stage2 output from URSA-MATH)             #
#   - This is a UrsaForTokenClassification model for step-level scoring       #
#   - Set PATH_TO_URSA_RM to the model directory                               #
#                                                                              #
# Step 3: Prepare MMathCoT-1M stage3 dataset                                   #
#   - For the current machine, the default path points to the converted full   #
#     Phase 1 manifest under /data/LightRFT/tmp/ursa_stage3/                  #
#   - Dataset format (JSON/JSONL):                                             #
#     {                                                                        #
#       "prompt": "math question text",                                       #
#       "images": ["path/to/image1.jpg", ...],  # optional                    #
#       "label": "math_prm",                    # or "math_prm_combined"      #
#       "reference": "ground truth answer"      # optional                    #
#     }                                                                        #
#   - Set PATH_TO_YOUR_MATH_DATASET to the dataset directory                  #
#                                                                              #
# Step 4: Configure training hyperparameters (Part 2)                          #
#   - Phase 3 baseline uses math_prm only: reward = min(step_scores)          #
#   - PS-GRPO drop-moment logic is NOT part of this script yet                #
#   - You can override all key hyperparameters and paths via environment vars #
#                                                                              #
# Step 5: Run training                                                         #
#   bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh                       #
#                                                                              #
# Key differences from URSA-MATH original implementation:                      #
#   - Uses LightRFT's FSDP/DeepSpeed training infrastructure                  #
#   - Integrates with vLLM/SGLang-compatible rollout engines                  #
#   - Co-locates reward model with actor for memory efficiency                #
#   - All URSA model code is self-contained in examples/math_prm/ursa_model/  #
#                                                                              #
# Response format (enforced by system_prompt):                                 #
#   Step 1: <reasoning>                                                        #
#   Step 2: <reasoning>                                                        #
#   ...                                                                        #
#   †Answer: <final answer>                                                    #
#                                                                              #
# URSA-8B-RM scoring protocol (Phase 3 baseline):                              #
#   - Scans for "Step N:" headings in the response                            #
#   - Inserts Cyrillic ' и' (U+0438) marker at each step boundary            #
#   - Single forward pass yields per-step probabilities                       #
#   - Minimum step score used as final sequence reward                        #
#                                                                              #
# Ablations / variants:                                                        #
#   - label="math_prm": PRM-only reward (Phase 3 baseline)                    #
#   - label="math_prm_combined": PRM + rule-based accuracy ablation           #
#   - Adjust aggregation in reward_models.py MathPRMReward:                   #
#       "min"  – most conservative (default, PS-GRPO)                         #
#       "avg"  – softer, less sensitive to single bad step                    #
#       "last" – only final step score (similar to ORM)                       #
#                                                                              #
################################################################################

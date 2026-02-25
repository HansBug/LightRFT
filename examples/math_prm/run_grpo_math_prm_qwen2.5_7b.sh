#!/bin/bash
#
# LightRFT GRPO Training Script – Math PRM (URSA-8B-RM)
#
# Trains a Qwen2.5-7B-Instruct policy with a Process Reward Model (PRM)
# as the reward signal.  The PRM (URSA-8B-RM) scores each reasoning step
# of the model's chain-of-thought rather than only the final answer.
#
# Key differences from the ORM-based run_svkng_fsdp_qwenvl.sh:
#   - reward_pretrain points to a PRM (URSA-8B-RM) instead of ORM models.
#   - Dataset labels should be "math_prm" (or "math_prm_combined") to trigger
#     the PRM recipe in reward_models_utils.py.
#   - URSA-8B-RM is a text-only causal LM; engine mode is disabled for it
#     because step-level scoring requires direct logit access.
#   - --text_only flag is set since math data is purely textual.
#
# Step-scoring protocol (see MathPRMReward in reward_models.py):
#   1. The actor generates a chain-of-thought response.
#   2. The response is split by "\n\n" into reasoning steps.
#   3. Each step is appended with the Cyrillic tag "ки" (used by URSA-MATH).
#   4. A single forward pass through URSA-8B-RM yields per-step probabilities.
#   5. The minimum step score is used as the final sequence reward.
#

################################################################################
#                         Part 1: User Configuration                           #
# Update paths and keys to match your environment before running.              #
################################################################################

# --- Actor (policy) model ---
# A text-only Qwen2.5 chat / instruct model.
PATH_TO_YOUR_BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
# PATH_TO_YOUR_BASE_MODEL="/path/to/local/Qwen2.5-7B-Instruct"

# --- Reward model ---
# URSA-8B-RM: a step-level Process Reward Model for mathematical reasoning.
# Set to your local copy or a HuggingFace model name.
PATH_TO_URSA_RM="/path/to/URSA-8B-RM"
# Example HuggingFace name (verify the exact repo name before use):
# PATH_TO_URSA_RM="AI-MO/URSA-8B-RM"

# --- Dataset ---
# A math reasoning dataset whose samples have:
#   "prompt"  : the question (string)
#   "label"   : "math_prm"  →  triggers PRM-only reward
#               "math_prm_combined" → PRM + rule-based accuracy
#               "math_rule" → rule-only baseline (no PRM, good for ablation)
#   "label"   (optional): ground-truth answer string for rule-based component
# See examples/data_preprocess/ for preprocessing helpers.
PATH_TO_YOUR_MATH_DATASET="/path/to/your/math_prm_dataset"

# --- Experiment metadata ---
EXPERIMENT_NAME="lightrft-math-prm-grpo"

# --- W&B ---
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_PROJECT="LightRFT-MathPRM-Experiments"


################################################################################
#                       Part 2: Training Hyperparameters                       #
################################################################################

# --- GRPO ---
N_SAMPLES=8           # Responses per prompt (must be > 1 for group_norm).
EPISODE=10            # Total training episodes.
WARMUP=0.03           # LR warmup ratio.

# --- Batch sizes ---
RBS=128               # Rollout batch size (total across all GPUs).
TBS=128               # Training batch size.

# --- Optimisation ---
KL=0.001              # KL divergence coefficient.
LR=1e-6               # Actor learning rate.
MAX_LENGTH=4096       # Max total sequence length (prompt + generation).
PROMPT_MAX_LEN=1024   # Max prompt length.
GENERATE_MAX_LEN=3072 # Max generation length (leave room for CoT).


################################################################################
#                    Part 3: Distributed Training Setup                        #
################################################################################

export MLP_WORKER_NUM=1               # Number of nodes.
export MLP_WORKER_GPU=8               # GPUs per node.
export MLP_ROLE_INDEX=0               # Rank of this node.
export MLP_WORKER_0_HOST="localhost"  # Master node IP.
export MLP_WORKER_0_PORT=20092        # Master node port.

export MASTER_ADDR=$MLP_WORKER_0_HOST
export MASTER_PORT=$MLP_WORKER_0_PORT
export NNODES=$MLP_WORKER_NUM
export NODE_RANK=$MLP_ROLE_INDEX
export GPUS_PER_NODE=$MLP_WORKER_GPU

# vLLM/SGLang tensor-parallelism for the *actor* inference engine.
# URSA-8B-RM (8B params) runs on a single GPU; this controls the actor engine.
ENGINE_TP=2


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
export WANDB_MODE="offline"   # Set to "online" for real-time W&B logging.

# JSON config passed to --reward_pretrain.
# Format: '{"<type>": "<path>"}' where <type> must match a RewardModelType value.
# URSA-8B-RM is a text-only HF model → engine mode NOT recommended for PRM
# (requires logit access).  The builder in reward_models_utils.py ignores
# use_engine for math_prm and loads via HF directly.
REWARD_PRETRAIN_PATHS="{\"math_prm\":\"${PATH_TO_URSA_RM}\"}"

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
    --text_only \
    --save_trajectories \
    --num_trajectories_to_save 16 \
    --print_replay_buffer_stats \
    --loss_agg_mode "seq-mean-token-mean" \
    --fsdp \
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --save_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --ckpt_path "results/${EXPERIMENT_NAME}/${SAVE_MODEL_NAME}" \
    --micro_train_batch_size 4 \
    --train_batch_size ${TBS} \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size ${RBS} \
    --advantage_estimator "group_norm" \
    --max_epochs 1 \
    --num_episodes ${EPISODE} \
    --lr_warmup_ratio ${WARMUP} \
    --n_samples_per_prompt $N_SAMPLES \
    --prompt_max_len $PROMPT_MAX_LEN \
    --generate_max_len $GENERATE_MAX_LEN \
    --zero_stage 3 \
    --bf16 \
    --actor_learning_rate $LR \
    --use_kl_loss \
    --init_kl_coef $KL \
    --kl_estimator "k3" \
    --prompt_data "${PATH_TO_YOUR_MATH_DATASET}" \
    --input_key "prompt" \
    --label_key "label" \
    --apply_chat_template \
    --flash_attn \
    --gradient_checkpointing \
    --save_steps 20 \
    --max_ckpt_num 2 \
    --rm_use_engine \
    --engine_type sglang \
    --engine_mem_util 0.6 \
    --engine_tp_size $ENGINE_TP \
    --enable_engine_sleep \
    --system_prompt 'A conversation between the User and Assistant. The User asks a math question, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, the final answer MUST be on its own line prefixed with "†Answer:" (e.g. "†Answer: 42"). Example format: Step 1: ... Step 2: ... †Answer: .... This structured format is required for the process reward model to score each step.' \
    --l2 1.0e-2 \
    --adam_offload \
    --use_wandb "${WANDB_API_KEY}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_run_name "${WANDB_RUN_NAME}" \
    2>&1 | tee "rft_logs/${EXPERIMENT_NAME}/node${NODE_RANK}_${current_time}.log"


################################################################################
#                           Usage Instructions                                 #
#                                                                              #
# Step 1: Prepare your math dataset                                            #
#   The dataset must have at least two columns:                                #
#     "prompt" : the math question (string)                                    #
#     "label"  : one of "math_prm", "math_prm_combined", "math_rule"          #
#   Optionally include "reference" / "chosen" for rule-based components.      #
#                                                                              #
#   The response format enforced by the system_prompt above:                  #
#     Step 1: <reasoning>                                                      #
#     Step 2: <reasoning>                                                      #
#     ...                                                                      #
#     †Answer: <final answer>                                                  #
#                                                                              #
#   URSA-8B-RM identifies step boundaries by scanning for "Step N" headings   #
#   and inserts the Cyrillic marker ' и' (U+0438) at the end of each step.   #
#   If responses do not follow this format, the PRM returns 0.0 reward.       #
#                                                                              #
# Step 2: Configure Part 1 above                                               #
#   - PATH_TO_YOUR_BASE_MODEL : actor checkpoint                               #
#   - PATH_TO_URSA_RM         : URSA-8B-RM (or any compatible text PRM)       #
#   - PATH_TO_YOUR_MATH_DATASET : preprocessed dataset directory               #
#                                                                              #
# Step 3: Run                                                                  #
#   bash examples/math_prm/run_grpo_math_prm_qwen2.5_7b.sh                   #
#                                                                              #
# Ablations / variants:                                                        #
#   - Set label = "math_rule" to train with rule-based reward only (no PRM).  #
#   - Set label = "math_prm_combined" for PRM + rule-based accuracy.          #
#   - Adjust MathPRMReward aggregation in reward_models_utils.py:             #
#       "min"     – most conservative (default, recommended)                  #
#       "avg"     – softer, less sensitive to a single bad step               #
#       "last"    – only the final step score (close to ORM behaviour)         #
#   - The step format MUST be "Step N: ...\n†Answer: ..." for URSA-8B-RM.   #
#     The system_prompt above enforces this. Do NOT change it to \n\n format. #
#                                                                              #
################################################################################

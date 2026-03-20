# URSA-MATH Stage3 Migration to LightRFT

This document describes the migration of URSA-MATH stage3 training (PS-GRPO) to the LightRFT framework.

## Overview

URSA-MATH is a three-stage training framework for multimodal math reasoning:
- **Stage I**: Vision-language alignment + instruction fine-tuning → produces URSA-8B
- **Stage II**: Process reward model training → produces URSA-8B-RM
- **Stage III**: PS-GRPO online RL (this migration)

This migration brings stage3 training into LightRFT with all necessary components self-contained.

## Migrated Components

### 1. URSA Model Architecture (`ursa_model/`)

All URSA model code has been copied from URSA-MATH and is now self-contained:

```
examples/math_prm/ursa_model/
├── __init__.py                    # Model exports
├── modeling_ursa.py               # UrsaForConditionalGeneration, UrsaForTokenClassification
├── configuration_ursa.py          # UrsaConfig, VisionConfig, AlignerConfig
├── processing_ursa.py             # UrsaProcessor
├── image_processing_vlm.py        # VLMImageProcessor
├── clip_encoder.py                # HybridVisionTower (SAM-B + SigLIP-L)
├── projector.py                   # MlpProjector
├── sam.py                         # SAM vision encoder
└── siglip_vit.py                  # SigLIP vision encoder
```

**Key features:**
- `UrsaForConditionalGeneration`: Multimodal generation model (actor)
- `UrsaForTokenClassification`: Process reward model with per-token scoring head
- Hybrid vision tower: SAM-B (1024x1024) + SigLIP-L (384x384)
- MLP projector: Maps vision features to LLM embedding space

### 2. PRM Inference Logic (`tools/prm_infer_score.py`)

Copied from URSA-MATH inference code:
- `replace_specific_plus_minus_with_ki()`: Inserts Cyrillic ' и' (U+0438) step markers
- `single_inference()`: Extracts per-step logits and applies sigmoid
- `return_score()`: Aggregates step scores (min/avg strategies)

### 3. Reward Model Integration

**Modified files:**
- `reward_models_utils.py`: Updated `_load_ursa_prm_model()` to import from local `ursa_model/` instead of external URSA-MATH repo
- `reward_models.py`: Already contains `MathPRMReward` class that wraps URSA-8B-RM

**Key implementation details:**
- Step marker: Single Cyrillic character ' и' (U+0438), NOT 'ки'
- Response format: `Step 1: ...\nStep 2: ...\n†Answer: ...`
- Score extraction: Reads logits at ' и' token positions
- Image padding: 575 dummy tokens for vision feature alignment
- Aggregation: `min` (default, PS-GRPO), `avg`, or `last`

### 4. Training Script (`run_grpo_math_prm_ursa_8b.sh`)

New training script configured for URSA-8B:

**Key differences from Qwen2.5-7B script:**
- Actor: URSA-8B (multimodal) instead of Qwen2.5-7B-Instruct (text-only)
- Removed `--text_only` flag (enables multimodal mode)
- Added `--mixed_mm_data` flag for mixed text/image data
- Added `--images_key "images"` for image input
- Added `--freeze_prefix` to freeze vision encoder
- Added `--limit_mm_image_per_prompt 10`
- Default launcher hyperparameters now follow the explicit Stage 3 values documented in the local `URSA-MATH` repo:
  - `EPISODE=10`
  - `N_SAMPLES=8`
  - `RBS=128`, `TBS=128`
  - `MICRO_TRAIN_BATCH_SIZE=4`, `MICRO_ROLLOUT_BATCH_SIZE=4`
  - `LR=1e-6`, `KL=0.001`
  - `PROMPT_MAX_LEN=1024`, `GENERATE_MAX_LEN=3072`
- The original paper uses a one-time filtered `~15K` RL subset. Because that exact subset is not yet available locally, the launcher keeps the converted manifest path and uses `MAX_SAMPLES=15360` as a scale proxy by default.

## Usage

### Prerequisites

1. **URSA-8B model** (stage1 output)
   - Hybrid vision tower + Qwen2.5-Math-Instruct
   - Download or train from URSA-MATH stage1

2. **URSA-8B-RM reward model** (stage2 output)
   - UrsaForTokenClassification with per-token scoring
   - Download or train from URSA-MATH stage2

3. **MMathCoT-1M dataset** (paper uses a filtered ~15K Stage 3 subset)
   - Current local launcher uses the converted LightRFT manifest path and caps `MAX_SAMPLES` to `15360`
   - Format: `{"prompt": "...", "images": [...], "label": "math_psgrpo", "reference": "..."}`

### Running Training

```bash
# 1. Edit the script to set paths
vim examples/math_prm/run_grpo_math_prm_ursa_8b.sh

# Set these variables:
# - PATH_TO_YOUR_BASE_MODEL="/path/to/URSA-8B"
# - PATH_TO_URSA_RM="/path/to/URSA-8B-RM"
# - PATH_TO_YOUR_MATH_DATASET="/path/to/mmathcot_stage3_math_psgrpo.jsonl"

# 2. Run training
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

### Dataset Format

```json
{
  "prompt": "Solve the following geometry problem: ...",
  "images": ["path/to/diagram.jpg"],
  "label": "math_psgrpo",
  "reference": "42"
}
```

**Label options:**
- `"math_psgrpo"`: PS-GRPO reward (default Stage 3 path)
- `"math_prm"`: PRM-only reward baseline
- `"math_prm_combined"`: PRM + rule-based accuracy
- `"math_rule"`: Rule-only baseline (ablation)

### Response Format

The system prompt enforces this format for PRM scoring:

```
Step 1: <reasoning>
Step 2: <reasoning>
...
†Answer: <final answer>
```

**Critical:** Do NOT change to `\n\n` separated format. URSA-8B-RM requires `Step N:` headings.

## Architecture Details

### URSA-8B Model Structure

```
Input (text + images)
    ↓
Hybrid Vision Tower
├── SAM-B (1024×1024) → 1024-dim features
└── SigLIP-L (384×384) → 1024-dim features
    ↓
MLP Projector (concat → 3072-dim → 3584-dim)
    ↓
Qwen2.5-Math-Instruct (8B params)
    ↓
Output tokens
```

### URSA-8B-RM Scoring Protocol

```
Response: "Step 1: ... Step 2: ... †Answer: ..."
    ↓
Insert ' и' markers: "Step 1: ... и Step 2: ... и †Answer: ..."
    ↓
Tokenize + add 575 image padding tokens
    ↓
Forward pass through UrsaForTokenClassification
    ↓
Extract logits at ' и' positions
    ↓
Apply sigmoid → per-step scores
    ↓
Aggregate (min/avg/last) → final reward
```

## Key Differences from URSA-MATH Original

1. **Training Infrastructure**
   - URSA-MATH: Custom training loop
   - LightRFT: FSDP/DeepSpeed + SGLang inference engine

2. **Model Loading**
   - URSA-MATH: Direct imports from `models/ursa_model`
   - LightRFT: Self-contained in `examples/math_prm/ursa_model/`

3. **Reward Model Integration**
   - URSA-MATH: Standalone inference scripts
   - LightRFT: Integrated into `MathPRMReward` class with co-location

4. **Inference Engine**
   - URSA-MATH: HuggingFace generate
   - LightRFT: SGLang for 2-5x faster rollouts

## Troubleshooting

### Import Errors

If you see `ImportError: Cannot import UrsaForTokenClassification`:
- Verify `examples/math_prm/ursa_model/` exists with all files
- Check `reward_models_utils.py` imports from local `ursa_model`

### PRM Returns 0.0 Reward

If all rewards are 0.0:
- Check response format has `Step N:` headings
- Verify system prompt is not modified
- Ensure ' и' (U+0438) marker insertion works

### OOM Errors

If you run out of memory:
- Reduce `--micro_rollout_batch_size` (try 4 instead of 8)
- Reduce `--engine_tp_size` (try 1 instead of 2)
- Enable `--adam_offload` (already enabled by default)
- Reduce `--limit_mm_image_per_prompt` (try 5 instead of 10)

### Vision Encoder Not Frozen

If vision encoder trains (high memory usage):
- Verify `--freeze_prefix` flag is set
- Check logs for "Freezing parameters with prefix: visual"

## Performance Notes

**Expected improvements over vanilla GRPO:**
- PS-GRPO achieves 2.2x average improvement (6.8% vs 3.1%)
- MathVision: 5.4x improvement (9.8% vs 1.8%)

**Training speed:**
- SGLang inference: 2-5x faster than HuggingFace
- Co-located reward model: Saves GPU memory
- FSDP ZeRO-3: Efficient 8-GPU training

## References

- URSA-MATH paper: [arXiv:2501.04686](https://arxiv.org/abs/2501.04686)
- URSA-MATH repo: https://github.com/bytedance/URSA-MATH
- LightRFT repo: https://github.com/OpenRLHF/LightRFT

## License

The URSA model code is licensed under Apache 2.0 (Copyright 2025 Bytedance Ltd.).

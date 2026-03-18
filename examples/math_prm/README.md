<div align="center">

# Math PRM Training in LightRFT

URSA-MATH Stage 3 PS-GRPO training migrated into LightRFT.

</div>

## Overview

This directory is the active `math_prm` workspace for multimodal math reinforcement learning.

The current target is:

- actor: `URSA-8B`
- reward model: `URSA-RM-8B`
- algorithm: GRPO / PS-GRPO style online RL
- raw dataset: `MMathCoT-1M`
- training dataset schema: `prompt / images / reference / label`

This folder is no longer a generic SafeWork example. The scripts and docs here are for the URSA-MATH to LightRFT migration path.

## Runtime Baseline

The runtime baseline is frozen by `/data/LightRFT/Dockerfile`.

- Do not change the versions of pip packages installed there as part of normal debugging.
- Do not treat upgrading or downgrading `torch`, `deepspeed`, `vllm`, `flash_attn`, or `sglang` as routine fixes.
- Prefer solving issues in code, data conversion, and training arguments first.

## Directory Map

```text
examples/math_prm/
├── prepare_ursa_stage3_manifest.py   # Convert URSA raw jsonl into LightRFT prompt manifest
├── train_colocate.py                 # Main GRPO training entry
├── run_grpo_math_prm_ursa_8b.sh      # URSA-8B + URSA-RM-8B training launcher
├── reward_models.py                  # Reward implementations, including MathPRMReward
├── reward_models_utils.py            # Reward model loading and routing
├── prm_infer_score.py                # Step-level PRM scoring helpers
├── test_reward_models.py             # Reward-side tests
├── URSA_MIGRATION.md                 # Migration notes from URSA-MATH
└── ursa_model/                       # Self-contained URSA model code
```

## Local Resources

The current machine layout is:

```bash
URSA actor:      /home/ubuntu/URSA-MATH/checkpoints/URSA-8B
URSA reward:     /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B
MMathCoT-1M raw: /home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl
Image root:      /home/ubuntu/URSA-MATH/datasets/URSA-MATH/images
```

The converted manifest generated in Phase 1 is:

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl
```

Its summary is:

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.summary.json
```

## Dataset Preparation

### Raw Schema

URSA raw Stage 3 data is not directly consumable by `PromptDatasetVL`. The raw schema is:

```json
{
  "image_url": "...",
  "instruction": "...",
  "output": "..."
}
```

LightRFT training expects at least:

```json
{
  "prompt": "...",
  "images": ["/abs/path/to/image.png"],
  "reference": "...",
  "label": "math_prm"
}
```

### Use `prepare_ursa_stage3_manifest.py`

Run a small smoke conversion:

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

Run the full conversion with default paths:

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py
```

Important arguments:

- `--input-path`: raw `MMathCoT-1M` jsonl
- `--image-root`: root directory used to resolve `image_url`
- `--output-path`: converted manifest path
- `--summary-path`: conversion summary path
- `--label`: default `math_prm`
- `--prompt-mode question_only|instruction`: default `question_only`
- `--max-samples`: cap rows for smoke checks
- `--smoke-samples`: how many rows are fed into `PromptDatasetVL` validation

The script does three things:

1. Converts `instruction` into `prompt`
2. Converts `image_url` into absolute image paths under the URSA image root
3. Extracts the final answer after `†Answer:` into `reference`

It also fails fast on missing images and runs a lightweight `PromptDatasetVL` collate check.

## Training

Update the variables in `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` to the correct paths. On the current machine, the expected values are:

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl"
```

Then run:

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

The training launcher already wires:

- `--pretrain` to the URSA actor
- `--reward_pretrain` to the URSA reward model
- `--prompt_data` to the converted manifest
- `--images_key images`
- `--label_key label`
- `--apply_chat_template`

`reference_key` defaults to `reference` in `train_colocate.py`, so the generated manifest works without additional field remapping.

## Label Semantics

- `math_prm`: PRM-only reward path
- `math_prm_combined`: PRM plus rule-based correctness path

For the current Phase 0 and Phase 1 work, the default remains `math_prm`.

## Response Format Requirement

The URSA PRM expects the actor output to stay in this format:

```text
Step 1: ...
Step 2: ...
...
†Answer: ...
```

Do not switch this to plain paragraph CoT. The PRM scorer depends on step boundaries.

## Common Failure Modes

- Passing raw `MMathCoT-1M/train.jsonl` directly as `--prompt_data`: this is wrong; convert it first.
- Missing image files during conversion: `prepare_ursa_stage3_manifest.py` raises `FileNotFoundError` by design.
- Empty or malformed reward signals after rollout: verify the actor still emits `Step N:` and `†Answer:` format.
- Dependency drift: if something breaks after package changes, restore the Dockerfile baseline instead of normalizing drift in docs or scripts.

## Related Documents

- [`../../plan/MATH_PRM.md`](../../plan/MATH_PRM.md)
- [`./URSA_MIGRATION.md`](./URSA_MIGRATION.md)

## License

This project is licensed under the Apache 2.0 License. See [LICENSE](../../LICENSE) for details.

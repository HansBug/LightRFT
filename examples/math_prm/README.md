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
├── README.md                         # This file, focused on the current Stage 3 path
├── README_zh.md                      # Chinese version of this directory guide
├── URSA_MIGRATION.md                 # Migration notes from the original URSA-MATH repo
├── train_colocate.py                 # Main LightRFT training entry used by all current launchers
├── run_grpo_math_prm_ursa_8b.sh      # Main Stage 3 reproduction launcher
├── ursa_actor.py                     # URSA-specific actor wrapper for policy loading
├── reward_models.py                  # Reward implementations; active path is MathPRMReward / PS-GRPO logic
├── reward_models_utils.py            # Reward model loading, reward_fn routing, and label-to-recipe wiring
├── prepare_ursa_stage3_manifest.py   # Convert raw MMathCoT-1M jsonl into LightRFT manifest
├── prm_infer_score.py                # Step-level PRM scoring helper mirrored from URSA-MATH logic
├── prepare_ursa_engine_checkpoint.py # Optional wrapper builder for vLLM/SGLang-style engine experiments
├── sitecustomize.py                  # Local import/runtime compatibility hook for the URSA example stack
├── check_phase2_alignment.py         # Phase 2 scorer parity check against the URSA reference behavior
├── check_hf_rollout.py               # Minimal local-HF rollout validation for URSA
├── check_phase6_script_alignment.py  # Lightweight checker for Stage 3 launcher defaults
├── test_phase2_alignment.py          # Current regression tests for Phase 2/4/5/6 logic
├── run_phase3_smoke.sh               # Time-boxed Phase 3 smoke launcher
├── run_phase7_observation.sh         # Bounded full-data observation launcher
├── analyze_phase7_observation.py     # Offline analyzer for saved Phase 7 trajectories/logs
├── probe_rollout_speed_candidates.py # Performance probe for rollout-like generate modes without changing library code
└── ursa_model/                       # Self-contained URSA model code used by actor and PRM loading
```

## File Roles

### 1. Primary training path

- `run_grpo_math_prm_ursa_8b.sh`
  - Main launcher for the current URSA-MATH Stage 3 reproduction path.
  - Wires actor path, reward path, dataset path, FSDP settings, W&B, and rollout options.
- `train_colocate.py`
  - Real training entry used by `torchrun`.
  - Loads actor / reference / reward model / dataset / trainer and starts the LightRFT PPO-GRPO loop.
- `ursa_actor.py`
  - URSA-specific actor wrapper.
  - Makes LightRFT load `UrsaForConditionalGeneration` instead of a generic VLM auto-class.

### 2. Reward and scoring path

- `reward_models.py`
  - Contains all reward model classes used in this example directory.
  - The active Stage 3 path is `MathPRMReward` and the PS-GRPO reward mapping built on top of URSA-RM-8B.
  - Historical Qwen2VL multi-reward classes are still present in this file, but they are not part of the current URSA-MATH Stage 3 training path.
- `reward_models_utils.py`
  - Handles reward model loading and reward function dispatch.
  - Maps labels such as `math_prm` and `math_psgrpo` onto the current URSA reward path.
- `prm_infer_score.py`
  - Standalone helper for step-level PRM inference.
  - Useful when comparing LightRFT reward behavior against URSA-MATH reference scoring.

### 3. Data preparation and compatibility

- `prepare_ursa_stage3_manifest.py`
  - Converts raw `MMathCoT-1M` Stage 3 data into the `prompt / images / reference / label` schema expected by LightRFT.
  - Also performs a lightweight dataset/collate smoke check.
- `prepare_ursa_engine_checkpoint.py`
  - Optional helper for engine experiments.
  - Builds an engine-friendly wrapper checkpoint with the local URSA model code and `auto_map` metadata so vLLM/SGLang can at least attempt to load URSA.
- `sitecustomize.py`
  - Local runtime/import hook used to keep this example stack compatible under the frozen Docker baseline.

### 4. Validation, smoke, and observation tools

- `check_phase2_alignment.py`
  - Verifies that LightRFT `MathPRMReward` remains aligned with the URSA reference scorer on a concrete sample.
- `check_hf_rollout.py`
  - Minimal end-to-end validation for LightRFT local `hf` rollout.
  - Compares `gather_and_generate()` output against direct `actor.generate()`.
- `check_phase6_script_alignment.py`
  - Static checker that confirms the Stage 3 launcher still matches the intended defaults.
- `test_phase2_alignment.py`
  - Current regression test file for the URSA Stage 3 path.
  - Covers alignment, reward mapping, answer extraction, rollout helper behavior, and related utilities.
- `run_phase3_smoke.sh`
  - Time-boxed Phase 3 smoke launcher for “can it run, does it trend normally, and do we clean up GPUs afterward”.
- `run_phase7_observation.sh`
  - Bounded full-data observation launcher for later-stage analysis.
- `analyze_phase7_observation.py`
  - Offline analyzer for saved trajectories and training logs.
  - Computes the Phase 7 health checklist and PRM image-ablation summary.
- `probe_rollout_speed_candidates.py`
  - Minimal benchmark for rollout-like decode speed.
  - Used to compare `fsdp_train_gc`, `fsdp_train_no_gc`, `fsdp_eval_no_gc`, and `raw_eval_no_gc` without modifying `lightrft/` itself.

### 5. Self-contained URSA runtime

- `ursa_model/`
  - Local copy of the URSA model stack needed by both the actor and the PRM.
  - Includes config, processor, image processor, projector, vision backbones, and model definitions.
  - This directory is what allows the current Stage 3 path to run without depending on importing code directly from the external URSA-MATH repo.

## Current Entry Points

If you only care about the active Stage 3 reproduction path, the files you usually need are:

- `run_grpo_math_prm_ursa_8b.sh`
- `train_colocate.py`
- `reward_models.py`
- `reward_models_utils.py`
- `prepare_ursa_stage3_manifest.py`
- `check_hf_rollout.py`
- `test_phase2_alignment.py`

Everything else in this directory is either:

- a one-off compatibility helper,
- a smoke/observation tool,
- or part of the self-contained URSA runtime.

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

# CLAUDE.md

Note: `AGENTS.md` is a symlink to `CLAUDE.md`, so they are the same file. Do not edit both separately or duplicate the same change.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About LightRFT

LightRFT is a reinforcement fine-tuning (RFT) framework for LLMs and VLMs, built on top of [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF). It supports GRPO, GSPO, GMPO, Dr.GRPO, DAPO, REINFORCE++, CPGD, and FIRE Sampling algorithms, with vLLM/SGLang inference engines and FSDP/DeepSpeed training strategies.

## Common Commands

### Installation
```bash
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt  # for linting/formatting
```

### Code Formatting & Linting
```bash
make format    # YAPF formatting (line length 120)
make fcheck    # Flake8 linting
```

### Documentation
```bash
make docs       # Build Sphinx HTML docs → docs/build/index.html
make docs-live  # Live preview at http://localhost:8000
```

### Running Tests
Tests are in `lightrft/models/tests/` and use pytest:
```bash
python -m pytest lightrft/models/tests/test_actor_language.py
python -m pytest lightrft/models/tests/test_actor_vl.py
python -m pytest lightrft/models/tests/test_actorvl_fused_linear_logprob.py
```

### Running a Training Example
```bash
# Preprocess dataset first (example for GSM8K):
python examples/data_preprocess/gsm8k_lightrft.py --local_save_dir /path/to/output

# Then launch training (8 GPUs, single node):
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# VLM example (Geo3K):
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

Training is launched via `torchrun` with `train_colocate.py` inside each example directory.

### Docker
```bash
make dbuild   # Builds opendilab/lightrft:v<VERSION>
make dpush    # Pushes to Docker Hub
```

## Architecture Overview

### Core Training Flow

The typical RLHF training loop works as follows:
1. **Experience generation** (`FastExperienceMaker`): runs policy model via vLLM/SGLang, scores with reward models, computes advantages.
2. **Training** (`SPMDPPOTrainer`): updates the actor using PPO-style policy loss; optionally co-locates reward models on the same GPUs.
3. **Strategy** (`DeepspeedStrategy` / `FSDPV2Strategy`): wraps models for distributed training with a uniform API.

### Real End-to-End Control Flow in Current Code

The most useful way to understand the framework is to follow the actual runtime chain used by the example scripts:

1. **Training script assembles everything**  
   Example: `examples/gsm8k_geo3k/train_colocate.py`
   - Parse args
   - Build strategy with `get_strategy(args)`
   - Build actor / critic / reward models / initial model
   - Build tokenizer / processor / datasets / dataloaders
   - Call `strategy.setup_inference_engine(...)`
   - Instantiate `SPMDPPOTrainer` or `SPMDPPOTrainerVL`
   - Call `trainer.fit(...)`

2. **`trainer.fit()` drives the outer loop**  
   In `ppo_trainer.py` / `ppo_trainer_vl.py`, `fit()` iterates:
   - sample prompt batch
   - call `experience_maker.make_experience_list(...)`
   - append experiences into replay buffer
   - run `ppo_train(...)`
   - clear replay buffer
   - log / save / update KL controller

3. **`FastExperienceMaker` builds trainable experiences**  
   `FastExperienceMaker.make_experience_list(...)` is the real rollout pipeline:
   - `generate_samples(...)`: call vLLM/SGLang to generate responses
   - shard-parallel preprocess via `strategy.sp_data_processor.preprocess(...)`
   - `_make_experience_list_by_model(...)`: run actor / initial model / critic / reward models
   - shard-parallel postprocess
   - `_process_experiences(...)`: reward shaping, dynamic filtering, overlong penalty
   - `_compute_advantages_and_returns(...)`: KL-adjusted reward, returns, advantages

4. **Trainer consumes `Experience` objects and updates models**  
   `training_step_actor(...)` and `training_step_critic(...)` in `ppo_trainer.py` / `ppo_trainer_vl.py`:
   - actor forward -> fresh logprobs
   - `PolicyLoss` computes PPO/CPGD-style loss
   - optional KL loss term / PTX loss / aux loss
   - `strategy.backward(...)`
   - `strategy.optimizer_step(...)`

5. **Updated actor weights are pushed back to the rollout engine**  
   After `ppo_train(...)`, `SPMDPPOTrainerBase` calls:
   - `strategy.update_engine_weights(actor)`
   This is important: the training model and the rollout engine are separate objects.

### Layered Architecture

Think of the codebase as six layers that interact in one direction:

1. **Experiment assembly layer**: example `train_colocate.py` scripts
2. **Distributed/runtime layer**: `lightrft/strategy/`
3. **Training loop layer**: `lightrft/trainer/ppo_trainer*.py`, `spmd_ppo_trainer.py`
4. **Experience construction layer**: `fast_exp_maker.py`, `experience_maker*.py`
5. **Model/loss layer**: `lightrft/models/`
6. **Dataset/schema layer**: `lightrft/datasets/`

When customizing LightRFT, the right question is usually not "where is algorithm X?" but:
- Is this change about rollout?
- reward shaping?
- advantage computation?
- policy loss?
- distributed execution?
- data schema?

Many named algorithms in the repo are implemented as a composition across these layers rather than as a standalone trainer class.

### Key Modules

**`lightrft/trainer/`** — Core training logic:
- `spmd_ppo_trainer.py`: The primary trainer (`SPMDPPOTrainer`, `SPMDPPOTrainerVL`). Extends `PPOTrainer` with SPMD/tensor-parallel support. This is the "entry point" for understanding how training works end-to-end.
- `fast_exp_maker.py`: `FastExperienceMaker` — handles rollout generation via vLLM/SGLang, reward aggregation, model-side experience assembly, reward shaping, and advantage computation. The most important extension points are `generate_samples()`, `_process_experiences()`, `_compute_advantages_and_returns()`, and `_make_experience_list_by_model()`.
- `advantage_calculator.py`: Pluggable advantage estimators (GAE, Group Norm/GRPO, RLOO, REINFORCE++, CPGD). Use `get_advantage_calculator()` factory.
- `ppo_trainer.py` / `ppo_trainer_vl.py`: Base PPO trainer (ABC) for LLM and VLM respectively.
- `experience_maker.py`: `NaiveExperienceMaker` base class; `FastExperienceMaker` inherits from this.
- `replay_buffer.py` / `replay_buffer_vl.py`: Experience replay buffers with packing support.

**`lightrft/strategy/`** — Distributed training abstraction:
- `strategy_base.py`: `StrategyBase` ABC with `backward()`, `optimizer_step()`, `save_ckpt()` API.
- `strategy.py`: `get_strategy(args)` factory — picks DeepSpeed or FSDP based on `args.fsdp`.
- `deepspeed/deepspeed.py`: DeepSpeed ZeRO (Stage 1/2/3) strategy.
- `fsdp/fsdpv2.py`: FSDP v2 strategy.
- `config.py`: `StrategyConfig` dataclass (typed access to all strategy params; use `StrategyConfig.from_args(args)` to construct).
- `fake_strategy.py`: `FakeStrategy` for single-process unit testing without distributed setup.

**`lightrft/models/`** — Model wrappers:
- `actor_language.py`: LLM actor wrapping HuggingFace causal LM.
- `actor_vl.py` / `actor_al.py`: VLM and audio-LM actors.
- `actor_modality.py`: `ActorModality` base that both LLM and VLM actors extend.
- `loss.py`: `PolicyLoss` (PPO/GSPO/GMPO/Dr.GRPO/DAPO/Token-Level Policy variants all flow through here), `ValueLoss`, `GPTLMLoss`.
- `srm_vl.py` / `srm_al.py`: Scalar reward model wrappers.
- `grm_vl.py`: Generative reward model (VLM).
- `monkey_patch/`: Patches for distributed training compatibility.

**`lightrft/datasets/`** — Dataset handlers, each implementing a dataset-specific preprocessing interface. `prompts_dataset.py` (LLM) and `prompts_dataset_vl.py` (VLM) are the main training datasets; others are for reward model training.

**`lightrft/utils/`**:
- `cli_args.py`: `add_arguments()` adds engine/FSDP/logging CLI args to any `argparse.ArgumentParser`.
- `remote_rm_utils.py`: Utilities for calling remote reward model HTTP APIs.
- `trajectory_saver.py`: Saves rollout trajectories for analysis.
- `processor.py`: HuggingFace tokenizer/processor wrapper.

### Algorithm Extension Points

| To change... | Edit... |
|---|---|
| Policy loss objective / clipping rule | `lightrft/models/loss.py` → `PolicyLoss.forward()` |
| Advantage estimation method | `lightrft/trainer/advantage_calculator.py` |
| Rollout generation / FIRE sampling | `lightrft/trainer/fast_exp_maker.py` → `generate_samples()` |
| Reward shaping / normalization / overlong penalty | `lightrft/trainer/fast_exp_maker.py` → `_process_experiences()` / `_compute_advantages_and_returns()` |
| Reward model execution / aggregation | `lightrft/trainer/fast_exp_maker.py` → `RewardComputationEngine` |
| Model-side experience construction | `lightrft/trainer/fast_exp_maker.py` → `_make_experience_list_by_model()` |
| Distributed training backend | `lightrft/strategy/` |
| Dataset schema / prompt-image-reference extraction | `lightrft/datasets/prompts_dataset.py` / `prompts_dataset_vl.py` |

### Training Entry Points (examples)

Each example has its own `train_colocate.py`. They share the same general structure:
1. Parse args via `argparse` + `lightrft.utils.cli_args.add_arguments`
2. Build strategy via `get_strategy(args)`
3. Load actor, reference model, reward models
4. Instantiate `SPMDPPOTrainer` (or VL variant)
5. Call `trainer.fit()`

### Important Implementation Reality Notes

These points matter when modifying the code:

- **Current mainline training path uses `FastExperienceMaker`, not `NaiveExperienceMaker`.**  
  If you are changing rollout, reward shaping, or advantage logic for real training, start with `fast_exp_maker.py`.

- **The training actor and rollout engine are separate.**  
  Updating the PyTorch actor is not enough; the trainer later calls `strategy.update_engine_weights(actor)` to push weights into vLLM/SGLang.

- **Algorithms are distributed across layers.**  
  For example:
  - GRPO / group norm / RLOO / REINFORCE++ are primarily in `advantage_calculator.py`
  - DAPO-like behavior is currently represented mainly by dynamic sampling and overlong reward shaping in `fast_exp_maker.py`
  - CPGD is split between `advantage_calculator.py` and `PolicyLoss`

- **Not every algorithm name in docs corresponds to a fully separate implementation path.**  
  Some feature flags are partially wired through scripts/trainers but do not yet have a full dedicated implementation branch in `PolicyLoss`. Check the actual call chain before assuming a flag is active.

- **If a new runtime parameter is consumed via `self.strategy.config.xxx`, you must update `StrategyConfig`.**  
  Adding only an argparse flag is not enough if the fast path reads from `StrategyConfig.from_args(args)`.

## Commit Style

Follow the dominant repository convention from recent history, using Conventional-Commit-style subjects:
- Prefer `type(scope): imperative summary`
- Common types in this repository: `feature`, `fix`, `polish`, `docs`, `style`, `refactor`
- Example: `feature(trainer): add CPGD advantage estimator`
- Keep `type` and `scope` lowercase when present
- Omit the scope only when the change genuinely spans the whole repository
- Write the summary as a concise imperative phrase starting with a lowercase verb such as `add`, `update`, `improve`, `align`, or `clean up`
- Do not add a trailing period to the subject line
- For non-trivial changes, add a blank line and then a body
- In the body, prefer a short overview paragraph first, followed by `-` bullet points for concrete changes, tests, compatibility notes, docs updates, or behavior clarifications
- When a bullet wraps, continue it on the next indented line instead of starting a new bullet
- Preserve standard trailers when applicable, especially `Co-Authored-By: Name <email>`
- Merge commits should keep the generated history style, such as `Merge branch 'main' into dev/...` or `Merge pull request #52 from ...`

## PR Checklist

Before opening a PR, run:
```bash
make format
make fcheck
```

---

## Deep Dive: Module Functionality & Extension Guide

This section provides detailed analysis of each major module and practical guidance for extending the framework with custom algorithms, models, and training strategies.

### 1. Trainer Module (`lightrft/trainer/`)

The trainer module orchestrates the entire RLHF training loop, from experience generation to policy updates.

#### Core Components

**`SPMDPPOTrainer` / `SPMDPPOTrainerVL`** (`spmd_ppo_trainer.py`)
- **What it does**: Main training coordinator supporting SPMD (Single Program Multiple Data) and tensor parallelism. Manages the full training loop: experience generation → advantage computation → policy/value updates → logging.
- **Key features**:
  - Co-location of reward models with actor on same GPUs for memory efficiency
  - Support for both local and remote reward models (HTTP API)
  - Automatic KL divergence monitoring and adaptive KL penalty
  - Checkpoint saving/loading with strategy-aware state management
  - Integration with vLLM/SGLang for fast inference
  - Pushes updated actor weights back to rollout engine after PPO training
- **Extension points**:
  - Override `training_step()` to customize the training loop (e.g., add auxiliary losses)
  - Override `training_step_actor()` / `training_step_critic()` to change actor/critic updates
  - Extend `ppo_train()` if you need a different replay-buffer-to-update schedule
  - Add custom logging through `save_logs_and_checkpoints()`
  - Extend trajectory saving / checkpoint timing in `SPMDPPOTrainerBase`

**`FastExperienceMaker`** (`fast_exp_maker.py`)
- **What it does**: Generates rollout experiences using vLLM/SGLang inference engines. This is the real "experience pipeline" in current training. It handles:
  - text/VLM preprocessing
  - rollout engine generation
  - actor/reference/critic forward passes
  - reward model execution and aggregation
  - KL computation
  - reward shaping
  - advantage/return computation
  - final packing into `Experience` / `ExperienceVL`
- **Key features**:
  - Multimodal data processing (text, images, videos)
  - Multiple reward model aggregation (weighted sum, product, min/max)
  - FIRE sampling support for improved exploration
  - Running reward normalization across batches
  - Sample packing for training efficiency
  - Shard-parallel preprocess/postprocess through `strategy.sp_data_processor`
- **Extension points**:
  - **Add new sampling strategies**: Modify `generate_samples()` to implement custom sampling (e.g., beam search variants, constrained decoding)
  - **Custom reward model execution/aggregation**: Edit `RewardComputationEngine`
  - **New reward preprocessing**: Extend `_process_experiences()`
  - **Custom advantage computation**: Integrate new advantage calculators via `get_advantage_calculator()`
  - **Custom model-side bookkeeping**: Extend `_preprocess_sample()` / `_pack_experience()`

**Example: Adding a new sampling strategy**
```python
# In fast_exp_maker.py, modify generate_samples()
def generate_samples(self, prompts, **kwargs):
    # Your custom sampling logic
    if self.args.use_custom_sampling:
        outputs = self._custom_sampling_strategy(prompts, **kwargs)
    else:
        outputs = self.inference_engine.generate(prompts, **kwargs)
    return outputs
```

**`AdvantageCalculator`** (`advantage_calculator.py`)
- **What it does**: Pluggable advantage estimation with multiple algorithms (GAE, GRPO, RLOO, REINFORCE++, CPGD).
- **Key features**:
  - Unified interface via abstract base class
  - Reward preprocessing (whitening, clipping, group normalization)
  - Support for both token-level and sequence-level advantages
- **Extension points**:
  - **Add new advantage methods**: Create a new class inheriting from `AdvantageCalculator` or `BaseREINFORCECalculator`
  - Implement `preprocess_rewards()` for custom reward preprocessing
  - Implement `compute()` for advantage/return computation
  - Register in `get_advantage_calculator()` factory function
  - If needed, also update `normalize_advantages_cross_batch()` behavior for your estimator

**Example: Adding a custom advantage calculator**
```python
class MyCustomAdvantageCalculator(BaseREINFORCECalculator):
    def compute(self, rewards, values, action_mask, **kwargs):
        # Your custom advantage computation
        advantages = self._my_custom_algorithm(rewards, values)
        returns = rewards + advantages  # Example
        return advantages, returns

# Register in get_advantage_calculator()
def get_advantage_calculator(config):
    if config.advantage_estimator == "my_custom":
        return MyCustomAdvantageCalculator(config)
    # ... existing cases
```

**`ReplayBuffer` / `ReplayBufferVL`** (`replay_buffer.py`, `replay_buffer_vl.py`)
- **What it does**: Stores and samples experiences for training with optional sample packing.
- **Important detail**: This is closer to an experience cache/rebatcher than a classic off-policy RL replay buffer.
- **Extension points**:
  - Override `make_experience_batch()` to customize batch construction
  - Change `append()` / `normalize()` behavior if your algorithm needs different per-item statistics

---

### 2. Models Module (`lightrft/models/`)

The models module provides wrappers around HuggingFace models with RLHF-specific functionality.

#### Core Components

**`ActorLanguage` / `ActorVL` / `ActorAL`** (`actor_language.py`, `actor_vl.py`, `actor_al.py`)
- **What it does**: Wraps HuggingFace models for policy training. Computes log probabilities, handles generation, and manages modality-specific inputs.
- **Key features**:
  - Automatic handling of attention masks and position IDs
  - Support for gradient checkpointing and LoRA
  - Modality-aware parameter filtering (text-only vs. multimodal)
  - Integration with vLLM/SGLang for inference
  - Optional action entropy output for high-entropy token filtering
- **Extension points**:
  - **Add new modalities**: Create a new actor class inheriting from the base actor, define modality in `ActorModality` enum
  - **Custom forward pass**: Override `forward()` to add auxiliary outputs (e.g., uncertainty estimates)
  - **Custom generation**: Override `generate()` for specialized decoding strategies
  - **Model-specific preprocessing**: Add preprocessing logic in `__init__()` or `forward()`

**Example: Adding a new modality**
```python
# In actor_modality.py
class ActorModality(Enum):
    LANGUAGE_ONLY = "text"
    VISION_LANGUAGE = "vision"
    AUDIO_LANGUAGE = "audio"
    MY_NEW_MODALITY = "my_modality"  # Add your modality

MODALITY_PARAMETERS = {
    # ... existing mappings
    ActorModality.MY_NEW_MODALITY: {
        "my_special_input",
        "my_other_input",
    },
}

# Create actor_my_modality.py
class ActorMyModality(ActorLanguage):  # Or inherit from appropriate base
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = ActorModality.MY_NEW_MODALITY

    def forward(self, sequences, attention_mask, my_special_input=None, **kwargs):
        # Your custom forward logic
        pass
```

**`PolicyLoss`** (`loss.py`)
- **What it does**: Unified policy loss for actor optimization. In the current code path, the concrete implemented branches are standard PPO clipping, CPGD-style clipping, and optional entropy-mask-based token selection.
- **Key features**:
  - Clipped surrogate objective (PPO)
  - Asymmetric clipping (CPGD)
  - High-entropy token masking for efficient training
  - Automatic clip fraction logging
- **Extension points**:
  - **Add new policy objectives**: Modify `forward()` to implement new clipping strategies or loss formulations
  - **Custom masking**: Add new masking strategies beyond entropy-based filtering
  - **Token-level vs. sequence-level**: Adjust aggregation logic for different granularities
  - If your objective needs additional per-sample metadata, also update trainer `training_step_actor()`

**Example: Adding a new policy loss variant**
```python
# In loss.py, modify PolicyLoss.forward()
def forward(self, log_probs, old_log_probs, advantages, action_mask, **kwargs):
    if self.use_my_custom_loss:
        # Your custom loss computation
        loss = self._compute_my_custom_loss(log_probs, old_log_probs, advantages)
    else:
        # Existing PPO/CPGD logic
        loss = self._compute_standard_loss(...)
    return loss
```

**Reward Models** (`srm_vl.py`, `grm_vl.py`, `srm_al.py`)
- **What it does**: Wraps reward models for scoring generations. Supports scalar rewards (SRM) and generative rewards (GRM).
- **Extension points**:
  - **Add new reward model types**: Create new wrapper classes for different reward architectures
  - **Custom reward aggregation**: Implement multi-objective reward functions
  - **Process reward models**: Extend for token-level reward prediction (see `PRMLoss` in `loss.py`)

---

### 3. Strategy Module (`lightrft/strategy/`)

The strategy module abstracts distributed training backends (DeepSpeed, FSDP) behind a uniform API.

#### Core Components

**`StrategyBase`** (`strategy_base.py`)
- **What it does**: Abstract base class defining the training strategy interface. All strategies (DeepSpeed, FSDP, FakeStrategy) implement this API.
- **Key methods**:
  - `prepare_model()`: Wraps model for distributed training
  - `backward()`: Computes gradients
  - `optimizer_step()`: Updates parameters with gradient clipping
  - `save_ckpt()` / `load_ckpt()`: Checkpoint management
  - `setup_inference_engine()`: Initializes vLLM/SGLang
  - `gather_and_generate()`: gathers prompts across ranks and calls rollout engine
  - `update_engine_weights()`: broadcasts actor weights into rollout engine
- **Extension points**:
  - **Add new backends**: Create a new strategy class inheriting from `StrategyBase`
  - Implement all abstract methods for your backend
  - Register in `get_strategy()` factory function
  - If your backend changes rollout runtime behavior, also inspect `engine_generate_local()` / `gather_and_generate()`

**`DeepspeedStrategy`** (`deepspeed/deepspeed.py`)
- **What it does**: DeepSpeed ZeRO (Stage 1/2/3) implementation with automatic mixed precision.
- **Key features**: ZeRO optimizer, gradient accumulation, pipeline parallelism support
- **Extension points**: Override `_configure_deepspeed()` to customize DeepSpeed config

**`FSDPV2Strategy`** (`fsdp/fsdpv2.py`)
- **What it does**: PyTorch FSDP v2 implementation with flexible sharding strategies.
- **Key features**: FULL_SHARD, HYBRID_SHARD, NO_SHARD modes; CPU offloading; mixed precision
- **Extension points**: Override `_wrap_model()` to customize FSDP wrapping policy

**`FakeStrategy`** (`fake_strategy.py`)
- **What it does**: Single-process strategy for unit testing without distributed setup.
- **Use case**: Testing trainer logic, debugging algorithms, rapid prototyping

**`StrategyConfig`** (`config.py`)
- **What it does**: Typed configuration dataclass for all strategy parameters.
- **Usage**: `config = StrategyConfig.from_args(args)` converts argparse namespace to typed config
- **Extension points**: Add new fields for custom strategy parameters

**Example: Adding a new training strategy**
```python
# Create strategy/my_backend/my_strategy.py
class MyCustomStrategy(StrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize your backend

    def prepare_model(self, model, optimizer):
        # Wrap model with your backend
        return wrapped_model, wrapped_optimizer

    def backward(self, loss, model, optimizer):
        # Custom backward pass
        pass

    # Implement other abstract methods...

# Register in strategy.py
def get_strategy(args):
    if args.use_my_backend:
        return MyCustomStrategy(...)
    # ... existing cases
```

---

### 4. Datasets Module (`lightrft/datasets/`)

Dataset handlers for different data formats and tasks.

#### Core Components

**`PromptDataset` / `PromptDatasetVL`** (`prompts_dataset.py`, `prompts_dataset_vl.py`)
- **What it does**: Main training datasets for LLM and VLM respectively. Loads prompts and optional images/videos.
- **Data format**: Expects JSON/JSONL with fields like `prompt`, `images`, `videos`, `label`, `reference`
- **Extension points**:
  - **Add new data formats**: Create new dataset classes inheriting from `torch.utils.data.Dataset`
  - **Custom schema normalization**: Extend `preprocess_data(...)`
  - **Multi-turn / chat-template behavior**: Extend prompt rendering logic in the dataset layer
  - **Do not put reward logic here** unless it is strictly schema extraction; algorithmic reward logic belongs in trainer/experience maker

**Example: Adding a custom dataset**
```python
class MyCustomDataset(PromptDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # Add custom preprocessing
        item['prompt'] = self._my_custom_transform(item['prompt'])
        return item
```

---

### 5. Utils Module (`lightrft/utils/`)

Utility functions and helpers.

#### Key Components

**`cli_args.py`**: Centralized CLI argument definitions. Call `add_arguments(parser)` to add all framework args to your argparse parser.

**`remote_rm_utils.py`**: Utilities for calling remote reward model HTTP APIs. Useful for large reward models that don't fit on training GPUs.

**`trajectory_saver.py`**: Saves rollout trajectories (prompts, generations, rewards) for offline analysis and debugging.

**`processor.py`**: Unified wrapper for HuggingFace tokenizers and processors (handles both text-only and multimodal).

---

## Common Extension Scenarios

### Scenario 1: Adding a New RL Algorithm (e.g., GRPO variant)

1. **Advantage computation**: Create new calculator in `advantage_calculator.py`
2. **Policy loss**: Add loss variant in `loss.py` → `PolicyLoss.forward()`
3. **Reward preprocessing / shaping**: Modify `FastExperienceMaker._process_experiences()` or `_compute_advantages_and_returns()`
4. **CLI args**: Add algorithm-specific flags in your training script
5. **Config plumbing**: Add the new fields to `StrategyConfig` if the fast path reads them from `self.strategy.config`
6. **Test**: Use `FakeStrategy` for single-process testing

### Scenario 2: Adding a New Model Architecture

1. **Actor wrapper**: Create `actor_my_model.py` inheriting from `ActorLanguage` or `ActorVL`
2. **Modality definition**: Add to `ActorModality` enum if new modality
3. **Parameter routing**: Update modality-based extra-kwarg routing in `fast_exp_maker.py`
4. **Processor**: Ensure tokenizer/processor is compatible in `utils/processor.py`
5. **Monkey patches**: Add any model-specific patches in `models/monkey_patch/`
6. **Test**: Write unit tests in `models/tests/`

### Scenario 3: Adding a New Distributed Backend

1. **Strategy class**: Create new strategy inheriting from `StrategyBase`
2. **Implement interface**: All abstract methods (prepare, backward, optimizer_step, save/load)
3. **Register**: Add to `get_strategy()` factory in `strategy.py`
4. **Config**: Add backend-specific args to `StrategyConfig`
5. **Test**: Verify with multi-GPU setup

### Scenario 4: Custom Reward Function

1. **Local reward model**: Create wrapper in `models/` (e.g., `my_reward_model.py`)
2. **Remote reward model**: Implement HTTP endpoint, use `remote_rm_utils.py`
3. **Integration**: Pass reward model / `reward_fn` / label map / recipe to trainer
4. **Aggregation**: Modify `RewardComputationEngine._aggregate_rewards()`
5. **If needed**: Add post-reward shaping in `_process_experiences()`

### Scenario 5: New Sampling Strategy (e.g., Constrained Decoding)

1. **Modify generation**: Edit `FastExperienceMaker.generate_samples()`
2. **vLLM/SGLang config**: Add sampling parameters to inference engine setup
3. **CLI args**: Add flags for your sampling strategy
4. **Test**: Verify generation quality with small model

---

## Debugging Tips

- **Use `FakeStrategy`**: Test trainer logic without distributed setup
- **Enable trajectory saving**: Set `--save_trajectory` to inspect rollouts
- **Check reward distributions**: Monitor reward stats in logs
- **Gradient norms**: Watch for exploding/vanishing gradients
- **KL divergence**: Ensure KL stays within reasonable bounds (< 0.5 typically)
- **Advantage whitening**: Verify advantages are normalized (mean ≈ 0, std ≈ 1)
- **Verify engine sync**: If rollouts look stale after training, check whether `update_engine_weights()` is being reached
- **Check `StrategyConfig`**: If a new flag seems ignored, confirm it exists in `strategy/config.py`
- **For VLM issues**: Inspect image-token / pixel-value consistency checks in `ppo_trainer_vl.py` and `fast_exp_maker.py`

---

## Performance Optimization

- **Sample packing**: Enable with `--packing_samples` for variable-length sequences
- **vLLM/SGLang**: Use for 2-5x faster inference vs. HuggingFace
- **Gradient checkpointing**: Enable with `--gradient_checkpointing` to reduce memory
- **Mixed precision**: Use `--bf16` or `--fp16` for faster training
- **Co-located reward models**: Set `--colocate_reward_model` to save GPU memory
- **Batch size tuning**: Increase `--rollout_batch_size` and `--micro_train_batch_size` until OOM

---

## Testing Your Extensions

1. **Unit tests**: Add tests in appropriate `tests/` directory
2. **Integration test**: Run small-scale training (1 GPU, 100 steps)
3. **Convergence test**: Verify algorithm converges on toy task (e.g., GSM8K subset)
4. **Scaling test**: Test multi-GPU setup with your extension
5. **Benchmark**: Compare performance vs. baseline implementation

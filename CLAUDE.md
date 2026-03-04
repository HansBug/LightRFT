# CLAUDE.md

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

### Key Modules

**`lightrft/trainer/`** — Core training logic:
- `spmd_ppo_trainer.py`: The primary trainer (`SPMDPPOTrainer`, `SPMDPPOTrainerVL`). Extends `PPOTrainer` with SPMD/tensor-parallel support. This is the "entry point" for understanding how training works end-to-end.
- `fast_exp_maker.py`: `FastExperienceMaker` — handles rollout generation via vLLM/SGLang, reward aggregation, and advantage computation. The `generate_samples()` and `_get_return_advs()` methods are the main algorithm extension points.
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
| Policy loss objective (GRPO/GSPO/GMPO/DAPO/Dr.GRPO) | `lightrft/models/loss.py` → `PolicyLoss.forward()` |
| Advantage estimation method | `lightrft/trainer/advantage_calculator.py` |
| Rollout generation / FIRE sampling | `lightrft/trainer/fast_exp_maker.py` → `generate_samples()` |
| Reward aggregation / normalization | `lightrft/trainer/fast_exp_maker.py` → `_get_return_advs()` |
| Distributed training backend | `lightrft/strategy/` |

### Training Entry Points (examples)

Each example has its own `train_colocate.py`. They share the same general structure:
1. Parse args via `argparse` + `lightrft.utils.cli_args.add_arguments`
2. Build strategy via `get_strategy(args)`
3. Load actor, reference model, reward models
4. Instantiate `SPMDPPOTrainer` (or VL variant)
5. Call `trainer.fit()`

## Commit Style

Follow Conventional Commits: `type(scope): description`
- Common types: `feature`, `fix`, `polish`, `docs`, `style`, `refactor`
- Example: `feature(trainer): add CPGD advantage estimator`

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
- **Extension points**:
  - Override `training_step()` to customize the training loop (e.g., add auxiliary losses)
  - Override `_learn()` to modify how policy/value updates are computed
  - Add custom logging by extending `_log_metrics()`
  - Implement custom checkpoint logic in `save_checkpoint()` / `load_checkpoint()`

**`FastExperienceMaker`** (`fast_exp_maker.py`)
- **What it does**: Generates rollout experiences using vLLM/SGLang inference engines. Handles prompt processing, generation, reward computation, and advantage estimation.
- **Key features**:
  - Multimodal data processing (text, images, videos)
  - Multiple reward model aggregation (weighted sum, product, min/max)
  - FIRE sampling support for improved exploration
  - Running reward normalization across batches
  - Sample packing for training efficiency
- **Extension points**:
  - **Add new sampling strategies**: Modify `generate_samples()` to implement custom sampling (e.g., beam search variants, constrained decoding)
  - **Custom reward aggregation**: Edit `_get_return_advs()` to change how multiple rewards are combined
  - **New reward preprocessing**: Add methods in `_get_return_advs()` for custom reward normalization/transformation
  - **Custom advantage computation**: Integrate new advantage calculators via `get_advantage_calculator()`

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
- **Extension points**: Override `make_experience_batch()` to customize batch construction or add data augmentation.

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
- **What it does**: Unified policy loss supporting PPO, CPGD, DAPO, and high-entropy token filtering.
- **Key features**:
  - Clipped surrogate objective (PPO)
  - Asymmetric clipping (CPGD)
  - High-entropy token masking for efficient training
  - Automatic clip fraction logging
- **Extension points**:
  - **Add new policy objectives**: Modify `forward()` to implement new clipping strategies or loss formulations
  - **Custom masking**: Add new masking strategies beyond entropy-based filtering
  - **Token-level vs. sequence-level**: Adjust aggregation logic for different granularities

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
- **Extension points**:
  - **Add new backends**: Create a new strategy class inheriting from `StrategyBase`
  - Implement all abstract methods for your backend
  - Register in `get_strategy()` factory function

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
  - **Custom preprocessing**: Override `__getitem__()` to add data augmentation or filtering
  - **Multi-turn conversations**: Extend to handle conversation history

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
3. **Reward preprocessing**: Modify `FastExperienceMaker._get_return_advs()`
4. **CLI args**: Add algorithm-specific flags in your training script
5. **Test**: Use `FakeStrategy` for single-process testing

### Scenario 2: Adding a New Model Architecture

1. **Actor wrapper**: Create `actor_my_model.py` inheriting from `ActorLanguage` or `ActorVL`
2. **Modality definition**: Add to `ActorModality` enum if new modality
3. **Processor**: Ensure tokenizer/processor is compatible in `utils/processor.py`
4. **Monkey patches**: Add any model-specific patches in `models/monkey_patch/`
5. **Test**: Write unit tests in `models/tests/`

### Scenario 3: Adding a New Distributed Backend

1. **Strategy class**: Create new strategy inheriting from `StrategyBase`
2. **Implement interface**: All abstract methods (prepare, backward, optimizer_step, save/load)
3. **Register**: Add to `get_strategy()` factory in `strategy.py`
4. **Config**: Add backend-specific args to `StrategyConfig`
5. **Test**: Verify with multi-GPU setup

### Scenario 4: Custom Reward Function

1. **Local reward model**: Create wrapper in `models/` (e.g., `my_reward_model.py`)
2. **Remote reward model**: Implement HTTP endpoint, use `remote_rm_utils.py`
3. **Integration**: Pass reward model to trainer, configure in `FastExperienceMaker`
4. **Aggregation**: Modify reward combination logic in `_get_return_advs()`

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

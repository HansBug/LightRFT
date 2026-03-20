# CLAUDE.md

注意：`AGENTS.md` 是指向 `CLAUDE.md` 的符号链接，因此它们是同一个文件。不要分别编辑两者，也不要重复进行相同修改。

本文件为 Claude Code（claude.ai/code）在此仓库中处理代码时提供指导。

## 关于 LightRFT

LightRFT 是一个面向 LLM 和 VLM 的强化微调（RFT）框架，构建于 [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) 之上。它支持 GRPO、GSPO、GMPO、Dr.GRPO、DAPO、REINFORCE++、CPGD 和 FIRE Sampling 算法，并提供 vLLM/SGLang 推理引擎与 FSDP/DeepSpeed 训练策略。

## 常用命令

### 安装
```bash
pip install -r requirements.txt
pip install -e .
pip install -r requirements-dev.txt  # 用于 lint/格式化
```

### 代码格式化与 Lint
```bash
make format    # YAPF 格式化（行长 120）
make fcheck    # Flake8 lint 检查
```

### 文档
```bash
make docs       # 构建 Sphinx HTML 文档 → docs/build/index.html
make docs-live  # 实时预览，地址为 http://localhost:8000
```

### 运行测试
测试位于 `lightrft/models/tests/` 中，并使用 pytest：
```bash
python -m pytest lightrft/models/tests/test_actor_language.py
python -m pytest lightrft/models/tests/test_actor_vl.py
python -m pytest lightrft/models/tests/test_actorvl_fused_linear_logprob.py
```

### 运行训练示例
```bash
# 先对数据集进行预处理（以 GSM8K 为例）：
python examples/data_preprocess/gsm8k_lightrft.py --local_save_dir /path/to/output

# 然后启动训练（8 张 GPU，单节点）：
bash examples/gsm8k_geo3k/run_grpo_gsm8k_qwen2.5_0.5b.sh

# VLM 示例（Geo3K）：
bash examples/gsm8k_geo3k/run_grpo_geo3k_qwen2.5_vl_7b.sh
```

训练通过各示例目录中的 `train_colocate.py`，使用 `torchrun` 启动。

### Docker
```bash
make dbuild   # 构建 opendilab/lightrft:v<VERSION>
make dpush    # 推送到 Docker Hub
```

## 架构概览

### 核心训练流程

典型的 RLHF 训练循环工作方式如下：
1. **经验生成**（`FastExperienceMaker`）：通过 vLLM/SGLang 运行策略模型，使用奖励模型打分，并计算 advantage。
2. **训练**（`SPMDPPOTrainer`）：使用 PPO 风格的策略损失更新 actor；可选地将奖励模型与 actor 共置于同一组 GPU 上。
3. **策略**（`DeepspeedStrategy` / `FSDPV2Strategy`）：以统一 API 包装模型，用于分布式训练。

### 当前代码中的真实端到端控制流

理解该框架最有用的方式，是沿着示例脚本实际使用的运行链路去看：

1. **训练脚本组装一切**  
   示例：`examples/gsm8k_geo3k/train_colocate.py`
   - 解析参数
   - 使用 `get_strategy(args)` 构建策略
   - 构建 actor / critic / 奖励模型 / 初始模型
   - 构建 tokenizer / processor / 数据集 / dataloader
   - 调用 `strategy.setup_inference_engine(...)`
   - 实例化 `SPMDPPOTrainer` 或 `SPMDPPOTrainerVL`
   - 调用 `trainer.fit(...)`

2. **`trainer.fit()` 驱动外层循环**  
   在 `ppo_trainer.py` / `ppo_trainer_vl.py` 中，`fit()` 会迭代执行：
   - 采样 prompt 批次
   - 调用 `experience_maker.make_experience_list(...)`
   - 将经验追加到 replay buffer
   - 运行 `ppo_train(...)`
   - 清空 replay buffer
   - 记录日志 / 保存 / 更新 KL 控制器

3. **`FastExperienceMaker` 构建可训练经验**  
   `FastExperienceMaker.make_experience_list(...)` 是真实的 rollout 流水线：
   - `generate_samples(...)`：调用 vLLM/SGLang 生成响应
   - 通过 `strategy.sp_data_processor.preprocess(...)` 做分片并行预处理
   - `_make_experience_list_by_model(...)`：运行 actor / 初始模型 / critic / 奖励模型
   - 分片并行后处理
   - `_process_experiences(...)`：奖励塑形、动态过滤、超长惩罚
   - `_compute_advantages_and_returns(...)`：KL 调整后的奖励、returns、advantages

4. **Trainer 消费 `Experience` 对象并更新模型**  
   `ppo_trainer.py` / `ppo_trainer_vl.py` 中的 `training_step_actor(...)` 和 `training_step_critic(...)`：
   - actor 前向传播 -> 最新 logprobs
   - `PolicyLoss` 计算 PPO/CPGD 风格损失
   - 可选的 KL 损失项 / PTX 损失 / 辅助损失
   - `strategy.backward(...)`
   - `strategy.optimizer_step(...)`

5. **更新后的 actor 权重会被推回 rollout 引擎**  
   在 `ppo_train(...)` 之后，`SPMDPPOTrainerBase` 会调用：
   - `strategy.update_engine_weights(actor)`
   这一点很重要：训练模型和 rollout 引擎是两个独立对象。

### 分层架构

可以把代码库理解为六个单向交互的层次：

1. **实验组装层**：示例 `train_colocate.py` 脚本
2. **分布式/运行时层**：`lightrft/strategy/`
3. **训练循环层**：`lightrft/trainer/ppo_trainer*.py`、`spmd_ppo_trainer.py`
4. **经验构建层**：`fast_exp_maker.py`、`experience_maker*.py`
5. **模型/损失层**：`lightrft/models/`
6. **数据集/schema 层**：`lightrft/datasets/`

在定制 LightRFT 时，正确的问题通常不是“算法 X 在哪里？”，而是：
- 这项改动与 rollout 有关吗？
- 与奖励塑形有关吗？
- 与 advantage 计算有关吗？
- 与策略损失有关吗？
- 与分布式执行有关吗？
- 与数据 schema 有关吗？

仓库中许多具名算法，都是跨这些层组合实现的，而不是作为独立 trainer 类存在。

### 关键模块

**`lightrft/trainer/`** — 核心训练逻辑：
- `spmd_ppo_trainer.py`：主 Trainer（`SPMDPPOTrainer`、`SPMDPPOTrainerVL`）。在 `PPOTrainer` 基础上扩展了 SPMD/张量并行支持。这里是理解训练端到端工作方式的“入口点”。
- `fast_exp_maker.py`：`FastExperienceMaker` —— 负责通过 vLLM/SGLang 进行 rollout 生成、奖励聚合、模型侧经验组装、奖励塑形和 advantage 计算。最重要的扩展点是 `generate_samples()`、`_process_experiences()`、`_compute_advantages_and_returns()` 和 `_make_experience_list_by_model()`。
- `advantage_calculator.py`：可插拔的 advantage 估计器（GAE、Group Norm/GRPO、RLOO、REINFORCE++、CPGD）。使用 `get_advantage_calculator()` 工厂。
- `ppo_trainer.py` / `ppo_trainer_vl.py`：分别面向 LLM 和 VLM 的基础 PPO trainer（ABC）。
- `experience_maker.py`：`NaiveExperienceMaker` 基类；`FastExperienceMaker` 继承自它。
- `replay_buffer.py` / `replay_buffer_vl.py`：支持 packing 的经验回放缓冲区。

**`lightrft/strategy/`** — 分布式训练抽象：
- `strategy_base.py`：带有 `backward()`、`optimizer_step()`、`save_ckpt()` API 的 `StrategyBase` 抽象基类。
- `strategy.py`：`get_strategy(args)` 工厂 —— 根据 `args.fsdp` 选择 DeepSpeed 或 FSDP。
- `deepspeed/deepspeed.py`：DeepSpeed ZeRO（Stage 1/2/3）策略。
- `fsdp/fsdpv2.py`：FSDP v2 策略。
- `config.py`：`StrategyConfig` 数据类（为所有策略参数提供类型化访问；使用 `StrategyConfig.from_args(args)` 构造）。
- `fake_strategy.py`：`FakeStrategy`，用于无需分布式环境的单进程单元测试。

**`lightrft/models/`** — 模型包装器：
- `actor_language.py`：包装 HuggingFace causal LM 的 LLM actor。
- `actor_vl.py` / `actor_al.py`：VLM 和音频-LM actor。
- `actor_modality.py`：LLM 和 VLM actor 都会扩展的 `ActorModality` 基类。
- `loss.py`：`PolicyLoss`（PPO/GSPO/GMPO/Dr.GRPO/DAPO/Token-Level Policy 变体都流经这里）、`ValueLoss`、`GPTLMLoss`。
- `srm_vl.py` / `srm_al.py`：标量奖励模型包装器。
- `grm_vl.py`：生成式奖励模型（VLM）。
- `monkey_patch/`：用于分布式训练兼容性的补丁。

**`lightrft/datasets/`** — 数据集处理器，每个处理器都实现特定于数据集的预处理接口。`prompts_dataset.py`（LLM）和 `prompts_dataset_vl.py`（VLM）是主要训练数据集；其他则用于奖励模型训练。

**`lightrft/utils/`**：
- `cli_args.py`：`add_arguments()` 会将 engine/FSDP/logging CLI 参数添加到任意 `argparse.ArgumentParser` 中。
- `remote_rm_utils.py`：调用远程奖励模型 HTTP API 的工具函数。
- `trajectory_saver.py`：保存 rollout 轨迹用于分析。
- `processor.py`：HuggingFace tokenizer/processor 包装器。

### 算法扩展点

| 若要修改…… | 编辑…… |
|---|---|
| 策略损失目标 / 裁剪规则 | `lightrft/models/loss.py` → `PolicyLoss.forward()` |
| Advantage 估计方法 | `lightrft/trainer/advantage_calculator.py` |
| Rollout 生成 / FIRE sampling | `lightrft/trainer/fast_exp_maker.py` → `generate_samples()` |
| 奖励塑形 / 归一化 / 超长惩罚 | `lightrft/trainer/fast_exp_maker.py` → `_process_experiences()` / `_compute_advantages_and_returns()` |
| 奖励模型执行 / 聚合 | `lightrft/trainer/fast_exp_maker.py` → `RewardComputationEngine` |
| 模型侧经验构建 | `lightrft/trainer/fast_exp_maker.py` → `_make_experience_list_by_model()` |
| 分布式训练后端 | `lightrft/strategy/` |
| 数据集 schema / prompt-image-reference 提取 | `lightrft/datasets/prompts_dataset.py` / `prompts_dataset_vl.py` |

### 训练入口（示例）

每个示例都有自己的 `train_colocate.py`。它们共享相同的一般结构：
1. 通过 `argparse` + `lightrft.utils.cli_args.add_arguments` 解析参数
2. 通过 `get_strategy(args)` 构建策略
3. 加载 actor、reference model、奖励模型
4. 实例化 `SPMDPPOTrainer`（或 VL 变体）
5. 调用 `trainer.fit()`

### 重要实现现实说明

在修改代码时，以下几点很关键：

- **当前主线训练路径使用的是 `FastExperienceMaker`，不是 `NaiveExperienceMaker`。**  
  如果你是在为真实训练修改 rollout、奖励塑形或 advantage 逻辑，请从 `fast_exp_maker.py` 开始。

- **训练 actor 和 rollout 引擎是分离的。**  
  仅更新 PyTorch actor 还不够；trainer 之后还会调用 `strategy.update_engine_weights(actor)` 将权重推送到 vLLM/SGLang。

- **算法分布在多个层中。**  
  例如：
  - GRPO / group norm / RLOO / REINFORCE++ 主要位于 `advantage_calculator.py`
  - 类 DAPO 行为目前主要体现为 `fast_exp_maker.py` 中的动态采样和超长奖励塑形
  - CPGD 被拆分在 `advantage_calculator.py` 和 `PolicyLoss` 之间

- **文档中的每个算法名称都不一定对应一条完全独立的实现路径。**  
  某些功能开关虽然部分接入了脚本/trainer，但在 `PolicyLoss` 中尚未拥有完整的专用实现分支。在假定某个 flag 已生效之前，请先检查实际调用链。

- **如果某个新的运行时参数通过 `self.strategy.config.xxx` 被消费，你必须更新 `StrategyConfig`。**  
  如果快速路径是从 `StrategyConfig.from_args(args)` 读取配置，那么仅添加 argparse flag 是不够的。

## 提交风格

遵循该仓库近期历史中的主流约定，使用 Conventional-Commit 风格的主题行：
- 优先使用 `type(scope): imperative summary`
- 本仓库常见的类型有：`feature`、`fix`、`polish`、`docs`、`style`、`refactor`
- 示例：`feature(trainer): add CPGD advantage estimator`
- 若存在 `type` 和 `scope`，请保持小写
- 仅当改动确实横跨整个仓库时才省略 scope
- 主题行应写成简洁的祈使短语，以小写动词开头，例如 `add`、`update`、`improve`、`align` 或 `clean up`
- 主题行末尾不要加句号
- 对于非平凡改动，添加一个空行后再写正文
- 在正文中，优先先写一个简短概述段落，然后使用 `-` 项目符号列出具体改动、测试、兼容性说明、文档更新或行为澄清
- 当项目符号换行时，应在下一行继续缩进，而不是开始新的项目符号
- 适用时保留标准 trailer，尤其是 `Co-Authored-By: Name <email>`
- 合并提交应保留生成的历史风格，例如 `Merge branch 'main' into dev/...` 或 `Merge pull request #52 from ...`

## PR 检查清单

在打开 PR 之前，运行：
```bash
make format
make fcheck
```

## Math PRM 分支协作约定

当前仓库里与 Math PRM 相关的长期协作分支如下：

- `archive/math_prm_train_full_20260320`
  - 2026-03-20 的完整快照，只读备份，不要重写，不要在上面继续开发。
- `dev/math_prm_train_working`
  - 当前完整开发分支。
  - 这里允许保留脚本、测试、迁移说明、plan、tmp、AGENTS/CLAUDE 等全部辅助材料。
  - 日常开发默认在这个分支上进行。
- `dev/math_prm_train`
  - 当前发往 `opendilab/main` 的精简 PR 分支。
  - 这里只保留当前 upstream PR 需要的最小 Stage 3 训练面。

### `dev/math_prm_train` 当前默认保留面

以下路径默认属于 `dev/math_prm_train` 的稳定 PR 面，除非用户明确要求，否则不要主动扩大：

- `examples/math_prm/README.md`
- `examples/math_prm/README_zh.md`
- `examples/math_prm/train_colocate.py`
- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- `examples/math_prm/reward_models.py`
- `examples/math_prm/reward_models_utils.py`
- `examples/math_prm/sitecustomize.py`
- `examples/math_prm/ursa_actor.py`
- `examples/math_prm/ursa_model/`
- `examples/math_prm/tools/__init__.py`
- `examples/math_prm/tools/prepare_ursa_stage3_manifest.py`
- `examples/math_prm/tools/prepare_ursa_engine_checkpoint.py`
- `lightrft/models/actor_language.py`
- `lightrft/models/actor_vl.py`
- `lightrft/strategy/config.py`
- `lightrft/strategy/fake_strategy.py`
- `lightrft/strategy/strategy_base.py`
- `lightrft/strategy/vllm_utils/__init__.py`
- `lightrft/trainer/fast_exp_maker.py`
- `lightrft/trainer/ppo_trainer_vl.py`
- `lightrft/trainer/spmd_ppo_trainer.py`
- `lightrft/utils/cli_args.py`
- `lightrft/utils/math_prm_output.py`
- `requirements.txt`

下列内容默认留在 `dev/math_prm_train_working`，不要在未得到明确指令时搬进 `dev/math_prm_train`：

- `AGENTS.md` / `CLAUDE.md`
- `plan/`
- `tmp/`
- `examples/math_prm/URSA_MIGRATION.md`
- `examples/math_prm/tools/` 下除 `prepare_ursa_stage3_manifest.py`、`prepare_ursa_engine_checkpoint.py` 之外的辅助脚本

### 从 `dev/math_prm_train_working` 搬运到 `dev/math_prm_train` 的规则

- 默认在 `dev/math_prm_train_working` 上开发，不要把 `dev/math_prm_train` 当作日常开发分支。
- 不要执行 `merge dev/math_prm_train_working -> dev/math_prm_train`。
- 不要为了搬运改动而把 `dev/math_prm_train` rebase 到 `dev/math_prm_train_working` 上。
- 如果待迁移改动已经是干净 commit，优先使用 `git cherry-pick -x <commit>`。
- 如果一个 commit 同时包含主线代码和辅助材料，优先先在 `dev/math_prm_train_working` 上拆 commit，再搬运。
- 如果不能先拆 commit，则在 `dev/math_prm_train` 上使用 `git cherry-pick -n <commit>`，然后只保留目标路径或目标 hunk，再重新提交。
- 如果迁移单位天然是“路径集合”而不是 commit，则从 `dev/math_prm_train` 新建临时分支后，使用 `git restore --source=dev/math_prm_train_working -- <paths>` 选择性恢复，再提交。
- 若 reviewer 只在 `dev/math_prm_train` 上提出修正，但这些修正也应保留在完整开发分支，请把该修正反向迁回 `dev/math_prm_train_working`。
- 后续这类搬运工作可以直接交给 AI 执行，但 AI 必须遵守本节约定，而不是直接合并两个分支。

### 搬运后的校验步骤

每次把内容搬进 `dev/math_prm_train` 后，至少做以下检查：

- `git status --short` 必须干净。
- `git diff --name-only main/main...dev/math_prm_train` 只应包含本次准备进入 upstream PR 的目标路径。
- `git diff --check main/main...dev/math_prm_train` 不应出现空白错误。
- 如果改动触及 `examples/math_prm/README.md`、`examples/math_prm/README_zh.md` 或 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`，要用 `rg` 检查它们是否引用了未搬运的脚本、临时文档或 plan 路径。
- 如果新增或修改了 Python 文件，优先做轻量语法检查；如果 `compileall` 受现有 `__pycache__` 目录权限限制，要在交付说明里明确写出该限制。
- 只要存在合适的最小测试，就运行；如果没跑测试，必须明确说明没跑。
- 如需同时维护两个工作现场，优先使用 `git worktree`，不要反复切换分支后手工清理工作树。

---

## 深入解析：模块功能与扩展指南

本节提供对各个主要模块的详细分析，以及扩展框架以支持自定义算法、模型和训练策略的实用指导。

### 1. Trainer 模块（`lightrft/trainer/`）

Trainer 模块编排整个 RLHF 训练循环，从经验生成到策略更新。

#### 核心组件

**`SPMDPPOTrainer` / `SPMDPPOTrainerVL`**（`spmd_ppo_trainer.py`）
- **作用**：支持 SPMD（Single Program Multiple Data）和张量并行的主训练协调器。管理完整训练循环：经验生成 → advantage 计算 → 策略/价值更新 → 日志记录。
- **关键特性**：
  - 将奖励模型与 actor 共置在同一组 GPU 上，以提升内存效率
  - 同时支持本地和远程奖励模型（HTTP API）
  - 自动监控 KL 散度并施加自适应 KL 惩罚
  - 带有策略感知状态管理的检查点保存/加载
  - 集成 vLLM/SGLang 以实现快速推理
  - 在 PPO 训练后将更新后的 actor 权重推回 rollout 引擎
- **扩展点**：
  - 重写 `training_step()` 以定制训练循环（例如添加辅助损失）
  - 重写 `training_step_actor()` / `training_step_critic()` 以更改 actor/critic 更新
  - 如果你需要不同的 replay-buffer 到 update 调度，可扩展 `ppo_train()`
  - 通过 `save_logs_and_checkpoints()` 添加自定义日志
  - 在 `SPMDPPOTrainerBase` 中扩展轨迹保存 / 检查点时机

**`FastExperienceMaker`**（`fast_exp_maker.py`）
- **作用**：使用 vLLM/SGLang 推理引擎生成 rollout 经验。这是当前训练中的真实“经验流水线”。它处理：
  - 文本/VLM 预处理
  - rollout 引擎生成
  - actor/reference/critic 前向传播
  - 奖励模型执行与聚合
  - KL 计算
  - 奖励塑形
  - advantage/return 计算
  - 最终打包为 `Experience` / `ExperienceVL`
- **关键特性**：
  - 多模态数据处理（文本、图像、视频）
  - 多奖励模型聚合（加权和、乘积、最小值/最大值）
  - 支持 FIRE sampling 以提升探索
  - 跨 batch 的运行中奖励归一化
  - 用于训练效率的样本 packing
  - 通过 `strategy.sp_data_processor` 进行分片并行预处理/后处理
- **扩展点**：
  - **添加新的采样策略**：修改 `generate_samples()` 以实现自定义采样（例如 beam search 变体、约束解码）
  - **自定义奖励模型执行/聚合**：编辑 `RewardComputationEngine`
  - **新的奖励预处理**：扩展 `_process_experiences()`
  - **自定义 advantage 计算**：通过 `get_advantage_calculator()` 集成新的 advantage 计算器
  - **自定义模型侧记账逻辑**：扩展 `_preprocess_sample()` / `_pack_experience()`

**示例：添加一种新的采样策略**
```python
# 在 fast_exp_maker.py 中，修改 generate_samples()
def generate_samples(self, prompts, **kwargs):
    # 你的自定义采样逻辑
    if self.args.use_custom_sampling:
        outputs = self._custom_sampling_strategy(prompts, **kwargs)
    else:
        outputs = self.inference_engine.generate(prompts, **kwargs)
    return outputs
```

**`AdvantageCalculator`**（`advantage_calculator.py`）
- **作用**：支持多种算法（GAE、GRPO、RLOO、REINFORCE++、CPGD）的可插拔 advantage 估计。
- **关键特性**：
  - 通过抽象基类提供统一接口
  - 奖励预处理（白化、裁剪、组归一化）
  - 同时支持 token 级和 sequence 级 advantage
- **扩展点**：
  - **添加新的 advantage 方法**：创建一个继承自 `AdvantageCalculator` 或 `BaseREINFORCECalculator` 的新类
  - 为自定义奖励预处理实现 `preprocess_rewards()`
  - 为 advantage/return 计算实现 `compute()`
  - 在 `get_advantage_calculator()` 工厂函数中注册
  - 如有需要，也要更新你的估计器在 `normalize_advantages_cross_batch()` 中的行为

**示例：添加一个自定义 advantage 计算器**
```python
class MyCustomAdvantageCalculator(BaseREINFORCECalculator):
    def compute(self, rewards, values, action_mask, **kwargs):
        # 你的自定义 advantage 计算
        advantages = self._my_custom_algorithm(rewards, values)
        returns = rewards + advantages  # 示例
        return advantages, returns

# 在 get_advantage_calculator() 中注册
def get_advantage_calculator(config):
    if config.advantage_estimator == "my_custom":
        return MyCustomAdvantageCalculator(config)
    # ... 现有分支
```

**`ReplayBuffer` / `ReplayBufferVL`**（`replay_buffer.py`、`replay_buffer_vl.py`）
- **作用**：存储并采样训练经验，可选支持样本 packing。
- **重要细节**：它更接近经验缓存/重分批器，而不是经典的离策略 RL replay buffer。
- **扩展点**：
  - 重写 `make_experience_batch()` 以定制 batch 构建
  - 如果你的算法需要不同的逐项统计，可修改 `append()` / `normalize()` 行为

---

### 2. Models 模块（`lightrft/models/`）

Models 模块提供围绕 HuggingFace 模型的包装器，并加入 RLHF 特有功能。

#### 核心组件

**`ActorLanguage` / `ActorVL` / `ActorAL`**（`actor_language.py`、`actor_vl.py`、`actor_al.py`）
- **作用**：为策略训练包装 HuggingFace 模型。负责计算对数概率、处理生成，以及管理特定模态的输入。
- **关键特性**：
  - 自动处理 attention mask 和 position ID
  - 支持 gradient checkpointing 和 LoRA
  - 具备模态感知的参数过滤（纯文本 vs. 多模态）
  - 与 vLLM/SGLang 集成用于推理
  - 可选输出 action entropy，用于高熵 token 过滤
- **扩展点**：
  - **添加新模态**：创建一个继承基础 actor 的新 actor 类，并在 `ActorModality` 枚举中定义该模态
  - **自定义前向传播**：重写 `forward()` 以添加辅助输出（例如不确定性估计）
  - **自定义生成**：重写 `generate()` 以支持专用解码策略
  - **模型特定预处理**：在 `__init__()` 或 `forward()` 中加入预处理逻辑

**示例：添加一种新模态**
```python
# 在 actor_modality.py 中
class ActorModality(Enum):
    LANGUAGE_ONLY = "text"
    VISION_LANGUAGE = "vision"
    AUDIO_LANGUAGE = "audio"
    MY_NEW_MODALITY = "my_modality"  # 添加你的模态

MODALITY_PARAMETERS = {
    # ... 现有映射
    ActorModality.MY_NEW_MODALITY: {
        "my_special_input",
        "my_other_input",
    },
}

# 创建 actor_my_modality.py
class ActorMyModality(ActorLanguage):  # 或继承合适的基类
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.modality = ActorModality.MY_NEW_MODALITY

    def forward(self, sequences, attention_mask, my_special_input=None, **kwargs):
        # 你的自定义前向逻辑
        pass
```

**`PolicyLoss`**（`loss.py`）
- **作用**：用于 actor 优化的统一策略损失。在当前代码路径中，已具体实现的分支包括标准 PPO 裁剪、CPGD 风格裁剪，以及基于 entropy mask 的可选 token 选择。
- **关键特性**：
  - 裁剪的替代目标（PPO）
  - 非对称裁剪（CPGD）
  - 用于高效训练的高熵 token mask
  - 自动记录 clip fraction
- **扩展点**：
  - **添加新的策略目标**：修改 `forward()` 以实现新的裁剪策略或损失形式
  - **自定义掩码**：添加超出 entropy-based filtering 之外的新掩码策略
  - **Token 级 vs. sequence 级**：针对不同粒度调整聚合逻辑
  - 如果你的目标需要额外的逐样本元数据，也请同步更新 trainer 中的 `training_step_actor()`

**示例：添加新的策略损失变体**
```python
# 在 loss.py 中，修改 PolicyLoss.forward()
def forward(self, log_probs, old_log_probs, advantages, action_mask, **kwargs):
    if self.use_my_custom_loss:
        # 你的自定义损失计算
        loss = self._compute_my_custom_loss(log_probs, old_log_probs, advantages)
    else:
        # 现有 PPO/CPGD 逻辑
        loss = self._compute_standard_loss(...)
    return loss
```

**奖励模型**（`srm_vl.py`、`grm_vl.py`、`srm_al.py`）
- **作用**：包装用于给生成结果打分的奖励模型。支持标量奖励（SRM）和生成式奖励（GRM）。
- **扩展点**：
  - **添加新的奖励模型类型**：为不同奖励架构创建新的包装类
  - **自定义奖励聚合**：实现多目标奖励函数
  - **处理式奖励模型**：扩展以支持 token 级奖励预测（参见 `loss.py` 中的 `PRMLoss`）

---

### 3. Strategy 模块（`lightrft/strategy/`）

Strategy 模块在统一 API 之后抽象了分布式训练后端（DeepSpeed、FSDP）。

#### 核心组件

**`StrategyBase`**（`strategy_base.py`）
- **作用**：定义训练策略接口的抽象基类。所有策略（DeepSpeed、FSDP、FakeStrategy）都实现该 API。
- **关键方法**：
  - `prepare_model()`：为分布式训练包装模型
  - `backward()`：计算梯度
  - `optimizer_step()`：带梯度裁剪地更新参数
  - `save_ckpt()` / `load_ckpt()`：检查点管理
  - `setup_inference_engine()`：初始化 vLLM/SGLang
  - `gather_and_generate()`：跨 rank 聚合 prompt 并调用 rollout 引擎
  - `update_engine_weights()`：将 actor 权重广播到 rollout 引擎中
- **扩展点**：
  - **添加新的后端**：创建一个继承自 `StrategyBase` 的新策略类
  - 为你的后端实现所有抽象方法
  - 在 `get_strategy()` 工厂函数中注册
  - 如果你的后端会改变 rollout 运行时行为，也请检查 `engine_generate_local()` / `gather_and_generate()`

**`DeepspeedStrategy`**（`deepspeed/deepspeed.py`）
- **作用**：带自动混合精度的 DeepSpeed ZeRO（Stage 1/2/3）实现。
- **关键特性**：ZeRO 优化器、梯度累积、pipeline parallelism 支持
- **扩展点**：重写 `_configure_deepspeed()` 以定制 DeepSpeed 配置

**`FSDPV2Strategy`**（`fsdp/fsdpv2.py`）
- **作用**：带灵活分片策略的 PyTorch FSDP v2 实现。
- **关键特性**：FULL_SHARD、HYBRID_SHARD、NO_SHARD 模式；CPU offloading；混合精度
- **扩展点**：重写 `_wrap_model()` 以定制 FSDP 包装策略

**`FakeStrategy`**（`fake_strategy.py`）
- **作用**：用于单元测试的单进程策略，无需分布式环境。
- **使用场景**：测试 trainer 逻辑、调试算法、快速原型开发

**`StrategyConfig`**（`config.py`）
- **作用**：面向所有策略参数的类型化配置数据类。
- **用法**：`config = StrategyConfig.from_args(args)` 会将 argparse namespace 转换为类型化配置
- **扩展点**：为自定义策略参数添加新字段

**示例：添加一种新的训练策略**
```python
# 创建 strategy/my_backend/my_strategy.py
class MyCustomStrategy(StrategyBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 初始化你的后端

    def prepare_model(self, model, optimizer):
        # 使用你的后端包装模型
        return wrapped_model, wrapped_optimizer

    def backward(self, loss, model, optimizer):
        # 自定义反向传播
        pass

    # 实现其他抽象方法...

# 在 strategy.py 中注册
def get_strategy(args):
    if args.use_my_backend:
        return MyCustomStrategy(...)
    # ... 现有分支
```

---

### 4. Datasets 模块（`lightrft/datasets/`）

面向不同数据格式和任务的数据集处理器。

#### 核心组件

**`PromptDataset` / `PromptDatasetVL`**（`prompts_dataset.py`、`prompts_dataset_vl.py`）
- **作用**：分别面向 LLM 和 VLM 的主训练数据集。加载 prompt 以及可选的图像/视频。
- **数据格式**：期望 JSON/JSONL，字段如 `prompt`、`images`、`videos`、`label`、`reference`
- **扩展点**：
  - **添加新的数据格式**：创建继承自 `torch.utils.data.Dataset` 的新数据集类
  - **自定义 schema 归一化**：扩展 `preprocess_data(...)`
  - **多轮 / chat-template 行为**：在数据集层扩展 prompt 渲染逻辑
  - **不要在这里放奖励逻辑**，除非它严格属于 schema 提取；算法性的奖励逻辑应放在 trainer/experience maker 中

**示例：添加一个自定义数据集**
```python
class MyCustomDataset(PromptDataset):
    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        # 添加自定义预处理
        item['prompt'] = self._my_custom_transform(item['prompt'])
        return item
```

---

### 5. Utils 模块（`lightrft/utils/`）

工具函数与辅助组件。

#### 关键组件

**`cli_args.py`**：集中式 CLI 参数定义。调用 `add_arguments(parser)` 以将所有框架参数添加到你的 argparse parser 中。

**`remote_rm_utils.py`**：用于调用远程奖励模型 HTTP API 的工具函数。适合那些无法放入训练 GPU 的大型奖励模型。

**`trajectory_saver.py`**：保存 rollout 轨迹（prompts、generations、rewards），用于离线分析与调试。

**`processor.py`**：统一的 HuggingFace tokenizer 和 processor 包装器（同时处理纯文本与多模态）。

---

## 常见扩展场景

### 场景 1：添加新的 RL 算法（例如 GRPO 变体）

1. **Advantage 计算**：在 `advantage_calculator.py` 中创建新的计算器
2. **策略损失**：在 `loss.py` → `PolicyLoss.forward()` 中添加损失变体
3. **奖励预处理 / 塑形**：修改 `FastExperienceMaker._process_experiences()` 或 `_compute_advantages_and_returns()`
4. **CLI 参数**：在你的训练脚本中添加算法专用 flag
5. **配置打通**：如果快速路径会从 `self.strategy.config` 读取新字段，则将这些字段添加到 `StrategyConfig`
6. **测试**：使用 `FakeStrategy` 进行单进程测试

### 场景 2：添加新的模型架构

1. **Actor 包装器**：创建继承自 `ActorLanguage` 或 `ActorVL` 的 `actor_my_model.py`
2. **模态定义**：如果是新模态，则添加到 `ActorModality` 枚举中
3. **参数路由**：更新 `fast_exp_maker.py` 中按模态划分的额外 kwarg 路由
4. **Processor**：确保 `utils/processor.py` 中 tokenizer/processor 兼容
5. **Monkey patch**：在 `models/monkey_patch/` 中添加任何模型特定补丁
6. **测试**：在 `models/tests/` 中编写单元测试

### 场景 3：添加新的分布式后端

1. **Strategy 类**：创建继承自 `StrategyBase` 的新策略
2. **实现接口**：实现所有抽象方法（prepare、backward、optimizer_step、save/load）
3. **注册**：添加到 `strategy.py` 中的 `get_strategy()` 工厂
4. **配置**：将后端特定参数添加到 `StrategyConfig`
5. **测试**：在多 GPU 环境中验证

### 场景 4：自定义奖励函数

1. **本地奖励模型**：在 `models/` 中创建包装器（例如 `my_reward_model.py`）
2. **远程奖励模型**：实现 HTTP 端点，使用 `remote_rm_utils.py`
3. **集成**：将奖励模型 / `reward_fn` / label map / recipe 传给 trainer
4. **聚合**：修改 `RewardComputationEngine._aggregate_rewards()`
5. **如有需要**：在 `_process_experiences()` 中加入奖励后的塑形

### 场景 5：新的采样策略（例如约束解码）

1. **修改生成**：编辑 `FastExperienceMaker.generate_samples()`
2. **vLLM/SGLang 配置**：向推理引擎设置中添加采样参数
3. **CLI 参数**：为你的采样策略添加 flag
4. **测试**：用小模型验证生成质量

---

## 调试提示

- **使用 `FakeStrategy`**：在无需分布式环境的情况下测试 trainer 逻辑
- **启用轨迹保存**：设置 `--save_trajectory` 以检查 rollout
- **检查奖励分布**：在日志中监控奖励统计信息
- **梯度范数**：关注梯度爆炸/消失
- **KL 散度**：确保 KL 保持在合理范围内（通常 < 0.5）
- **Advantage 白化**：验证 advantage 已归一化（均值 ≈ 0，标准差 ≈ 1）
- **验证引擎同步**：如果训练后 rollout 看起来仍然很旧，检查是否执行到了 `update_engine_weights()`
- **检查 `StrategyConfig`**：如果某个新 flag 看起来被忽略了，请确认它已存在于 `strategy/config.py` 中
- **对于 VLM 问题**：检查 `ppo_trainer_vl.py` 和 `fast_exp_maker.py` 中 image-token / pixel-value 一致性检查

---

## 性能优化

- **样本 packing**：对变长序列启用 `--packing_samples`
- **vLLM/SGLang**：相较 HuggingFace，可获得 2-5 倍更快的推理
- **Gradient checkpointing**：启用 `--gradient_checkpointing` 以减少内存占用
- **混合精度**：使用 `--bf16` 或 `--fp16` 以提升训练速度
- **奖励模型共置**：设置 `--colocate_reward_model` 以节省 GPU 内存
- **批大小调优**：增大 `--rollout_batch_size` 和 `--micro_train_batch_size` 直到 OOM

---

## 测试你的扩展

1. **单元测试**：在合适的 `tests/` 目录中添加测试
2. **集成测试**：运行小规模训练（1 张 GPU，100 步）
3. **收敛测试**：验证算法能在玩具任务上收敛（例如 GSM8K 子集）
4. **扩展性测试**：使用你的扩展测试多 GPU 环境
5. **基准测试**：与基线实现比较性能

<div align="center">

# LightRFT 中的 Math PRM 训练

URSA-MATH Stage 3 PS-GRPO 训练迁移到 LightRFT 的实现目录。

</div>

## 概述

这个目录现在是实际使用中的 `math_prm` 工作目录，目标不是通用 SafeWork 示例，而是当前这条链路：

- actor：`URSA-8B`
- reward model：`URSA-RM-8B`
- 算法：GRPO / PS-GRPO 风格在线强化学习
- 原始数据：`MMathCoT-1M`
- 训练数据 schema：`prompt / images / reference / label`

## 运行时基线

运行时基线由 `/data/LightRFT/Dockerfile` 冻结。

- Dockerfile 里已经安装的 pip 包版本不要擅自改。
- 不要把升级或降级 `torch`、`deepspeed`、`vllm`、`flash_attn`、`sglang` 当成常规修复手段。
- 优先从代码、数据转换、训练参数三层排查问题。

## 目录说明

```text
examples/math_prm/
├── README.md                         # 当前 Stage 3 路径的英文说明
├── README_zh.md                      # 当前目录的中文结构说明
├── URSA_MIGRATION.md                 # 从原始 URSA-MATH repo 迁移到 LightRFT 的说明
├── train_colocate.py                 # 当前训练主入口
├── run_grpo_math_prm_ursa_8b.sh      # 当前 Stage 3 复现实验主脚本
├── ursa_actor.py                     # URSA policy model 的自定义 actor 包装
├── reward_models.py                  # reward 实现；当前主路径是 MathPRMReward / PS-GRPO
├── reward_models_utils.py            # reward model 加载、label 路由和 reward_fn 组装
├── prepare_ursa_stage3_manifest.py   # 把原始 MMathCoT-1M Stage 3 数据转换成 LightRFT manifest
├── prm_infer_score.py                # step-level PRM 打分辅助脚本
├── prepare_ursa_engine_checkpoint.py # 给 vLLM/SGLang 试验准备 wrapper checkpoint 的辅助脚本
├── sitecustomize.py                  # 当前 example 栈的本地兼容性补丁入口
├── check_phase2_alignment.py         # Phase 2 打分对齐检查脚本
├── check_hf_rollout.py               # 本地 HF rollout 最小链路校验
├── check_phase6_script_alignment.py  # Stage 3 启动脚本默认配置检查器
├── test_phase2_alignment.py          # 当前 URSA Stage 3 路径的回归测试
├── run_phase3_smoke.sh               # 限时 Phase 3 smoke 试跑脚本
├── run_phase7_observation.sh         # 全量 bounded observation 启动脚本
├── analyze_phase7_observation.py     # Phase 7 离线分析脚本
├── probe_rollout_speed_candidates.py # 不改库代码时的 rollout 速度对照脚本
└── ursa_model/                       # 自包含的 URSA 模型代码
```

## 文件职责

### 1. 主训练路径

- `run_grpo_math_prm_ursa_8b.sh`
  - 当前 URSA-MATH Stage 3 复现实验的主启动脚本。
  - 负责拼 actor、reward、数据、FSDP、W&B 和 rollout 相关参数。
- `train_colocate.py`
  - 被 `torchrun` 直接调用的真实训练入口。
  - 负责加载 actor / reference / reward model / dataset / trainer，并启动 LightRFT 训练循环。
- `ursa_actor.py`
  - URSA 专用 actor 包装。
  - 让 LightRFT 用 `UrsaForConditionalGeneration` 来加载 policy 模型。

### 2. Reward 与打分路径

- `reward_models.py`
  - 当前目录下所有 reward model 实现都在这里。
  - 现在真正活跃的是 `MathPRMReward` 和基于 URSA-RM-8B 的 PS-GRPO reward 映射。
  - 文件里还保留了一些历史的 Qwen2VL 多 reward 类，但它们已经不属于当前 URSA-MATH Stage 3 主路径。
- `reward_models_utils.py`
  - 负责 reward model 的加载、label 到 recipe 的映射，以及 reward_fn 组装。
  - 当前 `math_prm` / `math_psgrpo` 的路由逻辑都在这里。
- `prm_infer_score.py`
  - 独立的 step-level PRM 打分辅助脚本。
  - 适合在 LightRFT 行为和 URSA-MATH 参考实现之间做单点对比。

### 3. 数据准备与兼容性

- `prepare_ursa_stage3_manifest.py`
  - 把原始 `MMathCoT-1M` Stage 3 数据转换成 LightRFT 需要的 `prompt / images / reference / label` schema。
  - 同时会做一次轻量级 dataset/collate smoke 检查。
- `prepare_ursa_engine_checkpoint.py`
  - 不是当前 `hf` 主线必需，但仍然用于 engine 试验。
  - 它会构造带本地 `ursa_model` 代码和 `auto_map` 元数据的 wrapper checkpoint，供 vLLM/SGLang 尝试加载 URSA。
- `sitecustomize.py`
  - 在当前冻结 Docker 基线下，为 example 目录提供本地运行时兼容性补丁。

### 4. 校验、smoke 与观测工具

- `check_phase2_alignment.py`
  - 检查 LightRFT 的 `MathPRMReward` 是否和 URSA 参考 scorer 保持一致。
- `check_hf_rollout.py`
  - 本地 `hf` rollout 的最小链路校验。
  - 会把 `gather_and_generate()` 的输出和直接 `actor.generate()` 做对比。
- `check_phase6_script_alignment.py`
  - 静态检查当前 Stage 3 启动脚本默认值是否仍然对齐。
- `test_phase2_alignment.py`
  - 当前 URSA Stage 3 主路径的回归测试集合。
  - 覆盖 scorer 对齐、reward 映射、答案抽取、rollout 辅助逻辑等。
- `run_phase3_smoke.sh`
  - 限时 Phase 3 smoke 试跑脚本。
  - 用来验证“能否正常起训、指标是否合理、结束后 GPU 是否清干净”。
- `run_phase7_observation.sh`
  - bounded full-data observation 启动脚本。
- `analyze_phase7_observation.py`
  - 对 Phase 7 保存下来的 trajectories 和训练日志做离线分析。
  - 用于计算 health checklist 和 PRM 图像消融结果。
- `probe_rollout_speed_candidates.py`
  - 不修改 `lightrft/` 主链时，用来比较几种 rollout-like decode 运行形态速度的最小测速脚本。
  - 目前主要用来确认 `gradient_checkpointing` 是 rollout 速度问题的主因。

### 5. 自包含 URSA 运行时

- `ursa_model/`
  - 本地复制的 URSA 模型栈。
  - 包含 config、processor、image processor、projector、vision tower 和 model 定义。
  - 这部分是当前 Stage 3 路径能脱离外部 URSA-MATH repo 直接运行的基础。

## 当前真正需要关注的入口

如果你只关心当前 URSA-MATH Stage 3 复现主线，通常只需要重点看这些文件：

- `run_grpo_math_prm_ursa_8b.sh`
- `train_colocate.py`
- `reward_models.py`
- `reward_models_utils.py`
- `prepare_ursa_stage3_manifest.py`
- `check_hf_rollout.py`
- `test_phase2_alignment.py`

目录中其他文件大多属于：

- 兼容性辅助脚本，
- smoke / observation / profiling 工具，
- 或自包含的 URSA 运行时代码。

## 本机资源路径

当前机器上的关键资源路径如下：

```bash
URSA actor:      /home/ubuntu/URSA-MATH/checkpoints/URSA-8B
URSA reward:     /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B
MMathCoT-1M raw: /home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl
Image root:      /home/ubuntu/URSA-MATH/datasets/URSA-MATH/images
```

Phase 1 生成好的转换结果在：

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl
```

对应 summary 在：

```bash
/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.summary.json
```

## 数据准备

### 原始 schema

URSA 的 raw Stage 3 数据不能直接喂给 `PromptDatasetVL`。raw schema 是：

```json
{
  "image_url": "...",
  "instruction": "...",
  "output": "..."
}
```

LightRFT 训练期望的最小 schema 是：

```json
{
  "prompt": "...",
  "images": ["/abs/path/to/image.png"],
  "reference": "...",
  "label": "math_prm"
}
```

### 使用 `prepare_ursa_stage3_manifest.py`

先做小样本 smoke：

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

直接跑全量转换：

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py
```

关键参数：

- `--input-path`：原始 `MMathCoT-1M` jsonl
- `--image-root`：解析 `image_url` 时使用的图片根目录
- `--output-path`：转换后 manifest 路径
- `--summary-path`：统计结果路径
- `--label`：默认 `math_prm`
- `--prompt-mode question_only|instruction`：默认 `question_only`
- `--max-samples`：只处理前 N 条，适合 smoke
- `--smoke-samples`：转换完之后拿前 N 条做 `PromptDatasetVL` 校验

这个脚本会完成三件事：

1. 把 `instruction` 转成 `prompt`
2. 把 `image_url` 展开成 URSA 图片树下的绝对路径
3. 把 `output` 里 `†Answer:` 后面的最终答案提取成 `reference`

同时它会对缺图直接 fail fast，并执行一次轻量级 dataset/collate 校验。

## 训练方法

先把 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 里的变量改成正确路径。对当前机器，预期值是：

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl"
```

然后执行：

```bash
bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh
```

这个训练脚本已经接好了：

- `--pretrain` 指向 URSA actor
- `--reward_pretrain` 指向 URSA reward model
- `--prompt_data` 指向转换后的 manifest
- `--images_key images`
- `--label_key label`
- `--apply_chat_template`

`train_colocate.py` 里 `reference_key` 默认就是 `reference`，所以当前生成出的 manifest 可以直接用。

## Label 语义

- `math_prm`：只走 PRM reward
- `math_prm_combined`：PRM + 规则正确率 reward

当前 Phase 0 和 Phase 1 默认仍然使用 `math_prm`。

## 输出格式要求

URSA PRM 依赖 actor 输出保持下面这个格式：

```text
Step 1: ...
Step 2: ...
...
†Answer: ...
```

不要改成普通段落式 CoT。PRM 打分要靠 step 边界。

## 常见错误

- 直接把 raw `MMathCoT-1M/train.jsonl` 当成 `--prompt_data`：这是错误用法，必须先转换。
- 转换时遇到缺图：`prepare_ursa_stage3_manifest.py` 直接抛 `FileNotFoundError`，这是预期行为。
- rollout 后 reward 异常或全零：先检查模型输出里是否还保留 `Step N:` 和 `†Answer:`。
- 环境漂移：如果问题来自包版本变动，应恢复 Dockerfile 基线，而不是继续在文档和脚本里适配漂移环境。

## 相关文档

- [`../../plan/MATH_PRM.md`](../../plan/MATH_PRM.md)
- [`./URSA_MIGRATION.md`](./URSA_MIGRATION.md)

## 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](../../LICENSE)。

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
├── prepare_ursa_stage3_manifest.py   # 把 URSA raw jsonl 转成 LightRFT prompt manifest
├── train_colocate.py                 # 主训练入口
├── run_grpo_math_prm_ursa_8b.sh      # URSA-8B + URSA-RM-8B 训练脚本
├── reward_models.py                  # reward 实现，含 MathPRMReward
├── reward_models_utils.py            # reward model 加载与路由
├── prm_infer_score.py                # step-level PRM 打分逻辑
├── test_reward_models.py             # reward 侧测试
├── URSA_MIGRATION.md                 # 从 URSA-MATH 迁移过来的说明
└── ursa_model/                       # 自包含的 URSA 模型代码
```

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

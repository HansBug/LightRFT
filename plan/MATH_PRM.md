# URSA Stage 3 在 LightRFT 中的分阶段任务计划

## 1. 目标与边界

本计划只针对 URSA 论文中的 **Stage 3** 复现与工程落地，不包含：

- Stage 1：URSA-8B 的视觉语言对齐与数学指令微调
- Stage 2：URSA-RM-8B 的 PRM 训练

本次要完成的是：

- 使用已经训练好的 `URSA-8B` 作为 policy model
- 使用已经训练好的 `URSA-RM-8B` 作为 verifier / PRM
- 在 LightRFT 中接通多模态 Stage 3 RL 训练链路
- 将 reward 语义推进到论文中的 **PS-GRPO**
- 逐步把数据流程与训练脚本对齐到论文设定

当前阶段的执行策略需要特别明确：

- **前期先不做“从完整 `MMathCoT-1M` 中抽取 `20K` 候选，再筛到 `15.3K`”的静态筛选流程**
- **先用全量可用数据把训练链路、行为对齐、奖励语义和数据载入跑通**
- **等基础链路稳定后，再补论文中的筛选流程**

换句话说，当前优先级是：

1. 跑通
2. 对齐行为
3. 对齐 reward 公式
4. 再补数据筛选

---

## 2. 论文复现基线

根据论文与现有资料，Stage 3 需要对齐的核心点有：

### 2.1 模型

- policy model：`URSA-8B`
- verifier / PRM：`URSA-RM-8B`

### 2.2 数据

- 你当前持有的原始 RL 数据是完整的 `MMathCoT-1M`
- 论文中的 Stage 3 不是直接用完整 `1M` 训练，而是先从 `MMathCoT-1M` 中抽取 `20K` 候选，再经静态筛选后得到约 `15.3K` 样本
- 但本计划前期阶段先不实现筛选，而是先用全量数据验证训练与 reward 链路

### 2.3 Stage 3 奖励语义

论文的最终奖励不是简单的 `min(step_score)` 或 `avg(step_score)`。

论文使用的是 **PS-GRPO**：

1. PRM 输出 step-level reward sequence
2. 检测 reward sequence 中是否存在明显下降，即 `drop-moment`
3. 结合最终答案正确性，构造最终 rollout reward

公式约束如下：

- `rho = 0.3`
- `gamma = 0.5`
- 正确且无明显下降：`R = 1`
- 正确但有下降：`R = 1 - gamma = 0.5`
- 错误：`R = 0`

### 2.4 论文中可参考的 Stage 3 超参

- Epochs：`2`
- Learning Rate：`2e-6`
- Temperature：`1.0`
- Rollout number per prompt：`8`
- Prompt Max Length：`6048`
- Output Max Length：`3072`
- Precision：`bf16`
- Train Batch Size：`512`
- KL Coefficient：`0.003`

这些超参在前期不要求一次性全部严格对齐，但需要作为后续 phase 的对齐目标。

### 2.5 本机现有资源、路径、状态与使用方式

以下结论基于逐份核对 `/home/ubuntu/URSA-MATH` 下的核心文档：

- `README.md`
- `RUN_GUIDE.md`
- `DATASET_LOAD.md`
- `CODE_EXAMPLES.md`
- `GUIDE.md`
- `paper.md`
- `paper_zh.md`
- `PAPER.md`
- `STAGE3_REPRODUCTION_PLAN.md`
- `report-2026-03-12.md`
- `20260312-URSA-MATH-reading-log.md`
- `DOCUMENTATION_INDEX.md`
- `checkpoints/URSA-8B/README.md`
- `checkpoints/URSA-RM-8B/README.md`

#### 2.5.1 模型 checkpoint

`URSA-8B`

- 本机路径：`/home/ubuntu/URSA-MATH/checkpoints/URSA-8B`
- 当前状态：目录已就绪；存在 `model-00001-of-00007.safetensors` 到 `model-00007-of-00007.safetensors`；`model.safetensors.index.json` 的 `metadata.total_size=32179133444`；磁盘占用约 `30G`
- 架构标识：`config.json` 中 `architectures=["UrsaForConditionalGeneration"]`，`model_type="ursa"`
- 直接使用方式：
  - 原仓库 torch 示例：`python /home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example.py --device cuda:0`
  - 原仓库 standalone 示例：`python /home/ubuntu/URSA-MATH/examples/run_ursa_8b_torch_example_standalone.py --device cuda:0`
  - 原仓库 vLLM 路径：先执行 `bash /home/ubuntu/URSA-MATH/start.sh`，再用 `/home/ubuntu/URSA-MATH/inference/vllm_infer.py` 或 `inference/start_vllm_infer.sh`，其中 `--model` 指向该目录
- LightRFT 接入位置：`examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 中的 `--pretrain` 应直接指向该目录

`URSA-RM-8B`

- 本机路径：`/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B`
- 当前状态：目录已就绪；存在 `model-00001-of-00007.safetensors` 到 `model-00007-of-00007.safetensors`；`model.safetensors.index.json` 的 `metadata.total_size=32167793672`；磁盘占用约 `30G`
- 架构标识：`config.json` 中 `architectures=["UrsaForTokenClassification"]`，`model_type="ursa"`
- 直接使用方式：
  - 原仓库 RM 示例：`python /home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example.py --device cuda:0`
  - 原仓库 standalone RM 示例：`python /home/ubuntu/URSA-MATH/examples/run_ursa_rm_8b_score_example_standalone.py --device cuda:0`
  - 原仓库打分入口：`/home/ubuntu/URSA-MATH/inference/prm_infer_score.py`
- LightRFT 接入位置：`examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 中的 `--reward_pretrain '{"math_prm":"..."}'` 应直接指向该目录
- 额外说明：`reward_models_utils.py` 当前会以 HF 方式直连加载 `UrsaForTokenClassification`，这和 Stage 3 需要的 logit 级 step score 读取方向一致

环境与依赖

- URSA 侧已提供验证过的依赖文件：`/home/ubuntu/URSA-MATH/requirements.txt`
- 文档记录的推荐环境是 `conda activate ursa`
- `requirements.txt` 中已固定 `torch==2.5.1+cu124`、`transformers==4.45.2`，并说明如需仓库内嵌 vLLM 路径需额外执行一次 `bash /home/ubuntu/URSA-MATH/start.sh`
- `report-2026-03-12.md` 里已有 `URSA-8B` / `URSA-RM-8B` 示例脚本在该仓库环境下跑通过的记录；本计划仍应把 LightRFT 侧链路视为待再次 smoke test

Docker 环境基线约束

- 当前仓库唯一的环境基线文件是 `/data/LightRFT/Dockerfile`
- 其中已经明确安装的 pip 依赖、版本和安装顺序应视为 **冻结基线**
- 后续为了接 URSA Stage 3，**不允许** 擅自修改这些包的版本、替换它们、删除它们，或打乱其安装顺序
- 如果后续出现训练、推理、编译或兼容性问题，默认优先通过代码适配、数据适配、脚本参数适配来解决，而不是漂移 Docker 基线
- 只有当任务本身被明确升级为“环境迁移 / Docker 基线重做”时，才允许讨论改动这些包；否则一律以保本一致性为先

当前需要视为冻结的 Docker pip 基线包括：

- `torch==2.9.0`
- `torchvision`
- `torchaudio`
- `ninja`
- `deepspeed==0.18.3`
- `vllm==0.13.0`
- `datasets`
- `librosa`
- `peft`
- `tensorboard`
- `decord`
- `easydict`
- `matplotlib`
- `wandb`
- `mathruler`
- `pylatexenc`
- `flash_attn==2.8.3` 对应 wheel
- `sglang==0.5.6.post2`

这条约束的工程含义需要明确：

- 后续文档、脚本、排障记录中如果提到依赖问题，要先区分“代码问题 / 数据问题 / 参数问题”与“环境基线问题”
- 在没有显式环境迁移任务之前，不把“升级 vLLM / 升级 DeepSpeed / 降级 torch / 改 flash-attn 版本”视为正常修复手段
- Stage 3 复现应建立在当前 Dockerfile 已定义好的运行时基线之上

#### 2.5.2 数据集与图片资产

`MMathCoT-1M`

- 本机路径：`/home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl`
- 当前状态：文件已存在；`wc -l` 为 `1019059`；磁盘占用约 `821M`
- 当前 schema：原始字段是 `image_url / instruction / output`
- 角色定位：这是 Stage 3 policy / RL 的原始数据源；当前计划前期也是从这里出发，而不是先做 `20K -> 15.3K` 筛选
- 图像前缀覆盖：`MathV-360k`、`Multimath`、`DataEngine_Geometry`、`Mavis_Extra`、`VarsityTutors`、`Geo170K`

`DualMath-1.1M`

- 本机路径：`/home/ubuntu/URSA-MATH/datasets/URSA-MATH/DualMath-1.1M/train.jsonl`
- 当前状态：文件已存在；`wc -l` 为 `1100779`；磁盘占用约 `1.8G`
- 当前 schema：原始字段也是 `image_url / instruction / output`
- 角色定位：这是 Stage 2 的 PRM 训练数据与 PRM 行为校验数据，不是 Stage 3 rollout prompt 的直接输入
- 图像前缀覆盖：`MathV-360k`、`Multimath`、`Mavis_Extra`、`DataEngine_Geometry`、`Geo170K`、`VarsityTutors`

图片资产

- 本机根路径：`/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images`
- 当前状态：目录已就绪；磁盘占用约 `32G`
- 已确认存在的关键前缀：
  - `MathV-360k -> data_images`（软链）
  - `Multimath/RGB_images -> ../RGB_images`（软链）
  - `DataEngine_Geometry`
  - `Geo170K`
  - `Mavis_Extra`
  - `VarsityTutors`
- 这与 `DATASET_LOAD.md` 中要求的最终目录布局一致，说明 raw `image_url` 在本机上已具备被解析成真实文件路径的基础

当前缺口

- `/home/ubuntu/URSA-MATH` 下当前没有现成的 Stage 3 `15K` 筛选子集
- `/home/ubuntu/URSA-MATH` 下当前也没有现成可直接喂给 LightRFT 的 `prompt / images / reference / label` 版 Stage 3 manifest
- 也就是说，当前不是“缺模型或缺原始数据”，而是“缺一层从 URSA raw schema 到 LightRFT 训练 schema 的转换”

#### 2.5.3 在 URSA-MATH 仓库中的直接使用方式

数据集验证与载入脚本都已经在原仓库里准备好了，可以直接拿本机现有资源做检查：

- 构造兼容 manifest 并预览载入：`python /home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py`
- 随机抽样检查字段完整性与图片可读性：`python /home/ubuntu/URSA-MATH/examples/validate_dataset_random_loading.py --mode sample --sample-size 10000`
- 走原始推理入口做端到端校验：`python /home/ubuntu/URSA-MATH/examples/validate_dataset_entrypoints.py --policy-model /home/ubuntu/URSA-MATH/checkpoints/URSA-8B --prm-model /home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B`

这些脚本的作用需要明确区分：

- 它们主要验证的是 `URSA-MATH` 原仓库现有 inference loader 能否吃到兼容 manifest
- 它们不是 LightRFT Stage 3 训练用的最终数据格式生成脚本

#### 2.5.4 在 LightRFT 中应如何接入

LightRFT 当前 Stage 3 训练脚本的关键参数要求是：

- `--pretrain`：指向 `URSA-8B`
- `--reward_pretrain`：指向 `URSA-RM-8B`
- `--prompt_data`：指向训练集
- `--input_key "prompt"`
- `--images_key "images"`
- `--reference_key "reference"`
- `--label_key "label"`

这意味着当前不能把 `/home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl` 原样直接传给 `--prompt_data`，因为它的 raw schema 是：

- `image_url`
- `instruction`
- `output`

而 LightRFT 当前希望消费的是至少如下 schema：

```json
{
  "prompt": "...",
  "images": ["/abs/path/to/image.png"],
  "reference": "...",
  "label": "math_prm"
}
```

当前阶段建议采用的转换语义应固定为：

- `prompt`：从 raw `instruction` 中抽出给 actor rollout 的题面文本
- `images`：由 `/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/<image_url>` 解析成绝对路径列表
- `reference`：从 raw `output` 中抽取 `†Answer:` 后的最终答案
- `label`：
  - `Phase 3` 先用 `math_prm`
  - `Phase 4+` 再切到 `math_psgrpo`

因此，当前脚本里最应该落成的本机实际路径不是占位符，而应是：

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/path/to/converted_lightrft_stage3_manifest.jsonl"
```

这里最后一个路径必须强调：

- 它应该是“转换后的 LightRFT 训练 manifest”
- 不是 raw `MMathCoT-1M/train.jsonl`
- 也不是 `DualMath-1.1M/train.jsonl`

#### 2.5.5 `prepare_ursa_stage3_manifest.py` 使用指南

当前仓库里已经补上了专门的转换脚本：

- 脚本路径：`/data/LightRFT/examples/math_prm/prepare_ursa_stage3_manifest.py`
- 作用：把 URSA raw `image_url / instruction / output` 转成 LightRFT 可直接训练的 `prompt / images / reference / label`
- 附加动作：脚本会同步做一次 `PromptDatasetVL` smoke 校验，避免只生成文件、不验证载入

默认输入输出

- raw 输入：`/home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl`
- 图片根目录：`/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images`
- 默认输出 manifest：`/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl`
- 默认 summary：`/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.summary.json`

脚本生成后的单条样本结构如下：

```json
{
  "data_source": "URSA-MATH/MMathCoT-1M",
  "prompt": "...",
  "images": [
    "/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/..."
  ],
  "reference": "...",
  "ground_truth": "...",
  "label": "math_prm",
  "reward_model": {
    "ground_truth": "..."
  },
  "extra_info": {
    "source_index": 0,
    "raw_image_url": "...",
    "image_prefix": "...",
    "prompt_mode": "question_only"
  }
}
```

最常用的两种跑法如下。

先做小样本 smoke：

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

直接做全量转换：

```bash
python examples/math_prm/prepare_ursa_stage3_manifest.py
```

当前脚本的关键参数含义需要明确：

- `--input-path`：原始 `MMathCoT-1M` jsonl 路径
- `--image-root`：`image_url` 对应的图片根目录
- `--output-path`：输出的 LightRFT manifest 路径
- `--summary-path`：输出的统计与校验 summary 路径
- `--label`：写入 manifest 的样本标签，当前默认是 `math_prm`
- `--prompt-mode`：
  - `question_only`：从 raw `instruction` 中抽题面，当前默认用这个
  - `instruction`：保留整个 raw instruction
- `--max-samples`：只转换前 N 条，适合 smoke
- `--smoke-samples`：转换完成后，拿前 N 条做 `PromptDatasetVL` dataset/collate 校验

转换后的产物应该这样接入训练：

```bash
PATH_TO_YOUR_BASE_MODEL="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
PATH_TO_URSA_RM="/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
PATH_TO_YOUR_MATH_DATASET="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl"
```

然后把最后一个路径传给：

- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 里的 `PATH_TO_YOUR_MATH_DATASET`
- 或者直接传给 `train_colocate.py` 的 `--prompt_data`

这几个使用注意点必须写清楚：

- 不允许把 raw `MMathCoT-1M/train.jsonl` 直接拿去做 `--prompt_data`
- 如果脚本发现缺图，会直接抛 `FileNotFoundError`，这是预期行为，不是可忽略 warning
- 当前默认 `label=math_prm`，意味着 reward 走 PRM-only 路径；如果要做混合 reward，再显式改 label
- 当前默认 `reference` 已经从 raw `output` 里的 `†Answer:` 抽出，可直接供 reward 侧使用
- 后续如果只改数据和脚本参数即可解决问题，就不要碰 `/data/LightRFT/Dockerfile` 中冻结的 pip 依赖版本
- 当前 Phase 1 已经实际跑通过一次全量转换，生成文件就是：
  - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl`
  - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.summary.json`

---

## 3. 当前仓库已有基础

当前 `examples/math_prm/` 已经具备以下能力：

- `train_colocate.py` 已能识别 URSA actor 并切换到 `UrsaActor`
- `reward_models_utils.py` 已具备 `URSA-RM-8B` 的基础加载路径
- `prm_infer_score.py` 已具备 step-level PRM score 提取逻辑
- `PromptDatasetVL` 已支持 `prompt / images / reference / label` 这类字段形态

因此当前不是从零开始，而是：

- actor 侧已有基础
- PRM 推理已有基础
- 多模态数据挂载已有基础

真正需要补的是：

- 行为对齐检查
- 多模态 PRM 真正生效
- PS-GRPO reward 公式
- 数据链路与训练脚本的阶段性对齐

---

## 4. 总体执行原则

### 原则 1：先保证主链路端到端可运行

前几阶段不先追求论文完全等价，而是先保证下面这条链路完整：

- 数据集载入
- 图像载入
- actor rollout
- PRM 打分
- reference 传递
- reward 计算
- trainer 消费 reward
- rollout engine 权重更新

### 原则 2：行为对齐本身就是一类显式任务

不能只看“代码能跑”。以下都属于必须显式检查的“行为对齐”项：

- URSA actor 的输入输出行为是否与原实现一致
- URSA-RM 的 step marker、图像处理、logit 读取方式是否一致
- 最终答案抽取与 correctness 判定是否与原始脚本/预期语义一致
- LightRFT 中 reward 的使用位置是否与论文定义一致

### 原则 3：前期先用全量数据，不先做筛选

当前计划中的前几个 phase：

- 不实现从完整 `MMathCoT-1M` 中抽取 `20K` 候选的步骤
- 不实现 8 次离线采样筛选
- 不先裁到 15.3K

先用全量可用数据验证：

- 数据 schema
- 图像路径
- reference 完整性
- reward 分布
- 训练是否稳定

### 原则 4：尽量少改 Trainer 主结构

优先修改：

- `examples/math_prm/reward_models.py`
- `examples/math_prm/reward_models_utils.py`
- `examples/math_prm/train_colocate.py`
- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

尽量不大改：

- `SPMDPPOTrainerVL`
- replay buffer 主结构
- advantage calculator 主结构

### 原则 5：Docker 基线保本一致

`/data/LightRFT/Dockerfile` 里已经安装的 pip 包、版本与安装顺序，默认都视为当前项目的运行时基线。

因此后续做 URSA Stage 3 时：

- 不把改 Docker 依赖版本作为常规调试手段
- 不为了临时修一个问题去升级/降级 `torch`、`deepspeed`、`vllm`、`flash_attn`、`sglang` 等关键包
- 所有复现、排障、对齐工作默认在现有 Docker 基线下完成
- 如确实怀疑环境本身有问题，需要单独立项说明是“环境迁移”，不能混在 Stage 3 常规开发里顺手改

---

## 5. 分阶段任务计划

下面按可执行顺序拆分 phase。每个 phase 都有明确目标、产出和 checklist。

### Phase 0：范围冻结与基线确认

状态：

- 已完成（`2026-03-18`）

目标：

- 明确当前阶段只做 Stage 3
- 明确前期不做数据筛选，先跑全量
- 明确哪些内容属于“必须对齐”，哪些内容可以延后

产出：

- 一份可执行的阶段计划
- 一份当前代码与论文要求的差异清单

Checklist：

- [x] 明确当前只做 `URSA-8B + URSA-RM-8B + Stage 3 RL`
- [x] 明确 Stage 1 / Stage 2 不在本轮范围内
- [x] 明确前期阶段不做“从完整 `MMathCoT-1M` 中抽 `20K` 候选再筛到 `15.3K`”的静态筛选
- [x] 明确前期先使用全量数据进行链路验证
- [x] 明确行为对齐属于必须检查项，而不是“可选优化”
- [x] 明确数据集载入、图像载入、reference 传递都属于必须检查项

已完成产出：

- 已在本文件第 `2.5` 节补齐 `/home/ubuntu/URSA-MATH` 的模型、数据集、图片树、原仓库使用方式、LightRFT 接入方式
- 已明确 `/data/LightRFT/Dockerfile` 是冻结运行时基线，后续默认不允许改 pip 包版本或安装顺序
- 已把当前阶段的范围冻结为：
  - 先用全量 `MMathCoT-1M`
  - 先完成 LightRFT schema 接线
  - 先不做论文 `20K -> 15.3K` 静态筛选

### Phase 1：数据集载入与样本 schema 打通

状态：

- 已完成（`2026-03-18`）

目标：

- 确保全量 Stage 3 数据能在 LightRFT 中被稳定读取
- 确保样本中的文本、图像、答案引用字段不会在载入过程中丢失

重点：

- 这一阶段先不关心筛选逻辑
- 先关心“数据能否完整进训练系统”

建议检查字段：

- `prompt`
- `images`
- `reference`
- `label`

Checklist：

- [x] 明确 `/home/ubuntu/URSA-MATH/datasets/URSA-MATH/MMathCoT-1M/train.jsonl` 与 `/home/ubuntu/URSA-MATH/datasets/URSA-MATH/DualMath-1.1M/train.jsonl` 的 raw schema 是 `image_url / instruction / output`
- [x] 明确 `MMathCoT-1M` 是 Stage 3 policy / RL 原始数据源，`DualMath-1.1M` 主要用于 PRM 训练或 PRM 行为校验
- [x] 明确 raw `train.jsonl` 不能直接作为 `--prompt_data` 传给 `PromptDatasetVL`
- [x] 先从全量 raw 数据生成一份 LightRFT 兼容 manifest：`prompt / images / reference / label`
- [x] 确认训练数据 JSON/JSONL schema 与 `PromptDatasetVL` 兼容
- [x] 确认 `prompt` 能正确进入训练 prompt 构建流程
- [x] 确认 `images` 字段能被 `PromptDatasetVL` 和图像预处理链正确读取
- [x] 确认图像路径格式稳定可用，优先绝对路径，并基于 `/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images` 展开
- [x] 确认缺图、坏图、空图样本的失败模式可观测
- [x] 确认 `reference` 能从数据集一路传递到 reward 计算入口
- [x] 确认 `label` 能用于区分 `math_prm` / `math_psgrpo` 等 reward 路径
- [x] 明确当前本机未发现现成 `15K` Stage 3 筛选集或现成 LightRFT-compatible manifest，需要自行构建
- [x] 确认全量数据中不存在大面积缺失 `reference` 的样本
- [x] 确认全量数据中图像字段的数量分布与格式分布
- [x] 确认 mixed multimodal 数据在 batch/collate 阶段不会丢字段
- [x] 先跑一轮小规模 dataset smoke test，打印单条样本和 batch 结构
- [x] 再跑全量数据扫描，统计字段完整率与异常样本数量

已完成产出：

- 新增转换脚本：`examples/math_prm/prepare_ursa_stage3_manifest.py`
- 已生成全量 Stage 3 LightRFT manifest：
  - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.jsonl`
- 已生成全量扫描 summary：
  - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.summary.json`
- 已生成图片随机抽样打开校验结果：
  - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_prm.image_sample_check.json`

本轮实测结果：

- `rows_seen = 1019059`
- `rows_written = 1019059`
- `prompt_fallback_rows = 763`
- `reference_fallback_rows = 2977`
- `empty_prompt_rows = 0`
- `empty_reference_rows = 0`
- `images_per_sample_counts = {"1": 1019059}`
- `image_prefix_counts`：
  - `MathV-360k = 319619`
  - `Multimath = 264205`
  - `DataEngine_Geometry = 186471`
  - `Mavis_Extra = 141406`
  - `VarsityTutors = 55162`
  - `Geo170K = 52196`
- 随机抽样 `10000` 条 manifest 记录，图片真实打开成功 `10000/10000`

字段链路确认：

- `PromptDatasetVL` 当前可以直接消费该 manifest 的 `prompt / images / reference / label`
- `PromptDatasetVL.collate_fn()` 会返回 `(prompts, images, refs, labels)`
- `ppo_trainer_vl.py` 会将 dataloader batch 作为 `all_prompts / all_images / all_references / all_labels` 传入经验构造器
- `experience_maker_vl.py` 与 `fast_exp_maker.py` 会继续把 `references / labels / raw_images / prompt_and_output` 送入 reward model forward

因此，Phase 1 当前已经完成的不是抽象分析，而是：

- raw schema 已被转成 LightRFT 训练 schema
- 全量数据已实际扫过
- 图片路径已实际解析成绝对路径
- 样本已通过 `PromptDatasetVL` smoke 校验
- `reference / label / images` 到 reward 入口的代码链路已确认存在

### Phase 2：URSA actor / PRM 载入与基础行为对齐

状态：

- 已完成（`2026-03-18`）

目标：

- 确认 LightRFT 中加载出的 `URSA-8B` 和 `URSA-RM-8B` 至少在关键行为上与原始实现一致
- 确认当前链路不是“能 import 就算通过”，而是核心行为一致

行为对齐范围：

- 模型加载
- processor 使用
- 图像输入处理
- step marker 插入
- logits 读取
- score 聚合语义

Checklist：

- [x] 确认 `train_colocate.py` 对 URSA actor 的识别逻辑稳定可复现
- [x] 确认 `UrsaActor` 的 tokenizer / processor / forward / generate 行为与原始 URSA 使用方式一致
- [x] 确认 `URSA-RM-8B` 强制走 HF 直连路径，而不是错误走 engine 路径
- [x] 确认 `UrsaProcessor` 使用方式与原实现保持一致
- [x] 确认 PRM 的图像输入不再被忽略，而是真实传入
- [x] 确认 step marker 仍然使用论文与原实现要求的特殊标记
- [x] 确认 `Step N:` 格式要求被保留，且不会被 chat template 破坏
- [x] 确认 `†Answer:` 的终答案格式要求被保留
- [x] 确认读取 step score 的 token 位置与原脚本一致
- [x] 确认图像占位 token / padding 逻辑与现有 URSA-RM 推理逻辑一致
- [x] 确认单条样本下，LightRFT 侧 PRM 输出与原始推理脚本输出可对比
- [x] 确认 `min / avg / last` 等聚合结果在对齐测试中可复现
- [x] 对齐失败时记录是文本格式问题、图像问题还是 processor 行为问题

已完成产出：

- `examples/math_prm/train_colocate.py`
  - 新增 `load_actor_tokenizer_processor()`，对 `URSA-8B` 显式走 `UrsaProcessor.from_pretrained(...)`
  - 避免继续误走 `AutoProcessor` 返回 tokenizer 的错误路径
  - 顺手修复训练入口顶层对不存在 `get_vlm_for_sequence_regression` 的硬依赖，改为仅在显式启用 critic 时再懒加载
- `examples/math_prm/reward_models_utils.py`
  - 修复 reward model shared-base 逻辑
  - `math_prm` 现在会强制走 `_load_ursa_prm_model(...)`
  - 即使命令行全局带了 `--rm_use_engine`，`URSA-RM-8B` 也不会再被错误预加载为 engine base
- `examples/math_prm/reward_models.py`
  - `MathPRMReward` 现在会真实消费 rollout 传下来的 `raw_images`
  - 从 `prompt_and_output` 抽 question 时会清理 `<|image|>` / `<image>` / vision placeholder token，确保 PRM 输入语义与原始 `prepare_input()` 一致
  - step marker 插入、`575` image pad 对齐、step-logit 读取位置以及 `min/avg/last` 聚合均保持与原脚本一致
- `examples/math_prm/ursa_model/`
  - 新增 `attrdict_compat.py`，使用 `easydict`/本地 fallback 兼容当前 Docker 基线中缺失的 `attrdict`
  - 修复 `modeling_ursa.py` 在当前 transformers 版本下访问 `_supports_sdpa` 时的初始化期异常
- 新增 Phase 2 对齐校验脚本：
  - `examples/math_prm/check_phase2_alignment.py`
- 新增 Phase 2 轻量单测：
  - `examples/math_prm/test_phase2_alignment.py`
- 新增单条样本 GPU 对齐结果：
  - `/data/LightRFT/tmp/ursa_stage3/phase2_alignment_smoke.json`

本轮实测结果：

- `UrsaProcessor.from_pretrained('/home/ubuntu/URSA-MATH/checkpoints/URSA-8B')` 在当前环境下已可直接加载，得到：
  - `processor = UrsaProcessor`
  - `tokenizer = Qwen2TokenizerFast`
  - `image_processor = VLMImageProcessor`
- 训练入口 `train_colocate.py` 已可被直接 import，`is_ursa_model('/home/ubuntu/URSA-MATH/checkpoints/URSA-8B') == True`
- `python -m unittest -q examples.math_prm.test_phase2_alignment`
  - 已通过（`4` 个测试）
- `python examples/math_prm/check_phase2_alignment.py --device cuda:0`
  - 已实际跑通 `URSA-RM-8B` 单条样本对齐
  - `prepared_input_match = true`
  - `reference.min = 0.87109375`
  - `lightrft.min = 0.87109375`
  - `reference.avg = 0.94140625`
  - `lightrft.avg = 0.94140625`
  - `delta.min = 0.0`
  - `delta.avg = 0.0`
  - `within_tolerance = true`

对齐失败分类约定：

- 文本格式问题：
  - `Step N:` 丢失
  - `†Answer:` 丢失
  - chat template 残留 vision placeholder 污染 question 文本
- 图像问题：
  - `raw_images` 未传入 PRM
  - 图片对象/路径未被正常转成单图输入
- processor 行为问题：
  - `AutoProcessor` 误返回 tokenizer
  - `UrsaProcessor` 未被显式加载
  - checkpoint 在当前 transformers 版本下初始化失败

### Phase 3：先跑通“全量数据 + 基础 reward”训练链路

目标：

- 在不引入数据筛选的前提下，先跑通一版完整训练
- 确认整个 RL 训练主链路没有结构性断点

说明：

- 这一阶段允许 reward 先保持“基础版 math_prm”
- 目标是先验证主链路可运行、日志可观测、显存和吞吐行为可接受

当前结果（2026-03-19）：

- 已完成一次严格按边界执行的 `time-boxed smoke run`：
  - 启动脚本：`/data/LightRFT/examples/math_prm/run_phase3_smoke.sh`
  - 训练日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_005513.log`
- 当前结论不是“脚本未崩即通过”，而是：
  - **主链路已打通**
  - **本次 smoke run 健康性判定为失败**
- 已确认打通的链路包括：
  - dataloader 正常起批
  - 多模态 actor rollout 正常返回
  - reward model 在训练环节拿到 `prompt_and_output / raw_images / references`
  - trainer 正常打印 `reward / kl / pg / response_length / total_length`
  - smoke run 结束后相关进程已清理，8 张 GPU 已回收为 `1 MiB`
- 这次判定为失败的直接原因包括：
  - 第一批 rollout 的 `step 0 generate length` 仍然是 `min=max=mean=1024`
  - 第一批样例生成出现 `StepStep 1:` 和大段 `"Camp Miniwauca"` 异常重复拖尾
  - 第二批样例虽然中前段推理和答案 `37` 看起来正确，但后续继续出现大段 `7777...` 拖尾，并重复输出 `†Answer: 37...`
  - `response_length` 持续在 `943 ~ 946` token，高于“健康 smoke”应有的紧凑回答形态
  - `format_reward=1.0` 不能掩盖生成文本已经明显跑偏，因此不能将本次试跑记为通过
- 这次 smoke run 中可确认的正向信号包括：
  - reward 不再是全 `0` / 全 `1` / 单一常数
  - `kl` 为有限值，且从 `0 -> 0.0251 -> 0.0585`
  - `pg` / `ret` / `model_reward_mean` 均有正常数值输出
  - 没有出现 OOM、死锁、图像载入失败或 reward 全零

Phase 3 范畴内修复后的第二次 smoke（2026-03-19）：

- 试跑日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_091453.log`
- 本轮修复保持在 Phase 3 范畴内，没有提前引入 Phase 4 的 correctness / drop-moment / `math_prm_combined`
- 本轮实际加入的修复包括：
  - 更严格的 system prompt，明确要求首个 `†Answer:` 后立即停止
  - 更保守的 smoke decode 参数：`temperature=0.8`、`top_p=0.95`、`top_k=50`、`repetition_penalty=1.05`、`no_repeat_ngram_size=4`
  - 仅对 `math_prm` / `math_prm_combined` 生效的 rollout 后处理：对首个 `†Answer:` 后的异常拖尾做截断，并清理 `StepStep 1:` / 重复 answer marker / 重复字符尾巴
- 这次 smoke 的直接结果是：
  - 第一批和第二批都触发了 postprocess 截断
  - 第一批 `sanitized 2/2 outputs`，平均裁掉 `927.5` token，最大裁掉 `953`
  - 第二批 `sanitized 2/2 outputs`，平均裁掉 `904.0` token，最大裁掉 `934`
  - 第一批样例已不再出现 `Camp Miniwauca` 长重复尾巴，输出恢复为干净的 `Step ... / †Answer: yes`
  - 第二批样例已不再出现 `7777...` 和重复 `†Answer:`，输出恢复为单个 `†Answer:` 行
  - 训练侧长度指标明显回落：
    - 第一批 `rollout_response_length=88`
    - 第二批 `rollout_response_length=117`
    - 汇总统计 `Response Length = 117.0 ± 42.4`，`hit_max=0.0000`
  - 训练侧 metrics 重新回到可读范围：
    - 第一批 train `glen=104/124`，`tlen=264/287`，`kl=0 -> 0.0234`
    - 第二批 train `glen=130/145`，`tlen=298/313`，`kl=0.0148 -> 0.0241`
  - smoke 结束后相关进程已清理，8 张 GPU 已再次回收到 `1 MiB`

但这次还不能简单记成“Phase 3 完全通过”，原因也需要明确保留：

- raw generate 在进入 postprocess 前，第一批仍然出现 `step 0 generate length = 1024`
- 第二批虽然格式已被修回正常，但样例解题内容本身仍然答错了，输出成了 `41` 而不是 reference `37`
- 本轮 reward 也明显偏低：
  - 第二批 `model_scores` 只有 `0.0192` 和 `0.0011`
  - 汇总 `Total Reward = 0.0101 ± 0.0128`
- 因此当前状态应记录为：
  - **Phase 3 的“明显异常长尾 / 乱码 / 重复 answer marker”问题，已在本阶段内被显著压下**
  - **Phase 3 的训练侧 metrics 已从“明显异常”恢复到“可继续观察与迭代”**
  - **但 raw generation 仍有强烈超长倾向，且答案正确性/PRM 分数仍偏弱，因此还不应直接视为最终健康通过**

Phase 3 范畴内修复后的第三次 smoke（2026-03-19）：

- 试跑日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_095529.log`
- 这次新增的修复重点不是 reward，而是把 stopping 再往前推到 HF 本地生成链路本身：
  - 让结构化 answer 规则在 generation 时就尝试强制收尾到 `eos`
  - 修正 local HF inference 里 `output_token_ids` 的截取方式，不再把 `eos` 后的整段 padding 一起误算成“生成长度”
- 本轮 smoke 在拿到足够判断健康度的首批日志后主动停止，并清理了相关进程与 GPU
- 这轮最关键的变化是：
  - 第一批 `step 0 generate length` 不再是伪 `1024`
  - 实际变为：`min=60`、`max=256`、`mean=120.25`、`median=105`
  - 说明 generation-time stopping / output trimming 已经把“首批 raw generate 结构性顶满上限”的问题显著压下
- 同时：
  - postprocess 只剩 `sanitized 1/2 outputs`
  - 而且只裁掉了 `2` 个 token
  - 这和上一轮动辄裁掉 `900+` token 相比，说明当前 stopping 已不再主要依赖后处理兜底
- 第一批样例输出仍保持干净的 `Step ... / †Answer: yes`
- 第一批 `rollout_response_length=87`，仍处于上一轮已经恢复出的健康短响应区间

因此，Phase 3 当前状态应再往前更新一档：

- **raw generation 顶满 `1024` 的主问题，在最新 smoke 首批里已被修复到基本可控**
- **当前主要剩余问题，已经从“结构化 stopping 异常”转移到“答案正确性弱、PRM reward 偏低”**
- **后续 Phase 3 继续迭代时，应把关注点更多放到 prompt/data 对齐与正确性质量，而不是继续把主要时间花在长尾乱码收尾上**

Phase 3 当前总结（2026-03-19）：

- 当前阶段状态可以概括为：
  - **训练主链路已打通**
  - **结构化输出与 stopping 问题已基本修住**
  - **还不能记为最终健康通过**
  - **剩余主问题已收敛为 correctness 偏弱、PRM reward 偏低**
- 也就是说，Phase 3 现在已经不是“脚本一跑就长尾乱码、完全不正常”，而是：
  - rollout / reward / PPO / cleanup 都能正常走通
  - `Step N:` / `†Answer:` 格式已经基本稳定
  - 生成长度已经回到合理区间
  - 但模型答案质量还没有稳定到可以直接宣告 Phase 3 结束

这几轮里实际采用的修复方法也需要统一记录为：

- 第一步：确认问题主要不在 Phase 4 缺失，而在 Phase 3 自身的 rollout / decode / stopping
  - 依据是最早那批异常输出发生在第一次 PPO 更新前
- 第二步：先用 Phase 3 范畴内的 postprocess 兜底
  - 对 `math_prm` / `math_prm_combined` 输出做首个 `†Answer:` 后截断
  - 清理 `StepStep 1:`、重复 `†Answer:`、重复字符尾巴、长重复词串
- 第三步：收紧 smoke decode 参数
  - 使用更保守的 `temperature / top_p / top_k / repetition_penalty / no_repeat_ngram_size`
- 第四步：把 stopping 再前推到 HF 本地生成阶段本身
  - 不再只依赖 postprocess 收尾
  - 在 generation-time 触发结构化 answer 规则并强制向 `eos` 收尾
- 第五步：修正 local HF inference 的输出截取
  - 不再把 `eos` 后面的 padding 误记进 `output_token_ids`
  - 从而让 `step 0 generate length` 和后续统计回到真实值

这些修复之后，问题演变可以概括为：

- 最开始的问题：
  - raw generate 固定顶满 `1024`
  - 输出里有 `Camp Miniwauca`、`7777...`、重复 `†Answer:`、`StepStep 1:`
- 中间状态：
  - 依靠 postprocess 已能把脏尾巴裁掉
  - 训练侧 `response_length` 已明显恢复
  - 但 raw generate 本身仍然假性顶满
- 当前状态：
  - raw generate 长度本身已降到合理范围
  - postprocess 只剩轻量补边角
  - Phase 3 的主要矛盾已经转成“如何把答案质量和 reward 拉起来”

对当前异常原因的判断（2026-03-19）：

- 当前判断是：**Phase 3 异常生成的主因，不是因为 Phase 4 的 PS-GRPO / correctness / drop-moment 还没上**
- 直接依据是：异常输出出现在第一次 PPO 更新之前
  - 训练日志 `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_005513.log` 中，先看到首批 rollout 的异常输出和 `step 0 generate length = 1024`
  - 真正 actor 训练日志是在之后才开始打印
  - 因此 `StepStep 1:`、`Camp Miniwauca`、`7777...` 这些异常，不可能是 Phase 4 reward 公式缺失“制造出来”的
- 但也需要明确：
  - **Phase 4 没上，确实会让这些异常更难被 reward 压下去**
  - 它是“放大器”或“缺少额外约束”，不是当前第一次 rollout 就异常的首因

当前更可能的原因排序：

- 高优先级怀疑：rollout stopping / decoding / structured-format 对齐不够
  - 首批 rollout 直接顶满 `generate_max_len=1024`
  - 说明当前 prompt + decode 行为下，模型不会稳定在首个 `†Answer:` 后自然停下
- 中高优先级怀疑：Phase 3 baseline reward 对“答完后继续拖尾”惩罚不够
  - 本阶段 reward 明确定义为 `min(step_scores)`
  - `math_prm` 路径下全局 `format_reward` 已不参与最终 reward
  - 因此如果前面几步和首个答案行看起来还像样，后面即使继续胡写，仍可能拿到不低分数
- 中优先级怀疑：full-data 样本分布与当前 system prompt 有一定错位
  - 例如日志里首条样本是图像问答式的 `Is the landscape flat?`
  - 但当前 system prompt 强约束成“math question + Step N + †Answer”
  - 这种 schema / prompt 张力会放大异常 completion
- 低优先级怀疑：单纯因为 Phase 4 算法还没实现
  - 这不足以解释“第一次 PPO 更新前就已经出现异常长尾”

因此，Phase 3 内的修复优先级需要固定为：

- 先修 prompt / stopping / truncation / decode hygiene
- 先确认首个 `†Answer:` 后能稳定停下，或至少能在训练侧被安全截断
- 先确认 `response_length`、样例输出质量、`reward / kl / loss` 重新朝正常方向前进
- **不要**把 Phase 4 的 correctness / drop-moment 提前混入，用 reward 升级去掩盖 Phase 3 的 rollout 结构问题
- 如果 Phase 3 在这些修复后仍然异常，再进入 Phase 4 时再评估 reward 盲区对训练趋势的二次影响

Phase 3 的试跑边界需要额外固定为：

- 允许直接启动 `bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- 但第一次以及后续每次“链路验证型”试跑，都必须是 **time-boxed smoke run**，不能无边界长时间挂在 GPU 上观察
- 默认 wall-clock 上限设为 `20` 分钟；如果在这之前已经出现明确报错，或已经拿到足够判断链路是否健康的首批训练日志，则应立即停止，不继续空耗 GPU
- 这里“足够判断趋势是否正常”的最小观察目标包括：
  - dataloader 正常起批
  - actor rollout 正常返回
  - reward 正常产生且不是大面积常数
  - trainer 正常打印至少一轮可读日志（如 reward / kl / response length / loss 等）
  - 没有立即出现 OOM、死锁、hang 住、图像载入失败、reward 全零等结构性错误
- 这里还需要明确一条判断原则：
  - **不能只看“训练脚本没炸”**
  - 还必须显式观察训练过程中的关键 metrics，判断它们是不是在朝“基本正常”的方向前进，而不是虽然还能跑日志、但 reward / kl / loss / response format / response length 已经明显异常
- Phase 3 smoke run 中至少应重点观察以下 metrics 及其变化趋势：
  - `reward`：不能长时间恒为单一常数，也不能大面积塌成全 `0` / 全 `1`
  - `kl`：不能从一开始就异常飙高到明显失控，也不能完全异常为 `0`
  - `response_length` / `total_length`：不能大面积异常过短、空响应、或持续顶满长度上限
  - `loss`：应为有限值，不能持续 `nan` / `inf`，也不能一开始就明显爆炸
  - 生成格式质量：`Step N:` / `†Answer:` 不能大面积丢失
  - 如日志中可见：`grad_norm` / `clip_ratio` / `advantage` / `reward std` 等，也应确认没有明显异常塌缩或爆炸
- 如果观察到的情况是：
  - 脚本虽然还在跑，但 metrics 明显异常
  - 或 metrics 没有朝正常区间收敛，而是在持续恶化
  - 或 rollout 输出已经明显偏离预期格式
  那么本次 smoke run 仍应判定为 **失败**，不能因为“训练没崩”就把它算作 Phase 3 通过
- 也就是说，Phase 3 的第一次运行目标不是“尽量多跑”，而是“在严格时间边界内确认主链路是否健康”

Phase 3 smoke run 的退出与清理要求也需要固定为：

- 一旦达到时间上限、确认趋势正常、或出现报错，必须立即结束本次试跑
- 结束后必须清理本次试跑拉起的全部相关进程，避免残留 `torchrun`、`python` 子进程、rollout engine、reward model 进程继续占用 GPU
- 清理完成后必须再确认一次 GPU 已释放，再进行下一轮修改或下一次试跑
- 如果后续需要更长时间观察，也应在 Phase 3 smoke run 通过后，再显式发起第二轮“更长时长试跑”，而不是把第一次链路验证直接跑成无上限全量训练

Phase 3 的 reward 方案需要明确固定为：

- baseline reward 使用 `math_prm`
- `math_prm` 在这一阶段只表示：`URSA-RM-8B` 输出 `step_scores` 后做 `min(step_scores)` 聚合得到 sequence-level scalar reward
- 这一阶段 **不引入** PS-GRPO 的 `drop-moment`
- 这一阶段 **不引入** outcome correctness
- 这一阶段 **不引入** rule reward
- 这一阶段 **不引入** `math_prm_combined`

这样定义 Phase 3 的原因是：

- 先把多模态 actor rollout、PRM 打分、reward 回传、trainer 消费这条主链路跑通
- 避免在主链路尚未稳定时，把 correctness、drop-moment、规则判定等额外变量一起混进来
- 让后续 Phase 4/5 中 reward 语义升级时，能够清楚区分“主链路问题”和“reward 定义问题”

但需要特别注意：

- 当前 `mix_rewards()` 里有一个全局 `format_reward_fn()`，它基于 `<think>...</think>` 语义，与 URSA 的 `Step N:` / `†Answer:` 格式并不对齐
- 因此 Phase 3 中虽然名义上使用 `math_prm`，但实现上必须把 `math_prm` 路径下这个不兼容的 format reward 去掉、置零或改成显式不参与
- 否则 Phase 3 的 reward 会变成“PRM scalar + 一个不对齐的格式分”，这不符合本阶段的 baseline 定义

也就是说，Phase 3 的最终目标 reward 形态应当是：

- `reward = min(step_scores)`

而不是：

- `reward = min(step_scores) + format_reward`
- `reward = min(step_scores) + rule_reward`
- `reward = PS-GRPO reward`

Checklist：

- [x] 明确 Phase 3 baseline reward 使用 `math_prm`
- [x] 明确 `math_prm` 在本阶段的语义就是 `min(step_scores)`
- [x] 明确 Phase 3 不引入 `drop-moment`
- [x] 明确 Phase 3 不引入 outcome correctness
- [x] 明确 Phase 3 不引入 `math_prm_combined`
- [x] 明确 Phase 3 不引入 rule reward
- [x] 修正 `mix_rewards()` 中与 URSA 格式不对齐的全局 format reward，使其不参与 `math_prm` 路径
- [x] 确认全量数据可被 dataloader 稳定消费
- [x] 确认 actor rollout 能在多模态样本上正常生成
- [x] 确认 reward model 能在训练环节稳定收到 `prompt_and_output`
- [x] 确认 reward model 能在训练环节稳定收到 `raw_images`
- [x] 确认 reward model 能在训练环节稳定收到 `references`
- [x] 确认 reward 输出能被 trainer 正常记录和消费
- [x] 确认 reward 不会大面积恒为 0、恒为 1 或恒为同一常数
- [x] 确认生成结果满足 `Step N:` / `†Answer:` 的基本格式要求（经 Phase 3 stopping/truncation 修复后）
- [x] 确认 rollout 后的权重同步回推理引擎链路正常
- [x] Phase 3 的第一次训练验证固定采用 time-boxed smoke run，而不是无上限长跑
- [x] 为 smoke run 设置明确 wall-clock 限制（默认 `20` 分钟）
- [x] 在时间上限内至少观察到一轮足以判断链路健康度的训练日志
- [x] smoke run 期间显式检查关键 metrics 的方向性，而不是只确认脚本未报错
- [x] 确认 `reward / kl / loss / response_length / 格式质量` 没有明显朝异常方向发展（训练侧已恢复到可读区间）
- [x] 如果脚本未崩但 metrics 明显异常，明确将该次 smoke run 记为失败而不是通过
- [x] 确认至少能跑通 smoke test 训练
- [x] smoke run 结束后清理全部相关进程，避免残留进程持续占用 GPU
- [x] smoke run 结束后再次确认 GPU 已释放
- [ ] 确认再进行一次更长时长的全量训练试跑
- [x] 记录训练中的 OOM、死锁、图像载入异常、reward 异常分布等问题
- [x] 记录“Phase 3 异常并非主要由 Phase 4 缺失导致”的根因判断
- [x] 在 Phase 3 范畴内先完成 stopping / truncation / decode hygiene 修复
- [x] 在修复后重新验证首个 `†Answer:` 后不会继续异常拖尾
- [x] 在修复后重新验证 `response_length` 不再大面积顶满上限或长期停留在异常高位

### Phase 4：实现并对齐 PS-GRPO reward 语义

目标：

- 将 reward 从“PRM 聚合分数”推进到论文定义的 PS-GRPO
- 把 outcome correctness 与 process drop-moment 真正纳入 reward

需要落地的逻辑：

1. 从 response 中提取最终答案
2. 用 `reference` 判定答案是否正确
3. 从 `step_scores` 中检测 `drop-moment`
4. 按论文公式输出最终 reward

Phase 4 与 Phase 3 的关系需要明确为：

- Phase 3 的 `math_prm` 是链路打通用 baseline
- Phase 4 才是把 baseline reward 升级成论文定义的真实 reward
- Phase 4 之后应新增或切换到一个显式的 Stage 3 reward 路径，例如 `math_psgrpo`
- 也就是说，真正的 Stage 3 reward 不应继续混在“Phase 3 baseline 的 `math_prm` 语义”里含糊存在

建议在 Phase 4 中明确落地以下改法：

- 在 `examples/math_prm/reward_models.py` 中实现 PS-GRPO 所需的 step-level 过程信息、drop-moment 检测、final reward 计算
- 在 `examples/math_prm/reward_models_utils.py` 中新增 `math_psgrpo` 的 recipe / label 路径
- 在 `examples/math_prm/reward_models_utils.py` 的 `mix_rewards()` / `reward_fn()` 中明确区分：
  - `math_prm` = Phase 3 baseline，只返回 `min(step_scores)`
  - `math_psgrpo` = Phase 4+ 的真实 Stage 3 reward
- 在 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 中把后续正式训练的 label 切到 `math_psgrpo`

Checklist：

- [ ] 明确 `MathPRMReward` 是直接输出最终 reward，还是输出结构化中间信息
- [ ] 确认可以拿到完整 `step_scores`
- [ ] 实现 relative drop 计算逻辑
- [ ] 实现 `rho = 0.3` 的 drop-moment 判定
- [ ] 实现最终答案抽取逻辑
- [ ] 实现 reference 标准化逻辑
- [ ] 实现 outcome correctness 判定逻辑
- [ ] 实现 `gamma = 0.5` 的最终 reward 映射
- [ ] 确认最终 reward 只取 `{1.0, 0.5, 0.0}` 或论文允许的等价值
- [ ] 对齐检查：正确且无 drop 的样本 reward 为 `1.0`
- [ ] 对齐检查：正确但有 drop 的样本 reward 为 `0.5`
- [ ] 对齐检查：错误样本 reward 为 `0.0`
- [ ] 对齐检查：与原始论文公式的变量定义一致，不混入额外 heuristic
- [ ] 日志中输出 step_scores、max_relative_drop、has_drop_moment、outcome_correct、final_reward 便于排查

### Phase 5：答案判定与行为对齐专项检查

目标：

- 解决“reward 公式写了，但行为不对”的问题
- 对齐 final answer 抽取、reference 归一化和 correctness 判断

说明：

- 这一阶段虽然属于 reward 实现的一部分，但值得独立成 phase
- 因为大量偏差都可能出在答案抽取和规则比较上

Checklist：

- [ ] 按题型梳理答案判定策略：选择题、数值题、公式题
- [ ] 优先复用现有规则工具，例如 `mathruler`
- [ ] 明确字符串比较只适用于哪些题型
- [ ] 确认最终答案抽取不会把中间步骤误识别为 final answer
- [ ] 确认 `†Answer:` 缺失时的失败策略和日志策略
- [ ] 确认 reference 为空、格式异常、题型不支持时的回退逻辑
- [ ] 抽样人工比对一批样本，确认 correctness 判定符合预期
- [ ] 对齐检查：同一条样本在原脚本与 LightRFT 中 correctness 结论一致

### Phase 6：训练脚本与运行参数阶段性对齐

目标：

- 把当前脚本从“可运行样例”推进到“Stage 3 复现脚本”
- 先做阶段性对齐，再逐步逼近论文 Table 14

说明：

- 这一阶段仍然允许先用全量数据
- 但脚本的命名、label、RM 路径、关键开关需要先对齐

Checklist：

- [ ] 将脚本中的 reward label 明确为目标 Stage 3 路径
- [ ] 将脚本中的模型占位路径替换为本机实际路径：`/home/ubuntu/URSA-MATH/checkpoints/URSA-8B` 与 `/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B`
- [ ] 将 `PATH_TO_YOUR_MATH_DATASET` 明确为转换后的 LightRFT manifest，而不是 raw `train.jsonl`
- [ ] 明确当前所有运行和排障都以 `/data/LightRFT/Dockerfile` 为冻结环境基线
- [ ] 明确 Dockerfile 中已安装的 pip 包版本与安装顺序后续不允许擅自改动，保持保本一致性
- [ ] 去掉与 PRM 直连需求冲突的 `rm_use_engine` 用法
- [ ] 明确 `freeze_prefix`、多模态开关、图像字段名等必要参数
- [ ] 明确如需先做资源 smoke test，可直接复用 `/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py` 与 `validate_dataset_entrypoints.py`
- [ ] 梳理当前脚本参数与论文 Table 14 的差异
- [ ] 优先对齐 `n_samples_per_prompt = 8`
- [ ] 优先对齐 `temperature = 1.0`
- [ ] 优先对齐 `init_kl_coef = 0.003`
- [ ] 优先对齐 `actor_learning_rate = 2e-6`
- [ ] 优先对齐 `prompt_max_len = 6048`
- [ ] 优先对齐 `generate_max_len = 3072`
- [ ] 评估当前硬件条件下是否能直接对齐 `train_batch_size = 512`
- [ ] 如果不能直接对齐，明确记录采用的等效梯度累积方案
- [ ] 确认脚本注释中明确写明“当前阶段先用全量数据，不做筛选”

### Phase 7：全量数据训练观测与稳定性验证

目标：

- 在“未做筛选”的全量数据上，观察真实训练行为
- 判断当前 reward 与数据链路是否足够稳定

需要重点关注：

- reward 分布
- 正确率分布
- drop-moment 分布
- 格式失败率
- 图像读取异常率
- KL 与 loss 是否稳定

Checklist：

- [ ] 统计训练中 reward 的均值、方差、分位数
- [ ] 统计 correctness 比例
- [ ] 统计 drop-moment 命中比例
- [ ] 统计无法抽取 final answer 的比例
- [ ] 统计图像读取失败比例
- [ ] 统计 PRM 推理失败比例
- [ ] 观察 KL 曲线是否异常
- [ ] 观察 actor loss 是否出现爆炸或塌缩
- [ ] 抽样检查生成文本是否持续满足 Stage 3 指定格式
- [ ] 抽样检查多模态样本是否真的影响 PRM 打分
- [ ] 输出需要在下一阶段修复的问题列表

### Phase 8：补论文中的数据筛选流程

目标：

- 在基础训练链路、行为对齐和 reward 语义稳定之后
- 再补论文里的“从完整 `MMathCoT-1M` 中抽 `20K` 候选 -> `8` 次采样 -> 过滤全对/全错 -> `15.3K`”流程

说明：

- 这是后置 phase，不是前置阻塞项
- 只有在前面阶段已经稳定后才值得做

Checklist：

- [ ] 新增离线数据准备脚本，而不是把筛选逻辑塞进训练主脚本
- [ ] 从完整 `MMathCoT-1M` 中按既定策略采样 `20K`
- [ ] 使用 `URSA-8B` 对每题采样 `8` 次
- [ ] 用 rule / reference 判定每次回答正确性
- [ ] 过滤掉“8 次全对”或“8 次全错”的题目
- [ ] 产出约 `15.3K` 的 Stage 3 训练集
- [ ] 确认产出数据继续满足 `prompt / images / reference / label`
- [ ] 确认筛选后数据仍能被 `PromptDatasetVL` 直接消费
- [ ] 在脚本与文档中把数据来源更新为“筛选后的 Stage 3 RL 集”

### Phase 9：论文复现收口

目标：

- 在筛选数据与训练脚本都稳定后，尽可能逼近论文设定
- 输出一版可复现、可说明、可追踪的 Stage 3 训练方案

Checklist：

- [ ] 梳理当前实现与论文剩余差异
- [ ] 明确哪些差异是工程折中，哪些是未完成项
- [ ] 更新文档说明当前 reward、数据、脚本、超参状态
- [ ] 整理最小复现实验步骤
- [ ] 整理 smoke test、全量训练、筛选数据训练三套运行方式
- [ ] 给出最终建议默认脚本和默认数据入口

---

## 6. 建议的文件改动映射

### `examples/math_prm/reward_models.py`

负责：

- 多模态 `MathPRMReward`
- Phase 3 baseline 所需的 `step_scores -> min(step_scores)` 聚合
- Phase 4 以后所需的 drop-moment 检测
- Phase 4 以后所需的 outcome correctness
- Phase 4 以后所需的最终 PS-GRPO reward

检查项：

- [ ] 输入是否拿到文本、图像、reference
- [ ] 输出是否包含排查所需的关键中间指标
- [ ] 多模态输入是否真的参与 PRM 推理
- [ ] 明确区分 Phase 3 baseline 输出与 Phase 4+ 的真实 Stage 3 reward 输出

### `examples/math_prm/reward_models_utils.py`

负责：

- PRM 的加载策略
- recipe / label 到 reward builder 的映射
- `math_prm` 与 `math_psgrpo` 路径区分
- Phase 3 baseline 与 Phase 4+ reward 升级路径的切换

检查项：

- [ ] `math_prm` / `math_psgrpo` 不走错误的 engine 路径
- [ ] 配置项能表达不同 reward 模式
- [ ] reward_fn 能拿到 `reference` 与 `raw_images`
- [ ] `math_prm` 只表示 Phase 3 baseline 的 `min(step_scores)`
- [ ] `math_psgrpo` 明确表示论文 Stage 3 的真实 reward
- [ ] `math_prm` 路径下不混入不对齐的 `<think>` 风格 format reward

### `examples/math_prm/train_colocate.py`

负责：

- actor 选择
- tokenizer / processor 准备
- 数据集与 dataloader 初始化
- reward model 组装

检查项：

- [ ] URSA actor 检测稳定
- [ ] 多模态数据参数配置完整
- [ ] dataset -> trainer -> reward 的字段链路完整

### `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

负责：

- 训练入口脚本
- 参数组织
- 论文超参与当前阶段参数的表达
- 不同阶段 reward label 的切换

检查项：

- [ ] 说明当前是否为“全量数据试跑版”
- [ ] 说明当前是否为“筛选数据复现版”
- [ ] 参数与 reward 路径不冲突
- [ ] Phase 3 试跑时明确使用 `math_prm`
- [ ] Phase 4+ 正式 Stage 3 训练时明确切换到 `math_psgrpo`

### `lightrft/trainer/fast_exp_maker.py`

负责：

- reward 计算时的字段传递
- 多模态样本在 rollout / reward 阶段的上下文保持

检查项：

- [ ] `prompt_and_output` 能传到 reward
- [ ] `raw_images` 能传到 reward
- [ ] `references` 能传到 reward
- [ ] reward metrics 如有需要可透传日志

---

## 7. 当前建议的推进顺序

如果按实际落地效率排序，建议这样推进：

1. 先做 `Phase 1` 和 `Phase 2`
2. 再做 `Phase 3`，用全量数据把链路跑通
3. 再做 `Phase 4` 和 `Phase 5`，把 reward 语义做对、行为对齐做实
4. 再做 `Phase 6` 和 `Phase 7`，把脚本和训练行为稳定下来
5. 最后做 `Phase 8`，补论文的数据筛选流程
6. 最后用 `Phase 9` 收口

这意味着当前最近的实际目标不是“马上做从完整 `MMathCoT-1M` 中筛出的 `15.3K` 过滤集”，而是：

- 先把全量数据训练主链路与 reward 行为跑通
- 先把行为对齐做实
- 先把多模态 PRM 与 correctness reward 做对

---

## 8. 本计划的当前结论

当前 LightRFT 已经有一定 URSA 接入基础，但离“可解释、可验证、可复现的 URSA Stage 3”还差几个关键环节：

1. 本机 `/home/ubuntu/URSA-MATH` 已经具备 `URSA-8B`、`URSA-RM-8B`、完整 `MMathCoT-1M`、完整 `DualMath-1.1M` 与完整图片树，当前主要缺口不是资源下载
2. 真正缺的是从 URSA raw schema 到 LightRFT `prompt / images / reference / label` schema 的转换与接线
3. `/data/LightRFT/Dockerfile` 中已安装的 pip 依赖、版本与顺序应视为冻结基线，后续默认不允许改动，必须保持保本一致性
4. 数据载入与字段传递仍然需要显式检查
5. 多模态 PRM 的真实行为仍然需要对齐检查
6. PS-GRPO reward 公式仍然需要真正落地
7. 当前阶段先用完整 `MMathCoT-1M` 全量数据试跑，而不是先做筛选
8. 数据筛选流程属于后置 phase，需要在基础链路稳定后再补

因此，本计划不是直接追求“论文最终形态一步到位”，而是明确分阶段推进：

- 先跑通
- 再对齐行为
- 再对齐 reward
- 最后补筛选数据

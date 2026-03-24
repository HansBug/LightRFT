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

### 2.4 本地 `URSA-MATH` 仓库中明确可参考的 Stage 3 超参与训练规模

- `n_samples_per_prompt = 8`
- `num_episodes = 10`
- `lr_warmup_ratio = 0.03`
- `actor_learning_rate = 1e-6`
- `init_kl_coef = 0.001`
- `rollout_batch_size = 128`
- `train_batch_size = 128`
- `micro_train_batch_size = 4`
- `micro_rollout_batch_size = 4`
- `prompt_max_len = 1024`
- `generate_max_len = 3072`
- `bf16`
- `gradient_checkpointing`
- `zero_stage = 3`
- `adam_offload`

训练规模方面：

- 论文写明 Stage 3 先从 `20K` 候选中做一次静态筛选，最终保留约 `15K+` RL 样本
- 本地 `/home/ubuntu/URSA-MATH` 里没有现成可直接给 LightRFT 使用的这份精确筛选子集
- 因此当前默认策略改成：
  - 继续使用转换后的 full manifest 路径
  - 但把 `MAX_SAMPLES` 默认压到 `15360`，作为当前最接近论文训练规模的代理配置

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

- 脚本路径：`/data/LightRFT/examples/math_prm/tools/prepare_ursa_stage3_manifest.py`
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
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py \
  --max-samples 32 \
  --output-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.jsonl \
  --summary-path /data/LightRFT/tmp/ursa_stage3/smoke_manifest.summary.json
```

直接做全量转换：

```bash
python examples/math_prm/tools/prepare_ursa_stage3_manifest.py
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

- 新增转换脚本：`examples/math_prm/tools/prepare_ursa_stage3_manifest.py`
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
  - `examples/math_prm/tools/check_phase2_alignment.py`
- 新增 Phase 2 轻量单测：
  - `examples/math_prm/tools/test_phase2_alignment.py`
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
- `python examples/math_prm/tools/check_phase2_alignment.py --device cuda:0`
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

按轮修复记录（2026-03-19）：

- 第 1 轮 smoke：
  - 启动脚本：`/data/LightRFT/examples/math_prm/tools/run_phase3_smoke.sh`
  - 训练日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_005513.log`
  - 结果：主链路已打通，但健康性判定失败
  - 暴露问题：
    - dataloader / rollout / reward / PPO / cleanup 已经都能走通
    - 但首批 `step 0 generate length` 仍是 `min=max=mean=1024`
    - 输出中出现 `StepStep 1:`、`Camp Miniwauca`、`7777...`、重复 `†Answer:`
    - `response_length` 持续在 `943 ~ 946`
  - 结论：当时的主要问题仍然是 stopping / 结构化输出异常，而不是训练脚本本身不能跑

- 第 2 轮 smoke：
  - 训练日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_091453.log`
  - 本轮修复保持在 Phase 3 范畴内，没有提前引入 Phase 4 的 correctness / drop-moment / `math_prm_combined`
  - 本轮修复动作：
    - 收紧 system prompt，明确要求首个 `†Answer:` 后立即停止
    - 收紧 smoke decode 参数：`temperature=0.8`、`top_p=0.95`、`top_k=50`、`repetition_penalty=1.05`、`no_repeat_ngram_size=4`
    - 给 `math_prm` / `math_prm_combined` 增加 rollout 后处理：首个 `†Answer:` 后截断，并清理 `StepStep 1:` / 重复 answer marker / 重复字符尾巴
  - 结果：
    - `Camp Miniwauca` / `7777...` / 重复 `†Answer:` 已被压下
    - 第一批 `rollout_response_length=88`
    - 第二批 `rollout_response_length=117`
    - 训练侧长度和 `kl` 已回到可读范围
  - 剩余问题：
    - raw generate 在进入 postprocess 前仍会顶满 `1024`
    - 内容正确性仍然偏弱，例如样例输出 `41` 而不是 reference `37`
    - reward 仍偏低

- 第 3 轮 smoke：
  - 训练日志：`/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_095529.log`
  - 本轮修复动作：
    - 把 stopping 往前推到 HF 本地生成阶段本身
    - 让结构化 answer 规则在 generation-time 就尝试收尾到 `eos`
    - 修正 local HF inference 的 `output_token_ids` 截取，不再把 `eos` 后 padding 误算成生成长度
  - 结果：
    - 首批 `step 0 generate length` 从伪 `1024` 降到 `min=60`、`max=256`、`mean=120.25`
    - postprocess 只剩 `sanitized 1/2 outputs`
    - 而且只裁掉 `2` 个 token
    - 第一批输出仍保持干净的 `Step ... / †Answer: yes`
    - 第一批 `rollout_response_length=87`
  - 结论：
    - Phase 3 的 stopping / 长尾乱码主问题已经基本压住
    - 剩余主要矛盾已经转成答案正确性和 PRM reward 偏低

Phase 3 现状（最新）：

- 当前状态可以概括为：
  - **训练主链路已打通**
  - **结构化输出与 stopping 问题已基本修住**
  - **还不能记为最终健康通过**
  - **剩余主问题已收敛为 correctness 偏弱、PRM reward 偏低**
- 也就是说，Phase 3 现在已经不是“脚本一跑就长尾乱码、完全不正常”，而是：
  - rollout / reward / PPO / cleanup 都能正常走通
  - `Step N:` / `†Answer:` 格式已经基本稳定
  - 生成长度已经回到合理区间
  - 但模型答案质量还没有稳定到可以直接宣告 Phase 3 结束

Phase 3 建议（最新）：

- 现在可以开始 `Phase 4`
- 直接原因是：Phase 3 当前剩下的问题，已经主要落在 `Phase 4` 的职责范围里，也就是 reward 对“答对/答错”的判别能力，而不再是 rollout 结构性故障
- 当前判断仍然保持：
  - **最早那批异常输出的首因，不是 Phase 4 没上**
  - 依据是异常输出发生在第一次 PPO 更新之前
  - 但在 Phase 3 的 stopping 问题已经基本压住之后，Phase 4 现在比之前更有机会真正发挥作用
- 当前对进入 Phase 4 的效果评估是：
  - `70% ~ 80%` 概率：会明显改善 reward 对错题/对题的区分能力
  - `50% ~ 60%` 概率：会带来可见的训练趋势改善
  - 但它更可能先改善“训练信号是否对”，不保证一上来就把模型答案质量整体拉正
- 因此当前建议是：
  - 保留 Phase 3 已完成的 stopping / truncation / decode hygiene 修复不回退
  - 在此基础上进入 Phase 4，实现 correctness + drop-moment + 最终 PS-GRPO reward 语义
  - 第一轮 Phase 4 smoke 中必须把 `min(step_scores)`、`outcome_correct`、`max_relative_drop`、`has_drop_moment`、`final_reward` 分开打日志
  - 如果 obvious wrong answers 仍然能拿到高 `final_reward`，优先检查答案抽取和 reference 对齐，而不是先怀疑 PPO 本身

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

状态：

- 已完成
- `math_prm` 保持为 Phase 3 baseline：reward = `min(step_scores)`
- `math_psgrpo` 已落地为 Phase 4 路径：reward = correctness + drop-moment 映射后的最终 PS-GRPO reward

本轮实际落地：

- `examples/math_prm/reward_models.py`
  - `MathPRMReward` 现在直接输出结构化结果，而不是只返回单个 scalar
  - 为避免破坏已有 Phase 2/Phase 3 工具链，`MathPRMReward.forward(...)` 保留了旧接口兼容：
    - 不传 `references/labels` 时仍返回 legacy tensor
    - Phase 4 新链路传 `references/labels` 时返回结构化 metrics
  - 对每条样本保留 `step_scores` 的聚合统计：`step_score_min` / `step_score_mean` / `step_score_last` / `step_count`
  - 新增 `rho = 0.3` 的 relative-drop 检测
  - 新增 `†Answer:` 优先的 final answer 抽取逻辑，缺失时回退到 `math_eval_utils.extract_answer(...)`
  - 复用 `lightrft/evaluation/math_eval_utils.py` 做 reference 标准化与 correctness 判断
  - 对 `math_psgrpo` 显式输出：
    - `outcome_correct`
    - `max_relative_drop`
    - `has_drop_moment`
    - `final_reward`
  - 最终 reward 映射已经固定为：
    - 正确且无 drop = `1.0`
    - 正确但有 drop = `0.5`
    - 错误 = `0.0`
- `examples/math_prm/reward_models_utils.py`
  - 新增 `math_psgrpo` recipe / label 路径
  - 保持 `math_prm` 与 `math_psgrpo` 语义分离
  - `math_psgrpo` 与 `math_prm` 一样都不再混入全局 format reward
- `lightrft/trainer/fast_exp_maker.py`
  - 单 RM 路径现在会保留 reward model 返回的辅助指标，不再在 reward aggregation 里丢失
- `lightrft/trainer/spmd_ppo_trainer.py`
  - 训练阶段会聚合并打印：
    - `outcome_correct`
    - `final_reward`
    - `max_relative_drop`
    - `has_drop_moment`
    - `step_score_min` / `step_score_mean` / `step_score_last` / `step_count`
- `lightrft/trainer/ppo_trainer_vl.py`
  - rollout/eval 统计也改为透传并聚合 reward_metrics
- `examples/math_prm/tools/prepare_ursa_stage3_manifest.py`
  - 默认 manifest label 已切到 `math_psgrpo`
  - 默认输出文件名也切到 `mmathcot_stage3_math_psgrpo.*`
- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
  - 默认训练数据路径切到 `math_psgrpo` manifest
  - 增加 dataset label 预检查，默认要求样本 label 为 `math_psgrpo`

验证：

- `python -m unittest -q examples.math_prm.test_phase2_alignment`
  - 通过
- 新增并通过的对齐点：
  - `math_psgrpo` label 在 `mix_rewards()` 中可正常走通
  - `MathPRMReward._compute_psgrpo_metrics(...)` 对三种 reward 映射都符合预期
  - `MathPRMReward.forward(...)` 在 `math_psgrpo` 标签下会返回结构化 metrics，并输出正确的 `0.5` 奖励样例
- `bash -n examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
  - 通过

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

- [x] 明确 `MathPRMReward` 是直接输出最终 reward，还是输出结构化中间信息
- [x] 确认可以拿到完整 `step_scores`
- [x] 实现 relative drop 计算逻辑
- [x] 实现 `rho = 0.3` 的 drop-moment 判定
- [x] 实现最终答案抽取逻辑
- [x] 实现 reference 标准化逻辑
- [x] 实现 outcome correctness 判定逻辑
- [x] 实现 `gamma = 0.5` 的最终 reward 映射
- [x] 确认最终 reward 只取 `{1.0, 0.5, 0.0}` 或论文允许的等价值
- [x] 对齐检查：正确且无 drop 的样本 reward 为 `1.0`
- [x] 对齐检查：正确但有 drop 的样本 reward 为 `0.5`
- [x] 对齐检查：错误样本 reward 为 `0.0`
- [x] 对齐检查：与原始论文公式的变量定义一致，不混入额外 heuristic
- [x] 日志中输出 `step_score_min` / `step_score_mean` / `step_score_last`、`max_relative_drop`、`has_drop_moment`、`outcome_correct`、`final_reward` 便于排查

### Phase 5：答案判定与行为对齐专项检查

状态：

- 已完成
- `math_psgrpo` 的 correctness 判断不再只是“抽个字符串比一下”
- 当前实现已经区分题型、限制 fallback、并把答案抽取失败显式暴露成指标

本轮实际落地：

- `examples/math_prm/reward_models.py`
  - 新增 reference 题型判断：
    - `multiple_choice`
    - `numeric`
    - `formula`
    - `text`
    - `missing`
  - 新增 `†Answer:` 优先的 final answer 抽取流程
  - 当缺失 `†Answer:` 时，不再对整段 response 做“最后一个数字”式的宽松提取
  - 缺失 `†Answer:` 时只允许两类显式 fallback：
    - `\\boxed{...}`
    - `<answer>...</answer>`
    - 最后一行存在显式 `Answer:` / `Final answer:` / `The answer is ...`
  - 如果以上都没有，则记为 `answer_extraction_failed=1`，避免把中间步骤误识别为最终答案
  - 数值/公式题优先复用 `mathruler.grader.grade_answer(...)`
  - `mathruler` 不命中时再回退到 `lightrft.evaluation.math_eval_utils.compare_answers(...)`
  - 额外输出并透传以下行为对齐指标：
    - `answer_tag_present`
    - `answer_extraction_failed`
    - `used_answer_fallback`
    - `reference_supported`
    - `used_mathruler`
    - `reference_type_id`
- `examples/math_prm/reward_models_utils.py`
  - `reward_fn()` / `mix_rewards()` 现在会保留 reward model 返回的辅助指标
  - 这样 `math_psgrpo` 的 correctness / answer extraction / drop-moment 指标在真实训练链路里不会再丢
- `lightrft/trainer/fast_exp_maker.py`
  - 多 RM / 单 RM 聚合都会把 `MathPRMReward` 的辅助指标一路带到 trainer 日志

验证：

- `python -m unittest -q examples.math_prm.test_phase2_alignment`
  - 通过
- 新增并通过的 Phase 5 对齐点：
  - `reward_fn()` 能保留 `math_psgrpo` 的答案对齐指标
  - `\\frac{1}{2}` 与 `1/2` 会通过 `mathruler` 判为正确
  - 缺失 `†Answer:` 且没有显式 final-answer 标记时，不会误把中间步骤里的 `37` 当成最终答案
  - 选择题在最后一行显式写 `The answer is B` 时可以作为受控 fallback
  - `reference` 为空时会被判为 `unsupported_reference`，不会伪造 correctness

目标：

- 解决“reward 公式写了，但行为不对”的问题
- 对齐 final answer 抽取、reference 归一化和 correctness 判断

说明：

- 这一阶段虽然属于 reward 实现的一部分，但值得独立成 phase
- 因为大量偏差都可能出在答案抽取和规则比较上

Checklist：

- [x] 按题型梳理答案判定策略：选择题、数值题、公式题
- [x] 优先复用现有规则工具，例如 `mathruler`
- [x] 明确字符串比较只适用于哪些题型
- [x] 确认最终答案抽取不会把中间步骤误识别为 final answer
- [x] 确认 `†Answer:` 缺失时的失败策略和日志策略
- [x] 确认 reference 为空、格式异常、题型不支持时的回退逻辑
- [x] 抽样人工比对一批样本，确认 correctness 判定符合预期
- [x] 对齐检查：同一条样本在原脚本与 LightRFT 中 correctness 结论一致

### Phase 6：训练脚本与运行参数阶段性对齐

状态：

- 已完成
- 当前 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 已从“能跑的样例脚本”推进到“当前机器上的 Stage 3 复现入口脚本”
- 这一阶段仍然先使用全量转换 manifest，不提前混入 Phase 8 的筛选数据逻辑

目标：

- 把当前脚本从“可运行样例”推进到“Stage 3 复现脚本”
- 把默认值切到本地 `URSA-MATH` 仓库中明确给出的 Stage 3 配置
- 对当前本机还缺失的“精确 15K 筛选子集”使用可解释的代理策略，而不是继续沿用更激进的旧默认值

说明：

- 这一阶段仍然允许先用全量数据
- 但脚本的命名、label、RM 路径、关键开关需要先对齐
- 这一阶段不改 `lightrft/` 核心训练语义，只收敛 example 入口、默认参数、注释和校验

本轮实际落地：

- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
  - 默认 reward label 保持为 `math_psgrpo`
  - 默认 actor / RM 路径显式固定为当前机器上的：
    - `/home/ubuntu/URSA-MATH/checkpoints/URSA-8B`
    - `/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B`
  - 默认数据入口显式固定为转换后的 LightRFT manifest：
    - `/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl`
  - 去掉了与 URSA PRM 直连需求冲突的 `--rm_use_engine`
  - 新增 preflight：
    - 检查 actor / RM / dataset / Dockerfile 路径是否存在
    - 检查 dataset label 是否与目标 reward path 一致
    - 显式打印当前 URSA Stage 3 默认值对齐快照
    - 显式打印当前本机 `train_batch_size=128` 的实现方式
  - 注释和使用说明里明确写明：
    - 当前所有运行和排障都以 `/data/LightRFT/Dockerfile` 为冻结环境基线
    - Dockerfile 中已安装的 pip 包版本与安装顺序不要擅自改动
    - 如需先做资源 smoke，可复用：
      - `/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py`
      - `/home/ubuntu/URSA-MATH/examples/validate_dataset_entrypoints.py`
    - Phase 3 baseline smoke 应走 `examples/math_prm/tools/run_phase3_smoke.sh`
  - 当前默认超参现在是：
    - `num_episodes = 10`
    - `n_samples_per_prompt = 8`
    - `temperature = 1.0`
    - `init_kl_coef = 0.001`
    - `actor_learning_rate = 1e-6`
    - `prompt_max_len = 1024`
    - `generate_max_len = 3072`
    - `rollout_batch_size = 128`
    - `train_batch_size = 128`
    - `max_samples = 15360`
  - 当前 8 卡机器上的 global batch 128 实现方式明确记录为：
    - `micro_train_batch_size = 4`
    - `world_size = 8`
    - `accumulated_gradient = 4`
    - 即 `4 x 8 x 4 = 128`
  - 关于数据规模的当前处理：
    - 论文的精确 Stage 3 训练集仍然是“`20K -> 15K+` 静态筛选后子集”
    - 本机尚未产出这个 LightRFT-compatible 子集
    - 因此先用 `MAX_SAMPLES = 15360` 作为训练规模代理
- `examples/math_prm/train_colocate.py`
  - example 默认值已同步到当前 Stage 3 入口语义：
    - `engine_type = hf`
    - `num_episodes = 10`
    - `rollout_batch_size = 128`
    - `micro_rollout_batch_size = 4`
    - `prompt_max_len = 1024`
    - `generate_max_len = 3072`
    - `max_samples = 15360`
    - `train_batch_size = 128`
    - `n_samples_per_prompt = 8`
    - `actor_learning_rate = 1e-6`
    - `init_kl_coef = 0.001`
    - `images_key = "images"`
  - 文档说明里也明确了：`rm_use_engine` 虽然仍是通用 flag，但 `math_prm/math_psgrpo` 的 URSA PRM 仍然走 HF 直连
- `examples/math_prm/tools/check_phase6_script_alignment.py`
  - 新增最小对齐校验脚本
  - 直接检查 launcher 默认值、注释、关键 flag、Docker baseline 说明和 batch 实现方式是否满足 Phase 6 约束

当前与最终论文复现仍保留的差异：

- 数据仍然是全量转换 manifest，不是 Phase 8 才会产出的筛选后约 `15.3K` RL 子集
- rollout backend 当前仍默认使用本地 `hf`，不是外部 `vllm/sglang`
- `rollout_batch_size` 仍保持工程上更稳的 `128`，没有在这一阶段强行和最终长跑策略一起改动

验证：

- `python examples/math_prm/tools/check_phase6_script_alignment.py`
  - 通过
- `python -m unittest -q examples.math_prm.test_phase2_alignment`
  - 通过
- `bash -n examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
  - 通过

Checklist：

- [x] 将脚本中的 reward label 明确为目标 Stage 3 路径
- [x] 将脚本中的模型占位路径替换为本机实际路径：`/home/ubuntu/URSA-MATH/checkpoints/URSA-8B` 与 `/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B`
- [x] 将 `PATH_TO_YOUR_MATH_DATASET` 明确为转换后的 LightRFT manifest，而不是 raw `train.jsonl`
- [x] 明确当前所有运行和排障都以 `/data/LightRFT/Dockerfile` 为冻结环境基线
- [x] 明确 Dockerfile 中已安装的 pip 包版本与安装顺序后续不允许擅自改动，保持保本一致性
- [x] 去掉与 PRM 直连需求冲突的 `rm_use_engine` 用法
- [x] 明确 `freeze_prefix`、多模态开关、图像字段名等必要参数
- [x] 明确如需先做资源 smoke test，可直接复用 `/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py` 与 `validate_dataset_entrypoints.py`
- [x] 梳理当前脚本参数与本地 `URSA-MATH` Stage 3 默认值的差异
- [x] 优先对齐 `n_samples_per_prompt = 8`
- [x] 优先对齐 `temperature = 1.0`
- [x] 优先对齐 `init_kl_coef = 0.001`
- [x] 优先对齐 `actor_learning_rate = 1e-6`
- [x] 优先对齐 `prompt_max_len = 1024`
- [x] 优先对齐 `generate_max_len = 3072`
- [x] 对齐 `rollout_batch_size = 128` 与 `train_batch_size = 128`
- [x] 记录当前 global batch `128` 的实现方式：`4 x 8 x 4 = 128`
- [x] 确认脚本注释中明确写明“当前仍使用 full manifest，但默认 `MAX_SAMPLES = 15360` 作为论文训练规模代理”

### Phase 7：全量数据训练观测与稳定性验证

状态：

- 已完成
- 最新 bounded full-data observation 已重新健康通过
- 训练链路、reward 链路、trajectory 保存和离线分析都已打通
- 之前的格式稳定性问题已定位并修复，根因详见 `plan/PHASE7_FORMAT_STABILITY_ANALYSIS.md`
- 当前仍有一个独立保留问题：本地 `hf` 多模态 rollout 速度明显偏慢，但已不再阻塞 bounded run 完成
- 推理性能拆解详见 `plan/PHASE7_HF_ROLLOUT_PERFORMANCE_ANALYSIS.md`
- 截至 `2026-03-21`，对 rollout 慢点的判断已进一步更新：
  - 早期 `gc`/训练态 actor 假设只解释了最初一层问题，不足以解释默认主训练入口当前的全部 slowdown
  - 最新主入口 30 分钟完整 profile 已确认：
    - 默认参数下，首个 `global step` 约 `1163.41s`
    - 其中 rollout generate 约 `989.93s`
    - PPO train 约 `81.04s`
    - 训练后权重同步约 `1.75s`
  - 当前真正的主慢点已经收敛到：
    - 本地 `hf` rollout 把扩展后的 `128` 个 rollout sample
    - 按 `local_hf_generate_max_batch_size = 4` 串行切成 `32` 个 generate chunk
    - 而每个 chunk 的 `model.generate` 本身又要约 `26 ~ 45s`
  - 这说明现在的问题核心已经不是 reward / PPO / structured stop，而是 local HF rollout generate 的吞吐与 chunk 组织方式
  - 完整 profile 的原始文件和详细结论，见 `plan/PHASE7_HF_ROLLOUT_PERFORMANCE_ANALYSIS.md` 新增的 `7.4`

本轮最终观测配置：

- 入口脚本：`examples/math_prm/tools/run_phase7_observation.sh`
- 真实运行时间戳：`20260319_233851`
- 结果目录：`results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_233851`
- 训练日志：`/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_233851.log`
- 观测摘要：`/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_233851.json`
- 轨迹文件：`results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_233851/trajectories/trajectories_step_1.json`
- 数据：全量 `math_psgrpo` manifest
- 资源形态：8 卡 bounded run
- 关键缩小参数：
  - `n_samples_per_prompt = 4`
  - `rollout_batch_size = 8`
  - `train_batch_size = 8`
  - `micro_train_batch_size = 1`
  - `micro_rollout_batch_size = 1`
  - `prompt_max_len = 4096`
  - `generate_max_len = 1024`
  - `max_samples = 8`

为完成 Phase 7 额外补的例子层修复：

- `lightrft/strategy/strategy_base.py`
  - 修复本地 `hf` rollout 在左 padding 批次中按统一 prompt 长度切输出，导致短 prompt 前缀生成 token 被截断的问题
  - 移除中间用于止血的 `per-sample fallback`，恢复 batched 多模态 rollout
- `lightrft/utils/math_prm_output.py`
  - 将 `math_psgrpo` 纳入 structured label
  - 补齐短代数 final answer 的 stop 判定
- `examples/math_prm/tools/check_hf_rollout.py`
  - 最小校验改成和真实 rollout 一致的 batched left-padding 直生基线
- `examples/math_prm/tools/test_phase2_alignment.py`
  - 补齐左 padding 输出切片、`math_psgrpo` structured stop、短代数 answer stop 的回归测试
- `examples/math_prm/tools/run_phase7_observation.sh`
  - 修复 `timeout + wait` 在 `set -e` 下无法继续分析的问题
  - 默认缺失 `math_psgrpo` manifest 时自动生成
  - 最终采用 8 卡 bounded observation，而不是 1 卡空跑配置
- `examples/math_prm/train_colocate.py`
  - reference model 的 FSDP `shard_size` 改为按当前 `world_size` 自适应，避免小 world size 调试时直接断在 `assert world_size % shard_size == 0`
  - 补上 `--trajectory_analysis` CLI，避免 `save_trajectories()` 在保存阶段因缺少字段报错
- `examples/math_prm/tools/analyze_phase7_observation.py`
  - 空跑 observation 不再被误判为 healthy
  - 多模态 impact check 改为“真实图像 vs 同尺寸白图”消融，避免 `raw_images=None` 这条路径失效

最终观测结果：

- 这轮最新 summary 已恢复为 `healthy_pass = true`
- 训练侧 progress 指标已经真实出现，不再是 `Episode 0it`
  - `pg` 共记录 `9` 个点，均值 `0.0006778`，范围 `[-0.0522, 0.17]`
  - `kl` 共记录 `9` 个点，均值 `0.0084111`，最大 `0.0181`
  - `rm` 共记录 `9` 个点，均值 `0.4862222`，范围 `[0.438, 0.5]`
- 真实 rollout 已完成并进入 train/save 阶段
  - `step 0 generate length`: `min=45, max=455, mean=148.875`
  - `rollout engine generation time (global max) = 661.9063s`
  - `math_prm_postprocess` 仅裁掉 answer 行后的少量尾巴：`sanitized 4/4`, `mean_trim_tokens=1.0`
- trajectory/summary 侧统计：
  - `num_trajectories = 4`
  - `rollout_reward.mean = 0.125`
  - `rollout_reward.variance = 0.046875`
  - `psgrpo_final_reward.mean = 0.125`
  - `correctness_ratio = 0.25`
  - `drop_moment_ratio = 1.0`
  - `answer_extraction_failure_ratio = 0.0`
  - `format_success_ratio = 1.0`
  - `prm_inference_failure_ratio = 0.0`
  - `image_read_failure_ratio = 0.0`（扫描前 `128` 条 manifest）
  - `multimodal_sample_count = 4`
  - 多模态 PRM 图像消融：
    - `checked_samples = 2`
    - `failed_samples = 0`
    - `mean_abs_delta = 0.815216064453125`

Phase 7 结论：

- Phase 7 的观测任务已经完成，当前已经能在全量 `math_psgrpo` manifest 上做 bounded full-data run，并稳定收集：
  - reward 分布
  - correctness/drop-moment 分布
  - KL / pg 训练趋势
  - 轨迹文件
  - 图像读取统计
  - 多模态 PRM 消融结果
- 当前健康性已通过，之前的格式稳定性问题已经收敛
- Phase 7 现在证明的是：
  - 当前 Stage 3 训练链路可观测、可分析、可保存
  - 真实 8 卡 bounded run 可以在 20 分钟内完成
  - 之前的格式异常主要是 rollout 集成 bug，而不是 URSA 本体问题
- Phase 7 同时也保留了一个需要后续单独处理的问题：
  - `hf` 多模态 rollout 仍然偏慢
  - bounded observation 虽然已经能完成，但这不能代表默认主训练入口已经健康
  - 最新主入口 30 分钟完整 profile 显示：
    - 默认参数下首轮 `rollout engine generation time = 989.9257s`
    - 单步总耗时约 `1163.41s/it`
    - 本地 rollout 实际是在串行跑 `32` 个 `batch_size=4` 的 `model.generate` chunk
  - `2026-03-20` 的 4 组最小化对照已经说明：
    - 单独 `URSA-8B` 本身不慢
    - `raw + train + gc on` 会从 `14.604s` 掉到 `306.014s`
    - `FSDP + gc off` 是 `32.977s`
    - `FSDP + gc on` 是 `254.804s`
  - 随后的更贴近真实 rollout 的 probe 进一步说明：
    - `fsdp_train_gc = 683.406s`
    - `fsdp_train_no_gc = 68.869s`
    - `fsdp_eval_no_gc = 65.816s`
    - `raw_eval_no_gc = 44.139s`
  - 但在真正的默认主训练入口里，最新 profile 进一步说明：
    - 当前最关键的提速杠杆已经收敛到 generate 吞吐与 chunk 组织
    - `structured_stop` 不是主因
    - PPO train 不是主因
    - rollout 后权重同步也不是主因
  - 新增脚本 `examples/math_prm/tools/probe_rollout_speed_candidates.py` 的职责，就是在不修改现有库代码的情况下，用更接近真实 rollout 的 workload 比较这些候选运行形态
  - 这说明当前 rollout 的主慢点已经不再模糊，详见 `plan/PHASE7_HF_ROLLOUT_PERFORMANCE_ANALYSIS.md`
  - 当前如果真的要继续改 rollout 代码，推荐顺序已经更新为：
    - 先基于完整主入口 profile 继续看 generate 路径
    - 优先优化 local HF generate 吞吐和 chunk 组织
    - 再决定 local HF 路径是否值得继续深挖，还是转向更 engine 化的 rollout 方案
- Phase 7 之后剩余的主问题已经切换成训练质量本身
  - 例如当前 `correctness_ratio` 仍只有 `0.25`
  - 以及 rollout 性能问题
  - 这些都不再属于格式稳定性问题

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

- [x] 统计训练中 reward 的均值、方差、分位数
- [x] 统计 correctness 比例
- [x] 统计 drop-moment 命中比例
- [x] 统计无法抽取 final answer 的比例
- [x] 统计图像读取失败比例
- [x] 统计 PRM 推理失败比例
- [x] 观察 KL 曲线是否异常
- [x] 观察 actor loss 是否出现爆炸或塌缩
- [x] 抽样检查生成文本是否持续满足 Stage 3 指定格式
- [x] 抽样检查多模态样本是否真的影响 PRM 打分
- [x] 输出需要在下一阶段修复的问题列表

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

- [x] 输入是否拿到文本、图像、reference
- [x] 输出是否包含排查所需的关键中间指标
- [x] 多模态输入是否真的参与 PRM 推理
- [x] 明确区分 Phase 3 baseline 输出与 Phase 4+ 的真实 Stage 3 reward 输出

### `examples/math_prm/reward_models_utils.py`

负责：

- PRM 的加载策略
- recipe / label 到 reward builder 的映射
- `math_prm` 与 `math_psgrpo` 路径区分
- Phase 3 baseline 与 Phase 4+ reward 升级路径的切换

检查项：

- [x] `math_prm` / `math_psgrpo` 不走错误的 engine 路径
- [x] 配置项能表达不同 reward 模式
- [x] reward model 输入能拿到 `reference` / `raw_images`，`reward_fn` 能拿到 `reference` 与 RM 辅助指标
- [x] `math_prm` 只表示 Phase 3 baseline 的 `min(step_scores)`
- [x] `math_psgrpo` 明确表示论文 Stage 3 的真实 reward
- [x] `math_prm` 路径下不混入不对齐的 `<think>` 风格 format reward

### `examples/math_prm/train_colocate.py`

负责：

- actor 选择
- tokenizer / processor 准备
- 数据集与 dataloader 初始化
- reward model 组装

检查项：

- [x] URSA actor 检测稳定
- [x] 多模态数据参数配置完整
- [x] dataset -> trainer -> reward 的字段链路完整

### `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

负责：

- 训练入口脚本
- 参数组织
- 论文超参与当前阶段参数的表达
- 不同阶段 reward label 的切换

检查项：

- [x] 说明当前是否为“全量数据试跑版”
- [ ] 说明当前是否为“筛选数据复现版”
- [x] 参数与 reward 路径不冲突
- [x] Phase 3 试跑时明确使用 `math_prm`
- [x] Phase 4+ 正式 Stage 3 训练时明确切换到 `math_psgrpo`

### `lightrft/trainer/fast_exp_maker.py`

负责：

- reward 计算时的字段传递
- 多模态样本在 rollout / reward 阶段的上下文保持

检查项：

- [x] `prompt_and_output` 能传到 reward
- [x] `raw_images` 能传到 reward
- [x] `references` 能传到 reward
- [x] reward metrics 如有需要可透传日志

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

---

## 9. W&B 主看板与 Runtime Eval 收口方案

本节只讨论当前 `math_prm` 主训练入口的线上观测面，不讨论论文 benchmark eval，也不讨论 W&B config 字段清理。

当前用户要求已经明确收敛为：

- `train/*` 主看板规模向参考 `Geo3K` run 靠拢
- `rollout/*` 主看板规模向参考 `Geo3K` run 靠拢
- `eval/*` 眼下只做 **runtime eval**
- runtime eval 的目标不是论文分数，而是 **训练趋势是否朝正确方向前进**
- 最终保留下来的指标，必须在 `examples/math_prm/README.md` 与 `examples/math_prm/README_zh.md` 中逐项解释

### 9.0 当前优先修复项目

在第 `9` 节范围内，当前优先级按以下顺序固定：

1. **精简 W&B 主看板并补上 runtime eval**
   - 目标：
     - 把 `train/*` 主面板收敛到参考 `Geo3K` run 同量级的约 `22` 项
     - 把 `rollout/*` 主面板收敛到 `7` 项
     - 补上不超过 `8` 项的 `eval/*` runtime eval
   - 交付：
     - allowlist 级别的日志收口
     - 独立于训练 rollout 的 runtime eval 生成参数
     - README 双语指标说明
   - 原因：
     - 当前主看板信息过宽，已经影响对真实训练状态的判断
     - 在没有 runtime eval 的情况下，只看 train/rollout 很难区分“真实改善”和“训练内波动”

2. **修复当前 `y9sulzln` 实验暴露出的训练质量与稳定性问题**
   - 目标：
     - 让默认 `math_psgrpo` 主入口不再停留在“前期可学到一点、中段失稳、后段部分恢复但未稳定收敛”的状态
   - 交付：
     - 对 run `y9sulzln` 的复盘结论
     - 明确该 run 反映出来的主问题与次问题
     - 将其转化为后续代码与实验层的修复目标
   - 原因：
     - 当前这轮 run 并不能支持“已经收敛”这个判断
     - 如果不先明确它坏在哪里，后面即使日志面板变干净，也只是在更清楚地看一个不健康训练

### 9.1 收口目标

上线目标按 namespace 控制为：

- `train/*`：目标对齐到参考 `Geo3K` run 的同款面板规模，**总量控制在约 `22` 项**
- `rollout/*`：**总量控制在 `7` 项**
- `eval/*`：**不超过 `8` 项**

这里的“对齐 `Geo3K` run”指的是：

- 主看板保留 PPO/GRPO 训练必须看的公共指标
- 不把 `math_prm` / `math_psgrpo` 的全部诊断量都堆进主面板
- 数量、层次和阅读负担与参考 run 接近

由于当前离线环境无法直接程序化读取参考 run 的完整字段清单，本计划先按“数量上限 + 指标类型 + 命名口径”对齐；真正落代码前，仍需要再对照一次参考 `Geo3K` 页面做最终字段核对。

### 9.2 主看板设计原则

主看板只保留三类指标：

1. PPO/GRPO 训练状态是否稳定
2. `math_psgrpo` reward 语义是否朝正确方向变化
3. 输出长度和答案抽取是否没有明显退化

明确不再把以下内容作为默认主看板的一部分：

- 全量 `reward_metrics/*`
- 全量 step-level 聚合统计
- 全量答案抽取/回退/比较器诊断
- 全量 `perf/experience_maker/*`
- 只在排障阶段短期有意义、但长期主看板可读性极差的调试项

### 9.3 `rollout/*` 保留集

`rollout/*` 目标收敛到以下 `7` 项：

1. `rollout/reward`
2. `rollout/reward_std`
3. `rollout/outcome_correct`
4. `rollout/has_drop_moment`
5. `rollout/model_reward`
6. `rollout/response_length`
7. `rollout/episode`

说明：

- 对当前默认 label `math_psgrpo`，`rollout/reward` 已经代表最终 PS-GRPO reward，不再额外保留同义的 `rollout/final_reward`
- `outcome_correct` 是“答案最终是否正确”
- `has_drop_moment` 是 `math_psgrpo` 特有的过程质量信号，必须保留
- `model_reward` 保留，是为了观察 PRM step-score 聚合值与最终 reward 是否开始脱钩
- `response_length` 保留，是为了防止 reward 上升只是长度策略漂移
- `episode` 保留，是为了和现有训练 UI 的阶段感保持一致

默认从主看板移除的 rollout 诊断项：

- `step_score_min`
- `step_score_mean`
- `step_score_last`
- `step_count`
- `max_relative_drop`
- `answer_tag_present`
- `answer_extraction_failed`
- `used_answer_fallback`
- `reference_supported`
- `used_mathruler`
- `reference_type_id`
- `format_reward`
- `rule_reward`

这些指标不应丢失语义，但应降级到以下位置之一：

- debug-only namespace
- trajectory / observation 离线分析
- 短期排障分支的临时仪表盘

#### 9.3.1 `rollout/*` 字段语义与趋势解释

以下解释默认建立在主线 label 为 `math_psgrpo` 的前提下。

##### `rollout/reward`

- 含义：
  - 当前 rollout 样本的最终 PS-GRPO reward 均值
  - 在默认 `math_psgrpo` 语义下，单样本 reward 主要落在 `{0.0, 0.5, 1.0}`
  - `1.0` 表示答案正确且没有 drop-moment
  - `0.5` 表示答案正确但有 drop-moment
  - `0.0` 表示答案错误
- 高/低意味着什么：
  - 高：整体 rollout 结果更接近“答对且过程稳定”
  - 低：更多样本落在答错或过程不稳定区间
- 趋势怎么看：
  - 持续上升：好，通常意味着 correctness 与 process quality 在同步改善
  - 长期横盘在很低位置：差，说明训练没有把样本推离错误区
  - 短期剧烈震荡：常见于 RL 早期，但如果长时间大幅震荡，通常表示优化噪声过大
- 联合解读：
  - `reward` 上升且 `outcome_correct` 同步上升：最健康
  - `reward` 上升但 `outcome_correct` 不动：需要警惕 reward 语义实现、答案抽取或 PRM 诊断口径是否漂移

##### `rollout/reward_std`

- 含义：
  - 当前 rollout 样本最终 reward 的标准差
- 高/低意味着什么：
  - 高：batch 内样本质量差异大，有的很好，有的很差
  - 低：batch 内 reward 更集中
- 趋势怎么看：
  - 训练早期偏高：正常，说明策略还在混合“对/错/半对”状态
  - 后期如果 `reward` 上升且 `reward_std` 缓慢下降：通常是收敛变稳定
  - 如果 `reward_std` 很低但 `reward` 也很低：坏的塌缩，表示大家都在一起答错
  - 如果 `reward_std` 很低且 `reward` 很高：好的收敛，表示大多数样本都稳定变好

##### `rollout/outcome_correct`

- 含义：
  - 最终答案是否正确的均值，本质上可看作 answer-level accuracy proxy
- 高/低意味着什么：
  - 高：更多样本最终答对
  - 低：更多样本最终答错
- 趋势怎么看：
  - 持续上升：明确是好信号
  - 长期不升：说明训练没有真正提升最终答案质量
  - 明显下降：应优先视为训练退化，而不是单纯波动
- 联合解读：
  - `outcome_correct` 上升但 `has_drop_moment` 也明显上升：说明“更常答对了”，但 reasoning path 还不稳

##### `rollout/has_drop_moment`

- 含义：
  - 触发相邻 step score 显著下跌事件的样本比例
  - 是 `math_psgrpo` 对过程稳定性的核心诊断
- 高/低意味着什么：
  - 高：更多样本在 reasoning path 中出现“明显掉点”
  - 低：过程稳定性更好
- 趋势怎么看：
  - 下降：好，表示过程质量更稳
  - 持续上升：差，表示模型虽然可能还能答对，但 reasoning chain 越来越脆
  - 长期维持高位：说明 process quality 一直没有被真正修好
- 联合解读：
  - `outcome_correct` 上升、`has_drop_moment` 下降：最理想
  - `outcome_correct` 上升、`has_drop_moment` 基本不变：可以接受，表示先把答案做对
  - `outcome_correct` 上升、`has_drop_moment` 大幅上升：短期可能可接受，但不应视为最终健康态

##### `rollout/model_reward`

- 含义：
  - PRM 对 step-level reasoning 的聚合分数
  - 它不是最终 PS-GRPO reward，而是 verifier 视角下的“过程质量/可信度”分数
- 高/低意味着什么：
  - 高：PRM 认为轨迹整体更可信
  - 低：PRM 认为轨迹更不可信
- 趋势怎么看：
  - 稳定上升：通常是好事
  - 上升但 `outcome_correct` 不升：要警惕 PRM 与最终答案 correctness 的脱钩
  - 持续下降：PRM 认为 reasoning path 质量在退化
- 联合解读：
  - `model_reward` 和 `outcome_correct` 同升：好
  - `model_reward` 升、`has_drop_moment` 也升：PRM 聚合分数可能被局部高分 step 抬高，但过程稳定性未同步改善

##### `rollout/response_length`

- 含义：
  - 生成回答的平均长度
- 高/低意味着什么：
  - 高：平均回答更长
  - 低：平均回答更短
- 趋势怎么看：
  - 前期适度上升：可能是模型开始学会完整输出 step-by-step 结构
  - 持续失控上升：坏，可能在靠冗长输出刷 reward 或拖慢训练
  - 突然塌缩很低：坏，可能格式断裂、提早停、答案抽取失败
- 联合解读：
  - `response_length` 上升且 `outcome_correct` 同升：可能是正常学习完整 CoT
  - `response_length` 上升但 `reward/outcome_correct` 不升：更像 verbosity drift

##### `rollout/episode`

- 含义：
  - 所处训练 episode
- 高/低意味着什么：
  - 只是训练进度锚点，没有“高更好/低更好”的含义
- 趋势怎么看：
  - 只用于给其它曲线提供阶段上下文

### 9.4 `train/*` 保留策略

`train/*` 目标对齐参考 `Geo3K` run 的同款面板规模，**约 `22` 项**。

这里不主张把 `math_prm` 专属诊断塞进 `train/*`，而是以“通用 PPO/GRPO 训练面”优先。保留策略如下：

#### 9.4.1 必保留核心项

以下项应视为不可删减的核心项：

1. `train/policy_loss`
2. `train/kl`
3. `train/actor_lr`
4. `train/critic_loss`
5. `train/critic_lr`
6. `train/values`
7. `train/reward`
8. `train/return`
9. `train/response_length`
10. `train/total_length`
11. `train/num_actions`
12. `train/episode`

说明：

- 这些项覆盖了 PPO/GRPO 最基本的优化、KL 约束、value 分支状态和长度状态
- `train/reward` / `train/return` / `train/response_length` 虽然来自 experience 侧，但在训练面里保留它们，有助于和参考 `Geo3K` run 的阅读习惯对齐

#### 9.4.2 预留给通用 PPO 诊断的补位项

为了把 `train/*` 规模收敛到参考 run 的约 `22` 项，可以优先保留或补齐下列 **通用 PPO** 诊断项，而不是引入 `math_prm` 特有杂项：

- `train/ptx_loss`
- `train/approx_kl`
- `train/clipfrac`
- `train/ratio_mean`
- `train/ratio_max`
- `train/advantages`
- `train/returns`
- `train/values_std`
- `train/reward_std`
- `train/return_std`

原则：

- 如果某项已经在当前代码链路中稳定存在，优先保留
- 如果某项不存在，但参考 `Geo3K` run 有同类 PPO 诊断项，则优先补这类公共项
- 不允许为了凑满 `22` 项而把 `math_psgrpo` 的所有 reward 诊断硬塞进 `train/*`

#### 9.4.2A 当前计划保留的 `train/*` 目标字段清单

为避免“约 `22` 项”在后续实现阶段再次发散，这里先把目标字段清单定为：

1. `train/policy_loss`
2. `train/kl`
3. `train/actor_lr`
4. `train/critic_loss`
5. `train/critic_lr`
6. `train/values`
7. `train/values_std`
8. `train/reward`
9. `train/reward_std`
10. `train/return`
11. `train/return_std`
12. `train/response_length`
13. `train/total_length`
14. `train/num_actions`
15. `train/approx_kl`
16. `train/clipfrac`
17. `train/ratio_mean`
18. `train/ratio_max`
19. `train/advantages`
20. `train/advantages_std`
21. `train/ptx_loss`
22. `train/episode`

补充约束：

- `global_step` 作为 step 轴锚点存在，但不计入“`22` 个主字段”名额
- `ptx_loss` 只在 PTX 打开时显示；若该 run 未启用 PTX，这个槽位可为空，不允许用 `math_prm` 特有长尾诊断去替补
- `values_std`、`reward_std`、`return_std`、`approx_kl`、`clipfrac`、`ratio_mean`、`ratio_max`、`advantages`、`advantages_std` 属于需要对齐或补齐的通用 PPO 诊断项

#### 9.4.2B `train/*` 字段语义与趋势解释

##### `train/policy_loss`

- 含义：
  - actor 的策略优化目标值
- 高/低意味着什么：
  - 不能简单理解成“越低越好”或“越高越好”
  - 它是 PPO 目标函数的数值表现，受 advantage、ratio clipping、batch 组成共同影响
- 趋势怎么看：
  - 平滑波动：正常
  - 绝对值突然大幅放大并持续震荡：坏，通常表示更新过猛或优势噪声过大
  - 长期几乎不动且 `reward` 也不动：可能学习停滞

##### `train/kl`

- 含义：
  - 当前策略与 reference/old policy 的偏离程度
- 高/低意味着什么：
  - 高：策略更新更激进
  - 低：策略更新更保守
- 趋势怎么看：
  - 中等、可控：最好
  - 持续尖峰：差，容易导致训练不稳
  - 长期过低且 `reward` 不涨：可能更新太保守，学不动
- 联合解读：
  - `kl` 升一点、`reward` 也升：常见且合理
  - `kl` 爆炸、`reward` 反而掉：典型坏信号

##### `train/actor_lr`

- 含义：
  - actor 学习率
- 高/低意味着什么：
  - 只是优化器调度状态
- 趋势怎么看：
  - 按 warmup/scheduler 计划变化：正常
  - 异常跳变：需要查调度器或恢复训练逻辑

##### `train/critic_loss`

- 含义：
  - critic 对 return 的拟合误差
- 高/低意味着什么：
  - 低：critic 拟合更好
  - 高：critic 拟合更差
- 趋势怎么看：
  - 逐步下降或维持稳定低位：好
  - 持续升高或剧烈抖动：差，value 分支不稳
  - 快速掉到极低但 `reward/return` 没改善：不一定是好事，也可能是 critic 学到退化常数解

##### `train/critic_lr`

- 含义：
  - critic 学习率
- 高/低意味着什么：
  - 只是优化器调度状态
- 趋势怎么看：
  - 应符合 scheduler 预期，不单独代表好坏

##### `train/values`

- 含义：
  - critic 预测的 value 均值
- 高/低意味着什么：
  - 代表 critic 认为当前样本 future return 的平均水平
- 趋势怎么看：
  - 应和 `train/return` 大体同向变化
  - 如果长期和 `train/return` 严重背离：critic 标定有问题

##### `train/values_std`

- 含义：
  - critic 预测值的离散程度
- 高/低意味着什么：
  - 高：critic 对不同样本的打分差异更大
  - 低：critic 预测更集中
- 趋势怎么看：
  - 应大体跟随 `return_std`
  - 如果 `return_std` 仍高，但 `values_std` 过低：可能是 critic 塌缩
  - 如果 `values_std` 无端暴涨：可能是 value 不稳

##### `train/reward`

- 含义：
  - 训练阶段消费到的样本 reward 均值
  - 对默认 `math_psgrpo` 来说，本质上仍然对应最终 PS-GRPO reward
- 高/低意味着什么：
  - 高：样本总体更接近答对且过程稳定
  - 低：样本总体更差
- 趋势怎么看：
  - 趋势解释与 `rollout/reward` 基本一致

##### `train/reward_std`

- 含义：
  - 训练 batch reward 的离散程度
- 高/低意味着什么：
  - 高：batch 内质量参差更大
  - 低：reward 更集中
- 趋势怎么看：
  - 解读方式与 `rollout/reward_std` 一致

##### `train/return`

- 含义：
  - PPO 实际优化时使用的 return 均值
  - 会受到 reward shaping、KL、advantage 计算路径影响
- 高/低意味着什么：
  - 高：训练目标视角下的总体回报更高
  - 低：训练目标视角下的总体回报更低
- 趋势怎么看：
  - 一般应与 `reward` 大体同向
  - 若明显偏离 `reward`，通常意味着 KL 或 shaping 在强烈介入

##### `train/return_std`

- 含义：
  - return 的离散程度
- 高/低意味着什么：
  - 高：训练目标值差异更大
  - 低：训练目标值更集中
- 趋势怎么看：
  - 早期偏高正常
  - 后期若 `return` 变好且 `return_std` 下降，通常代表收敛

##### `train/response_length`

- 含义：
  - 训练样本回答长度均值
- 高/低意味着什么：
  - 高：回答更长
  - 低：回答更短
- 趋势怎么看：
  - 不追求单调高或低，只追求“稳定在任务合理区间”
  - 快速上升且 `reward` 不升：疑似长度投机
  - 快速塌缩且 `answer_extraction_failed` 上升：疑似格式/停词异常

##### `train/total_length`

- 含义：
  - prompt + response 的总长度均值
- 高/低意味着什么：
  - 更多是序列预算和显存/吞吐上下文，不直接代表质量
- 趋势怎么看：
  - 和 `response_length` 一起看
  - 持续逼近上限会增加截断与吞吐风险

##### `train/num_actions`

- 含义：
  - 实际参与优化的 action token 数量均值
- 高/低意味着什么：
  - 高：每个样本可训练 token 更多
  - 低：每个样本可训练 token 更少
- 趋势怎么看：
  - 应与 `response_length` 基本同向
  - 如果与 `response_length` 明显背离，可能是 mask 或截断逻辑有问题

##### `train/approx_kl`

- 含义：
  - PPO 常见的近似 KL 诊断项
- 高/低意味着什么：
  - 高：策略相对旧策略改动更大
  - 低：策略改动更小
- 趋势怎么看：
  - 作用类似 `train/kl`
  - 与 `train/kl` 长期严重不一致时，要查实现与日志口径

##### `train/clipfrac`

- 含义：
  - PPO 更新中落入 clipping 区间的比例
- 高/低意味着什么：
  - 高：很多样本已在被 clipping，更新开始受限
  - 低：多数样本仍在未裁剪区域
- 趋势怎么看：
  - 中等水平通常最健康
  - 长期过高：说明更新过激，很多样本被硬裁
  - 长期接近零且 `reward` 不涨：可能更新过弱

##### `train/ratio_mean`

- 含义：
  - 新旧策略概率比值的均值
- 高/低意味着什么：
  - 接近 `1.0`：新旧策略相近
  - 明显偏离 `1.0`：策略变化更大
- 趋势怎么看：
  - 长期围绕 `1.0` 小幅波动：正常
  - 持续偏离 `1.0` 很多：通常意味着更新太猛或实现有问题

##### `train/ratio_max`

- 含义：
  - 新旧策略概率比值中的极端值
- 高/低意味着什么：
  - 高：存在极端更新样本
- 趋势怎么看：
  - 偶发尖峰可以接受
  - 高频大尖峰：坏，通常比 `ratio_mean` 更早暴露不稳定更新

##### `train/advantages`

- 含义：
  - advantage 的均值
- 高/低意味着什么：
  - 对启用 whitening / normalization 的路径，理论上应接近 `0`
  - 对未归一化路径，绝对值会更依赖 reward 尺度
- 趋势怎么看：
  - 若当前算法/配置会做 advantage normalization，则长期偏离 `0` 需要怀疑实现或日志口径
  - 若不做 normalization，则它更像 reward 尺度参考，不单独判断好坏

##### `train/advantages_std`

- 含义：
  - advantage 的离散程度
- 高/低意味着什么：
  - 高：样本梯度驱动力差异更大
  - 低：样本梯度驱动力更集中
- 趋势怎么看：
  - 对做 whitening 的路径，通常应稳定在一个受控范围
  - 持续暴涨：说明训练信号越来越噪

##### `train/ptx_loss`

- 含义：
  - PTX 辅助监督项损失，仅在混合 PTX 时存在
- 高/低意味着什么：
  - 低：监督拟合更好
  - 高：监督拟合更差
- 趋势怎么看：
  - 通常缓慢下降或稳定
  - 若突然升高，说明 RL 更新可能开始明显破坏语言建模/格式保持能力

##### `train/episode`

- 含义：
  - 训练 episode
- 高/低意味着什么：
  - 只是训练阶段锚点，没有质量好坏含义

#### 9.4.3 不应进入 `train/*` 主看板的内容

以下内容不应作为默认 `train/*` 主面板指标：

- `train/reward_metrics/*` 全量展开
- `train/final_reward`
- `train/max_relative_drop`
- `train/answer_tag_present`
- `train/answer_extraction_failed`
- `train/used_answer_fallback`
- `train/reference_supported`
- `train/used_mathruler`
- `train/reference_type_id`
- `train/perf/*`
- trajectory analysis 产出的长尾统计

### 9.5 Runtime Eval 的目标与边界

当前阶段的 `eval/*` 只做 **runtime eval**，不做论文 benchmark eval。

这里的 runtime eval 目标是：

- 在训练过程中给出一个比 rollout 更稳定的趋势信号
- 判断 reward / correctness / process-quality 是否同步向好
- 不把训练 loop 变成 benchmark runner

当前明确不做的事情：

- 不在这一步里接入 MathVista / MathVerse / GeoQA 等论文 benchmark
- 不把 eval 做成单独论文结论页
- 不把 test-time scaling / verifier reranking 逻辑塞进训练期 eval

### 9.6 Runtime Eval 数据来源

runtime eval 必须使用 **独立 held-out 数据**，而不是继续依赖“当前训练 manifest 上可能存在、也可能不存在的 split 名”。

推荐优先级：

1. 单独的 `eval_data` manifest
2. 若短期没有独立 manifest，再考虑从当前 manifest 中显式切出一个固定 held-out 子集

不建议继续依赖的方式：

- 仅靠 `eval_split=test` 去赌当前 JSONL / DatasetDict 里是否真的有 `test` split

原因：

- 当前 Stage 3 主线数据往往是单一 manifest 文件
- 这种情况下，“设了 `eval_split`”并不天然等于“真正拿到了 held-out eval 集”
- 如果 eval 数据和 train 数据语义上没有隔离，趋势会比真实泛化更乐观

### 9.7 Runtime Eval 生成策略

runtime eval 不应复用训练时的 rollout 采样设置。

必须和训练 rollout 解耦的项：

- `n_samples_per_prompt`
- `do_sample`
- `temperature`
- `top_p`
- `top_k`
- `repetition_penalty`
- `no_repeat_ngram_size`

当前推荐默认值：

- `eval_n_samples_per_prompt = 1`
- `eval_do_sample = False`
- `eval_temperature = 0.0` 或等价 greedy 配置
- `eval_top_p = 1.0`
- `eval_top_k = -1`
- `eval_repetition_penalty = 1.0`
- `eval_no_repeat_ngram_size = 0`

约束说明：

- runtime eval 的目标是“趋势稳定”，不是“模拟训练 rollout 分布”
- 如果继续沿用训练态的 `n_samples_per_prompt = 8` 与 `do_sample=True`，eval 曲线会被采样噪声严重污染
- eval 的 X 轴应继续沿用训练步数或 eval 次数，但生成策略必须是固定且低噪声的

### 9.8 `eval/*` 保留集

`eval/*` 先收敛到以下集合，**总量不超过 `8` 项**：

1. `eval/reward`
2. `eval/outcome_correct`
3. `eval/has_drop_moment`
4. `eval/model_reward`
5. `eval/response_length`
6. `eval/answer_extraction_failed`
7. `eval/train_step`
8. `eval/episode`

说明：

- `eval/reward`：当前 `math_psgrpo` 下的最终 reward 趋势
- `eval/outcome_correct`：是否真的更容易答对
- `eval/has_drop_moment`：过程质量是否在改善
- `eval/model_reward`：PRM 聚合分数是否同步改善
- `eval/response_length`：避免 reward/correctness 变化只是长度漂移
- `eval/answer_extraction_failed`：最近格式稳定性虽然已有改善，但短期内这个指标仍值得保留
- `eval/train_step` 与 `eval/episode`：作为上下文锚点保留

默认不进入 `eval/*` 主看板的内容：

- `step_score_min`
- `step_score_mean`
- `step_score_last`
- `step_count`
- `max_relative_drop`
- `answer_tag_present`
- `used_answer_fallback`
- `reference_supported`
- `used_mathruler`
- `reference_type_id`
- 全量 reward_metrics 展开

#### 9.8.1 `eval/*` 字段语义与趋势解释

runtime eval 的解释逻辑与 rollout 相似，但它的样本来源更稳定、生成策略更低噪声，因此更适合看中期趋势。

##### `eval/reward`

- 含义：
  - held-out runtime eval 集上的最终 PS-GRPO reward 均值
- 高/低意味着什么：
  - 高：在固定 held-out 集上，整体质量更好
  - 低：held-out 表现更差
- 趋势怎么看：
  - 稳定上升：最重要的正向信号之一
  - 长期横盘：训练可能没有带来真实泛化改善
  - 后期回落：要警惕过拟合或策略漂移

##### `eval/outcome_correct`

- 含义：
  - held-out eval 集上的最终答案正确率 proxy
- 高/低意味着什么：
  - 高：真实答对更多
  - 低：真实答对更少
- 趋势怎么看：
  - 持续上升：最好理解、也最值得信任
  - 如果它不上升，单看 `eval/reward` 上升不能算完全成功

##### `eval/has_drop_moment`

- 含义：
  - held-out eval 集上的 drop-moment 比例
- 高/低意味着什么：
  - 低更好
  - 高更差
- 趋势怎么看：
  - 下降：过程稳定性改善
  - 上升：过程质量退化

##### `eval/model_reward`

- 含义：
  - held-out eval 集上的 PRM 聚合分数
- 高/低意味着什么：
  - 高：PRM 视角认为 reasoning 更可信
  - 低：PRM 视角认为 reasoning 更不可信
- 趋势怎么看：
  - 应尽量和 `eval/outcome_correct` 同步改善
  - 若二者长期脱钩，要警惕 verifier 口径与最终答案质量不一致

##### `eval/response_length`

- 含义：
  - held-out eval 集上的回答长度均值
- 高/低意味着什么：
  - 本身没有单调好坏，只是上下文
- 趋势怎么看：
  - 应保持稳定或在合理区间轻微变化
  - 持续上升但 `eval/outcome_correct` 不升：大概率是 verbosity drift
  - 持续塌缩：大概率是格式、停词或输出截断问题

##### `eval/answer_extraction_failed`

- 含义：
  - held-out eval 集上答案抽取失败比例
- 高/低意味着什么：
  - 低更好
  - 高更差
- 趋势怎么看：
  - 应尽快下降并逼近 `0`
  - 若突然升高，通常应优先排查输出格式，而不是先怀疑 PPO 本身

##### `eval/train_step`

- 含义：
  - 本次 eval 对应的训练步
- 高/低意味着什么：
  - 只是上下文锚点

##### `eval/episode`

- 含义：
  - 本次 eval 所处 episode
- 高/低意味着什么：
  - 只是上下文锚点

### 9.8.2 Runtime Eval 健康趋势模板

对当前 `math_psgrpo` 主线，runtime eval 最健康的组合趋势应接近：

- `eval/reward` 上升
- `eval/outcome_correct` 上升
- `eval/has_drop_moment` 下降或至少不恶化
- `eval/model_reward` 上升
- `eval/answer_extraction_failed` 下降
- `eval/response_length` 保持在合理区间，不出现失控上升或突然塌缩

需要重点警惕的异常组合：

- `eval/reward` 上升，但 `eval/outcome_correct` 不升
  - 优先怀疑 reward 口径、答案抽取或 verifier 漂移
- `eval/outcome_correct` 上升，但 `eval/has_drop_moment` 也显著上升
  - 说明最终答案更常答对，但 reasoning 过程更脆
- `eval/response_length` 大幅上升，但 `eval/reward` / `eval/outcome_correct` 不升
  - 更像 verbosity drift，不像真实质量提升
- `eval/answer_extraction_failed` 回升
  - 优先排查输出格式、stop 条件和 answer marker

### 9.9 README 交付要求

当指标收口方案落代码后，必须同步更新：

- `examples/math_prm/README.md`
- `examples/math_prm/README_zh.md`

并新增一个专门的小节，例如：

- `W&B Metrics`
- `Runtime Eval`

README 中需要逐项解释最终保留下来的每个指标，至少说明：

1. 指标所在 namespace
2. 指标来自训练链路的哪个阶段
3. 对 `math_psgrpo` 来说它的语义是什么
4. 正常情况下应如何理解它的变化趋势
5. 为什么它被保留在主看板，而不是降级为 debug-only 指标

对 `math_psgrpo` 必须额外明确写清楚：

- `reward` 在默认主线下就是最终 PS-GRPO reward
- `reward` 与 `outcome_correct` 的区别
- `has_drop_moment` 为什么是主看板指标
- `model_reward` 与最终 reward 的关系

### 9.10 实施顺序

按风险和收益排序，建议这样推进：

1. 先做 `rollout/*` 与 `train/*` allowlist 收口
2. 再做 runtime eval 的独立生成参数与独立指标 allowlist
3. 再接入 held-out eval 数据入口
4. 最后更新 `README.md` / `README_zh.md`

原因：

- 先把主看板降噪，才能更清楚地判断 runtime eval 还缺什么
- 如果主看板本身仍然被大量调试指标污染，eval 设计也很容易继续扩张失控

### 9.11 验收标准

此轮改动完成后，至少应满足：

- W&B 主 run 中：
  - `train/*` 主看板规模约为 `22`
  - `rollout/*` 主看板规模为 `7`
  - `eval/*` 主看板规模不超过 `8`
- `math_psgrpo` 默认主线下：
  - `rollout/reward` 与最终 PS-GRPO reward 语义一致
  - 不再同时保留同义的 `rollout/final_reward`
- runtime eval：
  - 不再复用训练态 `n_samples_per_prompt`
  - 不再复用训练态随机采样配置
  - 可以稳定反映趋势，而不是高噪声采样分布
- 文档：
  - `README.md` 与 `README_zh.md` 对最终保留指标逐项解释
  - 明确指出哪些诊断项被降为 debug-only 指标

### 9.12 Run `y9sulzln` 复盘与后续修复目标

本小节记录对当前实验：

- W&B run: `hansbug/LightRFT-URSA8B-Stage3/y9sulzln`
- URL: `https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/y9sulzln`

的观察结论，并将它转化为后续需要修复的项目。

#### 9.12.1 当前状态判断

截至本次检查时，该 run 仍处于：

- `state = running`
- `rollout/global_step = 154`
- `rollout/episode = 1`

因此当前不能把它判断为“已经收敛”。

更准确的判断是：

- 这轮训练 **没有形成稳定收敛平台**
- 它呈现出“前期表现尚可 -> 中段明显失稳 -> 后段部分恢复，但未回到最好窗口”的形态
- 当前更像“仍在波动学习，且健康性不足”，而不是“已经训练完成并收敛”

#### 9.12.2 关键观测结论

按 `rollout/*` 核心指标粗分窗口，表现为：

##### 前段（早期窗口）

- `rollout/reward` 均值约 `0.448`
- `rollout/outcome_correct` 均值约 `0.491`
- `rollout/has_drop_moment` 均值约 `0.443`

说明：

- 训练前段并非完全学不动
- 模型曾经达到过一个中等可接受的质量区间

##### 中段（失稳窗口）

- `rollout/reward` 均值掉到约 `0.381`
- `rollout/outcome_correct` 均值掉到约 `0.411`
- `train/kl` 均值抬到约 `195`

说明：

- 中段出现了明显退化
- 不是“稳定上升后收敛”，而是训练中途把已经学到的一部分东西又损失掉了

##### 后段（部分恢复窗口）

- `rollout/reward` 均值约 `0.419`
- `rollout/outcome_correct` 均值约 `0.454`
- 相比中段有恢复，但仍低于前段均值

说明：

- 后段不是持续恶化，而是能恢复一部分
- 但恢复后仍未超过最好窗口，因此不能视为已收敛

#### 9.12.3 最好窗口与最差窗口

##### 最好窗口特征

在靠后但非最后的局部窗口中，曾出现：

- `rollout/reward ≈ 0.50 ~ 0.53`
- `rollout/outcome_correct ≈ 0.55 ~ 0.61`
- `rollout/has_drop_moment ≈ 0.30 ~ 0.46`

说明：

- 当前链路并不是完全达不到较好状态
- 模型一度能进入“答案质量和过程稳定性同时改善”的窗口
- 但这个窗口没有稳定保持住

##### 最差窗口特征

最差点表现非常典型：

- `rollout/reward = 0.0078`
- `rollout/outcome_correct = 0.0156`
- `rollout/has_drop_moment = 0.7969`
- `rollout/model_reward = 0.1263`
- `rollout/response_length = 318.5`
- `rollout/answer_extraction_failed = 0.1875`

说明：

- 这不是单纯“答案暂时答错一点”
- 而是：
  - 最终答案质量很差
  - 过程质量显著变差
  - 输出长度失控拉长
  - 答案抽取失败比例抬升

#### 9.12.4 当前 run 反映出的主问题排序

基于这轮 run，本计划将后续修复问题按如下优先级排序：

##### 主问题 1：输出格式与答案抽取仍不够稳定

证据：

- `response_length` 高时，`answer_extraction_failed` 也明显更高
- 最差窗口中出现了“更长输出 + 更高抽取失败 + 更差 reward/correctness”的组合

当前判断：

- 这轮里“输出变长”更多时候不像是学会了更完整的 CoT
- 更像是：
  - structured output 不稳
  - stop 条件不稳
  - 最终 answer marker 或抽取链路偶尔失效

修复目标：

- 降低 `answer_extraction_failed`
- 把 `response_length` 拉回稳定合理区间
- 避免长输出与质量下降继续绑定出现

##### 主问题 2：`math_psgrpo` 的过程质量仍未真正稳定下来

证据：

- 即使在后段恢复后，`has_drop_moment` 仍经常在 `0.4 ~ 0.5` 一带
- 最差窗口中 `has_drop_moment` 更是抬到极高水平

当前判断：

- 当前训练并不是只缺“把答案做对”
- 同时还缺“把 reasoning path 稳下来”
- 这说明 `math_psgrpo` 路径虽然已经接通，但还没有把策略稳定推向“正确且不掉点”的区间

修复目标：

- 让 `outcome_correct` 上升时，`has_drop_moment` 不再同步恶化
- 逐步把 `has_drop_moment` 压到明显低于当前 run 的水平

##### 次问题 3：训练稳定性一般，KL 中段有异常尖峰

证据：

- 中段出现明显 `train/kl` 尖峰
- 历史中存在极端高值，远超正常可读范围

当前判断：

- `kl` 不是这轮 run 唯一的问题来源
- 但它说明训练中段确实存在更新不稳
- 中段质量回撤与 KL 异常并存，不能忽略

修复目标：

- 降低中段大尖峰出现概率
- 避免“曾经学到一点 -> 中途大幅回撤 -> 后面再缓慢补回来”的训练轨迹

#### 9.12.5 当前 run 不支持哪些乐观结论

基于以上观察，当前 run **不支持** 以下结论：

- “已经收敛”
- “reward 在稳定上升”
- “当前默认主入口已经健康”
- “只差加 eval，不差训练质量本身”

当前 run 最多只能支持：

- 当前链路能跑通
- 当前链路能偶尔进入较好窗口
- 但训练质量和稳定性都还没有被真正固定住

#### 9.12.6 将该 run 转化为后续修复项目

从工程推进上，应把 `y9sulzln` 暴露的问题转化成一个明确的后续项目：

项目名建议：

- `修复 math_prm 默认主入口的训练质量与稳定性`

该项目必须排在：

- “精简 W&B 主看板并补上 runtime eval”

之后立即推进，而不是继续往后拖。

原因：

- 如果不先把日志面板收口，很难高效判断训练到底坏在哪里
- 但如果只做日志收口，不把这轮 run 暴露出的训练质量问题立为明确修复项，后续仍可能只是更清楚地观察一个不健康训练

#### 9.12.7 后续修复时的重点观察项

当后续开始修复这类训练质量问题时，至少应持续盯住：

- `rollout/reward`
- `rollout/outcome_correct`
- `rollout/has_drop_moment`
- `rollout/model_reward`
- `rollout/response_length`
- `rollout/answer_extraction_failed`
- `train/kl`

理想修复结果应表现为：

- `reward` 上升
- `outcome_correct` 上升
- `has_drop_moment` 下降
- `answer_extraction_failed` 下降
- `response_length` 回到稳定合理区间
- `kl` 不再出现中段极端尖峰

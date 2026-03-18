# URSA Stage 3 在 LightRFT 中的复现方案

## 1. 目标范围

本方案只针对 URSA 论文的 **Stage 3** 复现，不包含：

- Stage 1：URSA-8B 的视觉语言对齐与数学指令微调
- Stage 2：URSA-RM-8B 的 PRM 训练

本次要复现的内容是：

- 使用 **已经训练好的 `URSA-8B`** 作为 policy model
- 使用 **已经训练好的 `URSA-RM-8B`** 作为 verifier / PRM
- 使用 **从 MMathCoT-1M 中筛出的 Stage 3 RL 数据**
- 在 LightRFT 框架下实现并运行 **PS-GRPO**

换句话说，这不是“从头做 URSA 三阶段”，而是只做：

- `URSA-8B + URSA-RM-8B + 15K RL subset + PS-GRPO`

---

## 2. 论文里 Stage 3 的核心要求

根据论文与仓库文档，Stage 3 的关键约束如下：

### 2.1 输入模型

- policy model：`URSA-8B`
- verifier / PRM：`URSA-RM-8B`

### 2.2 输入数据

- RL 数据来自 `MMathCoT-1M`
- 不是直接全量使用
- 论文做法是：
  1. 先从 `MMathCoT-1M` 中按 SFT 近似比例采样 20K
  2. 用 `URSA-8B` 对每题采样 8 次
  3. 过滤掉 8 次结果“全对”或“全错”的题
  4. 剩余约 `15.3K` 样本用于 vanilla GRPO 和 PS-GRPO

### 2.3 PRM 在 Stage 3 里的作用

PRM 不是直接把 step reward 均值塞进 RL 目标。

论文明确否定了两类 naive 用法：

- Variant 1：`outcome reward + mean(process reward)`
- Variant 2：直接把 step-level process reward 加进每一步 advantage

论文最终采用的是 **PS-GRPO**：

1. 让 PRM 产出 step-level reward sequence
2. 检测 reward sequence 中是否出现明显下降，即 `drop-moment`
3. 结合最终答案正确性构造最终 rollout reward

### 2.4 PS-GRPO 奖励公式

论文公式 5：

- `delta_p^i = max((r_p,j - r_p,j+1) / r_p,j) > rho`

其中：

- `rho = 0.3`

论文公式 6：

- 正确且无明显下降：`R = 1`
- 正确但有下降：`R = 1 - gamma`
- 错误：`R = 0`

其中：

- `gamma = 0.5`

### 2.5 论文 Table 14 中的 Stage 3 超参

从 PDF 中可确认的 Stage 3 超参如下：

- Epochs：`2`
- Learning Rate：`2e-6`
- Temperature：`1.0`
- Rollout number per prompt：`8`
- Prompt Max Length：`6048`
- Output Max Length：`3072`
- Precision：`bf16`
- Train Batch Size：`512`
- KL Coefficient：`0.003`
- Data size：`15K`
- Time Cost：约 `18h`

---

## 3. 当前 LightRFT 已有能力

当前 `examples/math_prm/` 目录已经具备部分 URSA Stage 3 所需基础能力。

### 3.1 已有 actor 加载能力

文件：

- `examples/math_prm/train_colocate.py`
- `examples/math_prm/ursa_actor.py`

现状：

- `train_colocate.py` 会检测 `args.pretrain` 是否为 URSA 模型
- 如果是，则改用 `UrsaActor`
- `UrsaActor` 内部使用 `UrsaForConditionalGeneration.from_pretrained(...)` 加载 `URSA-8B`

这意味着：

- **被训练模型的加载路径基本已有，不是当前阻塞点**

### 3.2 已有 PRM 模型加载能力

文件：

- `examples/math_prm/reward_models_utils.py`

现状：

- `_load_ursa_prm_model()` 会加载：
  - `UrsaForTokenClassification`
  - `UrsaProcessor`

也就是说：

- **`URSA-RM-8B` 的 HuggingFace 载入路径已经基本具备**

### 3.3 已有 PRM 推理核心逻辑

文件：

- `examples/math_prm/reward_models.py`
- `examples/math_prm/prm_infer_score.py`

现状：

- `MathPRMReward` 已实现：
  - `Step N:` 格式要求
  - 在 step 边界插入 ` и`
  - 读取 `UrsaForTokenClassification` 的 per-token scalar logits
  - 插入 575 个图像占位符用于 token 对齐
  - 支持 `min / avg / last` 三种 step score 聚合

这意味着：

- **PRM 的“单条 response 打 step 分数”能力已经有了**

### 3.4 已有多模态 prompt 数据挂载通路

文件：

- `lightrft/datasets/prompts_dataset_vl.py`

现状：

- 数据会被整理成：
  - `prompt`
  - `images`
  - `reference`
  - `label`

并通过 `PromptDatasetVL` 提供给训练主循环。

这意味着：

- **Stage 3 数据集的字段挂载方式已经有框架支持**

---

## 4. 当前实现与论文复现之间的关键差距

虽然 `examples/math_prm/` 已经接入了 URSA-RM，但它离论文 Stage 3 的真实复现还有几处关键差距。

### 4.1 差距一：当前实现不是 PS-GRPO，只是“PRM 分数聚合”

文件：

- `examples/math_prm/reward_models.py`
- `examples/math_prm/reward_models_utils.py`

现状：

- `MathPRMReward` 目前只输出单个 sequence-level scalar
- 默认是对 `step_scores` 做 `min` 聚合
- `RECIPE["math_prm"]` 也是直接把这个 model score 当奖励分量使用

问题：

- 这不是论文的 PS-GRPO
- 这只是“把 PRM 分数压成一个序列分数”
- 缺少：
  - step reward sequence 的显式使用
  - `drop-moment` 检测
  - 最终答案正确性判定
  - 公式 6 的奖励重建

结论：

- **当前 `math_prm` 更像“PRM-guided scalar GRPO”而不是论文 Stage 3**

### 4.2 差距二：当前 `MathPRMReward` 实际忽略了图片

文件：

- `examples/math_prm/reward_models.py`

现状：

- 当前 `MathPRMReward.forward()` 在调用 URSA-RM 时，固定使用 `[None]` 作为图像输入
- 注释中也明确写了 `raw_images` 被忽略

问题：

- 论文里的 `URSA-RM-8B` 是多模态 PRM
- Stage II 的一个核心价值就是用 MIE 学会识别视觉误解和图文不一致
- 如果 Stage 3 里实际不给图，PRM 就退化成“文本 verifier”

后果：

- 无法保留论文方法中“多模态 process supervision”的关键能力
- 这会直接偏离论文声称的 Stage 3 设置

结论：

- **如果目标是论文复现，这一块必须恢复为真实多模态 PRM 打分**

### 4.3 差距三：reward model 加载链路存在设计冲突

文件：

- `examples/math_prm/reward_models_utils.py`

现状：

- `build_math_prm()` 已明确说明：
  - PRM 不支持 engine 模式
  - 必须走 HF 直连，因为需要直接访问 per-token logits
- 但 `load_reward_models()` 中的共享 base 加载逻辑，目前会优先对每个 `cfg.path` 调 `_load_engine(...)`

问题：

- 如果脚本里开了 `--rm_use_engine`
- 那么 `math_prm` 这一路很可能会先走 `_load_engine()` 再传进 builder
- 这与 `build_math_prm()` 期望的 `UrsaForTokenClassification + UrsaProcessor` 不一致

结论：

- **`math_prm` 的 RM 加载逻辑需要单独排除 engine 路径**

### 4.4 差距四：Stage 3 的 outcome reward 还没有真正打通

文件：

- `examples/math_prm/reward_models_utils.py`
- `lightrft/trainer/fast_exp_maker.py`

现状：

- 现在 `reward_fn()` 可以接收 `refs`
- 数据集也能产出 `reference`
- 但当前 `math_prm` 这条链路没有真正落实“公式 6 里的最终答案正确性判定”

问题：

- 论文里的 `PS-GRPO reward` 不是 PRM 分数
- 它依赖：
  - 最终答案对不对
  - 过程有没有 `drop-moment`

如果没有 outcome correctness：

- 就只能得到 PRM sequence score
- 不能得到论文定义的 `R^i`

结论：

- **必须把 `reference -> outcome correctness -> final rollout reward` 这条链路接通**

### 4.5 差距五：当前示例脚本超参与论文不对齐

文件：

- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

现状：

- 当前脚本里有多处参数与论文 Table 14 不一致，例如：
  - `EPISODE=20`
  - `LR=1e-6`
  - `KL=0.01`
  - `PROMPT_MAX_LEN=1024`

而论文 Stage 3 明确是：

- `Epochs=2`
- `LR=2e-6`
- `KL=0.003`
- `Prompt Max Length=6048`
- `Output Max Length=3072`
- `N=8`
- `Train Batch Size=512`

结论：

- **当前脚本最多算“可运行样例”，不能直接算论文复现脚本**

---

## 5. 复现的总体设计原则

为了尽量减少改动范围，同时保证尽可能接近论文，我建议遵循以下原则：

### 原则 1：只做 Stage 3 所需最小改动

不去重做：

- PRM 训练器
- Stage 2 数据集
- PRM 训练 loss

只做：

- policy rollout
- PRM inference
- PS-GRPO reward shaping

### 原则 2：尽量沿用 LightRFT 的现有 RL 主链路

尽量不改：

- `SPMDPPOTrainerVL`
- replay buffer
- advantage calculator 主结构

优先改：

- reward model 输出
- reward aggregation
- Stage 3 专用脚本与参数

### 原则 3：先把 reward 逻辑做对，再调超参

如果 reward 公式不对：

- 再怎么调 batch、KL、lr 都不是论文复现

所以优先级应该是：

1. PRM 多模态正确打分
2. PS-GRPO 奖励公式正确
3. 数据筛选流程对齐
4. 超参对齐

---

## 6. 推荐实现路线

我建议分三层实现。

### 6.1 第一层：修正现有 `math_prm` 接入链路

目标：

- 让当前 URSA actor / URSA-RM 至少能按“论文语义”正确加载和打分

需要处理的点：

1. **修正 `math_prm` 的 reward model 加载**
   - `math_prm` 必须强制走 HF 载入
   - 不能复用 `_load_engine()` 路径

2. **修正 `MathPRMReward` 的图像输入**
   - 把 `raw_images` 真正传给 `UrsaProcessor`
   - 不再固定用 `[None]`

3. **确认 `reference` 贯通**
   - 确保 Stage 3 的 ground truth answer 能一路传到 reward 计算处

这一步完成后，才能说：

- “URSA-8B + URSA-RM-8B + 多模态样本”这条线是能正确运行的

### 6.2 第二层：实现论文里的 PS-GRPO reward

目标：

- 让 reward 语义与论文公式 5 / 6 一致

推荐实现方式：

#### 方案 A：推荐方案

让 `MathPRMReward` 直接输出最终的 PS-GRPO scalar reward。

内部流程如下：

1. 从完整对话中提取：
   - question
   - response
   - image
2. 使用真实多模态输入跑 `URSA-RM-8B`
3. 得到 `step_scores`
4. 计算：
   - `relative_drops`
   - `max_relative_drop`
   - `has_drop_moment`
5. 根据 `reference` 判定最终答案是否正确
6. 按论文公式 6 输出最终 reward：
   - `1.0`
   - `0.5`
   - `0.0`

优点：

- 对 LightRFT 主链路侵入最小
- 训练主循环仍然只接收 sequence-level scalar reward
- 最容易先跑通

缺点：

- reward model 与 reward shaping 耦合更强
- 通用性稍弱

#### 方案 B：更干净但更重

让 `MathPRMReward` 只输出结构化过程信息，例如：

- `step_scores`
- `avg_score`
- `min_score`
- `max_relative_drop`
- `has_drop_moment`

然后在 `reward_fn()` 或 `RewardComputationEngine` 中再把：

- PRM 过程信息
- outcome correctness

组合成最终 `PS-GRPO reward`。

优点：

- 职责边界更清楚
- 以后更容易扩展别的 PRM 方案

缺点：

- 需要改动 `fast_exp_maker.py` 的 reward 传递结构
- 改动面更大

结论：

- **如果目标是尽快复现论文 Stage 3，优先采用方案 A**

### 6.3 第三层：把 Stage 3 数据流程与脚本对齐论文

目标：

- 保证“训练什么数据、怎么训练”尽量贴近论文

需要做的事：

1. **离线准备 Stage 3 RL 数据**
   - 先从 `MMathCoT-1M` 采样 20K
   - 用 `URSA-8B` 每题采样 8 次
   - 过滤掉全对/全错样本
   - 得到约 `15.3K` 数据

2. **把这批数据固化为训练 JSONL**

建议字段格式：

```json
{
  "prompt": "题目文本",
  "images": ["/abs/path/to/img.png"],
  "reference": "标准答案",
  "label": "math_psgrpo"
}
```

3. **训练脚本改成论文参数风格**
   - 不再延用当前样例的默认值
   - 尽量对齐 Table 14

---

## 7. 数据集设计建议

### 7.1 Stage 3 训练集的最小字段

建议最终训练集每条记录包含：

- `prompt`
- `images`
- `reference`
- `label`

建议格式：

```json
{
  "prompt": "Please solve the following math problem ...",
  "images": ["/abs/path/to/xxx.png"],
  "reference": "42",
  "label": "math_psgrpo"
}
```

### 7.2 为什么图片路径建议使用绝对路径

LightRFT 当前训练侧没有像 URSA `inference/` 那样显式 `image_root` 参数。

底层图像加载最终走的是：

- `lightrft/trainer/image_utils.py`

其中会直接：

- `Image.open(path)`

因此建议：

- **训练数据里的 `images` 直接写可打开的绝对路径**

这样最稳妥，避免再引入额外的路径拼接逻辑。

### 7.3 outcome correctness 的 ground truth 建议

Stage 3 的最终 reward 必须依赖答案正确性，因此 `reference` 不能缺。

建议：

- 所有训练样本都补齐 `reference`
- 并在预处理阶段统一标准化

### 7.4 参考答案判定建议

如果训练数据题型混杂，不能简单只做字符串全等。

建议采用分层策略：

1. 选择题：
   - 直接比较选项

2. 数值题：
   - 尽量标准化成 canonical number string
   - 必要时允许简单等价比较

3. 可由现有规则库处理的数学答案：
   - 尽量复用 `mathruler` 或已有规则比较器

4. 如果题型差异很大：
   - 可以按题型拆 label，例如：
     - `math_psgrpo_choice`
     - `math_psgrpo_numeric`
     - `math_psgrpo_formula`

---

## 8. 建议的文件级改动范围

以下是建议的后续代码改动位置。

### 8.1 `examples/math_prm/reward_models.py`

用途：

- 实现真正的多模态 `MathPRMReward`
- 在其中补上：
  - step score sequence
  - drop-moment 检测
  - outcome correctness
  - PS-GRPO scalar reward

建议改动点：

- `MathPRMReward.forward()`
- 可能新增辅助方法：
  - 提取最终答案
  - 计算 relative drops
  - 检测 drop-moment
  - 计算最终 PS-GRPO reward

### 8.2 `examples/math_prm/reward_models_utils.py`

用途：

- 修正 `math_prm` 的加载路径
- 新增 Stage 3 专用 label / recipe

建议改动点：

- `load_reward_models()`
  - `math_prm` 不应走 engine base
- `RECIPE`
  - 新增 `math_psgrpo`
- `reward_fn()` / `mix_rewards()`
  - 视最终实现方案决定是否保留额外组合逻辑

### 8.3 `lightrft/trainer/fast_exp_maker.py`

用途：

- 确保 reward model 在计算 reward 时能拿到：
  - `prompt_and_output`
  - `raw_images`
  - `references`
  - `labels`

当前大框架已经基本支持这些字段，但需要在 Stage 3 路径下确认：

- 多模态 PRM 计算时数据没有丢
- 如果采用结构化 reward 输出，可能还需要扩展 reward_metrics 传递

### 8.4 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

用途：

- 从“样例脚本”改造成“论文 Stage 3 复现脚本”

需要改的方向：

- 参数改为论文 Table 14 风格
- 标签从 `math_prm` 改为 `math_psgrpo`
- 去掉错误的 `rm_use_engine` 使用方式
- 明确要求输入数据是“20K->8 sample->静态筛选后”的 RL 集

### 8.5 新增数据预处理脚本

建议新增一个 Stage 3 专用数据准备脚本，用于：

1. 从 `MMathCoT-1M` 中抽样 20K
2. 执行 8 次采样
3. 过滤全对/全错题
4. 产出 15.3K 左右的 RL JSONL

这一步建议离线做，不要塞进训练主脚本。

---

## 9. 对现有脚本 `run_grpo_math_prm_ursa_8b.sh` 的具体判断

当前脚本可以作为“URSA 接入演示”，但不能直接当论文复现脚本。

### 9.1 当前脚本的优点

- 已经指定了：
  - actor = URSA-8B
  - reward model = URSA-RM-8B
  - `Step N:` / `†Answer:` 的 system prompt
- 已经走：
  - FSDP
  - bf16
  - group_norm
  - SGLang actor rollout

### 9.2 当前脚本的主要问题

1. 写着“PS-GRPO”，但代码实际上还不是论文 PS-GRPO
2. 开了 `--rm_use_engine`，与 PRM 的 HF 直连需求冲突
3. 超参数和论文 Table 14 不一致
4. 数据说明里写的是“15K subset”，但论文要求的是“20K→8 sample→静态过滤后的 15K+”

### 9.3 后续脚本应改成什么样

复现脚本应该明确体现：

- policy：`URSA-8B`
- verifier：`URSA-RM-8B`
- label：`math_psgrpo`
- 数据：`stage3_filtered_15k.jsonl`
- 关键超参：
  - `num_episodes = 2`
  - `actor_learning_rate = 2e-6`
  - `init_kl_coef = 0.003`
  - `temperature = 1.0`
  - `n_samples_per_prompt = 8`
  - `prompt_max_len = 6048`
  - `generate_max_len = 3072`
  - `train_batch_size = 512`

---

## 10. 建议的实施顺序

建议后续按以下顺序推进：

### 第一步：修正 PRM 载入与多模态输入

目标：

- 保证 `URSA-RM-8B` 在 Stage 3 中真的是“多模态 PRM”

完成标志：

- reward model 一定按 HF 模型加载
- `raw_images` 真正进入 `UrsaProcessor`

### 第二步：实现 PS-GRPO reward

目标：

- 让 reward 不再是 `min(step_score)`，而是论文公式 6

完成标志：

- 能输出：
  - `step_scores`
  - `max_relative_drop`
  - `has_drop_moment`
  - `outcome_correct`
  - `final_ps_grpo_reward`

### 第三步：离线准备 Stage 3 RL 数据

目标：

- 拿到真正符合论文设置的 15K+ 训练集

完成标志：

- 已有一份 JSONL
- 每条数据都包含：
  - prompt
  - images
  - reference
  - label

### 第四步：改写训练脚本

目标：

- 把当前示例脚本改成论文复现脚本

完成标志：

- 参数与论文基本对齐
- 训练数据输入明确
- RM 加载方式正确

### 第五步：先跑 smoke test，再跑全量

建议先做：

- 100 条样本 smoke test
- 再上 1K
- 最后再全量 15K+

这样可以尽早发现：

- 图像路径问题
- PRM 输入格式问题
- reward 恒为 0 或恒为 1 的问题
- `Step N:` 格式失败问题

---

## 11. 本方案的最终判断

当前 LightRFT 仓库里：

- **URSA actor 的接入基础已经有了**
- **URSA-RM 的 step-level PRM 推理基础已经有了**
- **多模态 prompt 数据挂载通路也已经有了**

但距离“论文 Stage 3 复现”还差下面这些真正关键的东西：

1. **把 PRM 从“文本版 min 聚合分数”恢复成“真实多模态 verifier”**
2. **把 reward 从“PRM scalar”改成“论文定义的 PS-GRPO reward”**
3. **把数据从“任意 15K 子集”改成“论文的一次性静态过滤 15K+ RL 数据”**
4. **把训练脚本参数改成接近 Table 14 的配置**

因此，我的总体结论是：

- 这个仓库已经不是“从零开始适配 URSA”
- 但它也**还没有真正到“可直接复现论文 Stage 3”**
- 正确做法不是推倒重来，而是围绕 reward 侧与数据侧做一轮针对性修正

---

## 12. 下一步建议

如果继续往下推进，我建议下一轮直接做：

1. 输出“文件级修改清单”
2. 逐文件定义：
   - 为什么要改
   - 改到什么程度
   - 哪些地方不要碰
3. 再决定是否开始正式改代码

当前文档只给出方案，不包含代码修改。

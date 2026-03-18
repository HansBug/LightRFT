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

---

## 5. 分阶段任务计划

下面按可执行顺序拆分 phase。每个 phase 都有明确目标、产出和 checklist。

### Phase 0：范围冻结与基线确认

目标：

- 明确当前阶段只做 Stage 3
- 明确前期不做数据筛选，先跑全量
- 明确哪些内容属于“必须对齐”，哪些内容可以延后

产出：

- 一份可执行的阶段计划
- 一份当前代码与论文要求的差异清单

Checklist：

- [ ] 明确当前只做 `URSA-8B + URSA-RM-8B + Stage 3 RL`
- [ ] 明确 Stage 1 / Stage 2 不在本轮范围内
- [ ] 明确前期阶段不做“从完整 `MMathCoT-1M` 中抽 `20K` 候选再筛到 `15.3K`”的静态筛选
- [ ] 明确前期先使用全量数据进行链路验证
- [ ] 明确行为对齐属于必须检查项，而不是“可选优化”
- [ ] 明确数据集载入、图像载入、reference 传递都属于必须检查项

### Phase 1：数据集载入与样本 schema 打通

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

- [ ] 确认训练数据 JSON/JSONL schema 与 `PromptDatasetVL` 兼容
- [ ] 确认 `prompt` 能正确进入训练 prompt 构建流程
- [ ] 确认 `images` 字段能被 `PromptDatasetVL` 和图像预处理链正确读取
- [ ] 确认图像路径格式稳定可用，优先绝对路径
- [ ] 确认缺图、坏图、空图样本的失败模式可观测
- [ ] 确认 `reference` 能从数据集一路传递到 reward 计算入口
- [ ] 确认 `label` 能用于区分 `math_prm` / `math_psgrpo` 等 reward 路径
- [ ] 确认全量数据中不存在大面积缺失 `reference` 的样本
- [ ] 确认全量数据中图像字段的数量分布与格式分布
- [ ] 确认 mixed multimodal 数据在 batch/collate 阶段不会丢字段
- [ ] 先跑一轮小规模 dataset smoke test，打印单条样本和 batch 结构
- [ ] 再跑全量数据扫描，统计字段完整率与异常样本数量

### Phase 2：URSA actor / PRM 载入与基础行为对齐

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

- [ ] 确认 `train_colocate.py` 对 URSA actor 的识别逻辑稳定可复现
- [ ] 确认 `UrsaActor` 的 tokenizer / processor / forward / generate 行为与原始 URSA 使用方式一致
- [ ] 确认 `URSA-RM-8B` 强制走 HF 直连路径，而不是错误走 engine 路径
- [ ] 确认 `UrsaProcessor` 使用方式与原实现保持一致
- [ ] 确认 PRM 的图像输入不再被忽略，而是真实传入
- [ ] 确认 step marker 仍然使用论文与原实现要求的特殊标记
- [ ] 确认 `Step N:` 格式要求被保留，且不会被 chat template 破坏
- [ ] 确认 `†Answer:` 的终答案格式要求被保留
- [ ] 确认读取 step score 的 token 位置与原脚本一致
- [ ] 确认图像占位 token / padding 逻辑与现有 URSA-RM 推理逻辑一致
- [ ] 确认单条样本下，LightRFT 侧 PRM 输出与原始推理脚本输出可对比
- [ ] 确认 `min / avg / last` 等聚合结果在对齐测试中可复现
- [ ] 对齐失败时记录是文本格式问题、图像问题还是 processor 行为问题

### Phase 3：先跑通“全量数据 + 基础 reward”训练链路

目标：

- 在不引入数据筛选的前提下，先跑通一版完整训练
- 确认整个 RL 训练主链路没有结构性断点

说明：

- 这一阶段允许 reward 先保持“基础版 math_prm”
- 目标是先验证主链路可运行、日志可观测、显存和吞吐行为可接受

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

- [ ] 明确 Phase 3 baseline reward 使用 `math_prm`
- [ ] 明确 `math_prm` 在本阶段的语义就是 `min(step_scores)`
- [ ] 明确 Phase 3 不引入 `drop-moment`
- [ ] 明确 Phase 3 不引入 outcome correctness
- [ ] 明确 Phase 3 不引入 `math_prm_combined`
- [ ] 明确 Phase 3 不引入 rule reward
- [ ] 修正 `mix_rewards()` 中与 URSA 格式不对齐的全局 format reward，使其不参与 `math_prm` 路径
- [ ] 确认全量数据可被 dataloader 稳定消费
- [ ] 确认 actor rollout 能在多模态样本上正常生成
- [ ] 确认 reward model 能在训练环节稳定收到 `prompt_and_output`
- [ ] 确认 reward model 能在训练环节稳定收到 `raw_images`
- [ ] 确认 reward model 能在训练环节稳定收到 `references`
- [ ] 确认 reward 输出能被 trainer 正常记录和消费
- [ ] 确认 reward 不会大面积恒为 0、恒为 1 或恒为同一常数
- [ ] 确认生成结果满足 `Step N:` / `†Answer:` 的基本格式要求
- [ ] 确认 rollout 后的权重同步回推理引擎链路正常
- [ ] 确认至少能跑通 smoke test 训练
- [ ] 确认再进行一次更长时长的全量训练试跑
- [ ] 记录训练中的 OOM、死锁、图像载入异常、reward 异常分布等问题

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
- [ ] 去掉与 PRM 直连需求冲突的 `rm_use_engine` 用法
- [ ] 明确 `freeze_prefix`、多模态开关、图像字段名等必要参数
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

1. 数据载入与字段传递需要显式检查
2. 多模态 PRM 的真实行为需要对齐检查
3. PS-GRPO reward 公式需要真正落地
4. 当前阶段先用完整 `MMathCoT-1M` 全量数据试跑，而不是先做筛选
5. 数据筛选流程属于后置 phase，需要在基础链路稳定后再补

因此，本计划不是直接追求“论文最终形态一步到位”，而是明确分阶段推进：

- 先跑通
- 再对齐行为
- 再对齐 reward
- 最后补筛选数据

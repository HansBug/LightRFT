# Phase 7 格式稳定性问题分析

本文档记录 `Phase 7` bounded full-data observation 中暴露出来的“格式稳定性未通过”问题，解释这里的“格式稳定性”到底指什么，为什么当前不能把这轮训练视为健康 baseline，以及真实样本里具体出现了哪些异常。

## 1. 背景

`Phase 7` 最终真实观测运行如下：

- 训练日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_205242.log`
- 观测摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_205242.json`
- trajectory 文件：
  - `/data/LightRFT/results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_205243/trajectories/trajectories_step_1.json`

这轮 run 已经证明：

- `hf` rollout 正常工作
- `math_psgrpo` reward 正常参与训练
- PPO update 已真实发生
- trajectory 保存已经正常落盘
- 离线分析可以直接读取这些轨迹并做 Phase 7 指标统计

所以当前的问题不是“训练跑不起来”，而是“输出质量是否足够健康”。

## 2. 这里的“格式稳定性”是什么意思

这里的“格式稳定性”，不是只看回答里有没有出现 `Step` 和 `†Answer:`。

更准确地说，它指的是：

- 模型是否能持续使用 `Step N:` 结构输出推理
- 模型是否只输出一个最终 `†Answer:`
- 模型是否会在 `†Answer:` 后干净停止
- 模型是否会在尾部继续生成垃圾字符、重复 token、重复 answer marker
- 同样的格式要求下，是否大多数样本都能稳定满足，而不是“一部分正常、一部分漂移”

当前 `Phase 7` 的严格格式检查逻辑在：

- `/data/LightRFT/examples/math_prm/analyze_phase7_observation.py`

其中 `is_valid_stage3_format(text)` 的判定要点是：

- 至少出现一个 `Step N:`
- `†Answer:` 行必须恰好只有一个
- 最后一个非空行必须以 `†Answer:` 开头

注意：

- 这已经比训练中较宽松的 `format_reward` 更严格
- 但即便通过这个严格检查，也不一定等于“人工看起来已经完全干净”

## 3. 为什么当前会说“健康性未通过”

`Phase 7` summary 里最终结论是：

- `healthy_pass = false`

唯一保留下来的问题项是：

- `生成格式成功率偏低：format_success_ratio=0.7500`

也就是说：

- 保存下来的 `4` 条 trajectory 中
- 只有 `3` 条通过了当前的严格格式 validator
- 还有 `1` 条是明确硬失败

但从人工检查的角度看，实际上这 `3` 条“通过”的样本也并不干净，它们仍然带有明显尾巴污染，只是还没有触发严格 validator 的“唯一 `†Answer:` + 最后一行是 answer”硬失败条件。

所以：

- `0.75` 已经说明当前格式稳定性不达标
- 而真实体感质量其实比 `0.75` 这个数字表现得更差一些

## 4. 真实输出样本

以下样本都来自：

- `/data/LightRFT/results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_205243/trajectories/trajectories_step_1.json`

### 样本 0

```text
StepStep 1: Observe the image. The image shows a long expanse of a plain or field in the foreground.

Step 2: Analyze the horizon. The horizon line is clearly visible and shows a level plane.

Step 3: Consider the perspective. The perspective of the image is from a height, looking down towards the horizon, which further supports the observation of a flat landscape.

†Answer: yes7777777777777777777777777777777777...
```

问题：

- `StepStep 1:`，step header 已经出现轻微漂移
- `†Answer:` 后面不是干净的 `yes`
- `†Answer:` 后面继续拖着大段 `7777...`

当前严格 validator 仍把它判为 `True`，因为：

- 只有一个 `†Answer:`
- 最后一条非空行仍然是 `†Answer: ...`

但人工看，这显然不是一个健康、稳定的收尾格式。

### 样本 1

```text
StepStep 1: Observe the image. The image shows a clear sky with a cloud, two planes on the ground, and a plain backdrop.

Step 2: Analyze the backdrop. The backdrop is a relatively flat expanse of land, consistent with a plain.

Step 3: Consider the cloud. The cloud is small and doesn't significantly distract from the overall impression of a flat landscape.

Step 4: Conclude. Based on the observation and analysis, the landscape in the image is flat.

†Answer: yes, no need to answer777777777777777777777777...
```

问题：

- 仍然是 `StepStep 1:`
- `†Answer:` 行本身已经不干净，混入了 `no need to answer`
- 后面继续拖了长串 `7777...`

这条在严格 validator 下仍判为 `True`，但它依然不是“健康格式”。

### 样本 2

```text
StepStep 1: Observe the image provided. The image shows a desert landscape with a hilly area in the background.

Step 2: Analyze the landscape. The majority of the visible landscape is flat, but there are noticeable hills and a sparse distribution of vegetation.

Step 3: Determine the answer. Based on the observation and analysis, the landscape is not perfectly flat, but it is mostly flat.

†Answer: Mostly flatFrealth777877788888888888...

†Answer: Mostly flat777888888888888888...
```

这是当前最典型的“格式稳定性硬失败”样本。

问题：

- 再次出现 `StepStep 1:`
- 第一个 `†Answer:` 后尾巴已经污染
- 更严重的是又重新起了第二个 `†Answer:`

这条会被严格 validator 判为 `False`，因为：

- `†Answer:` 不再唯一

这类样本说明模型在输出 final answer 后没有稳定停住，而是继续生成，并重新开启了一段 answer。

### 样本 3

```text
StepStep 1: Observe the image. The image shows a plain with a light at the horizon and two airplanes.  The main focus is the helicopter.

Step 2: Analyze the context. The context is not explicitly provided in the prompt, but the image suggests a landscape.

Step 3: Determine the nature of the roles. The prompt does not provide individual roles, only that the answer is "no".

Step 4: Conclude the answer based on the lack of information. ...

†Answer: no777777777777777777777777777777...
```

问题：

- `StepStep 1:` 仍存在
- `†Answer:` 后仍有长尾垃圾 token

这条在严格 validator 下仍判为 `True`，但也不能算“真正干净”。

## 5. 为什么训练日志里的 `Format Reward = 1.0` 还不够

在训练日志里：

- `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_205242.log`

可以看到：

- `Format Reward      1.0000 ± 0.0000`

这并不代表“格式已经健康”。

原因是：

- 训练里的 `format_reward` 规则比较宽松
- `Phase 7` 离线分析里的 `is_valid_stage3_format(...)` 更严格
- 人工检查又比严格 validator 更能识别尾巴污染和弱漂移

所以当前存在三层标准：

1. 训练期较宽松的 `format_reward`
2. Phase 7 较严格的结构 validator
3. 人工直观质量检查

这三层不是同一个东西。

当前观测说明：

- 在训练期宽松规则下，4/4 看起来都“像是正确格式”
- 在 Phase 7 严格规则下，只有 3/4 通过
- 在人工直观质量上，这 4 条都还有明显格式污染问题

## 6. 当前问题的本质

当前不是：

- rollout 引擎坏了
- reward 模型坏了
- 图像没参与
- answer extraction 失效
- 训练一开始就崩

这些在 `Phase 7` 里都已经被证明不是主问题。

当前更准确的问题是：

- 模型已经学会了大致的 `Step ... / †Answer ...` 外形
- 但还没有学会稳定、干净地在 `†Answer:` 处收尾

所以它经常表现为：

- `StepStep 1:`
- `†Answer:` 后继续乱写
- 重复 token 尾巴，如 `7777...`
- 偶尔重新起一个第二次 `†Answer:`

也就是说：

- 现在的问题不再是“不会按格式输出”
- 而是“会按格式输出，但不能稳定、干净、可重复地按格式结束”

## 7. 对健康性结论的正确理解

因此，当前 `Phase 7` 的结论应该理解为：

- 训练链路已经可观测、可分析、可保存
- 当前已经能做真实的 bounded full-data run
- 当前 reward/trajectory/多模态检查都能给出有效统计
- 但这轮结果还不能作为“健康 baseline”

原因不是系统挂了，而是：

- 输出格式一致性还不够
- final answer 收尾还不稳定

## 8. 当前最明确的下一步

在进入后续阶段前，当前最值得优先处理的是：

- 继续提高 `format_success_ratio`
- 压掉 `†Answer:` 后的长尾乱码
- 压掉重复 `†Answer:`
- 进一步收敛 `StepStep 1:` 这类 header 漂移

换句话说，下一步要解决的已经不是“能不能跑”，而是“跑出来的东西能不能稳定像样”。

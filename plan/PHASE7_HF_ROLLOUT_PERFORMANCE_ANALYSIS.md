# Phase 7 HF Rollout 性能问题分析

本文档记录 `Phase 7` 在格式稳定性问题修复之后，仍然保留下来的推理性能问题。这里讨论的是：

- 为什么当前本地 `hf` 多模态 rollout 仍然明显偏慢
- 时间主要耗在什么地方
- 在当前约束下有没有希望继续优化
- 更可能有效的解法是什么

## 1. 当前结论

先给结论：

- 当前“超级无敌慢到 20 分钟内跑不出样本”的版本已经解决
- 但“本地 `hf` rollout 依然明显偏慢”的问题仍然存在
- 从最新真实运行看，主要瓶颈明确在 `rollout generate`
- 不是 PPO train、不是 reward model、也不是 checkpoint save 在主导总时长
- 在当前冻结环境里，最有价值的长期解法仍然是重新获得 `vllm/sglang` 级别的 rollout 支持
- 如果只能继续停留在 `hf` 路径里，可以做一些工程优化，但预期收益大概率有限

## 2. 证据：时间主要耗在 generate

最新健康通过的真实 run：

- 训练日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_233851.log`
- 观测摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_233851.json`

日志里的关键时间点：

- `23:41:28`
  - `Start VLM gather_and_generate ..., total prompts: 4`
- `23:52:30`
  - `step 0 generate length: {'total_samples': 32, ...}`
  - `***Rollout engine generation time (global max): 661.9063s`
- `23:53:25`
  - `DCP checkpoint saved ...`

把这几个时间点换算以后：

- `generate_s = 662`
- `post_generate_to_ckpt_s = 55`
- `total_s = 717`
- `generate_ratio = 662 / 717 = 0.9233`

也就是：

- 约 `92.3%` 的总耗时都花在 rollout generate
- generate 之后到训练、trajectory 保存、checkpoint 落盘合计只占约 `7.7%`

这已经足够说明：

- 现在的主性能瓶颈非常明确，就是本地 `hf` 的多模态生成

## 3. 进一步量化：吞吐确实偏低

同一轮日志里：

- `total_samples = 32`
- `mean_length = 148.875`

粗略估算总生成 token 数：

- `32 * 148.875 = 4764`

对应吞吐大约是：

- 全局生成吞吐：`4764 / 661.9063 ≈ 7.2 tokens/s`
- 按 8 张 GPU 粗分：`≈ 0.9 tokens/s/GPU`

这个速度对于当前规模的 VLM rollout 来说，明显偏慢。

这里不需要纠结这个数字是否“绝对精确到最后一位”，因为它已经足够说明问题：

- 当前不是略慢
- 而是明显慢

## 4. 当前不太像瓶颈的部分

从这轮日志看，下面这些并不是主瓶颈：

### 4.1 PPO train

日志中：

- `Train epoch [1/1]` 从第一条进度到 `100%`，只用了大约 `7` 秒

因此：

- actor/optimizer 训练不是当前主要耗时

### 4.2 reward model / initial model / critic 的 reload/offload

日志中：

- `after reload_model`
- `after offload_model`

这些阶段确实存在，但它们都发生在 generate 之后，而且整个 generate 后半段到 checkpoint 合计也只有约 `55` 秒。

因此：

- reload/offload 有成本
- 但不是当前最重的成本

### 4.3 trajectory 保存 / checkpoint 保存

日志中：

- `Saved 4 trajectories`
- `DCP checkpoint saved`

这些都发生在最后阶段，量级明显小于前面的 generate。

因此：

- 保存链路也不是当前主瓶颈

## 5. 为什么本地 HF rollout 会慢

当前更像是以下几个因素叠加：

### 5.1 当前走的是本地 HF actor.generate，不是专门的推理引擎

代码上：

- `/data/LightRFT/lightrft/strategy/strategy_base.py`

当前 `engine_type='hf'` 的语义是：

- 直接复用训练 actor 作为 inference engine
- 不会像 `vllm/sglang` 那样单独起一个专门的高吞吐推理 runtime

对应代码里也明确写着：

- `engine_type='hf' requires the prepared actor to be passed in`
- `Skip update engine weights for local HF engine because it reuses the actor directly.`

这条路径的优点是：

- 兼容 URSA
- 集成简单
- 能快速打通训练链路

缺点也很明显：

- 它不是为高吞吐 rollout 设计的

### 5.2 当前模型是多模态 URSA，不是纯文本 LLM

当前 rollout 不是只做语言 decode，还包含：

- 图像预处理
- 视觉 tower
- projector / aligner
- 语言模型 decode

这天然比纯文本模型更重。

即使格式 bug 已修复，这层成本仍然存在。

### 5.3 当前外部高性能 rollout 引擎在冻结环境里不可用

这个项目里最大的潜在性能杠杆，其实是：

- `vllm`
- `sglang`

但这两条路径当前在 URSA 上都没有稳定跑通，详见：

- `/data/LightRFT/plan/URSA_ROLLOUT_ENGINE_FAILURE_ANALYSIS.md`

也就是说，现在之所以还停留在 `hf` 路径，并不是因为 `hf` 更快，而是因为：

- 它是当前冻结环境里唯一真正可用的 rollout 路径

因此，从工程现实上讲：

- 当前性能问题有一部分是“被架构选择锁死”的
- 不是简单调几个采样参数就能完全抹平

### 5.4 当前 rollout 仍然在做完整 autoregressive decode

即使格式已经稳定，当前这轮的生成长度仍然是：

- `min=45`
- `max=455`
- `mean=148.875`

而且是多样本 rollout：

- `n_samples_per_prompt = 4`

这意味着：

- 当前仍然在做一批真正的、多样本、多 token 的多模态 decode
- 工作量本身并不小

## 6. 当前约束下，性能问题有没有希望继续改善

有希望，但要分层看。

### 6.1 高概率能改善的是“工作量”

例如：

- 再收紧 smoke / observation 的 decode budget
- 降低 `n_samples_per_prompt`
- 降低 `generate_max_len`
- 继续优化 stop 条件，让无效尾部更早停住

这类优化的特点是：

- 实现相对容易
- 风险较低
- 对 bounded run 很有帮助

但它们改善的是：

- “总要生成多少 token”

而不是：

- “每个 token 本身生成有多快”

所以这类办法对 smoke 很有效，但不能从根上把 `hf` 路径变成高吞吐引擎。

### 6.2 中等概率能改善的是 HF 路径里的工程细节

可能方向包括：

- 更细地 profile `actor.generate()`，区分 prefill 和 decode
- 看视觉 tower / projector 是否重复做了不必要工作
- 检查是否存在可以缓存的图像特征或 prompt 前缀特征
- 检查 batch 组织方式是否还能更高效

这些方向理论上可能带来收益，但问题在于：

- 需要额外 profiling 证据
- 改动复杂度高
- 对训练主链可能有兼容性风险

而且当前还没有证据表明这里只要做一个小改动就能拿到数量级提升。

因此：

- 有希望
- 但不应该对短期收益过度乐观

### 6.3 最大收益的方向仍然是外部推理引擎

如果能让 URSA 真正稳定跑在：

- `vllm`
- 或 `sglang`

那么才有可能从根上改善 rollout 吞吐。

这是当前我认为最有希望的长期方向。

但现实约束也很明确：

- 当前冻结环境下，这条路还没打通
- 不是一个“小修小补”级别的问题

所以它是：

- 高收益
- 高成本
- 当前被环境/架构兼容性阻塞

## 7. 我对可能解法的排序

按“当前约束下的现实可行性”排序，我会这样看：

### 方案 A：继续把 observation/smoke 的 decode 工作量压小

可做内容：

- 更激进地限制 `generate_max_len`
- 更激进地限制 `n_samples_per_prompt`
- 继续打磨 structured stop，让结束更早更稳定

优点：

- 简单
- 风险低
- 对 bounded run 立刻有效

缺点：

- 治标不治本
- 不会把 `hf` rollout 变成真正高吞吐

判断：

- 最现实
- 最适合作为“继续做实验的工程手段”

### 方案 B：专门 profile HF rollout，找二级热点

可做内容：

- 对 `actor.generate()` 前后加更细的时间统计
- 把 prefill / decode / image preprocess / RM / trainer 分开记时
- 看是否有特定子阶段异常重

优点：

- 能把问题看得更清楚
- 便于决定值不值得继续在 `hf` 路径上深挖

缺点：

- 本身不直接提速
- 需要额外分析轮次

判断：

- 很值得做
- 应该是如果后面继续碰性能，最先做的动作

### 方案 C：在 HF 路径里继续做深层工程优化

可做内容：

- 缓存图像特征
- 优化 batch 组织
- 避免重复 prefill
- 探索训练 actor 与推理 actor 的更轻量分离方式

优点：

- 如果命中，可能有明显收益

缺点：

- 改动复杂
- 风险高
- 可能破坏现有训练链
- 当前没有足够证据保证收益

判断：

- 有希望，但不应盲做

### 方案 D：重新打通 vLLM/SGLang rollout

优点：

- 理论收益最大
- 最接近真正的高吞吐推理方案

缺点：

- 当前是被环境和 URSA 架构支持卡住的
- 成本最高

判断：

- 长期最值得
- 短期最难

## 8. 最终判断

基于当前证据，我的判断是：

- 当前性能问题是真实存在的
- 但它已经不再阻塞 `Phase 7` 的 bounded 观测完成
- 如果只是为了继续推进后续 phase，可以先接受当前速度
- 如果目标是“把 Stage 3 训练做成长期高效可重复跑的方案”，那这个问题迟早还得处理

更具体地说：

- 短期：可以先接受，继续推进后续 reward / 数据 / 训练质量工作
- 中期：值得补一轮更细的 profiling
- 长期：最值得的方向仍然是重新获得 `vllm/sglang` 级别的 rollout 支持

## 9. 建议的下一步

如果后面要继续处理性能，我建议顺序是：

1. 先补更细的 rollout profiling，而不是直接猜优化点
2. 先看 HF 路径里是不是某个子阶段异常重
3. 如果 HF 路径没有明显低垂果子，就不要在这里无上限深挖
4. 转而评估重新支持 `vllm/sglang` 的成本与收益

因此，当前最合理的状态是：

- 问题已经盘清楚
- 不急着立刻解决
- 先把它作为“已知性能短板”记录下来，避免之后误判成格式或 reward 问题

# Phase 7 HF Rollout 性能问题分析

本文档记录 `Phase 7` 在格式稳定性问题基本修复之后，仍然保留下来的 rollout 性能问题。这里讨论的是：

- 为什么当前本地 `hf` 多模态 rollout 仍然明显偏慢
- 当前慢到什么程度，是否已经超出“正常的 HF 路径损耗”
- 单独运行 `URSA-8B` 的时候到底是不是也这么慢
- 在当前约束下，合理的 rollout 时间量级应该是多少
- 接下来应该如何最小化拆解问题，而不是盲目改代码

## 1. 当前结论

先给结论：

- 当前“超级无敌慢到 20 分钟内跑不出样本”的最早版本已经不是主状态
- 但“本地 `hf` rollout 依然极端偏慢”的问题仍然存在
- 按 `2026-03-20` 的新增基线看，这个慢法已经不是“HF 路径正常偏慢”，而是明显异常
- 单独运行 `URSA-8B` 并不会慢到几十分钟，更不会慢到一小时一个 step
- 当前真实训练里，一个 `global step` 跑到十几分钟乃至一小时，根因明确在 `rollout generate`
- 不是 PPO train、不是 reward model、也不是 checkpoint save 在主导总时长
- 当前最可疑的根因组合是：
  - 本地 `hf` rollout 直接复用 FSDP 分片训练 actor
  - rollout 期间仍带着训练态的 `gradient_checkpointing`
  - 运行时日志已经明确显示 KV cache 实际被关掉

## 2. 证据：时间主要耗在 generate

历史健康通过的 `Phase 7` run：

- 训练日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_233851.log`
- 观测摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_233851.json`

这轮已经显示：

- 约 `92.3%` 的总耗时都花在 rollout generate
- generate 之后到训练、trajectory 保存、checkpoint 落盘合计只占约 `7.7%`

这已经足够说明：

- 当前主性能瓶颈非常明确，就是本地 `hf` 的多模态生成

## 2.1 `2026-03-20` 最新 Stage 3 长跑日志：已经不是“略慢”，而是明显异常

新的真实日志：

- `/data/LightRFT/rft_logs/lightrft-ursa8b-stage3-psgrpo-2h-stability-v2/node0_20260320_091239.log`

这份日志比上一轮更能说明问题，因为它已经真实走到了多个 `global step`。

第一轮：

- `Start VLM gather_and_generate ..., total prompts: 16`
- `step 0 generate length: total_samples=128, mean_length=150.078125`
- `***Rollout engine generation time (global max): 802.7335s`
- `PPO Train TIMECOST 8.8986s`

第二轮：

- 仍然是 `total prompts: 16`
- `***Rollout engine generation time (global max): 3895.6270s`
- `PPO Train TIMECOST 7.8046s`

换句话说：

- 第一轮光 generate 就花了约 `13.4` 分钟
- 第二轮光 generate 就花了约 `64.9` 分钟
- 但 PPO train 始终只有 `8` 秒左右

所以“为什么一个多小时才一个 global step”这个问题，现在已经可以明确回答：

- **不是训练更新慢**
- **不是奖励模型慢**
- **而是 rollout generate 异常慢**
- **并且第二轮比第一轮又进一步膨胀，说明这不是一个稳定但略慢的系统**

## 3. 进一步量化：当前吞吐确实偏低

用第一轮日志里的数粗估：

- `total_samples = 128`
- `mean_length = 150.078125`

粗略总生成 token 数：

- `128 * 150.078125 ≈ 19210`

对应吞吐大约：

- 全局生成吞吐：`19210 / 802.7335 ≈ 23.9 tokens/s`

这里不需要纠结是否精确到最后一位，因为它已经足够说明问题：

- 当前不是略慢
- 而是明显慢

## 3.1 单独运行 URSA-8B 的直接推理基线

为了回答“单独运行 8B 模型的时候也这么慢吗”，我直接绕开 LightRFT rollout，只跑了：

- 单卡 `cuda:0`
- 原生 `URSA-8B`
- 直接 `model.generate()`
- 使用和训练接近的采样参数：
  - `temperature=1.0`
  - `top_p=1.0`
  - `repetition_penalty=1.05`
  - `no_repeat_ngram_size=4`
  - `max_new_tokens=1024`

选取的是当前 Stage 3 manifest 里的真实多模态样本。

结果如下：

- `batch_size=4`
  - `time_sec=9.703`
  - `generated_tokens_mean=131.0`
  - `generated_tokens_max=208`
  - `peak_mem_gb=19.91`
- `batch_size=8`
  - `time_sec=14.663`
  - `generated_tokens_mean=179.25`
  - `generated_tokens_max=315`
  - `peak_mem_gb=24.77`
- `batch_size=16`
  - `time_sec=16.481`
  - `generated_tokens_mean=151.375`
  - `generated_tokens_max=303`
  - `peak_mem_gb=34.49`

这组数据足够说明两件事：

- 单独运行 `URSA-8B` **不是**几十分钟量级
- 同等输出长度下，单卡直接推理是 **十几秒** 量级

因此，当前 rollout 的慢，不能用“URSA 本来就慢”来解释。

## 3.2 小批对照：训练态 + gradient checkpointing 确实会变慢，但仍不该慢到小时级

我还做了一个更小的控制实验，只看两条真实样本、`max_new_tokens=256`，比较训练态和 `gradient_checkpointing` 的影响：

- `eval_gc_off`
  - `time_sec=7.085`
  - `max_gen_tokens=151`
- `eval_gc_on`
  - `time_sec=9.731`
  - `max_gen_tokens=243`
- `train_gc_on`
  - `time_sec=33.198`
  - `max_gen_tokens=144`

这个结果说明：

- 单看 `gradient_checkpointing + train mode`，确实能把直接推理拖慢到 `2x~5x`
- 但它依然是 **几十秒** 量级
- 这还不足以单独解释当前真实 rollout 里 `802s` 甚至 `3895s` 的极端慢速

所以更合理的判断是：

- `gradient_checkpointing` 是重要因素
- 但不是唯一因素
- 很可能还叠加了 FSDP 分片训练 actor 直接参与 autoregressive decode 的通信/重组成本

## 4. 当前不太像瓶颈的部分

从真实训练日志看，下面这些不是主瓶颈：

### 4.1 PPO train

日志中：

- 第一轮 `PPO Train TIMECOST 8.8986s`
- 第二轮 `PPO Train TIMECOST 7.8046s`

因此：

- actor/optimizer 训练不是当前主要耗时

### 4.2 reward model / initial model / critic 的 reload/offload

日志中有：

- `after reload_model`
- `after offload_model`

这些阶段确实存在，但量级远小于前面的 generate。

因此：

- reload/offload 有成本
- 但不是当前最重的成本

### 4.3 trajectory 保存 / checkpoint 保存

这些都发生在最后阶段，量级明显小于前面的 generate。

因此：

- 保存链路也不是当前主瓶颈

## 5. 为什么本地 HF rollout 会慢

当前更像是以下几个因素叠加：

### 5.1 当前走的是本地 HF actor.generate，不是专门的推理引擎

代码上：

- `/data/LightRFT/lightrft/strategy/strategy_base.py:715`
- `/data/LightRFT/lightrft/strategy/strategy_base.py:719`

当前 `engine_type='hf'` 的语义是：

- 直接复用训练 actor 作为 inference engine
- 不会像 `vllm/sglang` 那样单独起一个专门的高吞吐推理 runtime

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

### 5.3 当前外部高性能 rollout 引擎在冻结环境里不可用

最大的潜在性能杠杆其实是：

- `vllm`
- `sglang`

但这两条路径当前在 URSA 上都没有稳定跑通，详见：

- `/data/LightRFT/plan/URSA_ROLLOUT_ENGINE_FAILURE_ANALYSIS.md`

也就是说，现在之所以停留在 `hf` 路径，并不是因为 `hf` 更快，而是因为：

- 它是当前冻结环境里唯一真正可用的 rollout 路径

### 5.4 当前 rollout 使用的是已经 FSDP 包装过的训练 actor

训练准备阶段里，actor 在进入 rollout 之前就已经被 `prepare_model(..., is_training=True)` 包成了 FSDP 训练模型：

- `/data/LightRFT/lightrft/strategy/strategy_base.py:561`
- `/data/LightRFT/lightrft/strategy/strategy_base.py:564`

而本地 `hf` rollout 的生成入口只是直接调用：

- `/data/LightRFT/lightrft/strategy/strategy_base.py:951`
- `/data/LightRFT/lightrft/strategy/strategy_base.py:952`
- `/data/LightRFT/lightrft/models/actor_vl.py:252`
- `/data/LightRFT/lightrft/models/actor_vl.py:276`

这意味着当前 rollout 非常可能是在：

- 一个 **FSDP 分片训练模型**
- 上面直接做 **autoregressive decode**

这和“单独加载一个完整的、纯推理用途的 URSA 模型”完全不是一条路径。

### 5.5 当前 rollout 期间 KV cache 实际失效

虽然 `ActorVL.generate()` 会把 `use_cache=True` 传给 `model.generate()`，但运行时日志已经明确显示 cache 实际被关闭：

- `/data/LightRFT/examples/math_prm/train_colocate.py:458`
- `/data/LightRFT/examples/math_prm/train_colocate.py:460`
- `/data/LightRFT/rft_logs/lightrft-ursa8b-stage3-psgrpo-2h-stability-v2/node0_20260320_091239.log:6868`
- `/data/LightRFT/rft_logs/lightrft-ursa8b-stage3-psgrpo-2h-stability-v2/node0_20260320_091239.log:6975`

日志原文说明：

- `use_cache=True is incompatible with gradient checkpointing. Setting use_cache=False.`
- `Caching is incompatible with gradient checkpointing in FSDPQwen2DecoderLayer. Setting past_key_values=None.`

这条证据非常关键，因为 autoregressive decode 在没有 KV cache 的情况下会明显变慢。

### 5.6 当前 rollout 的“合理时间”应该是什么量级

按当前脚本配置粗估：

- `rollout_batch_size=32`
- `world_size=8`
- 每个 rank 大约处理 `4` 个 prompt
- `n_samples_per_prompt=4`
- 所以每个 rank 大约会生成 `16` 条 response
- 当前 `LOCAL_HF_GENERATE_MAX_BATCH_SIZE=8`
- 所以每个 rank 大约分成 `2` 个本地 generate chunk

如果按上面的直接 URSA 基线看：

- `batch_size=8` 一次 generate 大约 `14.7s`
- 那每个 rank 生成 16 条 response，大约就是 `29.3s`

再保守加上：

- reward / postprocess / gather 等开销几十秒
- PPO train 约 `8s`

那么**不使用 vLLM/sglang、只走本地 HF** 时，一个 `global step` 的合理量级大致应该是：

- 乐观：`1` 分钟左右
- 保守：`1~3` 分钟
- 即使偏慢：`5` 分钟也已经算比较慢了

而当前真实日志里：

- 第一轮 generate = `802.7s`
- 第二轮 generate = `3895.6s`

因此，当前 rollout 并不是“HF 路径正常偏慢”，而是**明显异常偏慢**。

## 6. 当前约束下，性能问题有没有希望继续改善

有希望，但要分层看。

### 6.1 高概率能改善的是“工作量”

例如：

- 继续收紧 decode budget
- 降低 `n_samples_per_prompt`
- 降低 `generate_max_len`
- 继续优化 stop 条件，让无效尾部更早停住

这类优化改善的是：

- “总要生成多少 token”

而不是：

- “每个 token 本身生成有多快”

所以它们对 bounded run 很有帮助，但不能从根上把 `hf` 路径变成高吞吐引擎。

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

### 6.3 最大收益的方向仍然是外部推理引擎

如果能让 URSA 真正稳定跑在：

- `vllm`
- 或 `sglang`

那么才有可能从根上改善 rollout 吞吐。

这是当前我认为最有希望的长期方向。

## 7. 我对可能解法的排序

按“当前约束下的现实可行性”排序，我会这样看：

### 方案 A：继续把 observation/smoke 的 decode 工作量压小

优点：

- 简单
- 风险低
- 对 bounded run 立刻有效

缺点：

- 治标不治本
- 不会把 `hf` rollout 变成真正高吞吐

### 方案 B：专门 profile HF rollout，找二级热点

优点：

- 能把问题看得更清楚
- 便于决定值不值得继续在 `hf` 路径上深挖

缺点：

- 本身不直接提速
- 需要额外分析轮次

### 方案 C：在 HF 路径里继续做深层工程优化

优点：

- 如果命中，可能有明显收益

缺点：

- 改动复杂
- 风险高
- 可能破坏现有训练链

### 方案 D：重新打通 vLLM/SGLang rollout

优点：

- 理论收益最大
- 最接近真正的高吞吐推理方案

缺点：

- 当前被环境和 URSA 架构支持卡住
- 成本最高

## 7.1 下一步准备做的最小化对照矩阵

为了把“到底是哪一层慢”彻底拆清楚，后续优先做下面这几组对照。这里先把位子留出来，结果后补：

### 对照 A：raw URSA，单卡，`eval + gc off`

- 目的：
  - 建立最干净的单独推理基线
- 当前状态：
  - 已部分完成，见上面的直接基线
- 待补字段：
  - `batch_size=8` 重复测 `N` 次后的均值 / 方差
  - `batch_size=16` 重复测 `N` 次后的均值 / 方差
  - `tokens/s`

### 对照 B：raw URSA，单卡，`train + gc off`

- 目的：
  - 分离 “train mode 本身” 和 “gradient checkpointing” 的影响
- 当前状态：
  - 待补
- 待补字段：
  - `batch_size=8` 时间
  - `tokens/s`
  - 与对照 A 的倍率

### 对照 C：raw URSA，单卡，`train + gc on`

- 目的：
  - 量化 `gradient_checkpointing -> cache 失效` 带来的直接损耗
- 当前状态：
  - 已做过小批控制实验，确认会明显变慢
  - 但还缺和当前 Stage 3 workload 更接近的稳定基线
- 待补字段：
  - `batch_size=8` 时间
  - `tokens/s`
  - 与对照 A / B 的倍率

### 对照 D：LightRFT actor，单卡或最小分布式，`FSDP + gc off`

- 目的：
  - 单独看 FSDP 分片训练 actor 做 rollout 时的额外损耗
- 当前状态：
  - 待补
- 待补字段：
  - `batch_size=8` 时间
  - `tokens/s`
  - 与 raw URSA 的倍率

### 对照 E：LightRFT actor，单卡或最小分布式，`FSDP + gc on`

- 目的：
  - 复现最接近当前真实 rollout 的最小慢路径
- 当前状态：
  - 待补
- 待补字段：
  - `batch_size=8` 时间
  - `tokens/s`
  - 与当前真实训练日志是否同量级

### 对照 F：真实 rollout 8 卡长跑，首个 `global step`

- 目的：
  - 用真实训练环境验证最终量级
- 当前状态：
  - 已有异常慢日志
- 已知结果：
  - 第一轮 `generate=802.7335s`
  - 第二轮 `generate=3895.6270s`
- 待补字段：
  - 如果后续修复，再记录修复后的首轮 generate 耗时
  - 是否回到分钟级

## 8. 最终判断

基于当前证据，我的判断是：

- 当前性能问题是真实存在的
- 当前 slowdown 并不能用“URSA 本来就慢”来解释
- 当前一个 `global step` 跑到十几分钟乃至一小时，**不正常**
- 如果只是为了继续推进后续 phase，可以先接受当前速度做短时观测
- 但如果目标是“把 Stage 3 训练做成长期高效可重复跑的方案”，这个问题迟早还得处理

更具体地说：

- 短期：先把问题彻底拆清楚，不急着盲改
- 中期：优先做上面的最小化对照矩阵
- 长期：最值得的方向仍然是重新获得 `vllm/sglang` 级别的 rollout 支持

## 9. 建议的下一步

如果后面要继续处理性能，我建议顺序是：

1. 先做上面的最小化对照矩阵，而不是直接猜优化点
2. 先分离 `raw URSA`、`train mode`、`gradient_checkpointing`、`FSDP` 这几个因素
3. 如果确认真正的主杀伤项是 `FSDP + gc + local hf actor reuse`，再决定是否改 rollout 运行上下文
4. 如果 HF 路径没有明显低垂果子，就不要在这里无上限深挖
5. 转而评估重新支持 `vllm/sglang` 的成本与收益

因此，当前最合理的状态是：

- 问题已经基本盘清楚
- 现在还不急着直接改代码
- 先用最小化对照把慢点彻底拆清楚
- 先把它作为“已知性能短板”记录下来，避免之后误判成格式或 reward 问题

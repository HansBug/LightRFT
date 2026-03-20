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
- `2026-03-20` 新增的 4 组最小化对照已经把范围进一步缩小：
  - 单独 `URSA-8B` 并不慢
  - `gradient_checkpointing` 会把 decode 直接拖慢一个数量级
  - `FSDP` 会继续增加额外开销
  - 当前 rollout 的主问题已经基本定位到“训练态 actor 的 decode 形态”，不是模型本体速度
- `2026-03-20` 随后又新增了一个更贴近真实 Stage 3 rollout 的最小测速脚本：
  - `examples/math_prm/tools/probe_rollout_speed_candidates.py`
  - 它用 `8` 卡、每个 rank `16` 条 response、`chunk_size=8` 来模拟当前本地 `hf` rollout 的核心 decode 形态
  - 这轮结果进一步证明：最关键的提速杠杆不是 train/eval mode，而是 rollout 阶段必须去掉 `gradient_checkpointing`

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

## 3.2 四组最小化对照：主慢点已经缩到训练态 decode

为了把“到底是哪一层慢”彻底拆开，我补跑了 4 组和真实 Stage 3 workload 参数尽量对齐的最小化对照。统一条件是：

- 同一批 `math_psgrpo` 多模态样本
- 同一套 Stage 3 system prompt
- `batch_size = 8`
- `max_new_tokens = 1024`
- `temperature = 1.0`
- `top_p = 1.0`
- `repetition_penalty = 1.05`
- `no_repeat_ngram_size = 4`

结果如下：

### 对照 A：raw URSA，单卡，`eval + gc off`

- `time_sec = 14.604`
- `generated_tokens_mean = 144.5`
- `generated_tokens_max = 308`
- `tokens_per_sec = 79.155`
- `peak_mem_gb = 24.77`

### 对照 B：raw URSA，单卡，`train + gc on`

- `time_sec = 306.014`
- `generated_tokens_mean = 154.5`
- `generated_tokens_max = 328`
- `tokens_per_sec = 4.039`
- `peak_mem_gb = 24.80`

说明：

- 仅仅把单独 URSA 切到 `train + gradient_checkpointing`，就会从 `14.6s` 直接掉到 `306.0s`
- slowdown 大约是 `20.95x`
- 这一步本身已经足以说明：`gradient_checkpointing -> cache 失效` 是主杀伤项之一

### 对照 C：LightRFT actor，8 卡最小分布式，`FSDP + gc off`

- `time_sec = 32.977`
- `generated_tokens_mean = 110.375`
- `generated_tokens_max = 233`
- `tokens_per_sec = 26.776`
- `peak_mem_gb = 15.39`

说明：

- 单独 `FSDP` 也会变慢，但量级远小于 `gc`
- 相比最干净的 raw baseline，slowdown 大约是 `2.26x`

### 对照 D：LightRFT actor，8 卡最小分布式，`FSDP + gc on`

- `time_sec = 254.804`
- `generated_tokens_mean = 116.75`
- `generated_tokens_max = 261`
- `tokens_per_sec = 3.666`
- `peak_mem_gb = 15.42`

说明：

- 这是最接近当前真实 rollout 的最小可复现实验
- 结果已经直接落到了 `4` 分多钟量级
- 吞吐只有 `3.666 tok/s`
- 这和真实训练里“第一个 global step generate 直接十几分钟”的方向已经一致

### 这 4 组对照给出的结论

- 单独 `URSA-8B` 本身速度正常，十几秒量级
- `gradient_checkpointing` 是第一主因
- `FSDP` 是第二主因
- 两者叠加后，已经能把最小 rollout 近似场景拖到几分钟量级
- 真实训练里更夸张的 `802.7s / 3895.6s`，更像是在这个错误慢基线上又叠加了 batch 组织、同步和 straggler 问题

因此，现在可以比之前更明确地说：

- rollout 主问题已经基本缩到“训练态 actor 的 decode 形态”
- 不是 URSA 模型本身慢
- 不是 PPO train 慢
- 也不是 reward model 慢

## 3.2.1 新增 probe 脚本：更贴近真实 rollout 的 16-response/rank 近似实验

为了避免只停留在“小 batch 控制实验”，我新增了：

- `/data/LightRFT/examples/math_prm/tools/probe_rollout_speed_candidates.py`

这个脚本不是训练脚本，也不是库代码修改；它的职责是：

- 不修改现有 `lightrft/` 主链
- 直接复用当前 URSA 运行时和 LightRFT actor 包装
- 用更接近真实 rollout 的工作量去测：
  - `8` 卡
  - 每个 rank `16` 条 response
  - `chunk_size=8`
  - `max_new_tokens=1024`
- 把几种最关键的运行形态并排对比：
  - `fsdp_train_gc`
  - `fsdp_train_no_gc`
  - `fsdp_eval_no_gc`
  - `raw_eval_no_gc`

也就是说，这个脚本的目的不是“再写一套训练代码”，而是：

- 用最小代价回答“如果不改库，只改变 rollout 运行形态，速度能拉回多少”

### 这轮 probe 的实测结果

结果文件在：

- `/data/LightRFT/tmp/ursa_stage3/rollout_speed_probe/fsdp_train_gc.json`
- `/data/LightRFT/tmp/ursa_stage3/rollout_speed_probe/fsdp_train_no_gc.json`
- `/data/LightRFT/tmp/ursa_stage3/rollout_speed_probe/fsdp_eval_no_gc.json`
- `/data/LightRFT/tmp/ursa_stage3/rollout_speed_probe/raw_eval_no_gc.json`

按“更像真实 rollout”的口径，结果是：

- `fsdp_train_gc`
  - `time_sec = 683.406`
  - `generated_tokens_mean = 154.625`
  - `tokens_per_sec = 3.62`
- `fsdp_train_no_gc`
  - `time_sec = 68.869`
  - `generated_tokens_mean = 142.562`
  - `tokens_per_sec = 33.121`
- `fsdp_eval_no_gc`
  - `time_sec = 65.816`
  - `generated_tokens_mean = 142.562`
  - `tokens_per_sec = 34.657`
- `raw_eval_no_gc`
  - `time_sec = 44.139`
  - `generated_tokens_mean = 150.625`
  - `tokens_per_sec = 54.6`

### 这轮 probe 新增说明了什么

- `fsdp_train_gc = 683.406s` 已经非常贴近真实 Phase 7 里 `661.9063s` / `802.7335s` 的慢法
- 只要把 `gradient_checkpointing` 从 rollout decode 阶段拿掉，`FSDP` actor 也能从 `683.406s` 直接掉到 `68.869s`
- 这个改善接近 `10x`
- `train` 和 `eval` 本身差异很小：
  - `68.869s -> 65.816s`
  - 说明它们不是主矛盾
- 如果进一步改成“每卡完整 URSA 推理副本”，还能从 `65.816s` 再降到 `44.139s`
  - 但这部分收益只有约 `1.49x`

因此，这轮 probe 把优先级进一步压实成了：

1. rollout 阶段先去掉 `gradient_checkpointing`
2. 再决定是否值得为了额外 `1.5x` 左右收益去上更重的“独立推理副本”方案

## 3.3 小批对照：训练态 + gradient checkpointing 确实会变慢，但仍不该慢到小时级

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
  - 已完成
- 已知结果：
  - `batch_size=8`: `14.604s`
  - `tokens/s=79.155`

### 对照 B：raw URSA，单卡，`train + gc off`

- 目的：
  - 分离 “train mode 本身” 和 “gradient checkpointing” 的影响
- 当前状态：
  - 仍待补
- 待补字段：
  - `batch_size=8` 时间
  - `tokens/s`
  - 与对照 A 的倍率

### 对照 C：raw URSA，单卡，`train + gc on`

- 目的：
  - 量化 `gradient_checkpointing -> cache 失效` 带来的直接损耗
- 当前状态：
  - 已完成
- 已知结果：
  - `batch_size=8`: `306.014s`
  - `tokens/s=4.039`
  - 与对照 A 相比约 `20.95x` slowdown

### 对照 D：LightRFT actor，8 卡最小分布式，`FSDP + gc off`

- 目的：
  - 单独看 FSDP 分片训练 actor 做 rollout 时的额外损耗
- 当前状态：
  - 已完成
- 已知结果：
  - `batch_size=8`: `32.977s`
  - `tokens/s=26.776`
  - 与对照 A 相比约 `2.26x` slowdown

### 对照 E：LightRFT actor，8 卡最小分布式，`FSDP + gc on`

- 目的：
  - 复现最接近当前真实 rollout 的最小慢路径
- 当前状态：
  - 已完成
- 已知结果：
  - `batch_size=8`: `254.804s`
  - `tokens/s=3.666`
  - 已经落入“几分钟级 rollout”区间

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

## 7.2 当前更值得推进的架构方向：独立本地 HF rollout actor

在当前证据下，后续如果真的要动代码，最值得优先尝试的方向，不是继续让训练侧 actor 直接承担本地 `hf` rollout，而是把 rollout actor 从训练 actor 中解耦出来。

核心思路：

- 保留现有训练 actor，继续负责：
  - actor logprob 计算
  - backward / optimizer step
  - PPO 训练态参数更新
- 新增一个专门用于 rollout generate 的 `rollout_actor`
  - 不挂 optimizer
  - 明确设为 `eval()`
  - 明确关闭 `gradient_checkpointing`
  - 只作为本地 `hf` inference engine 使用

这样做的目标不是“彻底摆脱所有 FSDP 开销”，而是先切断当前最伤的耦合：

- rollout 不再直接复用训练态 actor
- rollout decode 不再被训练态 `gradient_checkpointing` 拖着跑
- rollout 的模型生命周期可以单独做 `wakeup/sleep/offload`

### 为什么这条路可行

目前 LightRFT 的 rollout 架构里：

- `vllm` / `sglang` 本来就是“训练 actor”和“rollout engine”分离
- `hf` 只是一个例外，它当前直接把 `inference_engine = actor`
- `update_engine_weights()` 也正因为这个实现而对 `hf` 直接 `skip`

因此从框架设计上看：

- “训练 actor”和“rollout inference object”分离，并不是新范式
- 恰恰相反，这是当前架构的常规形态
- 本地 `hf` 现在只是还没有补上这一层抽象

此外，当前经验构建流水线里：

- rollout generate
- actor/reference/critic/reward 的后续前向

是串行阶段，不是同时执行的。也就是说，独立 rollout actor 并不意味着一定要把“两份 actor”长期同时常驻在 GPU 上；在 FSDP 路径下，理论上可以在 generate 结束后立刻把 rollout actor offload，再把训练 actor / reference / reward 这条链路拉回来。

### 三种实现形态的取舍

#### 方案 A：全量 raw rollout 副本

做法：

- 新建一个完全独立的、纯推理用途的 raw HF actor
- 不做 FSDP 包装
- 直接拿它做 `generate()`

优点：

- 推理速度上限最高
- 最接近 `raw_eval_no_gc` 基线

缺点：

- 显存压力最大
- 权重同步成本最高
- 很可能需要把训练 actor 或其他模型 aggressively offload，才能在 8 卡实跑中站住

当前判断：

- 这是长期值得评估的形态
- 但不适合作为第一版实现

#### 方案 B：单独 FSDP rollout actor

做法：

- 新建一个独立的 rollout actor
- 与训练 actor 分离
- rollout actor 走 FSDP 包装，但明确 `gc off + eval`

优点：

- 比 raw full replica 更容易过显存账
- 已有 probe 结果说明，只要去掉 `gc`，性能就能比当前真实慢路径好一个数量级
- 更容易复用当前 FSDP 的 `offload_model()` / `reload_model()` 能力

缺点：

- 仍会保留一部分 FSDP decode 开销
- 速度上限不如 raw replica

当前判断：

- 这是最值得优先落地的第一版

#### 方案 C：独立 rollout worker / process

做法：

- 本地 `hf` 不再只是在 trainer 进程里多一个模型对象
- 而是专门起一个本地 rollout worker，训练侧只做权重同步和 RPC/进程间调用

优点：

- 架构上最接近真正的 engine 化方案
- 训练和 rollout 生命周期最彻底地隔离

缺点：

- 工程量明显更大
- 已经接近“做一个 mini local engine”

当前判断：

- 不适合现在第一步就做

## 7.3 推荐的第一版实施计划

### Phase A：先做 FSDP-first 的独立 rollout actor

第一版推荐目标：

- 新增一个本地 `hf` 分支形态，例如：
  - `engine_type=hf_separate`
  - 或 `engine_type=hf` + `hf_rollout_mode=separate`
- 在 `train_colocate.py` 中额外创建 `rollout_actor`
- `rollout_actor` 明确：
  - 不挂 optimizer
  - `gradient_checkpointing_disable()`
  - `eval()`
  - 仅用于 rollout generate

为什么先做这版：

- 它已经能切断当前“训练态 actor 直接 rollout”的耦合
- 它有望在不引入 raw full replica 显存风险的前提下，先吃掉当前最大头的性能问题

### Phase B：权重同步先追求稳，不先追求最激进的推理形态

第一版权重同步的推荐原则：

- 优先让 `rollout_actor` 与训练 actor 使用尽量一致的 FSDP 包装和 shard 布局
- 这样做的主要目的不是极限性能，而是让同步路径更简单

原因：

- 如果两边的包装和参数布局一致，第一版更有机会直接做 rank-local shard copy
- 比起先把训练 actor gather 成 full state 再灌给 rollout actor，这样更省同步成本，也更不容易再制造新的内存尖峰

换句话说，第一版更合理的目标是：

- **先把“共享训练 actor”改成“独立 rollout actor”**
- **再把 rollout actor 上的 `gc` 去掉**
- **不要在第一步就同时引入 full raw replica + full-state broadcast 这种最激进方案**

### Phase C：把 rollout actor 纳入现有 sleep/wakeup 生命周期

当前 `vllm` / `sglang` 已经有：

- `setup_inference_engine()`
- `wakeup_inference_engine()`
- `maybe_sleep_inference_engine()`
- `update_engine_weights()`

第一版独立 `hf` rollout actor 也应该沿用这套生命周期，而不是额外再造一套平行逻辑。

推荐的行为应该是：

- rollout 前：
  - wakeup / reload `rollout_actor`
  - 如有必要，先 offload 训练 actor，给 rollout 阶段让显存
- rollout 后：
  - offload / sleep `rollout_actor`
  - reload 训练 actor
  - 再进入 actor logprob / initial / critic / reward 这些后续阶段

这里有一个现实约束：

- 当前 `offload_model()` / `reload_model()` 现成实现只在 FSDP 策略里有
- DeepSpeed 路径目前没有同等基建

所以合理顺序应当是：

- 先在当前实际使用的 FSDP 路径上验证
- DeepSpeed 兼容放到后续

### Phase D：验证标准

第一版是否值得继续推进，不看“代码写得漂不漂亮”，只看这几条：

1. 首轮 `global step` 的 `rollout engine generation time`
   - 是否显著低于当前 `661.9s ~ 802.7s` 量级
2. 更贴近真实 workload 的 probe
   - 是否至少接近 `fsdp_eval_no_gc` / `fsdp_train_no_gc` 这一档
   - 而不再停留在 `fsdp_train_gc` 这一档
3. 格式稳定性
   - 不能为了提速重新引入 Phase 7 早期那类输出收集 bug
4. 显存
   - 不能因为多一份 rollout actor 而把整条链路推回 OOM

### Phase E：如果第一版仍然不够快，再决定是否上更激进方案

如果独立 FSDP rollout actor 已经把 generate 从数百秒拉回到几十秒或一两分钟，那么这条路就成立，后续可以再决定是否继续优化。

只有在下面这种情况下，才值得继续往更激进方案推进：

- 独立 rollout actor 已经实现
- `gc` 已经确定不再进入 rollout
- 但 rollout 仍显著慢于 `fsdp_eval_no_gc` 预期

这时再考虑：

- raw full replica
- 本地 rollout worker
- 重新评估 `vllm/sglang` 接入

## 8. 最终判断

基于当前证据，我的判断是：

- 当前性能问题是真实存在的
- 当前 slowdown 并不能用“URSA 本来就慢”来解释
- 当前一个 `global step` 跑到十几分钟乃至一小时，**不正常**
- 当前第一主因已经基本确定为：
  - `gradient_checkpointing` 让 decode 路径失去 KV cache
- 当前第二主因是：
  - 训练态 FSDP actor 直接承担 rollout generate
- 当前最值得优先尝试的工程方案是：
  - 新增独立本地 `hf` rollout actor，而不是继续复用训练 actor
- 如果只是为了继续推进后续 phase，可以先接受当前速度做短时观测
- 但如果目标是“把 Stage 3 训练做成长期高效可重复跑的方案”，这个问题迟早还得处理

更具体地说：

- 短期：先把问题彻底拆清楚，不急着盲改
- 中期：优先做上面的最小化对照矩阵
- 长期：最值得的方向仍然是重新获得 `vllm/sglang` 级别的 rollout 支持

## 9. 建议的下一步

如果后面要继续处理性能，我建议顺序是：

1. 先接受“问题已经定位到 rollout 架构，而不是 reward / 格式 / PPO”这个事实
2. 优先做上面 7.3 中的 FSDP-first 独立 rollout actor 方案，而不是继续在共享训练 actor 上做零碎修补
3. 实现后先用 probe 和首轮真实 `global step` 验证是否回到 `no_gc` 量级
4. 如果第一版仍然不够快，再决定是否升级到 raw full replica 或更彻底的 worker/engine 方案
5. 如果本地 `hf` 的收益已经触顶，再转而评估重新支持 `vllm/sglang` 的成本与收益

因此，当前最合理的状态是：

- 问题已经基本盘清楚
- 下一步可以开始改代码，但应当优先改 rollout 架构，而不是继续猜测单点参数
- 第一版应以“独立 rollout actor + gc off + 可 offload”作为目标
- 继续把它作为“已知性能短板”记录下来，避免之后误判成格式或 reward 问题

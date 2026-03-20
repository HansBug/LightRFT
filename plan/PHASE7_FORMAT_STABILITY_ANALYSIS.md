# Phase 7 格式稳定性问题排查与修复记录

本文档记录 `Phase 7` 从“格式稳定性未通过”到“真实观测重新健康通过”的完整排查过程。结论先说：

- 这不是 `URSA` 模型本身天然会输出乱码
- 真正根因在 `LightRFT` 本地 `hf` rollout 的多模态批量输出收集逻辑
- `math_psgrpo` 早期还漏掉了 `structured_answer_stop`
- 中间用过的 `per-sample fallback` 只能止血，不能作为最终方案，因为会把 8 卡观测拖到 timeout
- 最终修复后，`Phase 7` 最新观测已经恢复到 `healthy_pass = true`
- 但这不等于推理性能已经优秀；当前 8 卡 bounded run 仍然存在明显的 rollout 速度偏慢问题

## 1. 历史现象

最早暴露问题的真实观测是：

- 训练日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_205242.log`
- 观测摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_205242.json`
- 轨迹文件：
  - `/data/LightRFT/results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_205243/trajectories/trajectories_step_1.json`

这轮结论是：

- `healthy_pass = false`
- `format_success_ratio = 0.75`

真实样本里出现过这些问题：

- `StepStep 1:`
- `†Answer:` 后继续拖 `7777...`
- 重复第二个 `†Answer:`
- 明明已经给出 final answer，还继续往后生成尾巴

也就是说，当时不是“训练跑不起来”，而是“训练能跑，但输出收尾不稳定”。

## 2. 为什么怀疑不是 URSA 本体问题

我用和 `Phase 7` 完全相同的真实样本，在当前仓库兼容过的 URSA runtime 上单独跑过纯推理，对照过：

- `greedy_helpful`
- `greedy_stage3`
- `sample_stage3`
- 直接 `model.generate()`

观察到的现象是：

- 同一张图、同一条问题，在纯 `URSA` 推理下不会自然复现 `7777...`
- 也不会自然复现第二个 `†Answer:`
- 输出通常能干净停在 `†Answer: ...`

因此，问题更像是：

- `LightRFT` 侧 rollout / stopping / 输出收集链路的问题
- 而不是 `URSA` checkpoint 本身“天生坏掉”

## 3. 第一层问题：`math_psgrpo` 没有吃到 structured stop

早期排查时发现：

- `lightrft/utils/math_prm_output.py` 里 `MATH_PRM_STRUCTURED_LABELS` 只有
  - `math_prm`
  - `math_prm_combined`
- 但 `Phase 7` 实际跑的是
  - `math_psgrpo`

这意味着当时真实训练里：

- `structured_answer_stop` 实际没有启用
- `math_prm_postprocess` 的结构化清理也没有启用

这会直接放大：

- `†Answer:` 后继续生成
- 重复 answer marker
- 尾巴污染

这部分后来已修复：

- `math_psgrpo` 已加入 structured label 集合

## 4. 第二层问题：中间的 per-sample fallback 只是止血，不是根治

为了先验证问题是否来自 batched 多模态 generate，我一度在 `StrategyBase.engine_generate_local(...)` 里给 URSA 多模态 batched prompt 做了 `per-sample fallback`。

这个方案的效果是：

- 最小 `hf` rollout 校验能恢复正常输出
- `Step 1: Observe` 这类异常截断不再出现

但它有一个明显副作用：

- 8 卡 `Phase 7` 真实观测会变得非常慢

对应的失败 run 是：

- 日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_231136.log`
- 摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_231136.json`

这轮的结果是：

- `healthy_pass = false`
- `issues = ["本次 Phase 7 观测没有形成有效训练样本，trajectory 和训练进度指标都为空"]`

也就是说：

- fallback 修住了格式异常
- 但把真实 8 卡 observation 拖到了 20 分钟 timeout，不能作为最终实现

## 5. 最终根因：左 padding 批次里按统一 prompt 长度切输出，导致短样本前缀被截掉

最终定位到的真正根因在：

- `/data/LightRFT/lightrft/strategy/strategy_base.py`

旧逻辑大意是：

```python
output_start_idx = padded_input_ids.size(1)
total_length = int(attention_mask_out[idx].sum().item())
output_token_ids = sequences[idx, output_start_idx:total_length].tolist()
```

这里有一个关键问题：

- `padded_input_ids.size(1)` 是整个 batch 里统一的最大 prompt 长度
- 但 `attention_mask_out[idx].sum()` 统计的是“该行真实 prompt token 数 + 生成 token 数”
- 对于左 padding 后较短的那一行，`total_length` 会天然比 `output_start_idx + 真实生成长度` 小一截

结果就是：

- 短 prompt 行最前面的若干生成 token 被直接裁掉
- 于是外部看到的输出会像是只剩：
  - `Step 1: Observe`
  - 或者 answer 行前半截消失

这解释了两个之前看起来很奇怪的现象：

1. 为什么 pure URSA 单独生成没问题，但 LightRFT batched rollout 会异常截断
2. 为什么 `per-sample fallback` 能“修好”问题
   - 因为 batch size 退化成 1 以后，不再存在“统一最大 prompt 长度”和“每行真实 prompt 长度”之间的差值

所以真正的问题不是：

- batched generate 本身坏了

而是：

- batched generate 的**输出切片逻辑错了**

## 6. 最终修复

最终落地的修复分成四块：

1. `math_psgrpo` 纳入 structured label
   - 文件：`/data/LightRFT/lightrft/utils/math_prm_output.py`

2. `structured_answer_stop` 支持短代数答案
   - 例如 `†Answer: y = 41`
   - 文件：`/data/LightRFT/lightrft/utils/math_prm_output.py`

3. 修正本地 `hf` rollout 的输出切片
   - 先记录每行真实 `prompt_length`
   - 再按 `generated_length = total_length - prompt_length` 计算真实生成长度
   - 用 `output_start_idx : output_start_idx + generated_length` 回收 token
   - 文件：`/data/LightRFT/lightrft/strategy/strategy_base.py`

4. 删除中间用于止血的 `per-sample fallback`
   - 恢复真正的 batched `hf` rollout
   - 文件：`/data/LightRFT/lightrft/strategy/strategy_base.py`

同时补了回归验证：

- `/data/LightRFT/examples/math_prm/tools/test_phase2_alignment.py`
- `/data/LightRFT/examples/math_prm/tools/check_hf_rollout.py`

## 7. 修复后的验证结果

### 7.1 最小 HF rollout 校验

命令：

```bash
python examples/math_prm/tools/check_hf_rollout.py \
  --max-new-tokens 1024 \
  --output-json /data/LightRFT/tmp/ursa_stage3/hf_rollout_check_1024_batch_fix_v2.json
```

结果文件：

- `/data/LightRFT/tmp/ursa_stage3/hf_rollout_check_1024_batch_fix_v2.json`

关键结论：

- `success = true`
- `strict_structure_success = true`
- `all_match_direct_generate = true`
- `all_below_length_cap = true`

这说明：

- LightRFT 本地 `hf` rollout 已重新回到 batched 路径
- batched rollout 和 direct batched `actor.generate()` 已经逐 token 对齐
- 之前的多模态批量截断问题已经不再存在

### 7.2 真实 Phase 7 重新观测

最新成功 run：

- 日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_233851.log`
- 摘要：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_summary_20260319_233851.json`
- 轨迹：
  - `/data/LightRFT/results/lightrft-ursa8b-stage3-phase7-observation/lightrft-ursa8b-stage3-phase7-observation-ep1-kl0.003-lr2e-6-20260319_233851/trajectories/trajectories_step_1.json`

关键结果：

- `healthy_pass = true`
- `format_success_ratio = 1.0`
- `num_trajectories = 4`
- `correctness_ratio = 0.25`
- `drop_moment_ratio = 1.0`
- `answer_extraction_failure_ratio = 0.0`
- `prm_inference_failure_ratio = 0.0`
- `mean_abs_delta = 0.815216064453125`

训练日志里也已经重新出现：

- `step 0 generate length`
- `math_prm_postprocess`
- PPO train 进度
- trajectory 落盘
- checkpoint 落盘

并且这轮没有再发生：

- 空跑 20 分钟后没有样本
- `Step 1: Observe` 异常截断
- `format_success_ratio = 0.75`

## 8. 当前结论

截至最新 `20260319_233851` 这轮观测，可以给出比较明确的结论：

- `Phase 7` 之前的“格式稳定性未通过”，本质上是一个 rollout 集成 bug
- 这个 bug 已经被修复
- 当前 `Phase 7` 已重新回到 `healthy_pass = true`
- 当前“超级无敌满”的长度问题也已经不再是主问题
- 现在剩下的主要问题不再是格式稳定性，而是训练质量本身
  - 例如 `correctness_ratio` 仍然只有 `0.25`
- 同时还存在一个独立但真实的性能问题：
  - 最新日志里的 `rollout engine generation time (global max) = 661.9063s`
  - 整轮 `Episode [1/1]` 大约 `11分57秒`
  - 说明当前本地 `hf` 多模态 rollout 已经“能跑完”，但还远不能算快

## 9. 当前仍然存在的推理性能问题

最新健康通过 run 的性能现状如下：

- 日志：
  - `/data/LightRFT/tmp/ursa_stage3/phase7_observation/phase7_observation_20260319_233851.log`
- 关键信息：
  - `Start VLM gather_and_generate ..., total prompts: 4` 出现在 `23:41:28`
  - `step 0 generate length ...` 出现在 `23:52:30`
  - 也就是 rollout 生成本身大约用了 `11` 分钟
  - 日志里明确记录：`***Rollout engine generation time (global max): 661.9063s`

这说明：

- 当前真正阻塞 `Phase 7` 的已经不再是格式错误
- 而是本地 `hf` 多模态 batched rollout 的吞吐仍然偏低

这个问题和前面的格式 bug 不是一回事：

- 格式 bug 已经修复
- 但性能问题仍然存在

也就是说，当前状态应该理解成：

- 正确性层面：格式稳定性已经恢复
- 性能层面：推理依然偏慢，只是还没有慢到再次阻塞 bounded run

因此，后续工作的主线应该切换成：

- 不再继续为“为什么会 `StepStep` / `7777...` / 截断”投入排查成本
- 继续推进 reward、数据和训练质量本身的提升

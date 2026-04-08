# PR53 Phase 2: 为什么 `eval/model_reward` 高于 `eval/outcome_correct`，以及 W&B 指标到底怎么来的

本文专门回答 [PR53_FOLLOWUP_CHECKLIST.md](./PR53_FOLLOWUP_CHECKLIST.md) 里的 Phase 2 问题：

> 为什么 `eval/model_reward` 显著高于 `eval/outcome_correct` 各自的含义是什么？理清楚 wandb 中各项指标的具体实现方法

这份文档不再停留在“看曲线猜原因”的层面，而是沿着当前仓库里真实启用的 Stage 3 路径，把下面几件事逐段拆开：

1. runtime eval 到底是怎么跑的，它和训练 rollout 有什么不同。
2. `eval/*`、`rollout/*`、`train/*` 这些 W&B 指标各自从哪段代码来。
3. `eval/model_reward`、`eval/outcome_correct`、`eval/reward` 三者到底是不是同一个东西。
4. 为什么当前实验里 `eval/model_reward > eval/outcome_correct` 是可能正常的，以及它和 URSA 论文里的 PS-GRPO 设计是什么关系。

本文聚焦的默认运行路径是：

- experiment: `exps/2026-04-01-default-real-online-hansbug-eval5`
- trainer: `examples/math_prm/math_prm_trainer.py::MathPRMSPMDPPOTrainerVL`
- reward label: `math_psgrpo`
- reward model: 单个 `URSA-RM-8B`
- train advantage estimator: `group_norm`
- runtime eval setting: `eval_n_samples_per_prompt=1`，当前实验 `eval_temperature=0.0`

## 太长不看

`eval/model_reward` 是 URSA-RM 的连续 proxy 分，`eval/outcome_correct` 是最终答案答对率，`eval/reward` 是 PS-GRPO 奖励均值：错=0，答对但有 drop=0.5，答对且无 drop=1。所以可能出现 `model_reward=0.97` 但 `outcome_correct=0,reward=0`，也可能 `outcome_correct=1` 但 `reward=0.5`。W&B 的 `eval/*` 只是对这些样本值直接求均值。

## 一页结论

- 当前 runtime eval 不是 paper 里的 BoN 评估，也不是 train rollout 的重放；它是 **每个 prompt 只生成 1 条回答** 的单样本评估。
- `eval/model_reward` 和 `eval/outcome_correct` 不在同一个语义空间里：
  - 前者是 PRM 的连续代理分。
  - 后者是最终答案对错的二值指标。
- 默认 `math_psgrpo` 路径下，`eval/reward` 看的是 `final_reward`，不是 `model_reward`。
- 因为 `final_reward <= outcome_correct` 是逐样本成立的，所以在相同 eval 样本集合上，**`eval/reward` 理应不高于 `eval/outcome_correct`**。
- `eval/model_reward > eval/outcome_correct` 本身不能直接判成 bug。它首先是一个 **proxy gap 候选信号**，而这恰恰是 URSA 论文提出 PS-GRPO 的原因之一。

## 1. 这句话到底在问什么

Phase 2 真正在问的，不是“为什么一个数大、一个数小”这么简单，而是四个层次的问题：

1. W&B 上每个指标是从哪段代码里打出来的。
2. 它们的统计口径到底是什么，是单 batch、整次 rollout，还是 eval 全集。
3. `model_reward`、`outcome_correct`、`reward` 这三个词在当前实现里是不是同义词。
4. 当前实验里看到的偏离，究竟是正常的 proxy gap，还是实现 bug。

如果这四件事不拆开，后面很容易出现三种误判：

- 把 `model_reward` 当成训练 reward。
- 把 `eval/reward` 当成“答对率”。
- 把 `rollout/*`、`train/*`、`eval/*` 三个命名空间里同名指标当成同一口径。

## 2. runtime eval 实际是怎么跑的

当前 runtime eval 不是训练时那套 `n_samples_per_prompt=8 + group_norm` 的路径，而是专门切进一个“单样本评估上下文”。

### 摘录：`examples/math_prm/math_prm_trainer.py`

```python
def _runtime_eval_context(self):
    original_generate_kwargs = self.generate_kwargs
    original_n_samples = self.strategy.args.n_samples_per_prompt
    original_advantage_estimator = self.strategy.args.advantage_estimator
    ...

    self.generate_kwargs = dict(self._eval_generate_kwargs)
    self.strategy.args.n_samples_per_prompt = max(1, int(getattr(self.strategy.args, "eval_n_samples_per_prompt", 1)))
    self.strategy.args.advantage_estimator = "reinforce"
    ...
    self.strategy.config.n_samples_per_prompt = self.strategy.args.n_samples_per_prompt
    self.strategy.config.advantage_estimator = "reinforce"
```

这段代码说明 runtime eval 有三个关键特点：

- `n_samples_per_prompt` 被强制切成 `1`。
- `advantage_estimator` 被临时切成 `reinforce`。
- `generate_kwargs` 被替换成 eval 专用解码参数。

当前实验 README 里也明确写了评估配置：

```text
- 评估配置：`eval_steps=5`，`max_eval_samples=500`，`eval_holdout_size=500`，`eval_temperature=0.0`
```

所以这次实验里的 runtime eval，本质上是：

- holdout eval set
- 每个题只采 1 条回答
- 当前实验还是 `eval_temperature=0.0`

也就是说，这里评估的是 **单次、近似贪心解码的 actor 表现**，不是：

- 训练 rollout 时 8 条采样中的组内表现
- 用 PRM 做 best-of-n 选择后的上界表现

这点和论文里的 BoN 分析一定要分开。

### 一个经常被忽略但很关键的细节

这里把 `advantage_estimator` 改成 `reinforce`，不是因为 eval 还在做 RL 更新，而是因为 eval 复用了 `experience_maker.make_experience_list(...)` 这条经验构建流水线。单样本 eval 下如果还保留 `group_norm`，组内标准化的语义就不成立了。

但对本问题最重要的是：

- runtime eval 最终记录到 W&B 的那些 `eval/*` 指标，**不是从 advantage 或 loss 算出来的**
- 它们来自 `experience.info["reward"]`、`experience.info["response_length"]` 和 `experience.info["reward_metrics"]`

所以你在 Phase 2 里关心的 `eval/model_reward`、`eval/outcome_correct`、`eval/reward`，语义上 **不依赖** 这个 `reinforce` 切换；它只是让 eval 流水线能在单样本上下文里跑通。

## 3. `eval/*` 指标的真实统计口径

runtime eval 的聚合入口在 `lightrft/trainer/ppo_trainer_vl.py::evaluate()`。

### 摘录：`lightrft/trainer/ppo_trainer_vl.py`

```python
all_rewards = []
reward_metric_values = defaultdict(list)
all_response_lengths = []

def extract_values(val):
    if isinstance(val, torch.Tensor):
        return val.view(-1).cpu().tolist()
    elif isinstance(val, (list, tuple)):
        return list(val)
    else:
        return [float(val)]
...
for i, experience in enumerate(
    self.experience_maker.make_experience_list(
        eval_prompts, eval_images, eval_videos, eval_references, eval_labels, **self.generate_kwargs
    )
):
    ...
    if hasattr(experience, 'info') and experience.info:
        info = experience.info
        if 'reward' in info:
            all_rewards.extend(extract_values(info['reward']))
        if 'response_length' in info:
            all_response_lengths.extend(extract_values(info['response_length']))

        if 'reward_metrics' in info:
            rm = info['reward_metrics']
            for key, value in rm.items():
                reward_metric_values[key].extend(extract_values(value))
...
compute_stats("reward", all_rewards)
for metric_name, values in reward_metric_values.items():
    compute_stats(metric_name, values)
compute_stats("response_length", all_response_lengths)
metrics["num_samples"] = len(all_rewards)
```

这说明 eval 统计口径非常直接：

- 先跑出 eval experiences。
- 对每个 experience，从 `info` 里把 `reward`、`response_length`、`reward_metrics[...]` 取出来。
- 最后对所有样本直接求均值。

几个重要结论：

- **`eval/*` 是样本均值，不是 loss。**
- **`eval/*` 不是看训练 batch，而是看 eval dataloader 跑出来的所有 eval samples。**
- **这里没有 `abs(mean_metric) > 1e-6` 之类的过滤逻辑。**

也就是说：

- 如果 `eval/outcome_correct=0.4028`，它就是这次 runtime eval 样本里 `outcome_correct` 的均值。
- 如果 `eval/model_reward=0.50`，它就是这次 runtime eval 样本里 `model_reward` 的均值。

不是别的。

### 多卡下的聚合方式也不是“mean of means”

`MathPRMSPMDPPOTrainerVL` 还额外做了一层跨 rank 聚合，而且是按 `num_samples` 加权的。

### 摘录：`examples/math_prm/math_prm_trainer.py`

```python
def _aggregate_eval_metrics(self, raw_eval_metrics: Dict[str, float]) -> Dict[str, float]:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return raw_eval_metrics

    gathered_metrics = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(gathered_metrics, raw_eval_metrics or {})

    total_samples = sum(float(metrics.get("num_samples", 0.0)) for metrics in gathered_metrics if metrics)
    if total_samples <= 0:
        return {}

    aggregated_metrics = {"num_samples": total_samples}
    ...
    for key in mean_keys:
        weighted_sum = 0.0
        for metrics in gathered_metrics:
            if not metrics or key not in metrics:
                continue
            weighted_sum += float(metrics["num_samples"]) * float(metrics[key])
        aggregated_metrics[key] = weighted_sum / total_samples
```

所以当前 eval 指标不是“每卡先求均值，再简单平均”，而是：

- 每卡先拿到自己那部分样本均值和样本数
- 再按样本数做 weighted average

这一层实现是合理的，不会凭空制造 `model_reward` 和 `outcome_correct` 的偏离。

## 4. W&B 的名字是怎么映射出来的

当前 `MathPRMSPMDPPOTrainerVL` 对三个命名空间单独做了 key remap。

### 摘录：`examples/math_prm/math_prm_trainer.py`

```python
_ROLLOUT_KEY_SOURCES = {
    "reward": ("rollout_reward", "step_reward_mean", "reward"),
    "reward_std": ("rollout_reward_std", "step_reward_std"),
    "outcome_correct": ("rollout_outcome_correct", "outcome_correct_mean", "reward_metrics/outcome_correct"),
    "has_drop_moment": ("rollout_has_drop_moment", "has_drop_moment_mean", "reward_metrics/has_drop_moment"),
    "model_reward": ("rollout_model_reward", "model_reward_mean", "reward_metrics/model_reward"),
    "response_length": ("rollout_response_length", "response_length_mean", "response_length"),
}

_TRAIN_KEY_SOURCES = {
    "policy_loss": ("policy_loss",),
    "kl": ("kl",),
    "actor_lr": ("actor_lr",),
    ...
    "reward": ("reward",),
    "reward_std": ("step_reward_std",),
    "return": ("return",),
    ...
}

_EVAL_KEY_SOURCES = {
    "reward": ("reward", "reward_mean"),
    "outcome_correct": ("outcome_correct", "outcome_correct_mean"),
    "has_drop_moment": ("has_drop_moment", "has_drop_moment_mean"),
    "model_reward": ("model_reward", "model_reward_mean"),
    "response_length": ("response_length", "response_length_mean"),
    "answer_extraction_failed": ("answer_extraction_failed", "answer_extraction_failed_mean"),
}
```

最终打到 W&B 时，则是统一加命名空间前缀：

### 摘录：`examples/math_prm/math_prm_trainer.py`

```python
for key, value in rollout_metrics.items():
    all_wandb_logs[f"rollout/{key}"] = value
...
for key, value in train_metrics.items():
    all_wandb_logs[f"train/{key}"] = value
...
for key, value in raw_eval_metrics.items():
    eval_logs[f"eval/{key}"] = value

eval_logs["eval/train_step"] = global_step
eval_logs["eval/episode"] = episode
```

所以 W&B 上看到的 `eval/model_reward` 不是底层原始名字，而是：

```text
raw_eval_metrics["model_reward_mean"]
  -> _EVAL_KEY_SOURCES remap 成 "model_reward"
  -> W&B 最终日志键 "eval/model_reward"
```

## 5. `rollout/*`、`train/*`、`eval/*` 为什么不能混着看

虽然名字很像，但这三个命名空间不是同一口径。

| W&B key | 样本来源 | 聚合位置 | 实际含义 |
| --- | --- | --- | --- |
| `rollout/reward` | 训练 prompt 的 rollout 样本 | rollout 后、优化前 | 这一轮收集到的 sequence reward 均值 |
| `rollout/model_reward` | 同上 | rollout 后、优化前 | 这一轮 rollout 上的 PRM 连续 proxy 均值 |
| `rollout/outcome_correct` | 同上 | rollout 后、优化前 | 这一轮 rollout 上的最终答对率 |
| `train/reward` | 当前 learn batch | actor training step 内 | 当前优化 batch 的 `experience.info["reward"]` 均值 |
| `train/kl` | 当前 learn batch | actor training step 内 | 当前优化 batch 的 KL 统计，按 response length 加权 |
| `eval/reward` | eval holdout prompt | runtime eval | eval 样本上的 `final_reward` 均值 |
| `eval/model_reward` | eval holdout prompt | runtime eval | eval 样本上的 PRM 连续 proxy 均值 |
| `eval/outcome_correct` | eval holdout prompt | runtime eval | eval 样本上的最终答对率 |
| `eval/answer_extraction_failed` | eval holdout prompt | runtime eval | eval 样本上的答案抽取失败比例 |

当前训练阶段的 `train/reward` 是在 actor training step 内从 `experience.info` 扁平化出来的。

### 摘录：`lightrft/trainer/ppo_trainer_vl.py`

```python
status = {"policy_loss": actor_loss.item(), "actor_lr": self.actor_scheduler.get_last_lr()[0]}
...
for k, v in experience.info.items():
    if k == "kl":
        if isinstance(v, torch.Tensor):
            weighted_kl = (v *
                           experience.info["response_length"]).sum() / experience.info["response_length"].sum()
            status[k] = weighted_kl.item()
        else:
            status[k] = v
        continue
    ...
    if isinstance(v, torch.Tensor):
        status[k] = v.float().mean().item()
```

所以：

- `train/reward` 是当前 learn batch 的 `experience.info["reward"]` 均值。
- `train/kl` 是当前 learn batch 的 KL，且按 response length 做了加权。

而 `rollout/*` 则是在 replay buffer 上重新聚合出来的：

### 摘录：`lightrft/trainer/ppo_trainer_vl.py`

```python
for item in self.replay_buffer.items:
    if hasattr(item, 'info') and item.info is not None and 'reward' in item.info:
        all_rewards.append(item.info['reward'])
    ...
    if (
        hasattr(item, 'info') and item.info is not None and 'reward_metrics' in item.info
        and item.info['reward_metrics'] is not None
    ):
        reward_metrics = item.info['reward_metrics']
        for key, value in reward_metrics.items():
            reward_metric_values[key].append(value)
...
rollout_status["rollout_reward"] = rewards_tensor.mean().item()
rollout_status["rollout_reward_std"] = rewards_tensor.std().item()
```

这意味着：

- `rollout/reward` 和 `train/reward` 虽然同名为 reward，但统计对象不同。
- `rollout/model_reward` 和 `eval/model_reward` 也不是同一分布，因为前者来自训练 rollout，后者来自 eval holdout。

## 6. 这几个关键指标各自到底是什么意思

Phase 2 最容易混淆的，就是下面五个量：

- `model_reward`
- `outcome_correct`
- `final_reward`
- `answer_extraction_failed`
- `response_length`

### 摘录：`examples/math_prm/reward_models.py`

```python
answer_eval = cls._evaluate_answer_alignment(response, reference)
outcome_correct = float(answer_eval["outcome_correct"])
max_relative_drop, has_drop_moment = cls._compute_relative_drop(step_scores)

final_reward = 0.0
if outcome_correct > 0.0:
    final_reward = 1.0 - cls._DROP_GAMMA if has_drop_moment else 1.0

return {
    "outcome_correct": outcome_correct,
    "accuracy_reward": outcome_correct,
    "max_relative_drop": max_relative_drop,
    "has_drop_moment": float(has_drop_moment),
    "final_reward": final_reward,
    "answer_tag_present": float(answer_eval["answer_tag_present"]),
    "answer_extraction_failed": float(answer_eval["answer_extraction_failed"]),
    ...
}
```

### 摘录：`examples/math_prm/reward_models.py`

```python
if step_scores.numel() == 0:
    aggregated_score = 0.0
elif self.aggregation == "min":
    aggregated_score = float(torch.min(step_scores).item())
elif self.aggregation in {"avg", "mean"}:
    aggregated_score = float(torch.mean(step_scores).item())
elif self.aggregation == "last":
    aggregated_score = float(step_scores[-1].item())

sequence_reward = psgrpo_metrics["final_reward"] if label == "math_psgrpo" else aggregated_score
batch_rewards.append(sequence_reward)
batch_metrics["model_reward"].append(aggregated_score)
```

从这里可以把几个量彻底拆开：

### 6.1 `eval/model_reward`

- 来源：`reward_metrics["model_reward"]`
- 语义：PRM 对 step score 聚合后的 **连续 proxy 分数**
- 当前默认聚合：`min(step_scores)`

它的核心特征是：

- 是连续值，不是 0/1。
- 反映的是“PRM 觉得这条过程像不像好过程”。
- 它默认 **不等于** 最终训练 reward。

### 6.2 `eval/outcome_correct`

- 来源：`reward_metrics["outcome_correct"]`
- 语义：最终答案是否答对
- 本质上是 0/1 指标的均值

它不是 PRM score，而是 final answer extraction + compare 的结果。

答案抽取的失败标记则来自下面这段逻辑：

### 摘录：`examples/math_prm/reward_models.py`

```python
details: Dict[str, Any] = {
    "predicted_answer": "",
    "answer_tag_present": False,
    "answer_extraction_failed": True,
    "used_answer_fallback": False,
    "extraction_source": "missing",
}
...
if "†Answer:" in response:
    ...
    predicted_answer = cls._extract_answer_from_candidate(candidate, reference_type)
    details["predicted_answer"] = predicted_answer
    details["answer_extraction_failed"] = predicted_answer == ""
    details["extraction_source"] = "dagger_answer"
    return details
...
explicit_fallbacks = [
    ("boxed", extract_boxed_answer(response)),
    ("tagged_answer", extract_answer_from_tags(response, "answer")),
]
```

所以：

- `answer_extraction_failed=1` 不是“这题做错了”
- 它更准确地说是“当前答案抽取链路没拿到可比较的最终答案”

### 6.3 `eval/reward`

- 来源：`experience.info["reward"]`
- 对默认 `math_psgrpo` 路径，它等于 `final_reward`
- 取值典型是 `0 / 0.5 / 1`

这一点是 Phase 2 的核心：

- 当前默认训练 label 是 `math_psgrpo`
- 所以 `sequence_reward` 不是 `aggregated_score`
- 而是上面的 `final_reward`

因此：

- `eval/reward` 看的不是 PRM 的连续 proxy 分
- 它看的是 **PS-GRPO 三档奖励**

### 6.4 `eval/answer_extraction_failed`

- 来源：`reward_metrics["answer_extraction_failed"]`
- 语义：答案抽取失败比例

它是用来判断“correctness 低到底是不是主要因为 answer extraction 崩了”的诊断项，而不是过程质量指标。

### 6.5 `eval/response_length`

- 来源：`experience.info["response_length"]`
- 语义：当前 eval response token 数均值

它只是长度统计，不带任何正确性或过程质量含义。

## 7. 为什么 `eval/model_reward` 可以显著高于 `eval/outcome_correct`

现在可以直接回答 Phase 2 最核心的问题了。

### 7.1 从实现上看，这两个量本来就不是同一种东西

- `model_reward` 关注的是 PRM 对过程局部质量的连续判断。
- `outcome_correct` 关注的是最终答案 exact match / grader 结果。

所以某条回答完全可能同时满足：

- 中间步骤写得很像样
- PRM 给出很高的连续分
- 但最后答案就是错

这种情况下，单样本层面就会出现：

```text
model_reward 高
outcome_correct = 0
final_reward = 0
```

### 7.2 真实例子：同一个 scorer 样本里就已经出现了这种情况

下面这个例子来自 Phase 1 文档里同一个 URSA-RM scorer 的真实 8 采样结果：

### 摘录：`PR53-phase1-prm-reward-to-grpo-loss.md`

```text
| sample | step_scores | model_reward(min) | outcome_correct | has_drop_moment | final_reward |
| 1 | `[0.984375, 0.96875, 0.914062]` | 0.914062 | 1 | 0 | 1.0 |
| 2 | `[0.964844, 0.451172, 0.425781]` | 0.425781 | 1 | 1 | 0.5 |
| 3 | `[0.96875, 0.9375, 0.294922]` | 0.294922 | 1 | 1 | 0.5 |
| 4 | `[0.984375, 0.972656, 0.777344]` | 0.777344 | 1 | 0 | 1.0 |
| 5 | `[0.890625, 0.141602]` | 0.141602 | 0 | 1 | 0.0 |
| 6 | `[0.960938, 0.769531]` | 0.769531 | 0 | 0 | 0.0 |
| 7 | `[0.984375, 0.972656]` | 0.972656 | 0 | 0 | 0.0 |
| 8 | `[0.988281]` | 0.988281 | 1 | 0 | 1.0 |
```

其中 sample 7 最典型：

```text
sample 7:
- model_reward(min)=0.972656
- outcome_correct=0
- final_reward=0.0
```

这正是 `eval/model_reward > eval/outcome_correct` 的最小反例：

- PRM 代理分非常高
- 但最终答案仍然错

只要这类样本比例不小，均值上就会出现：

```text
eval/model_reward 明显高于 eval/outcome_correct
```

### 7.3 从论文上看，这种偏离恰恰是 URSA 在 Stage 3 要规避的东西

URSA 论文自己就明确说过，直接把 scalar process reward 当学习目标会出问题。

### 摘录：`~/URSA-MATH/paper.md`

```text
We observe two highly significant conclusions from Figure 4:
(i) High susceptibility to reward hacking.
...
(ii) PRM’s length bias in rewarding.
...
This suggests that although the scalar reward from the PRM in online RL might be unreliable,
the relative quality of solutions it reveals is comparatively trustworthy.
```

附录里又进一步解释了“看起来过程正确，但并不真正帮助最终答案”的失败模式：

### 摘录：`~/URSA-MATH/paper.md`

```text
In online RL, models can easily recognize the patterns for obtaining process rewards,
leading to conservative analyses and concise responses as they sidestep PRM scrutiny.
...
models can easily focus on processes that seem "correct" in isolation.
However, these processes may not be genuinely helpful for the final outcome
and instead may lead the model to prioritize high process rewards over accuracy.
```

这几段话翻译成当前 LightRFT 的语义就是：

- `model_reward` 是有信息量的
- 但它的绝对数值不可靠，不能直接当成“答对概率”
- 所以 `model_reward > outcome_correct` 先验上就是可能发生的

### 7.4 一个最直观的均值例子

假设 100 个 eval 样本里：

- 40 个样本答对，平均 `model_reward=0.80`
- 60 个样本答错，但因为过程局部看起来像样，平均 `model_reward=0.45`

那就会得到：

```text
eval/outcome_correct = 40 / 100 = 0.40
eval/model_reward = (40 * 0.80 + 60 * 0.45) / 100 = 0.59
```

此时 `eval/model_reward` 显著高于 `eval/outcome_correct`，完全不需要任何 logging bug 就能成立。

## 8. 为什么 `eval/reward` 还会低于 `eval/outcome_correct`

这一点在当前实验里也确实发生了。README 里记录的是：

### 摘录：`README.md`

```text
- `eval/outcome_correct`：范围 **0.3794 ~ 0.4127**，最后一次 **0.4028**
- `eval/reward`：范围 **0.3490 ~ 0.3740**，最后一次 **0.3661**
- `eval/model_reward`：后 5 次均值 **0.5021**，高于 `eval/outcome_correct`
```

这不是异常，反而和当前实现完全一致。

### 8.1 因为 `final_reward` 对“答对但有 drop”的样本只给 0.5

### 摘录：`examples/math_prm/reward_models.py`

```python
final_reward = 0.0
if outcome_correct > 0.0:
    final_reward = 1.0 - cls._DROP_GAMMA if has_drop_moment else 1.0
```

而 `_DROP_GAMMA = 0.5`，所以：

- 错误答案 -> `final_reward=0`
- 正确答案但有 drop -> `final_reward=0.5`
- 正确答案且无 drop -> `final_reward=1`

### 8.2 这意味着在 `math_psgrpo` 下，逐样本都有 `final_reward <= outcome_correct`

因为 `outcome_correct` 是 0/1，而 `final_reward` 是：

```text
final_reward =
  0                  if outcome_correct = 0
  0.5 或 1.0         if outcome_correct = 1
```

所以对每个样本都成立：

```text
final_reward <= outcome_correct
```

在整批 eval 样本上取均值以后，仍然成立：

```text
eval/reward <= eval/outcome_correct
```

更具体一点，如果把“答对且无 drop”的比例记为 `p_good`，把“答对但有 drop”的比例记为 `p_drop`，那么：

```text
eval/outcome_correct = p_good + p_drop
eval/reward = p_good + 0.5 * p_drop
            = eval/outcome_correct - 0.5 * p_drop
```

所以：

- 只要存在一部分“答对但有 drop”的样本
- `eval/reward` 就会严格低于 `eval/outcome_correct`

### 8.3 真实例子：答对了也可能只拿 0.5

还是看 Phase 1 里的真实 scorer 结果，sample 2 和 sample 3：

### 摘录：`PR53-phase1-prm-reward-to-grpo-loss.md`

```text
| sample | step_scores | model_reward(min) | outcome_correct | has_drop_moment | final_reward |
| 2 | `[0.964844, 0.451172, 0.425781]` | 0.425781 | 1 | 1 | 0.5 |
| 3 | `[0.96875, 0.9375, 0.294922]` | 0.294922 | 1 | 1 | 0.5 |
```

这说明：

- `outcome_correct=1` 不代表 `reward=1`
- 当前实现里，只要出现 drop-moment，就会被降成 `0.5`

因此 README 里看到：

```text
eval/reward < eval/outcome_correct
```

恰恰是实现和设计对得上的表现。

## 9. 这和 URSA 论文、以及 `~/URSA-MATH` 的复现资料怎么对上

当前 LightRFT 的 Stage 3 设计，并不是“没想好所以顺手多打了几个指标”，而是和 URSA 对 PS-GRPO 的论证直接相关。

### 9.1 论文为什么不直接优化 scalar process reward

论文在 Section 4 先试了两种“把 scalar process reward 直接塞进 GRPO”的做法：

### 摘录：`~/URSA-MATH/paper.md`

```text
Variant 1: For i-th rollout, the reward is the sum of the outcome reward and the average process reward
Variant 2: ... a scalar process reward is assigned to the i-th rollout’s t-th step.
```

然后得出的结论是：

- 容易 reward hacking
- 有明显 length bias

因此论文转向 PS-GRPO：

- 不再把 `model_reward` 直接当训练目标
- 而是利用 PRM 的 **相对错误识别能力**
- 用 drop-moment 去修正 outcome reward

### 9.2 论文的 PS-GRPO 奖励和当前 LightRFT 实现是一致的

论文公式 6 的口径是：

### 摘录：`~/URSA-MATH/paper.md`

```text
R^i =
1,     if o^i is correct and δ_p^i < ρ
1-γ,   if o^i is correct and δ_p^i ≥ ρ
0,     otherwise
```

而 `~/URSA-MATH/STAGE3_REPRODUCTION_PLAN.md` 里对复现目标的描述也是：

### 摘录：`~/URSA-MATH/STAGE3_REPRODUCTION_PLAN.md`

```text
1. Drop-moment检测
   - 检测PRM奖励序列的显著下降点
   - 公式：`δ_i = max{(r_p,j - r_p,j+1) / r_p,j} > ρ`
   - 阈值：ρ = 0.3

2. 过程监督奖励计算
   - 三级奖励系统：
     - 答案正确 + 无drop-moment → R = 1.0
     - 答案正确 + 有drop-moment → R = 0.5 (γ=0.5)
     - 答案错误 → R = 0.0
```

这和当前仓库里的实现：

```python
final_reward = 1.0 - cls._DROP_GAMMA if has_drop_moment else 1.0
```

是对齐的。

### 9.3 但要注意：`STAGE3_REPRODUCTION_PLAN.md` 里有些表述代表的是“当时的计划状态”，不是当前 `/data/LightRFT` 的现实状态

例如该文档里有一段说：

```text
LightRFT目前只支持vanilla GRPO，需要添加PS-GRPO的核心创新
```

这更像是早期规划文档的口径。对今天这个仓库状态来说：

- `examples/math_prm/reward_models.py` 已经有 `math_psgrpo`
- `final_reward` / `has_drop_moment` / `outcome_correct` / `model_reward` 这些指标都已经接到主线上

所以读 `~/URSA-MATH` 时要分清两层信息：

- `paper.md` 给你的是设计动机和理论边界
- `STAGE3_REPRODUCTION_PLAN.md` 给你的是早期复现设计稿
- 当前 `/data/LightRFT` 代码则是今天真正生效的实现

### 9.4 当前 runtime eval 也不是论文 Figure 5(a) 的 BoN 评估

论文 Figure 5(a) 说明的是：

### 摘录：`~/URSA-MATH/paper.md`

```text
Figure (a) shows the BoN evaluation during GRPO training.
We select the best rollout using the mean value of process rewards.
```

而当前 LightRFT runtime eval 是：

- `n_samples_per_prompt=1`
- 当前实验 `eval_temperature=0.0`
- 不做 best-of-n 选择

所以：

- 当前 W&B 里的 `eval/model_reward` 不是论文 Figure 5(a) 的 BoN 选择分数
- 当前 `eval/outcome_correct` 也不是“挑过最优 rollout 后的 accuracy”

这也是为什么不能直接拿论文里 BoN 曲线和当前 runtime eval 曲线做一一对照。

## 10. W&B 上哪些现象是正常的，哪些更像 bug

把实现、论文和当前实验放到一起以后，边界就比较清楚了。

### 10.1 这些现象在当前实现下是“先验正常”的

- `eval/model_reward > eval/outcome_correct`
  - 正常的 proxy gap 候选现象。
- `eval/reward < eval/outcome_correct`
  - 在 `math_psgrpo` 下本来就应该经常发生。
- `rollout/model_reward` 和 `eval/model_reward` 数值不一致
  - 因为样本集和采样配置本来就不同。
- `train/reward`、`rollout/reward`、`eval/reward` 三者不相等
  - 因为它们不是同一批样本，也不是同一统计阶段。

### 10.2 这些现象才更值得优先怀疑实现或日志问题

- **如果同一套 `math_psgrpo` eval 样本上出现 `eval/reward > eval/outcome_correct`**
  - 这就值得重点排查，因为按当前 `final_reward` 定义不应该发生。
- **如果 `eval/answer_extraction_failed` 很高**
  - 那么 `eval/outcome_correct` 的解释就会被 extraction 噪声污染。
- **如果某些 rollout 指标明明应该有，但 W&B 里消失**
  - 需要想到 rollout 日志有过滤逻辑。

当前 rollout 日志里确实有一个“近零就不记”的分支：

### 摘录：`lightrft/trainer/ppo_trainer_vl.py`

```python
mean_metric = metric_tensor.mean().item()
if abs(mean_metric) > 1e-6:
    rollout_status[f"rollout_{metric_name}"] = mean_metric
```

此外，`spmd_ppo_trainer.py` 还会对全 0 的 `model_reward` / `rule_reward` 做跳过：

### 摘录：`lightrft/trainer/spmd_ppo_trainer.py`

```python
if metric_name in {"model_reward", "rule_reward"} and metric_tensor.abs().sum() == 0:
    continue
status_mean[f"{metric_name}_mean"] = metric_tensor.mean().item()
status_mean[f"{metric_name}_std"] = metric_tensor.std().item()
```

这两段逻辑会影响：

- 某些 `rollout/*`
- 某些 `train` 内部聚合状态

但 **不会影响 `eval/*` 的均值计算**，因为 eval 那条路径没有这个过滤。

## 11. 用当前实验 README 再回看一次，就会发现这些现象都顺了

README 已经给了当前实验的核心观察：

### 摘录：`README.md`

```text
- `eval/outcome_correct`：范围 **0.3794 ~ 0.4127**，最后一次 **0.4028**
- `eval/reward`：范围 **0.3490 ~ 0.3740**，最后一次 **0.3661**
- `eval/model_reward`：后 5 次均值 **0.5021**，高于 `eval/outcome_correct`
- `eval/answer_extraction_failed`：最低 **0.0397@step70**，最后一次 **0.0714**
```

把它翻译成实现语义，就是：

- `eval/model_reward≈0.50`
  - PRM 代理分并不低，说明很多回答“局部看起来像样”。
- `eval/outcome_correct≈0.40`
  - 但真正答对的最终答案比例还是只有四成左右。
- `eval/reward≈0.36`
  - 而且其中一部分“答对但有 drop”的样本被降成了 0.5，所以 reward 又比 correctness 低一截。
- `eval/answer_extraction_failed≈4%~7%`
  - extraction 不是完全没问题，但还不足以解释 correctness 的主矛盾。

这四个量放在一起时，README 里那句总结就有了坚实实现含义：

```text
proxy gap 还在
```

它不是一句抽象判断，而是当前代码和当前实验同时支持的结论。

## 12. 最终结论

把 Phase 2 压缩成一句最准确的话，就是：

- **`eval/model_reward` 是 PRM 连续代理分，`eval/outcome_correct` 是最终答对率，`eval/reward` 是 PS-GRPO 的离散训练奖励均值；三者不是一回事。**

再压缩成两条最重要的判断边界，就是：

1. **`eval/model_reward > eval/outcome_correct` 在当前实现下首先是正常的 proxy gap 信号，不足以单独判成 bug。**
2. **`eval/reward < eval/outcome_correct` 在当前 `math_psgrpo` 实现下反而是应该预期的，因为“答对但有 drop”样本只拿 0.5。**

如果后续要继续深挖，这个 Phase 2 文档已经把最关键的排查入口定出来了：

- 想查指标来源，看 `math_prm_trainer.py` 的 key remap。
- 想查 eval 均值怎么算，看 `ppo_trainer_vl.py::evaluate()`。
- 想查三个 reward 系列指标为何分叉，看 `reward_models.py`。
- 想判断“这是设计如此还是实现出错”，就把当前现象对照 `paper.md` 的 PS-GRPO 动机与边界来判断。

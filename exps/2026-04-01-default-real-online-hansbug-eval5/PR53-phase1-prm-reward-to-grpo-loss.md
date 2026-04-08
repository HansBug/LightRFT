# PR53 Phase 1: PRM 打分后是如何进入 GRPO Loss 的

本文专门回答 [PR53_FOLLOWUP_CHECKLIST.md](./PR53_FOLLOWUP_CHECKLIST.md) 里的 Phase 1 问题：

> reward细节，prm得到打分后，是如何用于grpo loss计算的

这份文档不再停留在“结论口径”层，而是沿着当前仓库里真实启用的 Stage 3 路径，把下面这条链路逐段拆开：

1. URSA-RM 如何从回答里抽出 step score。
2. step score 如何变成 sequence-level reward。
3. sequence-level reward 如何变成 GRPO 的 group-normalized advantage。
4. `PolicyLoss` 实际吃到的到底是什么。

本文聚焦的默认运行路径是：

- launcher: `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- train entry: `examples/math_prm/train_colocate.py`
- reward label: `math_psgrpo`
- reward model: 单个 `URSA-RM-8B`
- advantage estimator: `group_norm`
- rollout engine: 当前主线使用本地 `hf`

## 太长不看

默认 `math_psgrpo` 不是把 PRM 的连续分数直接拿去做 GRPO loss。真实链路是：URSA-RM 先从每个 `Step N:` 边界抽 `step_scores`，再聚合出连续 `model_reward` 仅用于日志；训练真正使用的是 `final_reward`，即答错=0，答对但出现 drop-moment=0.5，答对且无 drop=1。举例：某条回答 `step_scores=[0.96,0.45,0.43]`，虽然 `model_reward≈0.43`，但因答案答对且中间骤降，训练 reward 是 0.5，不是 0.43。随后 8 条同题采样的 `final_reward` 做组内标准化，变成 GRPO advantage，再挂到 token returns 上进入 PPO clipped loss。

## 一页结论

先把最重要的话说清楚：

- **默认 `math_psgrpo` 路径下，PRM 的连续分数 `model_reward` 不会直接进入 GRPO loss。**
- **真正进入训练的是 PS-GRPO 的离散 reward `final_reward ∈ {0, 0.5, 1}`。**
- **这个 `final_reward` 先做 group normalization，变成组内相对优势，再被广播到 token 维度，最后进入 PPO/GRPO clipped objective。**
- **`model_reward` 默认更像诊断指标，不是默认优化目标。**

如果把真实训练路径压成一行，就是：

```text
step_scores
  -> aggregated model_reward (默认 min，仅用于诊断)
  -> PS-GRPO final_reward = {0, 0.5, 1}
  -> group_norm 标准化
  -> 最后一个 token 挂 sequence reward + 每个 token 叠加 KL penalty
  -> cumulative returns / advantages
  -> PolicyLoss(PPO/GRPO clipped loss)
```

## 1. 训练入口的默认配置到底是什么

当前主线脚本已经把“这是 PS-GRPO 路径”写在注释和参数里了。

### 摘录：`examples/math_prm/run_grpo_math_prm_ursa_8b.sh`

```bash
#   - Algorithm: Phase 4 GRPO with PS-GRPO reward via math_psgrpo label
#
# Step-scoring protocol (see MathPRMReward in reward_models.py):
#   1. The actor generates a chain-of-thought response.
#   2. The response is formatted with "Step N:" headings and "†Answer:" prefix.
#   3. Each step boundary is marked with Cyrillic ' и' (U+0438) token.
#   4. A single forward pass through URSA-8B-RM yields per-step probabilities.
#   5. In Phase 4, MathPRMReward maps step scores + correctness to PS-GRPO reward.
```

```bash
PATH_TO_YOUR_MATH_DATASET="${PATH_TO_YOUR_MATH_DATASET:-/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl}"
EXPECTED_REWARD_LABEL="${EXPECTED_REWARD_LABEL:-math_psgrpo}"
```

```bash
    --reward_pretrain "${REWARD_PRETRAIN_PATHS}" \
    --advantage_estimator "group_norm" \
    --n_samples_per_prompt $N_SAMPLES \
    --init_kl_coef $KL \
    --local_hf_max_new_tokens ${LOCAL_HF_MAX_NEW_TOKENS} \
```

这说明当前默认配置不是：

- `math_prm` 连续 PRM 分数直接做 reward

而是：

- `math_psgrpo` 标签
- `group_norm` advantage estimator
- `n_samples_per_prompt=8`

这三个条件合在一起，决定了后面进入 loss 的不是裸的 PRM 分数，而是 **PS-GRPO 离散 reward 的组内相对化结果**。

### 摘录：`examples/math_prm/train_colocate.py`

```python
trainer = MathPRMSPMDPPOTrainerVL(
    ...
    reward_fn=reward_fn,
    reward_fn_label_map=label_map,
    reward_recipe=RECIPE,
    ...
    gamma=args.gamma,
    lambd=args.lambd,
    init_kl_coef=args.init_kl_coef,
    ...
)
```

```python
if args.advantage_estimator in ["rloo", "reinforce_baseline", "group_norm"]:
    assert args.n_samples_per_prompt > 1, f"{args.advantage_estimator} requires n_samples_per_prompt > 1"
```

这里要注意一个容易误解的点：

- `train_colocate.py` 确实把 `reward_fn` / `RECIPE` 也传进 trainer 了。
- **但默认 `single RM` 路径并不会走那条“多 reward 聚合”逻辑。**
- 真正默认启用的是“单个 `MathPRMReward` 直接返回 `score`”那条路径，后面会详细拆。

## 2. PRM 的 step score 到底是怎么抽出来的

当前 `MathPRMReward` 的核心工作，是把回答格式化成 URSA-RM 需要的“步骤边界 marker”输入，然后读取这些 marker 位置上的 token classification logits。

### 摘录：`examples/math_prm/reward_models.py`

```python
class MathPRMReward(nn.Module):
    _SYSTEM_PROMPT = "You are a helpful assistant."
    _PRM_PROMPT = (
        "You are given a problem and a step-by-step solution. "
        "You need to check the correctness of each step.\nQuestion:"
    )
    _IMAGE_PAD = 575
    _DROP_THRESHOLD = 0.3
    _DROP_GAMMA = 0.5
```

```python
def replace_specific_plus_minus_with_ki(text: str) -> str:
    pattern = r"Step \d+"
    matches = list(re.finditer(pattern, text))
    ...
    answer_start = text.find("†Answer:")
    ...
    text = text[:index] + " и" + text[index:]
```

```python
reward = self.model(**inputs).logits
input_ids = inputs["input_ids"].view(-1)
padding = torch.full((self._IMAGE_PAD,), -1, device=device)
input_ids_aligned = torch.cat((input_ids[:1], padding, input_ids[1:]))

reward_flat = reward.view(-1)
step_logits = reward_flat[input_ids_aligned == self.tag_id]
step_scores = torch.sigmoid(step_logits).view(-1)
```

这里的语义是：

1. 把 `Step N:` 和 `†Answer:` 边界前插入单 token 的 ` и`
2. 送进 `UrsaForTokenClassification`
3. 只取这些 ` и` 对应位置的 logit
4. 做 `sigmoid`
5. 得到一条 response 的 step-level scores

如果一条回答有 3 个 step，那么这里拿到的是形如：

```text
[0.98, 0.97, 0.91]
```

而不是一个单独的标量。

### 与公开 `URSA-MATH` 推理脚本的对齐证据

当前 LightRFT 这段逻辑不是拍脑袋重写的，而是和公开 `URSA-MATH` 仓库里的 PRM 推理逻辑直接对齐。

### 摘录：`~/URSA-MATH/inference/prm_infer_score.py`

```python
def return_score(scores: torch.Tensor, operation: str):
    if operation == 'min':
        if scores.numel() == 0:
            return torch.tensor([0.])
        scores = scores.view(-1)
        return torch.min(scores)
    elif operation == 'avg':
        ...
```

```python
def replace_specific_plus_minus_with_ki(text):
    pattern = r'Step \d+'
    ...
    answer_start = text.find('†Answer:')
    ...
    text = text[:index] + ' и' + text[index:]
```

```python
reward = model(**inputs).logits
input_ids = inputs['input_ids'].view(-1)
insert_values = torch.full((575,), -1).to(input_ids.device)
input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))
reward = reward.view(-1)[input_ids == tag_id[0]]
reward = torch.sigmoid(reward).view(-1)
min_score = return_score(reward, 'min')
avg_score = return_score(reward, 'avg')
```

LightRFT 的 `MathPRMReward` 基本上就是沿这条公开 scorer 路径做的工程化封装，并且仓库里还有专门的对齐测试：

- `examples/math_prm/tools/check_phase2_alignment.py`
- `examples/math_prm/tools/test_phase2_alignment.py`

## 3. step score 怎么变成 sequence reward

这一段是 Phase 1 最关键的地方。

当前实现里同时算了两种“序列级别”的东西：

1. **`aggregated_score`**
   - 来自 step score 的聚合，默认是 `min`
   - 也就是 `model_reward`
2. **`psgrpo_metrics["final_reward"]`**
   - 来自最终答案判对 + drop-moment 检测
   - 取值是 `0 / 0.5 / 1`

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
```

```python
sequence_reward = psgrpo_metrics["final_reward"] if label == "math_psgrpo" else aggregated_score
batch_rewards.append(sequence_reward)
batch_metrics["model_reward"].append(aggregated_score)
```

上面这两行就是默认训练口径的分水岭：

- `label == "math_psgrpo"` 时，真正返回给训练链路的 `score` 是 `final_reward`
- `aggregated_score` 仍然会被保存在 `reward_metrics["model_reward"]` 里

也就是说：

```text
默认 math_psgrpo:
  用 final_reward 训练
  用 model_reward 记日志

默认 math_prm:
  才是直接用 aggregated_score 训练
```

## 4. `final_reward = 0 / 0.5 / 1` 是怎么来的

这个部分是 PS-GRPO 的核心。

### 4.1 先判最终答案对不对

当前实现会从 `†Answer:` 或 fallback 规则里抽最终答案，然后和 `reference` 做 exact match / math grading。

### 摘录：`examples/math_prm/reward_models.py`

```python
def _evaluate_answer_alignment(cls, response: str, reference: Any) -> Dict[str, Any]:
    reference_type, reference_supported = cls._infer_reference_type(reference)
    extraction = cls._extract_final_answer_details(response, reference_type)
    outcome_correct, comparison_method = cls._compare_final_answer(
        extraction["predicted_answer"],
        reference,
        reference_type,
        reference_supported,
    )
    return {
        ...
        "outcome_correct": outcome_correct,
    }
```

### 4.2 再检测 PRM reward sequence 里有没有 drop-moment

### 摘录：`examples/math_prm/reward_models.py`

```python
def _compute_relative_drop(cls, step_scores: torch.Tensor) -> tuple[float, bool]:
    if step_scores.numel() < 2:
        return 0.0, False

    scores = step_scores.detach().float()
    prev_scores = scores[:-1]
    next_scores = scores[1:]
    denom = torch.clamp(prev_scores, min=1e-6)
    relative_drops = torch.clamp((prev_scores - next_scores) / denom, min=0.0)
    max_relative_drop = float(relative_drops.max().item()) if relative_drops.numel() else 0.0
    return max_relative_drop, max_relative_drop >= cls._DROP_THRESHOLD
```

当前阈值是：

```python
_DROP_THRESHOLD = 0.3
```

所以：

- 如果某两个相邻 step score 从 `0.96` 掉到 `0.45`
- 相对下降约是 `(0.96 - 0.45) / 0.96 = 0.53`
- 那就会触发 `has_drop_moment = 1`

### 4.3 最后映射成 PS-GRPO reward

### 摘录：`examples/math_prm/reward_models.py`

```python
def _compute_psgrpo_metrics(
    cls,
    response: str,
    reference: Any,
    step_scores: torch.Tensor,
) -> Dict[str, float]:
    answer_eval = cls._evaluate_answer_alignment(response, reference)
    outcome_correct = float(answer_eval["outcome_correct"])
    max_relative_drop, has_drop_moment = cls._compute_relative_drop(step_scores)

    final_reward = 0.0
    if outcome_correct > 0.0:
        final_reward = 1.0 - cls._DROP_GAMMA if has_drop_moment else 1.0
```

其中：

```python
_DROP_GAMMA = 0.5
```

所以当前 LightRFT 的默认 PS-GRPO reward 规则就是：

- 答错：`0.0`
- 答对且没有 drop：`1.0`
- 答对但有 drop：`0.5`

## 5. 这和论文里的 PS-GRPO 是怎么对应的

当前实现和论文要表达的核心思想是一致的：**不要直接把 PRM scalar reward 当优化目标，而是只利用它揭示出来的“过程是否出现可疑下跌”这个结构信号。**

### 摘录：`~/URSA-MATH/paper.md`

```text
Following most standard response-level and step-level reward modeling in RL ...
we examine two simple variants of GRPO with integrated scalar process rewards ...

Variant 1: ... the reward is the sum of the outcome reward and the average process reward
Variant 2: ... a scalar process reward is assigned to the i-th rollout’s t-th step.
```

```text
We observe two highly significant conclusions ...
(i) High susceptibility to reward hacking
(ii) PRM’s length bias in rewarding
```

```text
We introduce the concept of a “drop-moment” within the PRM’s reward sequence ...
a significant decrease in reward between consecutive steps indicates the occurrence of such a drop-moment.
```

论文里的公式对应关系也很直接。

### 摘录：`~/URSA-MATH/paper.md`

```text
δ_p^i = max{ (r_{p,j}^i - r_{p,j+1}^i) / r_{p,j}^i } > ρ
```

```text
R^i =
  1,       if o^i is correct and δ_p^i < ρ
  1 - γ,   if o^i is correct and δ_p^i ≥ ρ
  0,       otherwise
```

### 摘录：`~/URSA-MATH/paper_zh.md`

```text
A^i = (r^i - mean({r^j}_{j=1}^G)) / std({r^j}_{j=1}^G)
```

这三段和当前实现是高度同构的：

- `δ_p^i > ρ` 对应 `_compute_relative_drop()`
- `R^i ∈ {1, 1-γ, 0}` 对应 `_compute_psgrpo_metrics()`
- `A^i = (r^i - mean) / std` 对应 `GroupNormCalculator.preprocess_rewards()`

## 6. 这个 reward 是怎么进入 Experience 的

这里要分清“多 reward model 聚合路径”和“默认单 RM 路径”。

### 6.1 默认单 RM 路径

当前默认是单个 `URSA-RM-8B`，所以 `FastExperienceMaker` 会直接使用这个 reward model 输出的 `score`。

### 摘录：`lightrft/trainer/fast_exp_maker.py`

```python
if isinstance(rm_output, dict):
    score = torch.as_tensor(rm_output["score"], dtype=torch.float32, device=device)
    metrics = self._normalize_reward_metrics(rm_output, score.numel(), device)
else:
    score = torch.as_tensor(rm_output, dtype=torch.float32, device=device)
    metrics = None
```

```python
if is_multi_rm:
    rewards, reward_metrics = self.reward_fn(...)
    outputs[mb_idx].rewards = rewards
    outputs[mb_idx].reward_metrics = reward_metrics
else:
    outputs[mb_idx].rewards = same_batch_results[0].scores
    outputs[mb_idx].reward_metrics = same_batch_results[0].metrics
```

这段逻辑非常关键：

- **单 RM**：`outputs[mb_idx].rewards = rm_output["score"]`
- **多 RM**：才会调用 `reward_fn(...)`

所以在当前默认 Stage 3 路径下：

- `MathPRMReward.forward()` 里返回的 `score`
- 就是后面经验对象里的 `reward`

### 6.2 它是怎么进 `experience.info["reward"]` 的

### 摘录：`lightrft/trainer/fast_exp_maker.py`

```python
info = dict(
    kl=kl_mean,
    reward=output.rewards,
    response_length=output.response_length,
    total_length=output.total_length,
    num_actions=output.num_actions,
)

if output.reward_metrics is not None:
    info['reward_metrics'] = output.reward_metrics
```

这一步以后：

- `experience.info["reward"]` = 当前 sequence reward
- 对默认 `math_psgrpo` 而言，它就是 `final_reward`
- `experience.info["reward_metrics"]["model_reward"]` 才是连续 PRM 聚合分数

## 7. `reward_fn()/mix_rewards()` 为什么容易把人带偏

仓库里确实还有一个 `reward_models_utils.reward_fn()` / `mix_rewards()`，而且其中也定义了 `math_psgrpo` 这个 label。

### 摘录：`examples/math_prm/reward_models_utils.py`

```python
RECIPE: Dict[str, List[Tuple[str, Optional[str], float]]] = {
    "math_prm": [("model", "math_prm", 1.0)],
    "math_psgrpo": [("model", "math_prm", 1.0)],
    "math_prm_combined": [("model", "math_prm", 1.0), ("rule", None, 0.5)],
    "math_rule": [("rule", None, 1.0)],
}
```

```python
for reward_type, key, weight in recipe:
    if reward_type == "model":
        model_reward = weight * get_model_reward(key, index)
        reward_value += model_reward
        metrics_dict["model_reward"][index] += model_reward
```

如果只看这段，很容易得出错误结论：

- “`math_psgrpo` 不就是把 model reward 直接拿来当 reward 吗？”

但这只有在 **多 RM 聚合路径** 才会发生。

而默认主线是：

- `single RM`
- `FastExperienceMaker._aggregate_rewards(..., is_multi_rm=False)`
- 直接拿 `MathPRMReward.forward()` 返回的 `score`

所以默认训练路径里，真正生效的是：

- `reward_models.py` 里的 `sequence_reward = final_reward`

不是：

- `reward_models_utils.py` 里 `mix_rewards()` 的 `model_reward` 相加逻辑

这也是为什么只盯着 `reward_fn()` 做分析，常常会把链路看错。

## 8. `group_norm` 是怎么把 sequence reward 变成优势的

当前默认 `advantage_estimator="group_norm"`，对应 `GroupNormCalculator`。

### 摘录：`lightrft/trainer/advantage_calculator.py`

```python
class GroupNormCalculator(BaseREINFORCECalculator):
    ...
    rewards = rewards.reshape(-1, n_samples).to("cuda")
    rewards = (rewards - rewards.mean(-1, keepdim=True)) / (rewards.std(-1, keepdim=True) + 1e-9)

    rewards = rewards.flatten().to("cpu").chunk(len(experiences))
    return experiences, list(rewards)
```

这一步的输入是：

- `experience.info["reward"]`
- 也就是默认 `math_psgrpo` 下的 `final_reward`

不是：

- token-level PRM dense reward
- 也不是 `model_reward`

这一步做完以后，单个 prompt 下的 8 个样本会从：

```text
[1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
```

变成：

```text
[(1-μ)/σ, (0.5-μ)/σ, ..., (0-μ)/σ]
```

也就是“组内相对化后的序列奖励”。

### 为什么这一步很重要

因为 GRPO 不是在优化“绝对 reward”，而是在优化“同组中比别的 rollout 更值得学的程度”。

这意味着：

- `1.0` 不一定是“大正 advantage”，要看组内别的样本是什么
- `0.5` 不一定还是正 advantage
- `0.0` 不一定总是最差，但在默认三值奖励下通常会是负 advantage

## 9. group-normalized reward 是怎么变成 token-level returns 的

现在 reward 还是 sequence-level 标量。接下来它会被挂到 response 的最后一个有效 token 上，然后再叠加 token-level KL penalty。

### 摘录：`lightrft/models/utils.py`

```python
def compute_reward(
    r,
    kl_coef,
    kl,
    action_mask=None,
    num_actions=None,
    reward_clip_range=None,
):
    ...
    kl_reward = -kl_coef * kl
    eos_indices = action_mask.size(1) - 1 - action_mask.long().fliplr().argmax(dim=1, keepdim=True)
    last_reward = torch.zeros_like(kl).scatter_(dim=1, index=eos_indices, src=r.unsqueeze(1).to(kl.dtype))

    reward = last_reward + kl_reward
```

### 摘录：`lightrft/trainer/fast_exp_maker.py`

```python
final_reward = compute_reward(
    processed_reward,
    self.kl_ctl.value,
    experience.kl,
    action_mask=experience.action_mask,
    num_actions=experience.info["num_actions"],
)
```

这里的含义是：

- `processed_reward` 是 sequence-level reward
- 它只加在最后一个有效 action token 上
- 每个 token 还会额外收到 `-β * KL_t`

如果先忽略 KL，那么一个长度为 `T` 的 response，其 token-level reward 形状大概就是：

```text
[0, 0, 0, ..., 0, sequence_reward]
```

再通过 cumulative return 反推回去，整条轨迹的每个 token 都会继承这条 response 的序列级别学习信号。

## 10. 最后 `PolicyLoss` 实际吃到的是什么

默认 `group_norm` 会走 REINFORCE-style cumulative return，不用 critic baseline。

### 摘录：`lightrft/trainer/advantage_calculator.py`

```python
returns = self.get_cumulative_returns(final_reward, experience.action_mask, gamma)
advantages = deepcopy(returns)
```

然后进 `PolicyLoss`：

### 摘录：`lightrft/models/loss.py`

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
loss = -torch.min(surr1, surr2)
loss = masked_mean(loss, final_mask, dim=-1).mean()
```

所以默认路径里，`PolicyLoss` 吃到的是：

- `advantages`
- 这些 advantage 来自 cumulative returns
- cumulative returns 又来自“最后 token 的 sequence reward + 全 token KL penalty”
- sequence reward 的源头则是 `final_reward`

不是：

- 原始 step scores
- 也不是 `model_reward`

## 11. 一个真实跑出来的 8-sample 数字例子

下面这个例子不是手算出来的，而是我在 `lightrft` conda 环境下直接调用本地 `URSA-RM-8B` 和当前仓库的 `MathPRMReward` 跑出来的一个 mini-group。

运行口径：

- env: `conda run -n lightrft`
- scorer: `examples/math_prm/reward_models.py::MathPRMReward`
- model: `~/URSA-MATH/checkpoints/URSA-RM-8B`
- image: `~/URSA-MATH/figures/framework.png`
- question: `How many numbered training stages are shown in this diagram?`
- reference answer: `3`
- group size: `8`

### 11.1 原始结果

| sample | step_scores | model_reward(min) | outcome_correct | has_drop_moment | final_reward |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | `[0.984375, 0.96875, 0.914062]` | 0.914062 | 1 | 0 | 1.0 |
| 2 | `[0.964844, 0.451172, 0.425781]` | 0.425781 | 1 | 1 | 0.5 |
| 3 | `[0.96875, 0.9375, 0.294922]` | 0.294922 | 1 | 1 | 0.5 |
| 4 | `[0.984375, 0.972656, 0.777344]` | 0.777344 | 1 | 0 | 1.0 |
| 5 | `[0.890625, 0.141602]` | 0.141602 | 0 | 1 | 0.0 |
| 6 | `[0.960938, 0.769531]` | 0.769531 | 0 | 0 | 0.0 |
| 7 | `[0.984375, 0.972656]` | 0.972656 | 0 | 0 | 0.0 |
| 8 | `[0.988281]` | 0.988281 | 1 | 0 | 1.0 |

### 11.2 这组数据最值得注意的三个点

#### 点 A：`model_reward` 高，不代表会被高 reward 训练

sample 7 很典型：

- `model_reward(min)=0.972656`
- 但 `outcome_correct=0`
- 所以 `final_reward=0.0`

也就是说：

- PRM 觉得这条轨迹“看起来过程很像样”
- 但最终答案没对
- 默认 `math_psgrpo` 仍然会把它当作 `0 reward`

这正是 `eval/model_reward` 可能显著高于 `eval/outcome_correct` 的根本原因之一。

#### 点 B：`Format=1 + accuracy=1` 也不一定拿到 `1.0`

sample 2 和 sample 3 都是：

- `outcome_correct=1`
- 但 `has_drop_moment=1`
- 所以 `final_reward=0.5`

这就是你在 FOLLOWUP 里提到的那类现象：

- “明明答对了，为什么最终 reward 还是 0.5？”

直接原因就是：

- 有明显 `drop-moment`
- `_DROP_GAMMA = 0.5`

#### 点 C：`0.5` 在 GRPO 里不一定是正优势

这一组 8 条样本的 `final_reward` 是：

```text
[1.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0]
```

组均值和标准差是：

```text
mean = 0.5
std  = 0.46291
```

做完 group normalization 以后：

```text
[ 1.080123, 0.0, 0.0, 1.080123, -1.080123, -1.080123, -1.080123, 1.080123 ]
```

这意味着：

- `final_reward=1.0` 的样本获得正优势
- `final_reward=0.0` 的样本获得负优势
- **`final_reward=0.5` 的样本在这个 group 里恰好是 0 advantage**

这件事很关键，因为它解释了一个经常被误解的点：

- `0.5` 不是“已经明确鼓励”
- 在很多 batch 里，它更像“比错的好，但不一定比同组平均更值得学”

## 12. token-level 视角下，这个例子会变成什么

假设 sample 2 的 response 长度是 4 个有效 token，先忽略 KL。

它在 group_norm 后的序列 reward 是：

```text
0.0
```

那么 `compute_reward()` 后的 token-level reward 近似是：

```text
[0, 0, 0, 0]
```

再做 cumulative return：

```text
advantages ≈ [0, 0, 0, 0]
```

如果是 sample 1，group-normalized reward 是 `1.080123`，则近似变成：

```text
token rewards   = [0, 0, 0, 1.080123]
token returns   = [1.080123, 1.080123, 1.080123, 1.080123]
token advantages= [1.080123, 1.080123, 1.080123, 1.080123]
```

如果是 sample 5，group-normalized reward 是 `-1.080123`，则近似变成：

```text
token rewards   = [0, 0, 0, -1.080123]
token returns   = [-1.080123, -1.080123, -1.080123, -1.080123]
token advantages= [-1.080123, -1.080123, -1.080123, -1.080123]
```

再叠加默认很小的 KL penalty 后，这些值会稍微向 reference policy 回拉。

## 13. 那么 “PRM 分数到底有没有用于 loss” 应该怎么回答

最准确的说法不是简单地回答“有”或“没有”，而应该拆成两层：

### 13.1 有使用，但不是直接把连续分数拿来当 reward

PRM 的 step scores 在默认路径里至少参与了两件事：

1. 形成 `model_reward`
2. 检测 `drop-moment`

### 13.2 真正进入默认 GRPO loss 的，不是 `model_reward`

默认 `math_psgrpo` 的实际训练信号是：

```text
PRM step score sequence
  -> detect drop-moment
  -> combine with outcome correctness
  -> final_reward in {0, 0.5, 1}
  -> group_norm
  -> token-level returns/advantages
  -> PolicyLoss
```

所以更准确的结论是：

- **PRM 的“结构信号”被用于 loss**
- **PRM 的“连续 scalar reward”默认没有直接作为优化目标**

这正好和论文想避免的 failure mode 对上了：

- 论文不想让模型直接追逐 PRM scalar reward
- 当前实现也确实没有这样做

## 14. 这一结论能支持到什么程度，不能支持到什么程度

### 当前已经可以直接确认的

- 当前仓库默认 `math_psgrpo` 主线下，训练 reward 是 `final_reward`，不是 `model_reward`
- 当前仓库默认 `single RM` 主线下，不会经过 `mix_rewards()` 再把连续 model score 拼回训练 reward
- 当前 `group_norm` 确实是先对 sequence reward 做组内标准化，再进入 token-level returns / advantages
- 当前 `PolicyLoss` 吃到的是这些优势值，而不是原始 PRM 分数

### 当前仍然要保留边界感的

- 公开 `URSA-MATH` 仓库并没有直接放出完整 Stage 3 RL 训练代码
- 因此“LightRFT 当前实现是否与原始内部训练脚本逐行一致”不能只靠公开仓库完全证实
- 但就论文语义、公开 PRM 推理逻辑、当前 LightRFT 代码、现有对齐测试而言，**当前默认实现与论文宣称的 PS-GRPO 思想是明显一致的**

## 最后一句话

如果要把 Phase 1 压成一句可以直接复述给别人的话：

> 在当前 LightRFT 的默认 Stage 3 路径里，URSA-RM 先给出逐步 PRM 分数；这些分数默认不是直接当作 GRPO reward，而是先被用来检测是否出现 drop-moment，再和最终答案正确性一起压缩成 `0 / 0.5 / 1` 的 PS-GRPO reward。之后 trainer 对这组 reward 做组内标准化，生成 sequence-level 相对优势，再把它挂到 token-level returns/advantages 上，最后才送进 PPO/GRPO 的 clipped policy loss。

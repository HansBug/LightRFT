# Math PRM Eval Debug 记录

## 时间

- 记录日期：2026-03-29

## 背景

这次排查针对的是 `pjctfbbs` 这条 `math_psgrpo` 训练 run。

现象非常明确：

- W&B 上 `eval acc` 基本是一条直线。
- leader 的怀疑方向是两种：
  1. 模型其实没训练起来。
  2. eval 逻辑或者 eval 链路本身有 bug。

这份记录只做诊断归档，不改主线代码。

## 当前结论

当前结论已经可以写得比较明确：

1. **eval 不是没跑，确实在按步数触发。**
2. **eval 不是“面板显示问题”，本地日志里的 eval 输出和聚合指标都几乎完全不变。**
3. **训练权重不是没变。**
   - checkpoint 在持续保存；
   - step40 / step60 的 shard 内容不同；
   - 离线从 step40 / step60 checkpoint 直接 decode，同一 held-out 样本的输出会变化。
4. **最可能的问题点已经收敛到 separate local HF rollout actor 的权重同步链。**
   - 直接改 inference engine，自然会让 rollout 输出变化；
   - 直接改 actor，本体 direct generate 也会变化；
   - 但改完 actor 再调用 `update_engine_weights(actor)`，rollout 输出不跟着变化。

所以当前最强判断不是“模型没训练”，而是：

- **runtime eval 大概率没有真正吃到最新训练 actor 的权重。**
- **问题更像出在 `update_engine_weights(actor)` 到 separate rollout actor 的同步语义上。**

---

## 逻辑链条总览

这次诊断的完整逻辑链条是：

1. 先确认 eval 到底有没有触发。
2. 再确认 eval 结果是不是“真的完全不动”。
3. 然后排除 eval 数据集本身为空、切坏、或高度退化的可能。
4. 再排除“本地 HF rollout 路径根本不能工作”的可能。
5. 然后确认训练权重是否真的落盘、是否真的变化。
6. 最后做最小化同步探针，验证：
   - 改 inference engine 会不会影响 rollout；
   - 改 actor 本体会不会影响 direct generate；
   - 改 actor 再 sync 到 rollout，会不会影响 rollout。

走完这条链之后，问题边界已经非常清楚。

---

## 一、eval 确实在跑，而且本地日志里的值就是死的

### 1.1 eval 聚合指标在 step 5 和 step 60 完全相同

下面是本地日志里 `step 5` 的聚合 eval 输出：

```text
[StrategyINFO 03-25 13:51:16]  Aggregated runtime eval metrics (Step 5):
[StrategyINFO 03-25 13:51:16]    reward: 0.3601
[StrategyINFO 03-25 13:51:16]    outcome_correct: 0.3909
[StrategyINFO 03-25 13:51:16]    has_drop_moment: 0.4365
[StrategyINFO 03-25 13:51:16]    model_reward: 0.4994
[StrategyINFO 03-25 13:51:16]    response_length: 185.2699
[StrategyINFO 03-25 13:51:16]    answer_extraction_failed: 0.0714
```

下面是本地日志里 `step 60` 的聚合 eval 输出：

```text
[StrategyINFO 03-26 11:12:16]  Aggregated runtime eval metrics (Step 60):
[StrategyINFO 03-26 11:12:16]    reward: 0.3601
[StrategyINFO 03-26 11:12:16]    outcome_correct: 0.3909
[StrategyINFO 03-26 11:12:16]    has_drop_moment: 0.4365
[StrategyINFO 03-26 11:12:16]    model_reward: 0.4994
[StrategyINFO 03-26 11:12:16]    response_length: 185.2699
[StrategyINFO 03-26 11:12:16]    answer_extraction_failed: 0.0714
```

这条证据说明：

- eval 曲线“完全不动”不是 W&B 面板抽风。
- 本地 runtime eval 的聚合结果本身就是常数。

### 1.2 eval 打印出来的首个样本文本也完全不变

为了避免“聚合指标不变，但单样本文本其实在波动”的误判，我继续看了 eval 阶段打印的第一个样本输出。

`step 5` 的原始日志片段：

```text
[StrategyINFO 03-25 13:42:11]  eval phase: experience.sequences w skip_special_tokens:  ['system
A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.
user
 As shown in the figure, two roads of the same width are built on a rectangular ground with a width of 20 meters and a length of 30 meters, leaving the remaining part as farmland. If the area of the farmland needs to be 551 square meters, the width of the roads should be

[____ ]
A. 1 meter
B. 1.5 meters
C. 2 meters
D. 2.5 meters
Options:
assistant
Step 1: Let the width of the road be x meters.
Step 2: The total area of the rectangular ground is 20m * 30m = 600 square meters.
Step 3: The total area of the roads is 20x + 30x - x^2.
Step 4: The farmland area is 600 - (20x + 30x - x^2) = 551 square meters.
Step 5: 600 - 50x + x^2 = 551.
Step 6: x^2 - 50x + 49 = 0.
Step 7: Using the quadratic formula, x = [50 ± sqrt(50^2 - 4*1*49)] / 2*1.
Step 8: x = [50 ± sqrt(2500 - 196)] / 2.
Step 9: x = [50 ± sqrt(2304)] / 2.
Step 10: x = [50 ± 48] / 2.
Step 11: x = 49 or x = 1.
Step 12: Since the width of the road cannot be greater than the width of the rectangle, x = 1 meter.
†Answer: 1']
```

`step 20` 对应轮次的原始日志片段：

```text
[StrategyINFO 03-25 15:40:51]  eval phase: experience.sequences w skip_special_tokens:  ['system
A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.
user
 As shown in the figure, two roads of the same width are built on a rectangular ground with a width of 20 meters and a length of 30 meters, leaving the remaining part as farmland. If the area of the farmland needs to be 551 square meters, the width of the roads should be

[____ ]
A. 1 meter
B. 1.5 meters
C. 2 meters
D. 2.5 meters
Options:
assistant
Step 1: Let the width of the road be x meters.
Step 2: The total area of the rectangular ground is 20m * 30m = 600 square meters.
Step 3: The total area of the roads is 20x + 30x - x^2.
Step 4: The farmland area is 600 - (20x + 30x - x^2) = 551 square meters.
Step 5: 600 - 50x + x^2 = 551.
Step 6: x^2 - 50x + 49 = 0.
Step 7: Using the quadratic formula, x = [50 ± sqrt(50^2 - 4*1*49)] / 2*1.
Step 8: x = [50 ± sqrt(2500 - 196)] / 2.
Step 9: x = [50 ± sqrt(2304)] / 2.
Step 10: x = [50 ± 48] / 2.
Step 11: x = 49 or x = 1.
Step 12: Since the width of the road cannot be greater than the width of the rectangle, x = 1 meter.
†Answer: 1']
```

`step 60` 之后对应轮次的原始日志片段：

```text
[StrategyINFO 03-25 19:29:01]  eval phase: experience.sequences w skip_special_tokens:  ['system
A conversation between the User and Assistant. The User asks a question that may require mathematical or visual reasoning, and the Assistant solves it step by step. Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. After all steps, output exactly one final answer line prefixed with "†Answer:" (e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output any extra text, repeated answer markers, or additional steps.
user
 As shown in the figure, two roads of the same width are built on a rectangular ground with a width of 20 meters and a length of 30 meters, leaving the remaining part as farmland. If the area of the farmland needs to be 551 square meters, the width of the roads should be

[____ ]
A. 1 meter
B. 1.5 meters
C. 2 meters
D. 2.5 meters
Options:
assistant
Step 1: Let the width of the road be x meters.
Step 2: The total area of the rectangular ground is 20m * 30m = 600 square meters.
Step 3: The total area of the roads is 20x + 30x - x^2.
Step 4: The farmland area is 600 - (20x + 30x - x^2) = 551 square meters.
Step 5: 600 - 50x + x^2 = 551.
Step 6: x^2 - 50x + 49 = 0.
Step 7: Using the quadratic formula, x = [50 ± sqrt(50^2 - 4*1*49)] / 2*1.
Step 8: x = [50 ± sqrt(2500 - 196)] / 2.
Step 9: x = [50 ± sqrt(2304)] / 2.
Step 10: x = [50 ± 48] / 2.
Step 11: x = 49 or x = 1.
Step 12: Since the width of the road cannot be greater than the width of the rectangle, x = 1 meter.
†Answer: 1']
```

这条证据比“W&B 曲线平”更强：

- **固定 held-out 样本的原始生成文本本身就是死的。**

---

## 二、eval 数据集本身没有坏掉

### 2.1 held-out split 是稳定存在的

我按训练配置的相同规则单独切了一遍 held-out：

- `test_size = 500`
- `seed = 42`

输出如下：

```text
train_len= 1018559
eval_len= 500
{'i': 0, 'label': 'math_psgrpo', 'reference': '1', 'image': '/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/RGB_images/dd1c884f28781cf3c4dbcf192fbd3604.png', 'prompt_prefix': 'As shown in the figure, two roads of the same width are built on a rectangular ground with a width o'}
{'i': 1, 'label': 'math_psgrpo', 'reference': '11.0', 'image': '/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/DataEngine_Geometry/rule_base_geo_vision_dom/depth3/24629_vision_dom.jpg', 'prompt_prefix': 'AB equals to 11.0. What is the length of the side GH that forms the base of the isosceles triangle G'}
{'i': 2, 'label': 'math_psgrpo', 'reference': 'lobby adjacent to exhibit area', 'image': '/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/data_images/DocVQA/images/txpp0227_13.png', 'prompt_prefix': 'At which location will the coffee be served?'}
{'i': 3, 'label': 'math_psgrpo', 'reference': 'B', 'image': '/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/data_images/PMC-VQA/images/PMC8126771_fig01_443233.jpg', 'prompt_prefix': 'What is the significance of the cyan arrow shown in the image?\nChoices:\n(A)  Distance between the me'}
{'i': 4, 'label': 'math_psgrpo', 'reference': '27', 'image': '/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/RGB_images/2912ade8b6644db804037aaafffde6b3.png', 'prompt_prefix': 'There are ____ parallelograms in the image.'}
```

这条证据说明：

- held-out eval 数据集不是空的；
- 不是全重复样本；
- 也不是 label / reference 全都坏掉了。

### 2.2 eval 的首样本正好就是 held-out[0]

从上面的 split 输出看，第 0 个 held-out 样本就是：

- “two roads / farmland 551” 这题；
- reference 是 `1`；
- label 是 `math_psgrpo`。

而训练日志里反复打印的 eval 首样本，正好就是这道题。

这说明：

- 我后面做的离线对比和训练日志里的 runtime eval，确实是在看同一条 held-out 样本。

---

## 三、本地 HF rollout 路径本身不是“根本不能工作”

为了排除 “local HF rollout path 根本坏了” 这种更基础的问题，我先做了一个独立 smoke check。

输出如下：

```json
{
  "success": false,
  "strict_structure_success": false,
  "engine_type": "hf",
  "rollout_checks": {
    "engine_type_is_hf": true,
    "engine_reuses_actor": true,
    "num_outputs_match": true,
    "all_non_empty": true,
    "all_match_direct_generate": true,
    "all_below_length_cap": false
  },
  "quality_checks": {
    "all_have_step_marker": true,
    "all_have_answer_marker": false,
    "all_stop_condition_satisfied": false
  },
  "samples": [
    {
      "name": "vqa_flatness",
      "tokens_match_direct_generate": true,
      "generated_text": "Step 1: Observe the image provided.\nStep 2: Analyze the landscape in the image.\nStep 3: Determine if the landscape is flat or not.\nStep 4: Conclude that the landscape is not flat.\n\n†Answer: no"
    },
    {
      "name": "table_linear_eq",
      "tokens_match_direct_generate": true,
      "generated_text": "Step 1: Identify the problem: Find the y-value when x = 6 6 and the equation is y = 3x + 5.\n\nStep 2: Substitute the given value of x into the equation::  Substitute x = 7 into the equation y = 2x +"
    }
  ]
}
```

这里的重点不是 `success=false`，而是：

- `engine_type_is_hf: true`
- `engine_reuses_actor: true`
- `all_match_direct_generate: true`

这说明：

- local HF engine 的 `gather_and_generate()` 与 direct `actor.generate()` 至少在最小化场景下是对齐的；
- rollout 路径不是完全没接上。

所以问题不能简单归因为：

- “local HF rollout 根本不能生成”

---

## 四、训练权重不是没动，checkpoint 也不是没保存

### 4.1 checkpoint 在正常保存

日志原文如下：

```text
[StrategyINFO 03-25 19:37:50]  DCP checkpoint saved to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step20
[StrategyINFO 03-25 19:37:50]  client_state save to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step20/client_state.pt, content: {'consumed_samples': 2560}

[StrategyINFO 03-26 03:23:42]  DCP checkpoint saved to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step40
[StrategyINFO 03-26 03:23:42]  client_state save to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step40/client_state.pt, content: {'consumed_samples': 5120}

[StrategyINFO 03-26 11:12:24]  Deleted oldest ckpt results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step20
[StrategyINFO 03-26 11:12:39]  DCP checkpoint saved to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step60
[StrategyINFO 03-26 11:12:39]  client_state save to results/lightrft-ursa8b-stage3-psgrpo/.../_actor/global_step60/client_state.pt, content: {'consumed_samples': 7680}
```

这说明：

- step20 / step40 / step60 的 actor ckpt 都正常保存了；
- step60 保存时还删除了最旧的 step20，符合 `max_ckpt_num=2` 的配置预期。

### 4.2 checkpoint shard 的字节内容确实不同

我直接算了 `global_step40` 和 `global_step60` 的几个 shard 的 SHA256：

```text
314e6d93abf35ee0712fb1b9ca928f0dea9764e3e1da2ffa0f927b315874051b  __0_0.distcp (step40)
72d63d9f7db6b164dfe3b82a066e389cf494179214cafdbba61dc5c08ea490d4  __0_0.distcp (step60)

10d4e34ef4a2d0c86e214bfc208f310e04ba6f27d149d8f152bce73236a9f6f8  __1_0.distcp (step40)
55a67f62830f47345f1bd234d378c06287f575728bb7defa51137882aba439d4  __1_0.distcp (step60)

48e6021ea2f0941d56100f7d912c05418dd9c88d406ca4579f90f08216c13624  __2_0.distcp (step40)
8af6e408a998bdea35f05a800c51ac2dcc994ccd03cfc6bd4f4bf8241e4ce0b6  __2_0.distcp (step60)
```

这条证据说明：

- `global_step40` 和 `global_step60` 不是同一份旧权重重复写盘；
- 磁盘上的 actor 权重内容真的变了。

### 4.3 训练过程中也确实在调用权重同步

训练刚开始时就能看到：

```text
[StrategyINFO 03-25 11:51:50]  Finished update engine weights for separate local HF rollout actor {'total_s': 0.4636, 'keep_on_gpu': True, 'actor_offloaded': False, 'rollout_offloaded': False, 'offload_actor_s': 0.0, 'offload_rollout_s': 0.0, 'copy_state_s': 0.4534, 'prepare_s': 0.008, 'sync_clear_s': 0.0023}
```

后面在最小化 probe 中，这个同步操作也会持续出现，`copy_state_s` 大约在 `34-35s`。

所以从“流程有没有跑到”这个角度看：

- `update_engine_weights(actor)` 是有执行的；
- 但“执行了”并不等于“同步语义真的正确”。

---

## 五、离线从真实 checkpoint 直接 decode，同一 held-out 样本会变化

这是把问题从“训练没动”里真正拉开的关键证据。

我把 `global_step40` 和 `global_step60` 都转成了 HF 目录，然后按训练时相同的 held-out 规则：

- `eval_holdout_size = 500`
- `eval_holdout_seed = 42`

取前 8 个 held-out 样本，分别用：

- base model
- step40
- step60

做 greedy decode。

输出摘要如下：

```json
{
  "eval_holdout_size": 500,
  "eval_holdout_seed": 42,
  "limit": 8,
  "changed_counts": {
    "base_vs_step40": 6,
    "base_vs_step60": 6,
    "step40_vs_step60": 6
  }
}
```

也就是说：

- 8 个 held-out 样本里，有 6 个在 `base -> step40` 时文本变了；
- 有 6 个在 `base -> step60` 时文本变了；
- 有 6 个在 `step40 -> step60` 时文本变了。

第 0 个 held-out 样本正好就是 runtime eval 一直打印的 “two roads / farmland 551” 这题，它的离线结果如下：

```json
{
  "index": 0,
  "prompt_hash": "7bafd9fde35aa9b4",
  "reference": "1",
  "label": "math_psgrpo",
  "base_hash": "8e3c00da39819320",
  "step40_hash": "17d5d5a13c2e3557",
  "step60_hash": "8e3c00da39819320",
  "base_vs_step40_changed": true,
  "base_vs_step60_changed": false,
  "step40_vs_step60_changed": true,
  "base_preview": "Step 1: Let the width of the road be x meters.\nStep 2: The total\n†Answer: No answer found<|im_end|>",
  "step40_preview": "Step 1: Let the width of of the road be x meters.\nStep 2: The total\n†Answer: No answer found<|im_end|>",
  "step60_preview": "Step 1: Let the width of the road be x meters.\nStep 2: The total\n†Answer: No answer found<|im_end|>"
}
```

这条证据的意义非常直接：

- **真实训练 checkpoint 已经足够改变 held-out 样本的 greedy decode。**

因此，“训练更新太小，小到完全不影响行为”这条解释已经明显变弱了。

更准确的说法应该是：

- 如果 runtime eval 真在吃最新 actor，那至少一部分固定 held-out 样本的输出本该抖动。

---

## 六、最关键的最小化同步 probe

这部分是目前最能说明问题的位置。

### 6.1 直接改 inference engine，自然会让 rollout 输出变化

我对 separate rollout actor 的 `inference_engine` 直接做了极端扰动：

- 把 `lm_head` 清零

输出如下：

```json
{
  "sample_name": "vqa_flatness",
  "before_len": 64,
  "after_len": 64,
  "tokens_changed": true,
  "mutate_target": "inference_engine",
  "mutate_mode": "lm_head_zero",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>",
  "sync_stats": {
    "total_s": 34.3065,
    "copy_state_s": 34.2964
  }
}
```

这个结果说明：

- rollout 使用的推理对象本身是活的；
- 它当前参数一旦被直接改坏，生成会立刻变坏。

### 6.2 直接改 FSDP actor，本体 direct generate 也会变化

我又做了另一个探针：

- 把 FSDP actor 当前持有的参数全部清零；
- 不经过 rollout engine，直接调用 actor 自己的 `generate`。

输出如下：

```json
{
  "tokens_changed": true,
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be flat, with no significant hills or mountains visible.\n\nStep 3: Conclude. Based on the visual<|im_end|>",
  "after_text": "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<|im_end|>"
}
```

这个结果说明：

- 我对 actor 的极端扰动是有效的；
- FSDP actor 当前前向真正用到的那份参数，确实已经被改坏了；
- 因此“actor 其实没被改到”的解释站不住。

### 6.3 但改完 actor 再走 `update_engine_weights(actor)`，rollout 输出却不变

接下来是最关键的 probe：

1. 把 FSDP actor 全部清零；
2. 调用 `update_engine_weights(actor)`；
3. 再通过 separate rollout actor 跑 `gather_and_generate`。

输出如下：

```json
{
  "sample_name": "vqa_flatness",
  "before_len": 64,
  "after_len": 64,
  "tokens_changed": false,
  "mutate_target": "actor",
  "mutate_mode": "all_zero",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "sync_stats": {
    "total_s": 35.2631,
    "copy_state_s": 35.2526
  }
}
```

这条证据的逻辑含义非常强：

1. actor 自己已经被改坏；
2. inference engine 自己如果被改坏，rollout 会立刻变；
3. 但 actor 改坏后，通过 `update_engine_weights(actor)` 同步到 rollout actor，rollout 却完全不变。

也就是说：

- **问题不是 rollout engine 根本不能生成；**
- **问题也不是 actor 根本没变；**
- **问题出在 actor -> rollout actor 这条同步链。**

---

## 七、为什么这更像同步语义问题，而不是“训练更新太小”

如果只看线上曲线，确实有两种解释：

1. 训练没动；
2. 训练动了，但 eval 没吃到。

但加上上面的离线 checkpoint 对比和最小化 probe 后，这两种解释的权重已经不一样了。

### 7.1 “训练没动”为什么越来越站不住

因为下面三条同时成立：

1. checkpoint 在按 step20 / 40 / 60 保存；
2. shard 的 SHA256 明确不同；
3. 从 step40 / step60 checkpoint 离线 decode，同一 held-out 样本的输出会变化。

这三条已经足以说明：

- actor 训练权重不是完全冻结的；
- 至少对一部分 held-out 样本，行为层面已经有变化。

### 7.2 “eval 没吃到最新权重”为什么越来越像

因为下面三条也同时成立：

1. runtime eval 打印的首个 held-out 样本文本跨多个 eval 步完全一样；
2. 直接改 rollout inference engine 会立刻改输出；
3. 直接改 actor 再调用 `update_engine_weights(actor)`，却不会改 rollout 输出。

这套证据组合起来，最合理的解释就是：

- `update_engine_weights(actor)` 这条链虽然执行了，但没有把当前 actor 的真实有效权重同步到 separate rollout actor。

---

## 八、当前最可疑的代码位置

从代码层面看，最可疑的是 separate local HF rollout actor 的同步实现。

核心逻辑如下：

```python
def _copy_local_hf_rollout_actor_state(self, src_actor: nn.Module, dst_actor: nn.Module) -> None:
    if self._separate_hf_rollout_sync_param_pairs is None or self._separate_hf_rollout_sync_buffer_pairs is None:
        (
            self._separate_hf_rollout_sync_param_pairs,
            self._separate_hf_rollout_sync_buffer_pairs,
        ) = self._build_local_hf_rollout_actor_sync_plan(src_actor, dst_actor)

    for name, src_param, dst_param in self._separate_hf_rollout_sync_param_pairs:
        src_tensor = src_param.detach()
        if src_tensor.device != dst_param.device or src_tensor.dtype != dst_param.dtype:
            src_tensor = src_tensor.to(device=dst_param.device, dtype=dst_param.dtype)
        dst_param.detach().copy_(src_tensor)

    for name, src_buffer_ref, dst_buffer in self._separate_hf_rollout_sync_buffer_pairs:
        src_buffer = src_buffer_ref.detach()
        if src_buffer.device != dst_buffer.device or src_buffer.dtype != dst_buffer.dtype:
            src_buffer = src_buffer.to(device=dst_buffer.device, dtype=dst_buffer.dtype)
        dst_buffer.detach().copy_(src_buffer)
```

以及：

```python
def _sync_separate_hf_rollout_actor(self, actor: nn.Module) -> None:
    ...
    copy_state_t0 = time.time()
    self._copy_local_hf_rollout_actor_state(actor, self.inference_engine)
    copy_state_s = time.time() - copy_state_t0
    ...
    if keep_on_gpu:
        torch.cuda.synchronize()
        torch.distributed.barrier()
        self.inference_engine_status = EngineStatus.WAKEUP
```

入口则是：

```python
def update_engine_weights(self, actor):
    ...
    if self.inference_engine_type == "hf":
        if self._uses_separate_hf_rollout_actor():
            self._sync_separate_hf_rollout_actor(actor)
```

当前最可疑的问题点是：

- `src_param` / `dst_param` 在 FSDP `DTensor` 场景下，`detach().copy_()` 的语义可能并没有把真正参与 forward 的那份有效参数内容同步过去；
- 或者 source / destination 虽然名字一一对应，但当前复制到的对象并不是 rollout generate 实际使用的 live storage；
- 又或者 `DTensor` 在这里需要的是显式 local shard / full state materialization，而不是当前这种直接 `copy_`。

注意，这里最关键的一点不是“代码看起来对不对”，而是：

- **最小化 probe 已经说明这段同步的行为效果不对。**

---

## 九、现在我对问题的分析

### 9.1 我现在最认可的解释

当前我最认可的解释是：

- 训练 actor 确实在变；
- 真实 checkpoint 也确实在变；
- 但是 runtime eval 复用的是 separate local HF rollout actor；
- 这条 rollout actor 的权重同步没有正确反映当前 actor 的最新权重；
- 所以 eval 每次实际上都在用 stale rollout 权重做生成；
- 因而固定 held-out + greedy eval 的文本和指标全都变成了一条死线。

### 9.2 为什么我现在不再把主要怀疑放在“训练不收敛”

因为“不收敛”和“完全不抖”不是一回事。

如果只是训练效果不好，常见表现应该是：

- 指标有波动但不提升；
- 样本输出偶尔变化但总体不稳定；
- 甚至 reward / acc 在某些阶段变差。

但现在看到的是：

- held-out 样本的原始生成文本跨多个 eval step 逐字不变；
- 聚合指标跨多个 eval step 逐值不变；
- 而离线 checkpoint decode 又已经证明真实权重行为会变。

这更像“runtime eval 没有真正接上最新权重”，而不是“模型学不会”。

### 9.3 还没有完全闭环的地方

虽然当前结论已经很强，但还差两步可以进一步把锅钉死：

1. 现在最强的最小化 probe 是单卡 FSDP 环境下做的。
   - 它已经足够说明同步语义有问题；
   - 但还没在真实 8 卡 `torchrun` 拓扑下复跑。
2. 现在是用“输出有无变化”来判断同步是否生效。
   - 还没把每个关键参数在 sync 前后逐项做 local shard checksum。

这两步补上之后，根因就能从“高度怀疑”变成“几乎板上钉钉”。

---

## 十、下一步打算继续排查的部分

下一步我建议按下面顺序继续做，不要再大面积发散。

### 10.1 在真实 8 卡 `torchrun` 拓扑下复跑同样的最小化同步 probe

目标：

- 证明单卡最小化复现到的问题，在真实训练拓扑下同样存在。

要验证的点：

1. 直接改 rollout inference engine，输出应立刻变化。
2. 直接改 actor，本体 direct generate 应变化。
3. 改 actor 后走 `update_engine_weights(actor)`，rollout 输出是否仍然不变。

如果 8 卡下也复现，那么就可以非常强地说：

- 线上训练的 eval flatness 就是同步链问题，而不是单卡 probe 的偶然现象。

### 10.2 对关键参数做 sync 前后的 checksum / signature 对比

目标：

- 不再只看文本输出，而是直接看参数有没有真的同步过去。

优先对比的参数：

- `model.language_model.lm_head.weight`
- 几个 decoder layer 的 `q_proj.weight`
- 几个 decoder layer 的 `down_proj.weight`
- vision tower 中至少一个代表性参数

想确认的事情：

1. actor 参数在变；
2. rollout 参数在 sync 前不变；
3. 调用 `_copy_local_hf_rollout_actor_state(...)` 后，rollout 参数到底有没有跟 actor 对齐。

### 10.3 明确 `DTensor.copy_()` 在这个场景下的实际语义

目标：

- 查清当前同步代码到底是在 copy：
  - local shard；
  - global view；
  - 还是一个不参与 forward 的包装视图。

重点怀疑：

- `src_param.detach()` 和 `dst_param.detach()` 都是 `DTensor`；
- 在 FSDP 的 fully_shard 后，这样直接 `copy_` 未必等于把当前可生成的 live state 从 actor 同步到 rollout。

### 10.4 在主训练链上加一次最小日志埋点

如果后续开始改代码前还想再稳一点，我建议临时加最小日志：

1. 每次 `update_engine_weights(actor)` 前后，对固定参数打一个短 checksum。
2. eval 开始前，对 rollout actor 同一参数再打一个 checksum。
3. 再对固定 held-out 样本打一个短 hash。

这样可以直接把线上 run 的证据链串起来：

- actor checksum 变了；
- rollout checksum 没变；
- eval sample hash 不变。

这会是最干净的一套线上闭环证据。

---

## 十一、最终一句话版本

如果只用一句话概括当前诊断结果，那就是：

- **`pjctfbbs` 的 eval 完全不动，不是因为模型没训练，而是因为 runtime eval 大概率一直在用 stale 的 separate local HF rollout 权重；问题最像出在 `update_engine_weights(actor)` 的 FSDP / DTensor 同步语义上。**

---

## 十二、后续解决 Checklist

### 12.1 继续检查

- [ ] 在真实 8 卡 `torchrun` 拓扑下复跑最小化同步 probe，确认单卡复现不是偶然现象。
- [ ] 对 `model.language_model.lm_head.weight` 做 sync 前后的 local shard checksum，对比 actor 与 rollout actor 是否真的一致。
- [ ] 对 1 到 2 个 decoder layer 的代表性参数做同样 checksum，对比问题是否只出现在 `lm_head`，还是更普遍地存在于全部参数同步。
- [ ] 对 vision tower / aligner 各选一个代表性参数做 checksum，确认多模态部分是否同样存在同步失效。
- [ ] 明确 `src_param.detach()` / `dst_param.detach()` 在当前 FSDP `DTensor` 场景下拿到的到底是什么视图，确认是不是 copy 到了错误对象。
- [ ] 确认 rollout actor 在 `setup_inference_engine(...)` 后实际参与 `generate` 的 live module 与 `named_parameters()` 遍历出来的参数对象是不是同一套存储。
- [ ] 确认 `keep_on_gpu=True` 分支下，`torch.cuda.synchronize()` + `barrier()` 之后是否还缺少额外的 refresh / materialize 步骤。
- [ ] 把最小化 probe 再扩一版，直接在 sync 前后打印固定参数的短 signature，避免只靠文本输出来推断。

### 12.2 可以做的修复方向

- [ ] 把当前 `dst_param.detach().copy_(src_tensor)` 的同步逻辑替换成显式的 local shard copy，避免依赖 `DTensor.copy_` 的隐式语义。
- [ ] 如果 local shard copy 仍然不稳定，改成显式 gather 到 full state 后再同步到 rollout actor，先保证语义正确，再回头优化性能。
- [ ] 如果发现 source / destination 参数对象不是 live storage，改成沿着 rollout `generate` 实际使用的 module 路径拿参数，而不是只按 `named_parameters()` 对齐。
- [ ] 在 `update_engine_weights(actor)` 中增加一个可开关的 debug assert：同步后抽查若干关键参数的 checksum，不一致就直接报错。
- [ ] 在 runtime eval 前增加一个轻量 guardrail：打印固定参数短 hash 和固定 held-out 样本输出 hash，后续一眼就能看出是否又回到 stale 状态。
- [ ] 如果 separate local HF rollout actor 这条链短期内难以修稳，先准备一个保底修复方案：eval 临时直接复用 actor 本体做 local HF generate，不走 separate rollout actor，同步保证正确性。
- [ ] 如果 separate rollout 只在 eval 上出问题，也可以考虑训练 rollout 继续用 separate actor，但 runtime eval 切到 actor-direct path，先把监控可信度恢复。

### 12.3 修复后验证

- [ ] 修复后重新跑最小化 probe，验证“改 actor 后 sync，rollout 输出会跟着变化”。
- [ ] 修复后重新跑离线 held-out 对比，确认 runtime eval 的首样本文本会与对应 checkpoint decode 一致或至少同步变化。
- [ ] 修复后做一个极短训练 smoke run，确认 eval 指标不再是严格常数。
- [ ] 修复后检查 `copy_state_s`、显存占用、生成耗时，确认修复没有把同步成本推到不可接受的程度。
- [ ] 修复后保留一轮带 debug 日志的 run，确认线上链路证据闭环，然后再把额外 debug 日志关掉。

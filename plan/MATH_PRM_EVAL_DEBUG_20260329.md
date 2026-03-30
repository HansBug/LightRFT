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
4. **separate local HF rollout actor 的“参数 copy”本身不是完全失效。**
   - 单卡和真实 8 卡 `torchrun` 下，`lm_head` / decoder / vision / aligner 的 local shard signature 都能在 sync 后变化；
   - `named_parameters()`、module live attribute、`detach()` 在当前检查的参数上都指向同一份 local storage。
5. **更具体的问题已经收敛到 `hf_separate_rollout_keep_on_gpu=True` 分支缺少 refresh / materialize。**
   - 仅调用 `update_engine_weights(actor)` 后，固定样本 rollout 输出仍可能完全不变；
   - 但只要显式做一次 `reload_model(self.inference_engine)`，同样的同步结果就会立刻体现在 rollout 输出上；`offload + reload` 也能生效，但已经不是最小必要动作；
   - 仅把 `self.inference_engine_status` 改成 `SLEEPED` 并不会让新权重生效；
   - `keep_on_gpu=False` 的对照组天然会在下次 generate 前走 wakeup / reload，因此输出会立刻变化。

所以当前最强判断不是“模型没训练”，而是：

- **runtime eval 大概率没有真正吃到最新训练 actor 的权重。**
- **更准确地说，问题最像出在 `update_engine_weights(actor)` 之后，`keep_on_gpu=True` 这条 local HF rollout generate 路径没有把已同步的新权重 materialize 到实际生成所用的 GPU 态。**

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

- `src_param` / `dst_param` 的 `detach().copy_()` 从参数 signature 上看是生效的，但 `keep_on_gpu=True` 分支里，copy 完成之后并没有显式 refresh / reload rollout actor；
- 因而当前更像是：copy 写到了 rollout actor 持有的那份参数对象上，但 local HF generate 实际吃到的 GPU 态没有被重新 materialize；
- 反过来，`keep_on_gpu=False` 时下次 generate 会走 wakeup / reload，所以这条对照组能正常看到新参数。

注意，这里最关键的一点不是“代码看起来对不对”，而是：

- **最小化 probe 已经说明：问题不是“copy 完全失败”，而是“keep_on_gpu=True` 下 copy 之后 generate 仍然可能看不到新值”。**

---

## 九、现在我对问题的分析

### 9.1 我现在最认可的解释

当前我最认可的解释是：

- 训练 actor 确实在变；
- 真实 checkpoint 也确实在变；
- runtime eval 复用的是 separate local HF rollout actor；
- separate rollout actor 的参数 copy 本身是成功的；
- 但在 `hf_separate_rollout_keep_on_gpu=True` 下，`update_engine_weights(actor)` 后只做 `torch.cuda.synchronize()` + `barrier()` 还不够；
- rollout generate 很可能继续复用旧的 GPU materialization；
- 只有在 `offload + reload` 之后，新参数才会真正反映到输出上；
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

### 9.3 现在已经补完的关键闭环

这轮补充检查已经把之前还缺的两步补上了：

1. 真实 8 卡 `torchrun` 拓扑已经复跑。
2. 关键参数的 local shard checksum / signature 已经逐项对过。

补完之后，根因比之前更清楚了：

- 不是“`update_engine_weights(actor)` 完全没把参数写过去”；
- 而是“参数已经写过去了，但 `keep_on_gpu=True` 这条 generate 路径没有把新参数 refresh 到实际生成所用的 GPU 态”。

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

## 十一、补充检查：问题进一步收敛到 `keep_on_gpu=True` 缺 refresh

### 11.1 `detach()` / live storage 不是这次问题的主因

单卡 probe 里，`actor` 和 `rollout_actor` 的关键参数都呈现同样的关系：

```json
{
  "model.language_model.lm_head.weight": {
    "named_is_live_obj": true,
    "named_local_ptr_eq_live": true,
    "named_detach_ptr_eq_named": true,
    "live_detach_ptr_eq_live": true
  },
  "model.language_model.model.layers.0.self_attn.q_proj.weight": {
    "named_is_live_obj": true,
    "named_local_ptr_eq_live": true,
    "named_detach_ptr_eq_named": true,
    "live_detach_ptr_eq_live": true
  }
}
```

我又把同样的关系检查放到真实 8 卡 `torchrun` 的 rank0 上复核，结果仍然是：

```text
8gpu rank0 summary
actor_relations True
rollout_relations True
```

这说明：

- 目前检查到的关键参数上，`named_parameters()`、module live attribute、`detach()` 并没有明显脱节；
- 问题不再像之前那样指向“copy 到了错误对象”。

### 11.2 单卡 `keep_on_gpu=False` 对照组：参数变，输出也立刻变

单卡 `keep_on_gpu=False` 的 audit 同时做了参数 signature 和 generate 对照。

关键参数在 sync 后都变成了同一个全零 hash：

```text
single-card, keep_on_gpu=false
lm_head: baseline=23f00ebb3d91b639 -> post_sync=4fe7b59af6de3b66
layer0.q_proj: baseline=9d0fc8e9da63f4db -> post_sync=4fe7b59af6de3b66
layer27.down_proj: baseline=79635bede86ef493 -> post_sync=4fe7b59af6de3b66
vision.patch_embed: baseline=1f99b9fabb2a08b8 -> post_sync=4fe7b59af6de3b66
aligner.layers.1: baseline=0432e5c0ae3aceb4 -> post_sync=4fe7b59af6de3b66
```

对应 generate 输出也立刻变化：

```json
{
  "keep_rollout_on_gpu": false,
  "sync_stats": {
    "total_s": 30.6783,
    "keep_on_gpu": false,
    "actor_offloaded": true,
    "copy_state_s": 10.0303
  },
  "rollout_generate_before": {
    "text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>"
  },
  "rollout_generate_after": {
    "text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>"
  }
}
```

这说明：

- offload 分支下，参数 sync 和 generate 行为是一致的；
- 这里只要参数被改坏，下一次 rollout 就会立刻反映出来。

### 11.3 `keep_on_gpu=True`：不 refresh 时，参数已经变了，但输出仍然不变

我先把之前那个“无 refresh”的旧 probe 按原命令重新跑了一遍：

```json
{
  "tokens_changed": false,
  "mutate_target": "actor",
  "mutate_mode": "all_zero",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "sync_stats": {
    "total_s": 36.4599,
    "keep_on_gpu": true,
    "copy_state_s": 36.4478
  }
}
```

但另一条同样是 `keep_on_gpu=True`、且不做 refresh 的 signature probe 显示，rollout 参数其实已经变了：

```json
{
  "initial_actor_sig": {
    "sum": 4068.70166,
    "mean": 0.496668
  },
  "mutated_actor_sig": {
    "sum": 12260.701172,
    "mean": 1.496668
  },
  "pre_sync_rollout_sig": {
    "sum": 4068.70166,
    "mean": 0.496668
  },
  "post_sync_rollout_sig": {
    "sum": 12249.101562,
    "mean": 1.495252
  },
  "rollout_matches_actor_after_sync": false,
  "rollout_changed_after_sync": true,
  "sync_stats": {
    "keep_on_gpu": true,
    "copy_state_s": 33.8433
  }
}
```

这两条合在一起非常关键：

- **`keep_on_gpu=True` 下，参数 signature 已经在 sync 后变化；**
- **但如果不 refresh，固定样本 generate 仍然可能保持旧输出。**

所以现在已经不能再说“copy 没写进去”，而应该说：

- **copy 写进去了，但 generate 没看到。**

### 11.4 `keep_on_gpu=True`：最小必要 refresh 已经缩到 `reload_only`

我把 probe 扩成了 refresh matrix，分别验证：

1. `none`
2. `reload_only`
3. `force_sleep`
4. `offload_reload`

单卡下，`reload_only` 的关键结果如下：

```json
{
  "refresh_mode": "reload_only",
  "tokens_changed": true,
  "actions": [
    "reload_model(inference_engine)"
  ],
  "status_before": "WAKEUP",
  "status_after": "WAKEUP",
  "lm_head_baseline": "23f00ebb3d91b639",
  "lm_head_post_update": "23f00ebb3d91b639",
  "lm_head_post_refresh": "4fe7b59af6de3b66",
  "lm_head_post_generate": "4fe7b59af6de3b66",
  "sync_stats": {
    "total_s": 52.0542,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 52.0427,
    "prepare_s": 0.0105,
    "sync_clear_s": 0.001
  },
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>"
}
```

这条结果把“最小必要动作”缩得很明确：

- **不需要先 `offload`；**
- **单独 `reload_model(self.inference_engine)` 就足够让新权重生效。**

### 11.5 `keep_on_gpu=True`：单独把状态切到 `SLEEPED` 不够

同一组单卡 matrix 里，`force_sleep` 的结果如下：

```json
{
  "refresh_mode": "force_sleep",
  "tokens_changed": false,
  "actions": [
    "inference_engine_status=SLEEPED"
  ],
  "status_before": "WAKEUP",
  "status_after": "SLEEPED",
  "lm_head_baseline": "23f00ebb3d91b639",
  "lm_head_post_update": "23f00ebb3d91b639",
  "lm_head_post_refresh": "23f00ebb3d91b639",
  "lm_head_post_generate": "23f00ebb3d91b639",
  "sync_stats": {
    "total_s": 51.0959,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 51.0843,
    "prepare_s": 0.0101,
    "sync_clear_s": 0.0015
  },
  "after_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>"
}
```

这条结果直接推翻了我之前更弱的猜测：

- **问题不是“只要把 status 打回 `SLEEPED` 就行”；**
- **`SLEEPED` 本身不是修复，真正起作用的是后续的 `reload_model(self.inference_engine)`。**

### 11.6 `keep_on_gpu=True`：`offload + reload` 当然也能生效，但它不是最小必要动作

之前 audit 里验证过更强的 refresh：

1. `offload_model(self.inference_engine)`
2. `reload_model(self.inference_engine)`

对应关键片段如下：

```json
{
  "refresh_stats": {
    "performed_manual_offload_reload": true
  },
  "sync_stats": {
    "total_s": 33.7189,
    "keep_on_gpu": true,
    "copy_state_s": 33.7078
  },
  "rollout_generate_before": {
    "text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>"
  },
  "rollout_generate_after": {
    "text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>"
  }
}
```

所以现在这部分判断应该写成：

- `offload + reload` 能生效；
- 但**它已经不是最小必要动作**；
- 更直接、更贴近修复点的最小动作是：**在 `keep_on_gpu=True` 路径里显式 `reload_model(self.inference_engine)`。**

### 11.7 真实 8 卡 `torchrun` 拓扑下同样成立

我又把同样的 matrix 放到真实 8 卡 `torchrun` 下重跑了一次。

`reload_only` 的关键结果如下：

```json
{
  "refresh_mode": "reload_only",
  "tokens_changed": true,
  "actions": [
    "reload_model(inference_engine)"
  ],
  "status_before": "WAKEUP",
  "status_after": "WAKEUP",
  "lm_head_baseline": "23f00ebb3d91b639",
  "lm_head_post_update": "23f00ebb3d91b639",
  "lm_head_post_refresh": "4fe7b59af6de3b66",
  "lm_head_post_generate": "4fe7b59af6de3b66",
  "sync_stats": {
    "total_s": 3.1234,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 3.1117,
    "prepare_s": 0.0099,
    "sync_clear_s": 0.0018
  },
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>"
}
```

`force_sleep` 的关键结果如下：

```json
{
  "refresh_mode": "force_sleep",
  "tokens_changed": false,
  "actions": [
    "inference_engine_status=SLEEPED"
  ],
  "status_before": "WAKEUP",
  "status_after": "SLEEPED",
  "lm_head_baseline": "23f00ebb3d91b639",
  "lm_head_post_update": "23f00ebb3d91b639",
  "lm_head_post_refresh": "23f00ebb3d91b639",
  "lm_head_post_generate": "23f00ebb3d91b639",
  "sync_stats": {
    "total_s": 3.4714,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 2.1861,
    "prepare_s": 0.0097,
    "sync_clear_s": 1.2757
  },
  "after_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>"
}
```

这条证据非常关键，因为它说明：

- 这不是单卡 probe 的偶然现象；
- 真实 8 卡训练拓扑下，**`reload_only` 仍然足够，`force_sleep` 仍然不够。**

### 11.8 cached `src_param` 不是陈旧引用

为了排除“sync plan 缓存里拿着旧 actor 参数引用”这条怀疑，我单独做了一个静态探针。

结果如下：

```text
same_object True
cached_src_ptr 134411202854912
current_named_ptr 134411202854912
```

这说明：

- `_separate_hf_rollout_sync_param_pairs` 里缓存的 `src_param`；
- 和当前 `dict(actor.named_parameters())[name]`；
- **在至少 `model.language_model.lm_head.weight` 这条关键参数上就是同一个对象。**

我又补了一个 actor 侧 sanity check，确认 `for p in actor.parameters(): ...` 的修改并不会绕开 `named_parameters()` 看到的那份存储：

```text
named_hash_before 23f00ebb3d91b639
named_ptr_in_parameters True
named_hash_after_zero_module_params 4fe7b59af6de3b66
```

以及：

```text
named_hash_before 23f00ebb3d91b639
named_hash_after_zero_named 4fe7b59af6de3b66
```

所以现在至少可以明确排除两件事：

- 不是 `module.parameters()` 改了，但 `named_parameters()` 没改；
- 也不是 sync plan 的 `src_param` 简单地拿着一份过期 actor 引用。

更深一层的问题如果还要继续往下追，应该优先怀疑 destination / materialize 语义，而不是 source 引用失效。

### 11.9 为什么这能解释线上 eval flatness

现在回头看 `strategy_base.py` 里的分支逻辑，问题就更像了。

`update_engine_weights(actor)` 最终会走到：

```python
if keep_on_gpu:
    torch.cuda.synchronize()
    torch.distributed.barrier()
    self.inference_engine_status = EngineStatus.WAKEUP
else:
    self.inference_engine_status = EngineStatus.SLEEPED
    self.sync_and_clear_cache()
```

而 `keep_on_gpu=False` 的下一次 generate，会走 wakeup 分支里的：

```python
if self._uses_separate_hf_rollout_actor():
    if self.rollout_train_actor_is_on_gpu:
        self.offload_model(self.rollout_train_actor)
        self.rollout_train_actor_is_on_gpu = False
    self.reload_model(self.inference_engine)
    self._prepare_separate_hf_rollout_actor_for_generation()
```

也就是说：

- `keep_on_gpu=False` 这条链天然会 reload rollout actor；
- `keep_on_gpu=True` 这条链目前只做 synchronize / barrier，不做 reload；
- 即使手工把 `self.inference_engine_status` 改成 `SLEEPED`，`wakeup_inference_engine()` 在 `keep_on_gpu=True` 分支里也只是 `_prepare_separate_hf_rollout_actor_for_generation()`，不会真正执行 `reload_model(self.inference_engine)`；
- 而我的最小化 probe 已经证明：
  - 仅 synchronize / barrier 不足以让 generate 看见新值；
  - `reload_model(self.inference_engine)` 就足够让新值立刻生效；
  - 单独 `self.inference_engine_status = SLEEPED` 不足以让新值生效；
  - `offload + reload` 当然也能生效，但不是最小必要动作。

所以当前最合理的解释已经进一步收敛为：

- **线上 runtime eval flat，不是因为 actor 没训练、也不是因为 checkpoint 没变，而是因为 `hf_separate_rollout_keep_on_gpu=True` 这条链在 `update_engine_weights(actor)` 之后缺少 refresh / materialize，导致 eval 长时间复用 stale 的 GPU-resident rollout weights。**

---

## 十二、最终一句话版本

如果只用一句话概括当前诊断结果，那就是：

- **`pjctfbbs` 的 eval 完全不动，不是因为模型没训练，而是因为 `hf_separate_rollout_keep_on_gpu=True` 的 separate local HF rollout actor 在 `update_engine_weights(actor)` 后没有做足够的 refresh / materialize；参数 signature 已经变了，但 runtime eval 仍可能继续吃旧的 GPU rollout 权重，显式 `reload_model(self.inference_engine)` 后新权重才立刻生效，而单独把状态切成 `SLEEPED` 并不够。**

---

## 十三、后续解决 Checklist

### 13.1 继续检查

- [x] 在真实 8 卡 `torchrun` 拓扑下复跑最小化同步 probe，确认单卡复现不是偶然现象。
- [x] 对 `model.language_model.lm_head.weight` 做 sync 前后的 local shard checksum，对比 actor 与 rollout actor 是否真的一致。
- [x] 对 1 到 2 个 decoder layer 的代表性参数做同样 checksum，对比问题是否只出现在 `lm_head`，还是更普遍地存在于全部参数同步。
- [x] 对 vision tower / aligner 各选一个代表性参数做 checksum，确认多模态部分是否同样存在同步失效。
- [x] 明确 `src_param.detach()` / `dst_param.detach()` 在当前 FSDP `DTensor` 场景下拿到的到底是什么视图，确认是不是 copy 到了错误对象。
- [x] 确认 rollout actor 在 `setup_inference_engine(...)` 后实际参与 `generate` 的 live module 与 `named_parameters()` 遍历出来的参数对象是不是同一套存储。
- [x] 确认 `keep_on_gpu=True` 分支下，`torch.cuda.synchronize()` + `barrier()` 之后是否还缺少额外的 refresh / materialize 步骤。
- [x] 把最小化 probe 再扩一版，直接在 sync 前后打印固定参数的短 signature，避免只靠文本输出来推断。
- [x] 直接在同一个最小化脚本里同时打印：`post_update` / `post_refresh` 参数 hash，以及 refresh 前后样本输出，避免以后再靠多份 probe 拼结论。
- [ ] 沿 `runtime eval -> gather_and_generate -> engine_generate_local` 主链补最小日志，确认线上 eval 当次确实走的是 `keep_on_gpu=True` 且没有任何 `reload_model(self.inference_engine)`。
- [x] 进一步缩小 refresh 的最小必要动作，确认到底必须 `offload + reload`，还是单独 `reload_model(self.inference_engine)` / 其他 materialize 动作就够。
- [ ] 继续比较 cached `dst_param` 与当前 `dict(strategy.inference_engine.named_parameters())` 是否始终指向同一 live storage，确认问题是否进一步落在 destination / materialize 语义上。

### 13.2 可以做的修复方向

根据当前证据，优先级最高的方向已经不是重写 `DTensor.copy_`，而是先把 `keep_on_gpu=True` 分支的 refresh / materialize 语义补正确；`copy_` 重写退到次一级备选。

- [ ] 优先尝试最小修复：在 `keep_on_gpu=True` 分支的 sync 结束后显式调用 `reload_model(self.inference_engine)`，不要只做 `torch.cuda.synchronize()` + `barrier()`。
- [ ] 如果想复用现有 wakeup 路径，不要只把 `self.inference_engine_status` 改成 `SLEEPED`；probe 已证明这一步单独无效，必须保证后续真的执行 `reload_model(self.inference_engine)`。

- [ ] 把当前 `dst_param.detach().copy_(src_tensor)` 的同步逻辑替换成显式的 local shard copy，避免依赖 `DTensor.copy_` 的隐式语义。
- [ ] 如果 local shard copy 仍然不稳定，改成显式 gather 到 full state 后再同步到 rollout actor，先保证语义正确，再回头优化性能。
- [ ] 如果发现 source / destination 参数对象不是 live storage，改成沿着 rollout `generate` 实际使用的 module 路径拿参数，而不是只按 `named_parameters()` 对齐。
- [ ] 在 `update_engine_weights(actor)` 中增加一个可开关的 debug assert：同步后抽查若干关键参数的 checksum，不一致就直接报错。
- [ ] 如果上面的最小修复不够，再尝试在 `keep_on_gpu=True` 分支内显式调用 rollout actor refresh / reload，而不是只做 `torch.cuda.synchronize()` + `barrier()`。
- [ ] 在 runtime eval 前增加一个轻量 guardrail：打印固定参数短 hash 和固定 held-out 样本输出 hash，后续一眼就能看出是否又回到 stale 状态。
- [ ] 如果 separate local HF rollout actor 这条链短期内难以修稳，先准备一个保底修复方案：eval 临时直接复用 actor 本体做 local HF generate，不走 separate rollout actor，同步保证正确性。
- [ ] 如果 separate rollout 只在 eval 上出问题，也可以考虑训练 rollout 继续用 separate actor，但 runtime eval 切到 actor-direct path，先把监控可信度恢复。

### 13.3 修复后验证

- [x] 修复后重新跑最小化 probe，验证“改 actor 后 sync，rollout 输出会跟着变化”。
- [x] 修复后专门验证 `keep_on_gpu=True` 下“不做手工 refresh 也会立即反映新权重”。
- [ ] 修复后重新跑离线 held-out 对比，确认 runtime eval 的首样本文本会与对应 checkpoint decode 一致或至少同步变化。
- [ ] 修复后做一个极短训练 smoke run，确认 eval 指标不再是严格常数。
- [x] 修复后检查 `copy_state_s`、显存占用、生成耗时，确认没有把同步成本推到不可接受的程度。
- [ ] 修复后保留一轮带 debug 日志的 run，确认线上链路证据闭环，然后再把额外 debug 日志关掉。

---

## 十四、本轮修复计划

这轮不再继续只做诊断，而是按“最小修复 + 定向验收 + 回归对照”的顺序推进。

当前准备先落的最小修复是：

- 在 `hf_separate_rollout_keep_on_gpu=True` 的 separate local HF rollout actor 同步路径里，不再只做 `torch.cuda.synchronize()` + `barrier()`；
- 在 `_copy_local_hf_rollout_actor_state(actor, self.inference_engine)` 之后，显式执行 `reload_model(self.inference_engine)`，让 rollout generate 真正 materialize 到最新权重；
- 不采用“只把 `self.inference_engine_status` 设回 `SLEEPED`”的方案，因为 probe 已经证明这一步单独无效。

### 14.1 本轮执行 Checklist

* [x] 先在文档里固定这轮修复目标、假设和验收标准，避免边改边漂移。
* [x] 在 `strategy_base.py` 的 `keep_on_gpu=True` 分支里落最小修复，优先只补显式 `reload_model(self.inference_engine)`，不同时重写 `DTensor.copy_` 语义。
* [x] 跑单卡最小 probe，确认 `keep_on_gpu=True` 下即使不手工 refresh，`update_engine_weights(actor)` 之后 rollout 输出也会立刻变化。
* [x] 跑真实 8 卡最小 probe，确认多卡拓扑下也同样不需要额外手工 refresh。
* [x] 跑 `keep_on_gpu=False` 对照组，确认原本就正常的 offload / wakeup 路径没有被这次修复打坏。
* [x] 检查修复前后 `sync_stats`、显存和耗时，确认没有引入明显异常。
* [x] 本轮没有额外补 runtime eval 主链日志；现有最小 probe 已直接覆盖 `update_engine_weights -> gather_and_generate -> engine_generate_local` 的关键链路，足以判定修复是否命中。
* [x] 把修复结果、日志片段、是否验通、是否发现副作用全部补回本文档。
* [x] 最终提交并 push 修复代码和更新后的排查记录。

### 14.2 本轮验收标准

本轮只有同时满足下面几条，才算“修通”：

* [x] `keep_on_gpu=True` 下，`update_engine_weights(actor)` 后不再需要手工 `reload_only` / `offload_reload`，固定样本输出会直接变化。
* [x] 单独把状态切成 `SLEEPED` 仍然不是必要条件，说明修复点确实命中 materialize 本身，而不是偶然绕路。
* [x] 真实 8 卡 `torchrun` 下复现同样结论，不是单卡偶然现象。
* [x] `keep_on_gpu=False` 路径仍然保持原行为，没有功能回退。
* [x] Python 语法、最小 probe、目标链路验证都通过，没有新的明显报错或 OOM。

### 14.3 本轮修复结果

这轮修复真正改动的点很小，但命中了问题本体：

- 在 `keep_on_gpu=True` 的 separate local HF rollout actor 同步路径里，原来只做 `copy -> synchronize/barrier -> WAKEUP`；
- 现在改成 `copy -> reload_model(self.inference_engine) -> prepare -> synchronize/barrier -> WAKEUP`；
- 这条修改直接对应了之前 probe 证明过的最小必要动作。

问题点现在也可以写得更具体：

- 出问题的位置不是 `cached src_param`，也不是训练 actor 没更新；
- 真正的问题在于 `keep_on_gpu=True` 时，`_sync_separate_hf_rollout_actor(...)` 复制完状态后没有显式 rematerialize rollout actor；
- 而 `wakeup_inference_engine()` 在 `keep_on_gpu=True` 分支又会直接早退，不会替你补这次 reload；
- 所以旧的 GPU-resident rollout 权重会继续被 generate 复用，导致 eval 长时间看起来完全不动。

修复后的单卡 `keep_on_gpu=True` 最小 probe 结果如下：

```json
{
  "refresh_mode": "none",
  "tokens_changed": true,
  "lm_head_post_update": "4fe7b59af6de3b66",
  "lm_head_post_refresh": "4fe7b59af6de3b66",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>",
  "sync_stats": {
    "total_s": 0.4604,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "rollout_reloaded": true,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 0.3212,
    "reload_rollout_s": 0.1292,
    "prepare_s": 0.0094,
    "sync_clear_s": 0.0007
  }
}
```

这条结果说明：

- 修复后即使 `refresh_mode = "none"`，输出也已经直接变化；
- `lm_head_post_update` 和 `lm_head_post_refresh` 都是新 hash，说明不再依赖额外手工 refresh；
- 新增的 `rollout_reloaded = true` / `reload_rollout_s = 0.1292` 也表明这次 materialize 已经在同步链路内完成。

修复后的真实 8 卡 `torchrun` probe 结果如下：

```json
{
  "refresh_mode": "none",
  "tokens_changed": true,
  "lm_head_post_update": "4fe7b59af6de3b66",
  "lm_head_post_refresh": "4fe7b59af6de3b66",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>",
  "sync_stats": {
    "total_s": 0.4792,
    "keep_on_gpu": true,
    "actor_offloaded": false,
    "rollout_offloaded": false,
    "rollout_reloaded": true,
    "offload_actor_s": 0.0,
    "offload_rollout_s": 0.0,
    "copy_state_s": 0.3358,
    "reload_rollout_s": 0.1344,
    "prepare_s": 0.0081,
    "sync_clear_s": 0.0009
  }
}
```

这条结果说明：

- 修复不只是单卡偶然现象；
- 真实 8 卡训练拓扑下，`keep_on_gpu=True` 现在也不需要额外 `reload_only` / `offload_reload`；
- materialize 语义已经被真正补进主同步路径。

`keep_on_gpu=False` 的单卡回归对照结果如下：

```json
{
  "keep_rollout_on_gpu": false,
  "lm_head_baseline": "23f00ebb3d91b639",
  "lm_head_post_sync": "4fe7b59af6de3b66",
  "before_text": "Step 1: Observe the image. The image shows a landscape with a plane flying in the sky and another plane on the ground.\n\nStep 2: Analyze the landscape. The ground appears to be mostly flat, with some variations in elevation.\n\nStep 3: Determine if the landscape is flat. Based<|im_end|>",
  "after_text": "!!!!\"!!!#!!!$!!!%!!!&!!!'!!!(!!!)!!!*!!!+!!!,!!!-!!!.!!!/!!!0!!<|im_end|>",
  "sync_stats": {
    "total_s": 31.2658,
    "keep_on_gpu": false,
    "actor_offloaded": true,
    "rollout_offloaded": false,
    "rollout_reloaded": false,
    "offload_actor_s": 20.981,
    "offload_rollout_s": 0.0,
    "copy_state_s": 10.2742,
    "reload_rollout_s": 0.0,
    "prepare_s": 0.0101,
    "sync_clear_s": 0.0005
  }
}
```

这条对照组说明：

- 原本正常的 offload / wakeup 路径没有被这次修复打坏；
- `keep_on_gpu=False` 仍然维持原先行为，没有额外 reload，也没有功能回退；
- 所以这次补丁目前看是一个比较干净的定点修复，而不是通过扰动其他路径侥幸“修好”。

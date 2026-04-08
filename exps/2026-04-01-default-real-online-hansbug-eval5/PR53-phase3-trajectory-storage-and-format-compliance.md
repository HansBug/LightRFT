# PR53 Phase 3: eval 轨迹到底存没存、存在哪里，以及 actor 的格式遵循情况

本文专门回答 [PR53_FOLLOWUP_CHECKLIST.md](./PR53_FOLLOWUP_CHECKLIST.md) 里的 Phase 3 问题：

> eval轨迹存储与分析，actor回答的格式遵循情况分析

这份文档不只讲代码入口，还强制对上了当前实验的本地 log 和本地结果目录。也就是说，这里回答的不是“理论上会不会保存”，而是：

1. 这次 run 里，轨迹到底有没有真的保存。
2. 它们到底存到了哪个目录、什么文件名。
3. 保存的是 runtime eval 轨迹，还是训练/rollout 轨迹。
4. 文件里实际长什么样，能不能直接拿来做格式分析。
5. 当前这批本地样本里，`Step N:` / `†Answer:` 的格式遵循情况到底怎样。

本文聚焦的真实运行对象是：

- experiment: `default-real-online-hansbug-eval5`
- run dir: `results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638`
- log: `rft_logs/default-real-online-hansbug-eval5/node0_20260330_235638.log`

## 太长不看

当前主线不会把 runtime eval 单独存成轨迹；本地 log 能确认保存的是 checkpoint 时 replay buffer 里的 train/rollout 轨迹。这次 run 的文件真实落在 `results/.../trajectories/` 下，已有 `trajectories_step_20/40/60/80/100.json` 各 16 条，图片在同目录 `images/`。离线看这 80 条样本，格式成功率约 97.5%，但它们不是 eval 专用样本，而且部分 `reward_metrics` 在 shape mismatch 时不一定严格逐样本对齐。

## 一页结论

- **有存。** 这次 run 确实把轨迹写到了本地磁盘，不是只有代码里“理论支持”。
- **存的是训练/rollout 轨迹，不是 runtime eval 轨迹。**
- **真实路径已经在 log 和结果目录里对上：**
  - JSON：`.../trajectories/trajectories_step_20|40|60|80|100.json`
  - 图片：`.../trajectories/images/step100_exp12_sample0_img0.png` 这类文件
- **当前保存内容足够做格式分析**，因为里面有：
  - `pure_generated_text`
  - `full_sequence`
  - `image_paths`
  - `response_token_count`
  - `info.reward_metrics`
- **但边界也要说清楚：**
  - 这些不是 eval 专用轨迹，所以不能直接当作 runtime eval case study。
  - 在当前 packed / 形状不齐的场景下，部分 `reward_metrics` 在 JSON 里会保留成列表，未必严格对齐到单样本。

## 1. 这句话到底在问什么

Phase 3 实际上有两个层次：

1. **存储层面**
   - eval 期间的具体轨迹到底有没有被落盘？
   - 如果有，路径是什么，文件长什么样？
2. **分析层面**
   - actor 现在对 Stage 3 输出格式要求，到底是稳定遵循，还是经常依赖后处理兜底？

如果这两个层次不拆开，很容易出现两种混淆：

- 看到仓库里有 `trajectory_saver.py`，就误以为 runtime eval 一定有独立轨迹文件。
- 看到 W&B 的 `eval/answer_extraction_failed` 比例不高，就误以为“格式一定已经没问题”。

## 2. 当前轨迹保存机制在代码里到底挂在哪

### 2.1 保存目录是 `args.save_path/trajectories`

### 摘录：`lightrft/utils/trajectory_saver.py`

```python
if not hasattr(args, 'save_trajectories') or not args.save_trajectories:
    return None

save_dir = os.path.join(args.save_path, "trajectories")
...
return TrajectorySaver(
    save_dir=save_dir,
    tokenizer=tokenizer,
    save_images_separately=True,
    max_image_size=512,
```

这段代码把保存目录定死成：

```text
args.save_path / "trajectories"
```

而且图片默认单独保存：

- `save_images_separately=True`

所以如果这次实验真的开了 trajectory saving，文件就应该落在：

```text
results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/
```

### 2.2 触发时机不在 `evaluate()`，而在训练完成后的 `save_steps`

### 摘录：`lightrft/trainer/spmd_ppo_trainer.py`

```python
if global_steps % self.args.save_steps == 0:
    with self.profiler.section("learn/save_trajectories"):
        self.save_trajectories(global_steps)
```

```python
if self.trajectory_saver is not None and self.replay_buffer.items:
    output_path, stats = self.trajectory_saver.save_trajectories(
        experiences=self.replay_buffer.items,
        step=global_step,
        num_samples=self.num_trajectories_to_save,
        prefix="trajectories",
        compute_stats=self.args.trajectory_analysis
    )
```

这非常关键：

- `save_trajectories()` 吃的是 `self.replay_buffer.items`
- `self.replay_buffer.items` 是 rollout / train 阶段收集到的 experiences

所以当前主线保存的是：

- **训练轨迹 / rollout 轨迹**

不是：

- runtime eval 单独导出的轨迹

### 2.3 `evaluate()` 自己并不会调用 `trajectory_saver`

### 摘录：`lightrft/trainer/ppo_trainer_vl.py`

```python
# 3. CHECKPOINTING
if global_step % args.save_steps == 0:
    tag = f"global_step{global_step}"
    self._save_checkpoint(args, tag, client_states)
```

而在 math PRM trainer 这一层，eval 部分也只是记日志然后走 checkpoint：

### 摘录：`examples/math_prm/math_prm_trainer.py`

```python
for key, value in raw_eval_metrics.items():
    eval_logs[f"eval/{key}"] = value
...
if global_step % args.save_steps == 0:
    with self.profiler.phase("checkpoint"):
        with self.profiler.section("total"):
            tag = f"global_step{global_step}"
            self._save_checkpoint(args, tag, client_states)
```

这里没有任何：

- `save_eval_trajectories(...)`
- `trajectory_saver.save_trajectories(eval_...)`

之类的调用。

所以从代码链路上，已经可以先下一个明确结论：

- **当前仓库默认没有 runtime eval 专用轨迹导出。**

## 3. 这次 run 在本地 log 里能不能对上“真的保存了”

能，而且能对到非常具体。

### 3.1 先看启动参数：这次 run 确实打开了轨迹保存

### 摘录：`rft_logs/default-real-online-hansbug-eval5/node0_20260330_235638.log`

```text
Namespace(
  ...
  save_path='results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638',
  save_steps=20,
  save_trajectories=True,
  trajectory_analysis=False,
  num_trajectories_to_save=16,
  ...
)
```

这段 log 已经把三个关键事实写死了：

- `save_trajectories=True`
- `save_steps=20`
- `num_trajectories_to_save=16`

也就是说，这次 run 本来就配置成：

- 每逢 `global_step % 20 == 0`
- 从 replay buffer 抽样保存 16 条轨迹

### 3.2 再看保存日志：本地确实写出了 JSON 文件

### 摘录：同一份 `node0_20260330_235638.log`

```text
[TrajectorySaver] Saved 16 trajectories to results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_20.json
```

```text
[TrajectorySaver] Saved 16 trajectories to results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_100.json
```

这已经不是“推测会保存”，而是：

- trajectory saver 在 log 里明确说它保存了
- 文件名就是 `trajectories_step_<step>.json`

### 3.3 本地结果目录也能实物对上

我直接在当前 run 的结果目录里搜到的是：

### 摘录：`find .../trajectories -maxdepth 2`

```text
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/images
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_20.json
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_40.json
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_60.json
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_80.json
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_100.json
/data/LightRFT/results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/images/step100_exp12_sample0_img0.png
```

这一步把“存在哪里”彻底坐实了。

如果只用一句话回答用户那句“到底存在哪里了”，最准确的说法就是：

```text
存到了这次 run 的 results 目录下的 trajectories/ 子目录里，
JSON 在 trajectories_step_*.json，关联图片在 trajectories/images/。
```

## 4. 当前 run 一共存了多少、每个文件里有多少

我直接读了当前实验目录下的 5 个 trajectory JSON：

```text
trajectories_step_20.json   -> 16 条
trajectories_step_40.json   -> 16 条
trajectories_step_60.json   -> 16 条
trajectories_step_80.json   -> 16 条
trajectories_step_100.json  -> 16 条
```

总计：

```text
5 个文件 × 每个 16 条 = 80 条已保存轨迹
```

这和 log 里的：

- `save_steps=20`
- `num_trajectories_to_save=16`

是完全对上的。

## 5. 文件里到底存了什么字段

### 摘录：`lightrft/utils/trajectory_saver.py`

```python
traj_dict = {
    "global_step": step,
    "experience_index": exp_idx,
    "sample_in_exp": i,
    "full_sequence": decoded_sequences[i],
    "generated_text": generated_text,
    "pure_generated_text": pure_generated_text,
    "repeat_score": repeat_score,
    "reflection_pattern_score": reflection_pattern_score,
    "reflection_pattern_details": reflection_pattern_dict,
    "policy_entropy": policy_entropy,
    "response_token_count": response_token_count,
}
...
if hasattr(exp, 'info') and exp.info is not None:
    ...
    info_dict[key] = ...
    traj_dict["info"] = info_dict
```

### 摘录：同文件的图片保存部分

```python
if sample_images:
    traj_dict["has_images"] = True
    traj_dict["num_images"] = len(sample_images)
    if self.save_images_separately:
        image_paths = self._save_images(sample_images, step, exp_idx, i)
        traj_dict["image_paths"] = image_paths
```

所以当前 trajectory JSON 已经足够支撑 Phase 3 里最关心的分析：

- 要看格式：`pure_generated_text`
- 要看全上下文：`full_sequence`
- 要对图片 case：`image_paths`
- 要看长度：`response_token_count`
- 要看 reward / extraction 诊断：`info.reward_metrics`

## 6. 真实样例：本地 JSON 里到底长什么样

下面不是“我编的例子”，而是直接从当前 run 的 trajectory JSON 里抠出来的真实内容。

### 6.1 一个格式合规的真实样例

来源：

```text
results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_100.json
-> item 0
```

### 摘录：真实保存内容

```json
{
  "global_step": 100,
  "experience_index": 0,
  "sample_in_exp": 0,
  "pure_generated_text": "Step 1: Observe the given triangle and the line.\nStep 2: Determine the number of intersections between the line and the triangle.  The line does not intersect the triangle.\nStep 3: Count the number of intersections. There are 0 intersections.\n†Answer: 0",
  "image_paths": [
    "images/step100_exp0_sample0_img0.png"
  ],
  "response_token_count": 59,
  "reward_metrics": {
    "format_reward": [1.0, 1.0, 1.0, 1.0],
    "outcome_correct": [0.0, 0.0, 0.0, 1.0],
    "answer_tag_present": [1.0, 1.0, 1.0, 1.0],
    "answer_extraction_failed": [0.0, 0.0, 0.0, 0.0]
  }
}
```

从格式角度看，这条样本是标准的：

- 有 `Step 1/2/3`
- 有且只有一个 `†Answer:`
- 最后一行就是 `†Answer: 0`

### 6.2 一个真实的坏样例：中间混进了 `Answer 1:` / `Answer 2:`，最后才拼出 `†Answer:`

来源：

```text
results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_100.json
-> item 12
```

### 摘录：真实保存内容

```json
{
  "global_step": 100,
  "experience_index": 12,
  "sample_in_exp": 0,
  "pure_generated_text": "Step 1: Analyze the diagram, observing that shape ① is rotated about point A.\nStep 2: Observe shape ① as moving from its original position to its final position.\nStep 3: Shape ① is rotated around point A in a counterclockwise direction.\nStep 4: According to the rotation direction, shape ① is rotated 180° counterclockwise around point A.\nStep 5: Shape ① is rotated around point A.\nAnswer 1: \\boxed{①}\n\nStep 6: Analyze the diagram, observing that shape ② is rotated about point B.\nStep 7: Observe shape ② as moving from its original position to its final position.\nStep 8: Shape ② is rotated around point B in a clockwise direction.\nStep 9: According to the rotation direction, shape ② is rotated 180° clockwise around point B.\nStep 10: Shape ② is rotated around point B.\nAnswer 2: \\boxed{②}\n\nStep 11: Analyze the diagram, observing that shape ③ is rotated about point C.\nStep 12: Observe shape ③ as moving from its original position to its final position.\nStep 13: Shape ③ is rotated around point C in a clockwise direction.\nStep 14: According to the rotation direction, shape ③ is rotated 90° clockwise around point C.\nAnswer 3: \\boxed{③†Answer: ①",
  "image_paths": [
    "images/step100_exp12_sample0_img0.png"
  ],
  "response_token_count": 331
}
```

这条样本为什么是坏格式，一眼就能看出来：

- 中间冒出了 `Answer 1:` / `Answer 2:` / `Answer 3:`
- 最后虽然出现了 `†Answer:`，但它不是单独的最终答案行
- 末尾也不是标准的 `†Answer: ...`

这说明当前 actor 虽然大多数时候能遵循 Stage 3 格式，但并不是 100% 稳定。

而且这条样本对应的图片文件在本地也确实存在：

```text
trajectories/images/step100_exp12_sample0_img0.png
```

我已经实际检查过这个相对路径在 trajectory 目录下是存在的。

### 6.3 另一个真实坏样例：明显被截断，根本没走到 `†Answer:`

来源：

```text
results/default-real-online-hansbug-eval5/default-real-online-hansbug-eval5-ep10-kl0.001-lr1e-6-20260330_235638/trajectories/trajectories_step_80.json
-> item 6
```

### 摘录：真实保存内容

```text
Step 1: Rewrite the given equation into the standard quadratic equation standard.
...
Step 11: Solve for x by dividing both sides by \ln(2): x = \frac{\ln(4\sqrt{7}) + i(\pi/3)}{\ln(2)}.
Step 12: Calculate numerical values: x \approx \frac{
```

配套字段是：

```text
image_paths = ['images/step80_exp6_sample0_img0.png']
response_token_count = 511
```

这里最值得注意的是：

- `response_token_count = 511`
- 这几乎贴住了 `local_hf_max_new_tokens=512`

所以这条坏样例更像：

- **还没来得及收尾到 `†Answer:` 就撞上长度上限**

这件事和 Phase 4 的“截断 / eos / max_new_tokens”会直接连起来，但在 Phase 3 这里它已经足够说明：

- 当前格式失败并不只有“回答格式写错”
- 还有一类是“根本没自然收尾到答案行”

### 6.4 我已经把几张真实图片复制成文档 asset

为了让 Phase 3 文档本身就能直接看 case，我把当前 run 里几张真实 trajectory 图片复制到了：

```text
exps/2026-04-01-default-real-online-hansbug-eval5/assets/phase3/
```

现在这个目录下已经有：

```text
step100-exp3-sample0.png
step80-exp0-sample0.png
step20-exp8-sample0.png
step100-exp12-sample0.png
step80-exp6-sample0.png
```

下面这些图文例子，都是从这些 asset 和对应 JSON 一起摘出来的。

### 6.5 真实图片 + 对应文本 + 打分样例

先说明一个严格边界：

- 对下面前 3 个样例，我采用的是“`info.reward` 在 `final_reward` 列表里只出现一次”的 case。
- 在这种情况下，可以把同一索引位置上的 `step_score_* / outcome_correct / final_reward` 当作当前样本最可信的对应槽位。
- 对后面两个坏格式样例，由于 `final_reward` 全是 `0`，无法唯一反推对应槽位，所以我只展示 **保存态原样 metric 列表**，不伪装成精确单样本标量。

#### 样例 A：格式合规、答对、最终 reward=1.0

来源：

```text
trajectories_step_100.json -> item 3
asset -> assets/phase3/step100-exp3-sample0.png
```

![step100-exp3-sample0](assets/phase3/step100-exp3-sample0.png)

### 摘录：对应文本

```text
Step 1: Identify the type of figure. The problem involves two lines intersecting.

Step 2: Determine the number of intersection points. The two lines intersect at one point, indicated by the illustration.

Step 3: State the answer. There is 1 intersection point.

†Answer: 1
```

### 摘录：对应打分

```text
info.reward = 1.0
response_length = 64.0

可信对应槽位 = 3
step_score_min  = 0.99609375
step_score_mean = 0.99609375
step_score_last = 0.99609375
outcome_correct = 1.0
format_reward   = 1.0
final_reward    = 1.0
answer_extraction_failed = 0.0
```

这是最标准的“格式对 + 答对 + reward 满分”样本。

#### 样例 B：格式合规、答对，但因为过程问题只拿到 reward=0.5

来源：

```text
trajectories_step_80.json -> item 0
asset -> assets/phase3/step80-exp0-sample0.png
```

![step80-exp0-sample0](assets/phase3/step80-exp0-sample0.png)

### 摘录：对应文本

```text
Step 1: Simplify the complex fraction: 2/(-1+i).
...
Step 9: Verify with options:
...
Therefore, original sinlge answer to the problem is correct. Answer should be C.

†Answer: C
```

### 摘录：对应打分

```text
info.reward = 0.5
response_length = 465.0

可信对应槽位 = 0
step_score_min  = 0.388671875
step_score_mean = 0.5390625
step_score_last = 0.388671875
outcome_correct = 1.0
format_reward   = 1.0
final_reward    = 0.5
answer_extraction_failed = 0.0
```

这条样本很重要，因为它证明了：

- **文本格式完全合规**
- **最终答案也被判成正确**
- 但 **最终训练 reward 仍然只有 `0.5`**

也就是说，Phase 3 的格式分析不能替代 reward 分析。格式对，不代表 reward 一定是 1。

#### 样例 C：格式合规，但答错，最终 reward=0.0

来源：

```text
trajectories_step_20.json -> item 8
asset -> assets/phase3/step20-exp8-sample0.png
```

![step20-exp8-sample0](assets/phase3/step20-exp8-sample0.png)

### 摘录：对应文本

```text
Step 1: Identify the question: How many students received an A in Math.

Step 2: Locate the column representing Math grades in the table.

Step 3: Count the number of students who received an 'A' in Math. There are two students (Katrina and Chris) who received an 'A' in Math.

Step 4: State the final answer. Therefore, there are 2 students who received an A in Math.

†Answer: 2
```

### 摘录：对应打分

```text
info.reward = 0.0
response_length = 99.0

可信对应槽位 = 0
step_score_min  = 0.044677734375
step_score_mean = 0.6484375
step_score_last = 0.044677734375
outcome_correct = 0.0
format_reward   = 1.0
final_reward    = 0.0
answer_extraction_failed = 0.0
```

这条样本说明：

- **格式完全正确**
- **答案抽取也没失败**
- 但 **就是答错**

所以“格式遵循好了”不等于 correctness 已经解决。

#### 样例 D：图片已存、文本已存，但格式直接坏掉

来源：

```text
trajectories_step_100.json -> item 12
asset -> assets/phase3/step100-exp12-sample0.png
```

![step100-exp12-sample0](assets/phase3/step100-exp12-sample0.png)

### 摘录：对应文本

```text
Step 1: Analyze the diagram, observing that shape ① is rotated about point A.
...
Answer 1: \boxed{①}
...
Answer 2: \boxed{②}
...
Answer 3: \boxed{③†Answer: ①
```

### 摘录：保存态打分列表

```text
info.reward = 0.0
response_length = 332.0

保存态 step_score_min  = [0.55078125, 0.25390625, 0.12353515625, 0.16015625]
保存态 step_score_mean = [0.87890625, 0.396484375, 0.404296875, 0.361328125]
保存态 step_score_last = [0.65625, 0.25390625, 0.12353515625, 0.16015625]
保存态 outcome_correct = [0.0, 0.0, 0.0, 0.0]
保存态 format_reward   = [0.0, 1.0, 1.0, 1.0]
保存态 final_reward    = [0.0, 0.0, 0.0, 0.0]
```

这里我故意不写“单个精确 step score”，因为这条 item 的 `final_reward` 全是 0，无法靠唯一值把 metric 槽位反推回来。

但即使不做一一映射，这条样本也已经足够说明：

- 当前确实会出现 **格式直接坏掉** 的本地真实 case
- 而且这种坏法不是没写 step，而是 **中途偏离成 `Answer 1:` / `Answer 2:`**

#### 样例 E：图片已存、文本已存，但被长度上限截断到没写出 `†Answer:`

来源：

```text
trajectories_step_80.json -> item 6
asset -> assets/phase3/step80-exp6-sample0.png
```

![step80-exp6-sample0](assets/phase3/step80-exp6-sample0.png)

### 摘录：对应文本

```text
Step 1: Rewrite the given equation into the standard quadratic equation standard.
...
Step 11: Solve for x by dividing both sides by ln(2): ...
Step 12: Calculate numerical values: x ≈ \frac{
```

### 摘录：保存态打分列表

```text
info.reward = 0.0
response_length = 511.0

保存态 step_score_min  = [0.2109375, 0.048095703125, 0.283203125, 0.27734375]
保存态 outcome_correct = [0.0, 0.0, 0.0, 0.0]
保存态 format_reward   = [1.0, 1.0, 0.0, 1.0]
保存态 final_reward    = [0.0, 0.0, 0.0, 0.0]
保存态 answer_extraction_failed = [0.0, 0.0, 1.0, 0.0]
```

这条样本把另一个问题也坐实了：

- 它不是“写了错误的 `†Answer:`”
- 它是 **根本没走到 `†Answer:`，直接在 511 token 附近被截断**

所以 Phase 3 的格式问题，至少已经包含两类真实失效模式：

1. 模板漂移：`Answer 1:` / `Answer 2:` 这种非目标格式
2. 直接截断：还没写出最终答案行就结束

## 7. 当前这批已保存轨迹的格式遵循情况到底怎样

为了不只讲个案，我对当前 run 里这 5 个 trajectory JSON 做了离线格式统计。统计口径采用仓库里现成脚本 `examples/math_prm/tools/analyze_phase7_observation.py` 的同一套思路：

### 摘录：`examples/math_prm/tools/analyze_phase7_observation.py`

```python
def is_valid_stage3_format(text: str) -> bool:
    if not text:
        return False
    if not STEP_PATTERN.search(text):
        return False
    answer_lines = ANSWER_PATTERN.findall(text)
    if len(answer_lines) != 1:
        return False
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return False
    return non_empty_lines[-1].startswith("†Answer:")
```

也就是说，我这里的“格式成功”定义是：

- 至少有一个 `Step N:`
- 恰好一个 `†Answer:` 行
- 最后一条非空行以 `†Answer:` 开头

### 7.1 分 checkpoint 的统计结果

```text
trajectories_step_20.json   -> format_success_ratio = 1.0000
trajectories_step_40.json   -> format_success_ratio = 1.0000
trajectories_step_60.json   -> format_success_ratio = 1.0000
trajectories_step_80.json   -> format_success_ratio = 0.9375
trajectories_step_100.json  -> format_success_ratio = 0.9375
```

更完整地看：

```text
step20:  step_present=1.0, single_answer=1.0, final_answer_last=1.0, extraction_failed=0.0
step40:  step_present=1.0, single_answer=1.0, final_answer_last=1.0, extraction_failed=0.0
step60:  step_present=1.0, single_answer=1.0, final_answer_last=1.0, extraction_failed=0.0
step80:  step_present=1.0, single_answer=0.9375, final_answer_last=0.9375, extraction_failed=0.0
step100: step_present=1.0, single_answer=0.9375, final_answer_last=0.9375, extraction_failed=0.0
```

### 7.2 聚合后的总结果

把这 5 个文件合起来，总共 80 条样本，结果是：

```text
total = 80
step_present_ratio = 1.0000
single_answer_ratio = 0.9750
final_answer_last_ratio = 0.9750
format_success_ratio = 0.9750
multiple_answer_marker_ratio = 0.0000
answer_tail_extra_ratio = 0.0000
answer_extraction_failed_ratio = 0.0000
```

这组数字说明：

- `Step N:` 基本是稳的。
- 真正的问题不在“完全不按 step 写”。
- 当前格式失败主要发生在：
  - 没能以 `†Answer:` 作为最后一行稳定收尾
  - 或者根本被截断在答案前

换句话说，当前 actor 的格式遵循情况更像是：

- **大部分样本已经能遵循**
- **但在长输出 / 异常题型上仍然会漏收尾或偏离答案行模板**

## 8. 为什么这还不能直接回答“eval bad case 长什么样”

这里必须把边界讲死。

虽然当前这批本地轨迹已经能支持格式分析，但它们是：

- `replay_buffer.items`
- 也就是训练期间 rollout 收集到的样本

不是：

- runtime eval holdout 上的专用样本导出

这点从代码上已经确定：

- `save_trajectories()` 只挂在 `ppo_train()` 的 `save_steps`
- `evaluate()` 本身没有 trajectory saver 调用

而从结果目录上也能侧面验证：

- 我只搜到了 `trajectories_step_*.json`
- 没搜到任何 `eval_trajectory*.json`、`eval_samples*.json` 之类的 runtime eval 专用文件

所以当前 Phase 3 能支持的最准确结论是：

- **本地确实有轨迹，而且足够拿来分析 actor 的格式趋势**
- **但如果你要专门研究 runtime eval 的 bad case，现在还缺一条 eval 专用导出链路**

## 9. 当前保存文件还有一个重要 caveat：`reward_metrics` 未必严格逐样本对齐

这部分必须单独写出来，不然很容易被 JSON 表象误导。

### 9.1 本地 log 里已经出现了大量 shape mismatch 警告

### 摘录：`node0_20260330_235638.log` 在 step 100 保存时

```text
[TrajectorySaver] Warning: sequences is 1D tensor with shape torch.Size([216]) at step 100, exp_idx 0. Reshaping to 2D.
[TrajectorySaver] Warning: advantages has mismatched batch size 60, expected 1. Padding/truncating.
[TrajectorySaver] Warning: returns has mismatched batch size 60, expected 1. Padding/truncating.
[TrajectorySaver] Warning: action_log_probs has mismatched batch size 60, expected 1. Padding/truncating.
...
[TrajectorySaver] Warning: sequences is 1D tensor with shape torch.Size([508]) at step 100, exp_idx 12. Reshaping to 2D.
[TrajectorySaver] Warning: advantages has mismatched batch size 332, expected 1. Padding/truncating.
```

### 9.2 代码里对 shape mismatch 的处理也比较保守

### 摘录：`lightrft/utils/trajectory_saver.py`

```python
if len(tensor.shape) == 1:
    if tensor.shape[0] == expected_batch_size:
        return tensor
    else:
        print(
            f"[TrajectorySaver] Warning: {attr_name} has mismatched batch size {tensor.shape[0]}, expected {expected_batch_size}. Padding/truncating."
        )
        if tensor.shape[0] < expected_batch_size:
            ...
        else:
            return tensor[:expected_batch_size]
```

而 `reward_metrics` 的分支则是：

```python
if isinstance(metric_tensor, torch.Tensor) and len(metric_tensor.shape) > 0 and len(metric_tensor) == batch_size:
    metrics[metric_name] = self._tensor_to_list(metric_tensor[i])
else:
    metrics[metric_name] = self._tensor_to_list(metric_tensor) if isinstance(metric_tensor, torch.Tensor) else metric_tensor
```

这意味着：

- 只要 `reward_metrics` 的 tensor 第一维长度不等于 saver 眼里的 `batch_size`
- 它就不会被 slice 成单样本
- 而是整段 list 原样塞进这个 trajectory item

### 9.3 本地实物里已经能看到这个现象

比如 `trajectories_step_100.json` 的前几个 item，`format_reward` 都是：

```text
[1.0, 1.0, 1.0, 1.0]
```

`outcome_correct` / `final_reward` 也会以整段 list 的形式重复出现，而不是单个标量。

所以这里要下一个很重要的边界判断：

- **`pure_generated_text`、`image_paths`、`response_token_count` 这些字段可以放心拿来做格式与样例分析。**
- **`reward_metrics` 可以做辅助诊断，但在当前 saver shape mismatch 条件下，不应直接把每个 list 当作该单样本的精确标签。**

## 10. 最终结论

把 Phase 3 压缩成一句最准确的话，就是：

- **这次 run 的轨迹确实已经存到了本地，但存的是训练/rollout 轨迹，不是 runtime eval 专用轨迹；保存位置是 `results/.../trajectories/`，样例和图片文件都能在本地直接找到。**

再压缩成三个你后续最该记住的点：

1. **“到底存在哪里”已经坐实了：**
   - JSON 在 `.../trajectories/trajectories_step_20|40|60|80|100.json`
   - 图片在 `.../trajectories/images/`
2. **“当前格式表现如何”也已经有初步结论了：**
   - 这 80 条本地已保存样本里，格式成功率约 `97.5%`
   - 剩下的问题主要是截断未收尾到 `†Answer:`，或中途偏离出 `Answer 1:` 这类非标准形式
3. **“能不能直接代表 eval”必须保持克制：**
   - 当前还不能，因为 runtime eval 并没有单独导出轨迹
   - 如果后续要专门分析 eval bad case，应该单独给 `evaluate()` 接 trajectory export

也就是说，Phase 3 的问题现在已经可以拆成两半：

- **“轨迹到底存没存、存在哪里”这个问题，已经回答完了。**
- **“eval 侧 bad case 长什么样”这个问题，当前还缺 runtime eval 专用导出。**

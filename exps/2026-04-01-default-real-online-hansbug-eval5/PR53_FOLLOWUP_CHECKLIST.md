# PR53 + 2026-04-01 Follow-up Questions and Checklist

本文件不是“已经有答案的结论稿”，而是把讨论的每一行 TODO，先翻译成一个人类和 LLM 都能直接理解并执行的问题说明文档。

使用方式：

- 先看每个 Phase 里的“这句话到底在问什么”，确认问题边界。
- 再看“当前已经确认的实现”，避免重复排查已经知道的事实。
- 再看“当前不能直接下结论的点”，这些才是后续需要补证据的地方。
- 最后执行下面的 `* [ ]` checklist。

## 原始 TODO 总表

以下 checklist 保留原始拆分，先作为总入口：

* [ ] reward细节，prm得到打分后，是如何用于grpo loss计算的
* [ ] 为什么eval/model_reward 显著高于 eval/outcome_correct各自的含义是什么？理清楚wandb中各项指标的具体实现方法
* [ ] eval轨迹存储与分析，actor回答的格式遵循情况分析
* [ ] 修复截断的逻辑，eos token是否正确传入，max_new_tokens是否应该设置比较大
* [ ] train/kl一直上升的原因
  * [ ] Prm reward用于grpo loss等相关实现的bug
  * [ ] Kl weight是否过小，lr是否过大
* [ ] example图示是怎么得到的？
  * [ ] Format=1 accuracy=1 最终reward为什么是0.5

本文件基于以下上下文整理：

- PR `opendilab/LightRFT#53`
- `exps/2026-04-01-default-real-online-hansbug-eval5/README.md`
- 当前仓库中的 `examples/math_prm/*`
- 当前仓库中的 `lightrft/trainer/*`、`lightrft/strategy/*`、`lightrft/utils/math_prm_output.py`
- 公开可访问的 `URSA-MATH/URSA-MATH` 仓库入口与 `inference/prm_infer_score.py`

需要先统一的几个术语：

- `model_reward`：PRM 对整条回答给出的连续 proxy 分数。当前实现里来自 step score 聚合，默认聚合方式是 `min`。
- `outcome_correct`：最终答案是否答对。它不是 PRM 分数，而是 final answer extraction + exact match / grader 的结果。
- `final_reward`：PS-GRPO 风格的离散训练奖励。当前实现里典型取值是 `0 / 0.5 / 1`。
- `reward`：日志里看到的 `rollout/reward`、`eval/reward`、`train/reward`，本质上都是 `output.rewards` 的均值。对于默认 `math_psgrpo` 路径，它等于上面的 `final_reward`，但还没有加 token-level KL 惩罚。
- `KL penalty`：在 `compute_reward()` 里加到 token-level reward 上，用来算 returns / advantages。
- `train/kl`：rollout 阶段算出来并在训练日志里聚合的 KL 统计，不等于单个样本的 loss 项本身。

当前已经确认的几件事实：

- 默认 Stage 3 数据标签是 `math_psgrpo`。在这个标签下，真正进入 GRPO / `group_norm` 的 sequence reward 不是 `model_reward`，而是 `final_reward`。
- 当前 `eval/model_reward`、`eval/outcome_correct`、`eval/reward` 三者语义不同，不能直接拿数值大小做一一对应。
- 当前实现里 `Format=1`、`accuracy=1`、最终 reward 仍然是 `0.5` 的直接原因，是 `has_drop_moment=1` 且 `_DROP_GAMMA=0.5`。
- 公开的 `URSA-MATH` 仓库可以确认 PRM 打分入口里也使用了 `replace_specific_plus_minus_with_ki()` 和 `min` 聚合；但公开仓库里没有直接暴露完整 Stage 3 RL 训练脚本，所以凡是涉及 Stage 3 reward / KL / LR 的“和原 repo 完全对齐”结论，仍然需要补原始训练脚本或更直接的设计证据。

---

## Phase 1 - “reward细节，prm得到打分后，是如何用于grpo loss计算的”

### 这句话到底在问什么

这不是在问“PRM 会不会打分”，而是在问一条完整的训练链路：

1. PRM 是怎么从回答里抽 step score 的。
2. step score 是怎么变成 sequence-level reward 的。
3. sequence-level reward 是怎么再进入 GRPO / PPO 的 return 和 advantage 计算的。
4. 最后 `PolicyLoss` 实际吃到的 advantage，到底是连续 PRM 分、离散 PS-GRPO reward，还是两者混合。

如果这一条链路没有理清，后面所有关于 `eval/model_reward`、`train/kl`、`KL/LR`、`bug` 的讨论都会混口径。

### 当前已经确认的实现

- `examples/math_prm/reward_models.py::MathPRMReward.forward()`
  - 会先把回答中的 `Step N:` 边界转成 ` и` marker。
  - 然后从 `UrsaForTokenClassification` 的 logits 里取出 step 对应位置，做 `sigmoid` 得到 step score。
  - 对 step score 做聚合，默认是 `aggregation="min"`。
- 同一个 `forward()` 里还会额外算一组 PS-GRPO 指标：
  - `outcome_correct`
  - `max_relative_drop`
  - `has_drop_moment`
  - `final_reward`
- 对默认 `label == "math_psgrpo"`：
  - `sequence_reward = psgrpo_metrics["final_reward"]`
  - 也就是说，默认训练 reward 是 `final_reward`，不是 `aggregated_score`
  - `aggregated_score` 会被保存在 `reward_metrics["model_reward"]` 里，主要用于诊断
- 之后链路是：
  - `FastExperienceMaker._aggregate_rewards()` 把 `sequence_reward` 放进 `output.rewards`
  - `_build_experience()` 把它写进 `experience.info["reward"]`
  - `_compute_advantages_and_returns()` 里调用 `compute_reward(processed_reward, kl_coef, experience.kl, ...)`
  - 也就是说，sequence reward 先作为末 token reward，再叠加 token-level KL penalty
  - 默认 `advantage_estimator="group_norm"` 时，会走 `GroupNormCalculator`
  - 最终 advantage 再喂给 `PolicyLoss.forward()`

### 当前最容易误解的点

- `model_reward` 不是默认 `math_psgrpo` 的训练 reward。
- `reward_models_utils.py::reward_fn()` / `mix_rewards()` 虽然也定义了 `math_psgrpo` 路径，但默认单 RM 的真实训练路径并不会经过这里。
- 如果只盯着 `reward_fn()` 做分析，很容易分析错链路。

### 当前不能直接下结论的点

- 这种“默认只用离散 `final_reward` 训练，而连续 `model_reward` 只做日志”的做法，是否和原 Stage 3 设计完全一致，还需要拿原始训练脚本或更直接设计文档核对。
- 当前 `use_kl_loss` 打开后，是否等于“reward 侧加一次 KL，actor loss 侧再显式加一次 KL”，以及这是否是设计使然，需要在后续问题中专门确认。

### Checklist

* [ ] 画一张真实训练链路图，从 `MathPRMReward.forward()` 一直画到 `PolicyLoss.forward()`，把每一步的张量名字、shape、语义都标出来。
* [ ] 用一个真实 micro-batch 举例，列出 `step_scores`、`model_reward`、`final_reward`、`experience.info["reward"]`、`token-level final_reward`、`returns`、`advantages` 的对应值。
* [ ] 明确写出默认 `math_psgrpo` 路径和 `reward_fn()/mix_rewards()` 路径的区别，避免后续排查又把两条链路混在一起。
* [ ] 去原设计或原 repo 里补证据，确认默认 `math_psgrpo` 训练 reward 应该是离散 reward、连续 reward，还是二者混合。
* [ ] 给默认 `single RM + math_psgrpo + group_norm` 真实链路补最小回归测试，不要只测 `mix_rewards()`。

---

## Phase 2 - “为什么 eval/model_reward 显著高于 eval/outcome_correct 各自的含义是什么？理清楚 wandb 中各项指标的具体实现方法”

### 这句话到底在问什么

这句话不是单纯问“为什么一个大一个小”，而是在问：

1. W&B 里每个指标到底是从哪段代码打出来的。
2. 每个指标的统计口径是什么。
3. 为什么它们数值上会出现“看起来不一致”的情况。
4. 这种不一致是正常 proxy gap，还是实现 bug。

### 当前已经确认的实现

- runtime eval 入口：
  - `examples/math_prm/math_prm_trainer.py::_runtime_eval_context()`
  - 这里会把 eval 的 `n_samples_per_prompt` 改成 `1`
  - 会把 `advantage_estimator` 改成 `reinforce`
  - 会使用 eval 专用 decode 参数，当前实验里 `eval_temperature=0.0`
- eval 统计入口：
  - `lightrft/trainer/ppo_trainer_vl.py::evaluate()`
  - 它会遍历 eval batch，把 `experience.info["reward"]`、`response_length`、`reward_metrics[...]` 全部收集起来求 mean
- W&B 映射入口：
  - `examples/math_prm/math_prm_trainer.py::_EVAL_KEY_SOURCES`
- 目前几个关键指标的真实语义是：
  - `eval/model_reward`
    - 来自 `reward_metrics["model_reward"]`
    - 含义：PRM step score 聚合后的连续 proxy 分数
  - `eval/outcome_correct`
    - 来自 `reward_metrics["outcome_correct"]`
    - 含义：final answer 是否正确的比例
  - `eval/reward`
    - 来自 `experience.info["reward"]`
    - 对默认 `math_psgrpo` 路径，它等于 `final_reward`
    - 也就是已经吃掉 `drop_moment` 惩罚后的 sequence reward 均值
  - `eval/answer_extraction_failed`
    - 来自 `reward_metrics["answer_extraction_failed"]`
    - 含义：final answer 没有被正确抽出来的比例
  - `eval/response_length`
    - 来自 `experience.info["response_length"]`
    - 含义：response token 数

### 为什么当前会出现 `eval/model_reward` 高于 `eval/outcome_correct`

当前实现里，这种现象从语义上是可以成立的：

- `model_reward` 是连续 proxy，PRM 只要觉得每一步“看起来还行”，就可能给较高分。
- `outcome_correct` 是最终答案 exact match，只要最后答案错了，就是 `0`。
- 所以会出现：
  - reasoning 过程看起来像样
  - PRM 连续分数不低
  - 但最后答案还是错
  - 从而 `model_reward > outcome_correct`

这正是 README 里说的 proxy gap 的候选解释。

### 为什么当前 `eval/reward` 还会低于 `eval/outcome_correct`

因为默认 `math_psgrpo` 路径里：

- `eval/reward` 看的不是 `model_reward`
- 它看的是 `final_reward`
- 当前 `final_reward` 规则是：
  - 答错 -> `0`
  - 答对且没有 drop -> `1`
  - 答对但有 drop -> `0.5`

所以即使 `outcome_correct` 已经是 `1`，也有一部分样本会因为 `has_drop_moment=1` 被打成 `0.5`，于是整体均值就可能低于 `outcome_correct`。

### 当前不能直接下结论的点

- 当前 `model_reward > outcome_correct` 不能直接判成 bug；它可能只是 proxy gap。
- 但也不能反过来直接认定“完全合理”，因为还需要确认：
  - PRM 是否真的对视觉与步骤质量起作用
  - 当前 `final_reward` 是否与原设计一致
  - 当前 answer extraction 是否干净可靠

### Checklist

* [ ] 把 `rollout/*`、`train/*`、`eval/*` 的关键指标全部做成一页“指标名 -> 来源代码 -> 聚合方式 -> 业务含义”对照表。
* [ ] 单独写清 `eval/model_reward`、`eval/outcome_correct`、`eval/reward` 三者的区别，不允许再混用“reward”这个词。
* [ ] 抽取 20~50 个 eval 样本，分桶看 `model_reward 高但 outcome_correct 低` 的典型错误模式。
* [ ] 确认 `answer_extraction_failed` 是否足够低，足以支持“当前主要矛盾不是 extraction”这一说法。
* [ ] 检查 rollout log 里 `abs(mean_metric) > 1e-6` 的过滤逻辑，确认没有把某些重要但接近 0 的指标静默吞掉。

---

## Phase 3 - “eval轨迹存储与分析，actor回答的格式遵循情况分析”

### 这句话到底在问什么

这句话实际包含两个问题：

1. 现在有没有把 eval 期间的具体样本轨迹存下来，方便做 case study。
2. actor 当前对 Stage 3 格式要求的遵循情况，到底是稳定合规，还是主要靠后处理兜底。

### 当前已经确认的实现

- 当前仓库的 trajectory 保存入口是：
  - `lightrft/utils/trajectory_saver.py`
  - `lightrft/trainer/spmd_ppo_trainer.py::save_trajectories()`
  - `examples/math_prm/math_prm_trainer.py::save_trajectories()`
- 当前保存的是 replay buffer 里的训练经验轨迹
  - 也就是 train / rollout 期间形成的 `Experience`
  - 默认不是 runtime eval 专用轨迹
- 当前保存下来的轨迹里，已经有足够多的分析字段：
  - `full_sequence`
  - `generated_text`
  - `pure_generated_text`
  - `image_paths`
  - `info.reward_metrics`
  - `response_token_count`
- 当前可以直接用这些字段分析格式：
  - 有没有 `Step N:`
  - 有没有且仅有一个 `†Answer:`
  - `†Answer:` 后面有没有多余尾巴
  - answer extraction 是不是 fallback 才成功

### 当前最关键的现实限制

- 现在“训练轨迹保存”和“eval 轨迹保存”不是一回事。
- 所以如果要回答“eval bad case 到底长什么样”，当前不能只靠已有 `results/.../trajectories/*.json`。
- 需要二选一：
  - 给 runtime eval 单独加 trajectory 保存
  - 或者离线重跑 eval，把样本导出来

### 当前不能直接下结论的点

- 目前看到的格式问题，到底是 train 轨迹的现象，还是 eval 轨迹也同样存在，需要补 eval 侧证据。
- 当前 README 里“主要问题不是 extraction”是合理推测，但如果没有 eval 侧样本分析，仍然不够扎实。

### Checklist

* [ ] 明确区分“训练轨迹分析”和“eval 轨迹分析”，不要再把两者混成一个需求。
* [ ] 给 runtime eval 加一个最小可用的 trajectory 导出方案，至少保存 `prompt`、`pure_generated_text`、`image_paths`、`reward_metrics`。
* [ ] 按格式规则统计：`Step N:` 存在率、单个 `†Answer:` 存在率、answer 收尾合规率、answer 后多余文本比例。
* [ ] 按错误类型分桶：格式错、answer extraction 失败、格式对但 reasoning 错、格式对但视觉判断错。
* [ ] 评估是否把 `format_success_ratio`、`multiple_answer_marker_ratio`、`answer_tail_extra_text_ratio` 直接接入 W&B。

---

## Phase 4 - “修复截断的逻辑，eos token是否正确传入，max_new_tokens是否应该设置比较大”

### 这句话到底在问什么

这句话本质是在问“模型到底是真的学会在 `†Answer:` 后停住了，还是只是因为长度上限和后处理看起来像停住了”。

它拆开以后至少有四个子问题：

1. local HF rollout 时 `eos_token_id`、`pad_token_id` 有没有正确传进去。
2. `structured_answer_stop` 有没有在对的时机生效。
3. `sanitize_math_prm_response_text()` 有没有在事后大量截尾。
4. `max_new_tokens` / `local_hf_max_new_tokens` 是不是设得太小，导致大量样本都是硬截断而不是自然结束。

### 当前已经确认的实现

- HF rollout 参数入口：
  - `lightrft/trainer/fast_exp_maker.py`
  - 对 `engine_type == "hf"`，最终会把 `max_new_tokens` 变成：
    - `min(generate_kwargs["max_new_tokens"], local_hf_max_new_tokens)`
    - 也就是说 `local_hf_max_new_tokens` 是本地 HF rollout 专用硬上限
- `eos_token_id` / `pad_token_id` 获取入口：
  - `lightrft/strategy/strategy_base.py`
  - 优先从 inference tokenizer 取，取不到再从 model config 取
- 结构化 early stop 入口：
  - `lightrft/strategy/strategy_base.py::_StructuredAnswerEosLogitsProcessor`
  - 它会在检测到 `†Answer:` 且满足启发式 stop 条件时，强制把后续 token 变成 `eos`
- 事后清洗入口：
  - `lightrft/trainer/fast_exp_maker.py::_sanitize_structured_math_prm_outputs()`
  - 它会在生成完成后，把第一个 answer 行之后的垃圾尾巴裁掉
- 当前实验 README 和 `generation_control.json` 已经显示：
  - raw generation 长度长期贴着 `512`
  - sanitize ratio 很高
  - 这说明“后处理重截断”是真实存在的，不是猜测

### 当前最值得怀疑的实现点

- `ActorLanguage.process_sequences()` / `ActorVL.process_sequences()`
  - 会基于当前 attention mask 重新 scatter 一个 `eos_token_id`
  - 如果样本实际上只是 hit max length，而不是真自然停住，这一步可能会让轨迹看起来“像是有 eos 正常结束”
  - 这会掩盖真正的截断问题

### 当前不能直接下结论的点

- 不能只因为 `local_hf_max_new_tokens=512` 就简单说“应该调大”。
- 如果 stop 逻辑本身错了，把上限调大可能只会：
  - 让输出更长
  - 让 KL 更高
  - 让训练更慢
  - 但并不会自然修好格式
- 所以需要先确认：当前问题主要是 stop 逻辑失效，还是单纯上限太小。

### Checklist

* [ ] 抓 sanitize 之前的原始 `output_token_ids`，确认“自然停住”和“打满长度上限”各占多少比例。
* [ ] 验证 local HF rollout 实际传入的 `eos_token_id`、`pad_token_id` 是否和训练 actor / tokenizer 一致。
* [ ] 检查 `_StructuredAnswerEosLogitsProcessor` 的触发频率、marker 检测窗口和 answer tail stop 条件是否过晚或过松。
* [ ] 单独审查 `process_sequences()` 强制 scatter `eos` 的行为，确认它会不会把 hit-max-length 样本伪装成正常终止样本。
* [ ] 做受控实验：`local_hf_max_new_tokens = 512 / 1024 / 1536`，同时记录 raw length、sanitize ratio、`eval/outcome_correct`、OOM 风险和 step time。
* [ ] 在真正修 stop 逻辑前，不要只靠“把上限调大”当默认修复方案。

---

## Phase 5 - “train/kl一直上升的原因”

### 这句话到底在问什么

这句话是在问：当前 KL 飙升到底是哪一类问题主导的。

候选原因至少包括：

- policy drift 真的过大
- reward 设计鼓励了过长、过散的回答
- stop 逻辑失效导致生成一直撞长度上限
- `init_kl_coef` 太小
- `actor_learning_rate` 太大
- KL 监控口径和优化口径不是同一件事

### 当前已经确认的实现

- 当前 `train/kl` 的控制器：
  - `kl_target` 为空时走 `FixedKLController(init_kl_coef)`
  - 当前实验是 `init_kl_coef=0.001`
  - 当前不是 adaptive KL
- 当前 `train/kl` 的日志不是凭空来的：
  - 它来自 rollout 时对 `experience.info["kl"]` 的聚合
  - 在训练日志里又做了 response-length-weighted mean
- 当前实验中还有一个强相关信号：
  - raw generation 长期贴着 `512`
  - sanitize ratio 极高
  - 所以“长度失控”和“KL 上升”大概率不是独立现象

### 当前最需要警惕的一点

当前实现在 `use_kl_loss` 打开时，看起来同时存在两种 KL 约束：

- reward 侧：
  - `compute_reward()` 会把 `-kl_coef * kl` 加进 token-level reward
- actor loss 侧：
  - `ppo_trainer_vl.py` 里 `loss = actor_loss + kl_loss * self.kl_ctl.value + ...`

这是不是原设计要求的“双重 KL 约束”，当前还不能直接下结论。

### 当前不能直接下结论的点

- 不能只看 `train/kl` 上升就说“KL weight 太小”。
- 也不能只看 `train/kl` 上升就说“LR 太大”。
- 更不能在 stop / truncation 还没理顺前，就把所有问题都归因到超参。

### Checklist

* [ ] 把 `train/kl`、raw generation length、sanitize ratio、`eval/outcome_correct` 放到同一时间轴上，看相关性。
* [ ] 核对 `train/kl` 的统计口径和 actor loss 里显式 `kl_loss` 的统计口径是否一致。
* [ ] 明确当前 `use_kl_loss` 下是否存在“reward 侧 KL + loss 侧 KL”双重约束，以及这是不是有意设计。
* [ ] 抽样分析 KL 特别高的 batch，看看是否同时伴随长输出、格式尾巴、重复文本或视觉误判。
* [ ] 在 stop 问题没有查清前，不要只通过调 KL 来解释一切。

---

### 子问题 A - “Prm reward用于grpo loss等相关实现的bug？”

#### 这句话到底在问什么

这句话不是泛泛问“代码里有没有 bug”，而是在问：

- PRM reward 到 GRPO loss 这条链路里，是否存在实现和设计不一致的地方。
- 当前我们看到的现象，到底是设计就是这样，还是代码把它实现歪了。

#### 当前已经确认的实现

- 默认 `math_psgrpo` 真实训练路径并不经过 `reward_models_utils.py::reward_fn()`
- 默认路径里真正用于训练的是 `MathPRMReward.forward()` 直接返回的 `score`
- 对 `math_psgrpo`，这个 `score` 就是 `final_reward`

#### 当前最值得怀疑的几个点

- 只看 `reward_fn()/mix_rewards()` 很容易误判默认训练路径
- `model_reward` 作为日志指标保留了下来，但默认训练吃的是 `final_reward`
- `use_kl_loss` 可能造成 reward 侧和 loss 侧双重 KL
- 如果只看 W&B，不展开到代码，很容易把“设计上的离散 reward”误认为“PRM reward 没接上”

#### 当前不能直接下结论的点

- “默认训练只用离散 `final_reward` 而不用连续 `model_reward`”是否是 bug，必须先看原设计。
- 如果原设计本来就是 PS-GRPO 的离散 reward，那这不是 bug，而是设计选择。
- 如果原设计要求连续 PRM 分也进入 RL reward，那当前实现才可能有偏差。

#### Checklist

* [ ] 逐一核对 `math_prm`、`math_psgrpo`、`math_prm_combined`、`math_rule` 四种 label 的 reward 语义，不要把不同 label 的路径混起来。
* [ ] 画清“默认真实路径”和“辅助 `mix_rewards` 路径”的分叉点，确认当前怀疑的 bug 究竟在哪条链路上。
* [ ] 核对 `reward_metrics["model_reward"]`、`reward_metrics["final_reward"]`、`experience.info["reward"]` 是否被任何地方错误覆盖。
* [ ] 去原 Stage 3 设计或原 repo 训练脚本里补证据，确认默认训练到底应该吃哪种 reward。
* [ ] 任何最终确认的 bug，都必须配最小复现和回归测试，不接受只改逻辑不补测试。

---

### 子问题 B - “Kl weight是否过小，lr是否过大？”

#### 这句话到底在问什么

这句话是一个超参诊断问题，不是一个默认结论。

真正要回答的是：

- 当前看到的 KL 上升和 correctness 平台化，究竟更像是 KL 约束不足，还是学习率过大，还是两者都有。
- 如果要改超参，优先改哪个，为什么。

#### 当前已经确认的实现

- 当前实验配置来自：
  - `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- 关键值是：
  - `init_kl_coef = 0.001`
  - `actor_learning_rate = 1e-6`
  - `advantage_estimator = group_norm`
  - `local_hf_max_new_tokens = 512`
- 公开 `URSA-MATH` 仓库目前能确认的是 PRM 推理侧实现片段
- 但公开仓库里没有直接给出完整 Stage 3 RL 超参脚本，所以“和原 repo 对齐”的超参结论当前还缺直接证据

#### 当前不能直接下结论的点

- 不能在 stop / truncation 还没理清时，直接把问题全归因到 KL 或 LR。
- 也不能只凭单个长跑样本就拍板“0.001 太小”或“1e-6 太大”。
- 这类结论必须来自受控对比实验，而不是靠直觉。

#### Checklist

* [ ] 设计一个最小 2x2 对比实验：至少覆盖 `init_kl_coef` 和 `actor_learning_rate` 两个维度。
* [ ] 对比实验必须固定 seed、eval holdout、reward label、`n_samples_per_prompt`、`local_hf_max_new_tokens`、batch size 和 decode 参数。
* [ ] 每组实验至少记录：`train/kl`、`eval/model_reward`、`eval/outcome_correct`、`eval/reward`、raw length、sanitize ratio。
* [ ] 分开验证“只改 KL”和“只改 LR”两种情况，判断谁对 KL 漂移和 correctness 更敏感。
* [ ] 如果最终建议改超参，必须写清楚：改它是为了解哪一个机制问题，而不是只报一个看起来更稳的数字。

---

## Phase 6 - “example图示是怎么得到的？”

### 这句话到底在问什么

这句话是在问实验文档里的图和样例卡片能不能被别人复现。

它至少包含两类对象：

- 曲线图
  - `collect_metrics.png`
  - `eval_metrics.png`
  - `learn_metrics.png`
  - `generation_control.png`
- 样例图 / pair card
  - `step20_pair_card.png`
  - `step100_pair_card.png`
  - `step20_sample_good.png`
  - `step100_sample_bad.png`

### 当前已经确认的实现

- 曲线图的数据源基本已经在实验目录里：
  - `assets/history_rows.json`
  - `assets/generation_control.json`
  - `assets/run_analysis.json`
- 样例图的数据源也部分在实验目录里：
  - `assets/trajectory_examples.json`
  - 它明确记录了样本来自哪个 trajectory JSON、对应图片文件叫什么
- 但当前仓库里没有明显的自动生成 pair card 的脚本
  - 所以这部分大概率是手工制作，或脚本在仓库外

### 当前不能直接下结论的点

- 现在还不能说“所有图都可一键复现”。
- 至少样例卡片这部分，目前缺少明确的生成脚本或制作步骤。

### Checklist

* [ ] 给每张图补 provenance：输入数据、生成脚本或手工步骤、输出文件名。
* [ ] 验证 `generation_control.png` 是否完全可以由 `assets/generation_control.json` 复现。
* [ ] 验证 `collect/eval/learn` 三张曲线图是否完全可以由 `assets/history_rows.json` 复现。
* [ ] 核对 `trajectory_examples.json` 里记录的样例来源是否能完整对应到 README 里的图文样例。
* [ ] 如果当前没有样例卡片生成脚本，就补一个最小版本，至少能从 trajectory JSON + image path 生成可复现的样例卡片。
* [ ] 把图的生成过程补回实验文档，避免之后只能看到图片而不知道是怎么来的。

---

### 子问题 - “Format=1 accuracy=1 最终reward为什么是0.5”

#### 这句话到底在问什么

这句话是在问：

- 当前 reward 规则到底是什么。
- `format`、`accuracy`、`drop_moment` 三个量之间是怎样组合的。
- 为什么“格式对且答案对”不一定拿满分。

#### 当前已经确认的实现

当前逻辑在 `examples/math_prm/reward_models.py::_compute_psgrpo_metrics()`：

- 先算 `outcome_correct`
- 再算 `max_relative_drop`
- 只要 `max_relative_drop >= _DROP_THRESHOLD`，就有 `has_drop_moment = 1`
- 当前阈值和惩罚是：
  - `_DROP_THRESHOLD = 0.3`
  - `_DROP_GAMMA = 0.5`
- 最终 reward 规则是：
  - `outcome_correct == 0` -> `final_reward = 0`
  - `outcome_correct == 1 and has_drop_moment == 0` -> `final_reward = 1`
  - `outcome_correct == 1 and has_drop_moment == 1` -> `final_reward = 0.5`

#### 当前最容易误解的点

- `format_reward` 目前主要是诊断指标，不是默认 `math_psgrpo` 最终 reward 的直接加项。
- 所以“format=1”不代表会额外加 `1` 分。
- 这也是为什么：
  - `format=1`
  - `accuracy=1`
  - 但 `final_reward` 仍然可能是 `0.5`

#### 当前不能直接下结论的点

- 这套规则从实现上已经是自洽的。
- 但它是否与原设计完全一致，仍然需要对齐原 Stage 3 训练定义。
- 如果后面决定改 reward 规则，这不是“小修数值”，而是要同时改测试、日志解释和实验记录口径。

#### Checklist

* [ ] 在文档里明确写出当前 reward 公式，不要再只写口头解释。
* [ ] 用一个真实 trajectory 样例和一个单元测试同时证明这条规则。
* [ ] 核对 `_DROP_THRESHOLD=0.3` 和 `_DROP_GAMMA=0.5` 是否与原设计一致。
* [ ] 单独说明 `format_reward` 当前只是诊断项，不是默认最终 reward 的直接加项。
* [ ] 如果后续改动这条规则，必须同步更新代码、测试、README 和实验记录。

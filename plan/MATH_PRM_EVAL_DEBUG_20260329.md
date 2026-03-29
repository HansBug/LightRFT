# Math PRM Eval Debug 记录

## 时间

- 记录日期：2026-03-29

## 背景

近期 `math_prm` / `math_psgrpo` 主链路中，W&B run `pjctfbbs` 的 eval 曲线表现出“几乎绝对不动”的现象。

对应讨论结论是，优先沿下面两条假设排查：

1. 模型其实没有真正训练起来。
2. eval 逻辑或 eval 链路本身有 bug。

本记录只做诊断归档，不修改代码。

## 本次排查范围

- 线上 W&B run：
  - `pjctfbbs`
  - 同类历史 run：`51q21rc5`、`y9sulzln`、`unuz8bej`
  - 对照 run：`da0cx5ok`、`vo45u2kd`
- 本地日志：
  - `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log`
  - 以及同目录下历史 `psgrpo` 日志
- 本地结果目录：
  - `results/lightrft-ursa8b-stage3-psgrpo/...`
- 当前代码链路：
  - `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
  - `examples/math_prm/train_colocate.py`
  - `examples/math_prm/math_prm_trainer.py`
  - `lightrft/trainer/ppo_trainer_vl.py`
  - `lightrft/trainer/spmd_ppo_trainer.py`
  - `lightrft/strategy/strategy_base.py`
  - `lightrft/strategy/fsdp/fsdpv2.py`

## 结论先行

### 1. eval 不是没有触发

`pjctfbbs` 的本地日志里，runtime eval 明确在 `step 5/10/15/.../65` 都触发了。

日志中能看到：

- `Aggregated runtime eval metrics (Step 5)`
- `Aggregated runtime eval metrics (Step 10)`
- ...
- `Aggregated runtime eval metrics (Step 65)`

对应位置示例：

- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:8648`
- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:21068`
- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:22172`

### 2. 但 eval 聚合结果确实完全不变

`pjctfbbs` 本地日志里，聚合后的 eval 指标从 `step 5` 到 `step 65` 数值完全一样：

| step | reward | outcome_correct | has_drop_moment | model_reward | response_length | answer_extraction_failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.3601 | 0.3909 | 0.4365 | 0.4994 | 185.2699 | 0.0714 |
| 20 | 0.3601 | 0.3909 | 0.4365 | 0.4994 | 185.2699 | 0.0714 |
| 40 | 0.3601 | 0.3909 | 0.4365 | 0.4994 | 185.2699 | 0.0714 |
| 60 | 0.3601 | 0.3909 | 0.4365 | 0.4994 | 185.2699 | 0.0714 |
| 65 | 0.3601 | 0.3909 | 0.4365 | 0.4994 | 185.2699 | 0.0714 |

这说明“eval 是直线”不是纯 W&B 面板展示问题，本地日志中的聚合 eval 输出本身就是直线。

### 3. 训练侧并不是完全没动作

至少从训练链路和落盘结果看，不能把问题简单归因为“根本没训练”：

- rollout 侧指标在 run 过程中明显波动，不是整条 run 都是常数。
- `pjctfbbs` 在 `global_step20`、`global_step40`、`global_step60` 都保存了 actor checkpoint。
- 每个 step 后都调用了 `update_engine_weights()`，而且日志里反复打印了
  `Finished update engine weights for separate local HF rollout actor ... copy_state_s=...`
- `global_step40` 和 `global_step60` 的 checkpoint shard 文件哈希不同，说明磁盘上的 actor checkpoint 字节内容确实发生了变化。

### 4. 当前最合理的缩圈判断

当前问题已经缩到下面两类之一：

1. **训练 actor 确实在变化，但 eval 用到的模型/权重是 stale 的。**
2. **训练 actor 有变化，但变化不足以改变当前固定 held-out + greedy eval 的输出，导致 eval 聚合指标完全不变。**

第一类更像“链路问题”，第二类更像“训练有效但对 greedy held-out 没有行为变化”。

目前还没有做完最后一步验证，不能只凭现有证据在这两者之间下最终结论。

## `pjctfbbs` 的关键信息

### W&B / 本地路径映射

- W&B URL：`https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/pjctfbbs`
- 本地 W&B 目录：`/data/LightRFT/wandb/run-20260325_115151-pjctfbbs`
- 本地日志：`/data/LightRFT/rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log`
- 本地结果目录：`/data/LightRFT/results/lightrft-ursa8b-stage3-psgrpo/lightrft-ursa8b-stage3-psgrpo-ep10-kl0.001-lr1e-6-20260325_114828`

### 启动参数中和 eval 直接相关的部分

从日志里的 `Namespace(...)` 可确认，本次 run 的关键参数为：

- `engine_type='hf'`
- `fsdp=True`
- `hf_separate_rollout_actor=True`
- `hf_separate_rollout_keep_on_gpu=True`
- `eval_steps=5`
- `eval_data=None`
- `eval_split=''`
- `max_eval_samples=500`
- `eval_holdout_size=500`
- `eval_holdout_seed=42`
- `eval_n_samples_per_prompt=1`
- `eval_do_sample=False`
- `eval_temperature=0.0`
- `eval_generate_max_len=3072`

这意味着该 run 不是走显式 `eval_data`，而是走：

- 从 `prompt_data` 中按固定 seed 切出的 deterministic held-out subset
- `n_samples=1`
- greedy eval

这与 launcher 默认逻辑一致：

- `examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- `examples/math_prm/train_colocate.py` 中的 `split_runtime_eval_dataset(...)`

## 当前 eval 实现链路

### 1. 数据集来源

`train_colocate.py` 中：

- 如果没传 `eval_data` 且没传 `eval_split`，则从 `prompt_data` 切出固定 held-out eval。
- 具体实现是 `split_runtime_eval_dataset(...)`。

这条链路的语义是：

- eval 数据应当是固定的
- 每次 eval 用的是同一批 held-out 样本

### 2. eval 运行时上下文

`examples/math_prm/math_prm_trainer.py` 中：

- `_build_eval_generate_kwargs()` 会构建单独的 eval 生成参数
- `_runtime_eval_context()` 会在 eval 期间临时覆盖
  - `generate_kwargs`
  - `n_samples_per_prompt`
  - `advantage_estimator`

关键点：

- eval 时强制 `n_samples_per_prompt = eval_n_samples_per_prompt`
- eval 时把 `advantage_estimator` 临时改成 `reinforce`
- eval 结束后再恢复训练态配置

### 3. eval 实际执行逻辑

`lightrft/trainer/ppo_trainer_vl.py` 的 `evaluate()` 会：

1. 遍历 `eval_dataloader`
2. 调用 `self.experience_maker.make_experience_list(...)`
3. 从 `experience.info` 中收集 reward / reward_metrics / response_length
4. 计算 `*_mean`

也就是说，eval 并不是直接跑一个独立“推理-only”路径，而是复用了 `experience_maker` 的生成和奖励计算逻辑。

### 4. eval 的 W&B 记录语义

`examples/math_prm/math_prm_trainer.py` 里：

- `eval/*` 被定义到 `eval/train_step` 这条 X 轴
- `save_logs_and_checkpoints()` 中每到 `global_step % eval_steps == 0` 就执行 eval 并记录

因此这次 `pjctfbbs` 的问题，不是“没有 eval logging key”，而是“eval key 有了，但值恒定”。

## 2026-03-29 新增最小化 probe 结果

本节补充本轮新增的最小化局部验证，目标是把问题从“训练没动”与“eval 链路 stale”之间进一步定性。

### 1. 真实 checkpoint 的离线输出会变化

我先把 `pjctfbbs` 的两个 actor FSDP checkpoint 转成 HF 目录：

- `tmp/ursa_stage3/pjctfbbs_step40_hf`
- `tmp/ursa_stage3/pjctfbbs_step60_hf`

然后按训练时相同的 held-out 规则：

- `eval_holdout_size=500`
- `eval_holdout_seed=42`

从 `mmathcot_stage3_math_psgrpo.jsonl` 中取前 8 个 held-out eval 样本，分别用：

- base model
- `global_step40`
- `global_step60`

做 greedy decode，对比结果见：

- `tmp/ursa_stage3/compare_eval_checkpoint_decodes_limit8.json`

核心结果：

- `base_vs_step40`: `6/8` 个样本输出发生变化
- `base_vs_step60`: `6/8` 个样本输出发生变化
- `step40_vs_step60`: `6/8` 个样本输出发生变化

对应文件中：

- `tmp/ursa_stage3/compare_eval_checkpoint_decodes_limit8.json:12`
- `tmp/ursa_stage3/compare_eval_checkpoint_decodes_limit8.json:15`

其中第 0 个 held-out 样本就是日志里反复打印的那道“two roads / farmland 551”题。离线 decode 对比显示：

- base / step60：
  - `Step 1: Let the width of the road be x meters.`
- step40：
  - `Step 1: Let the width of of the road be x meters.`

对应文件位置：

- `tmp/ursa_stage3/compare_eval_checkpoint_decodes_limit8.json:19`
- `tmp/ursa_stage3/compare_eval_checkpoint_decodes_limit8.json:32`

这条证据非常关键：

- **真实训练 checkpoint 会改变 held-out eval 样本的解码文本。**
- 因而“模型完全没训练起来，所以 eval 才是一条死线”这一解释已经明显站不住。

### 2. 运行时 eval 日志里的相同 held-out 样本却始终不变

训练日志里，多次打印了 eval 阶段的首个样本文本。对于上面同一道“two roads / farmland 551”题，`step 5 / 20 / 65` 的日志内容完全一致：

- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:8551`
- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:9664`
- `rft_logs/lightrft-ursa8b-stage3-psgrpo/node0_20260325_114828.log:22079`

这说明：

- **训练过程中 runtime eval 实际打印出来的生成文本没有变。**
- 但离线从 `global_step40/60` checkpoint 直接解码，同一 held-out prompt 的输出已经变了。

两条证据合起来，已经足以说明问题不在“checkpoint 本身是否变化”，而在“runtime eval 当时吃到的到底是不是当前 actor 权重”。

### 3. 直接修改 inference engine，会立刻改变 rollout 输出

对单卡最小化 FSDP + separate rollout probe 做了一个强制扰动：

- 直接把 `strategy.inference_engine` 持有的 `lm_head` 清零

结果见：

- `tmp/ursa_stage3/probe_separate_rollout_generation_change_inference_zero.log`

关键结果：

- `tokens_changed: true`
- 生成文本从正常回答变成了明显损坏的 `!!!!...`

对应位置：

- `tmp/ursa_stage3/probe_separate_rollout_generation_change_inference_zero.log:1799`
- `tmp/ursa_stage3/probe_separate_rollout_generation_change_inference_zero.log:1809`

这证明：

- **rollout / eval 使用的 inference engine 对象本身是活的。**
- **它当前的参数如果被直接改坏，生成输出会立刻跟着变。**

### 4. 直接修改 FSDP actor，本体直接生成也会立刻变化

我又验证了另一件事：

- 对 FSDP actor 本身的 `DTensor` 参数做 `all_zero`
- 不走 rollout engine，直接调用 actor 自己的 `generate`

结果见：

- `tmp/ursa_stage3/probe_actor_direct_generation_after_zero.log`

关键结果：

- `tokens_changed: true`
- 输出从正常文本变成 `!!!!!!!!!!!!!!!!...`

对应位置：

- `tmp/ursa_stage3/probe_actor_direct_generation_after_zero.log:1009`
- `tmp/ursa_stage3/probe_actor_direct_generation_after_zero.log:1012`

这证明：

- **我对 actor 做的“清零”扰动本身是有效的。**
- **FSDP actor 当前持有的参数确实会影响它自己的直接生成。**

### 5. 但通过 `update_engine_weights(actor)` 同步过去，rollout 输出却不变

随后做了最关键的最小化同步 probe：

- 对 FSDP actor 做 `all_zero`
- 调用 `strategy.update_engine_weights(actor)`
- 再用 separate rollout actor 走 `gather_and_generate`

结果见：

- `tmp/ursa_stage3/probe_separate_rollout_generation_change_actor_all_zero.log`

关键结果：

- `tokens_changed: false`
- `copy_state_s` 仍然耗时约 `35s`
- 前后 rollout 文本完全相同

对应位置：

- `tmp/ursa_stage3/probe_separate_rollout_generation_change_actor_all_zero.log:1795`
- `tmp/ursa_stage3/probe_separate_rollout_generation_change_actor_all_zero.log:1804`
- `tmp/ursa_stage3/probe_separate_rollout_generation_change_actor_all_zero.log:1818`

这条证据与前两条放在一起就非常明确：

1. actor 自己变了会影响直接生成；
2. inference engine 自己变了会影响 rollout 生成；
3. **但 actor 变了以后，通过 `update_engine_weights(actor)` 同步到 rollout actor，并没有让 rollout 生成发生对应变化。**

这已经是一个最小化可复现的链路问题，不再是“怀疑”。

## 当前最可能的根因位置

当前最可疑的代码位置已经缩到：

- `lightrft/strategy/strategy_base.py:850`
- `lightrft/strategy/strategy_base.py:877`

也就是：

- `_copy_local_hf_rollout_actor_state(...)`
- `_sync_separate_hf_rollout_actor(...)`

尤其是 `_copy_local_hf_rollout_actor_state(...)` 的复制逻辑：

- `src_param.detach()`
- `dst_param.detach().copy_(src_tensor)`

对应代码：

- `lightrft/strategy/strategy_base.py:857`
- `lightrft/strategy/strategy_base.py:861`

结合上面的最小化复现，当前最合理的解释是：

- 在 FSDP / `DTensor` 参数场景下，这里的参数复制**没有把 actor 的当前有效权重正确同步到 separate rollout actor**。
- 因此日志里的
  - `Finished update engine weights for separate local HF rollout actor`
  - `copy_state_s=...`
  只能证明“执行了复制流程”，**不能证明复制语义真的成功了**。

## 现在的诊断结论

基于本轮新增证据，当前结论已经可以明确写成：

1. **训练权重确实在变化。**
   - checkpoint 文件哈希变化；
   - 离线从 `global_step40/60` 解码，held-out 样本输出明显变化。
2. **runtime eval 的生成结果却保持不变。**
   - 本地日志里同一道 held-out 题的 eval 输出跨多个 step 完全相同。
3. **问题点已经最小化复现到 `update_engine_weights(actor)` 这条同步链。**
   - 直接改 actor，会影响 actor 自己的直接生成；
   - 直接改 inference engine，会影响 rollout 生成；
   - 但 actor 改完后通过 `update_engine_weights(actor)` 同步，rollout 输出不跟着变。

因此，当前最强结论不是：

- “模型没训练”

而是：

- **separate local HF rollout actor 的权重同步链有问题，导致 runtime eval 基本吃的是 stale rollout 权重。**

## 还差什么

虽然主结论已经足够明确，但如果后续要继续把证据再补满，剩下两项仍可做：

1. 在 8 卡 `torchrun` 拓扑下复跑同样的最小化同步 probe，确认该问题在真实训练拓扑下同样复现。
2. 进一步把 `_copy_local_hf_rollout_actor_state(...)` 拆成“逐参数 local shard checksum”验证，精确确认到底是：
   - `DTensor.copy_` 本身语义不对；
   - 还是 source / destination 参数视图不是当前 forward 真正使用的那份存储。

但这两项已经属于“进一步加固证据”，不是当前诊断成立所必需的前提。

## 与“模型是否真的在训练”相关的证据

### 1. checkpoint 保存配置是明确开启的

launcher 中：

- `SAVE_STEPS=20`
- `MAX_CKPT_NUM=2`
- `--save_steps ${SAVE_STEPS}`
- `--ckpt_path results/...`

这意味着 checkpoint 正常应该每 20 step 保存一次，并且只保留最新 2 个。

### 2. `pjctfbbs` 的 checkpoint 保存记录

本地日志明确记录了：

- `global_step20`
- `global_step40`
- `global_step60`

并且在保存 `global_step60` 前删除了 `global_step20`。

所以当前磁盘上保留的是：

- `.../_actor/global_step40`
- `.../_actor/global_step60`

### 3. checkpoint 文件格式

当前 FSDP checkpoint 不是 HF `save_pretrained` 目录，而是 DCP shard：

- `.metadata`
- `__0_0.distcp`
- `__1_0.distcp`
- ...
- `__7_0.distcp`
- `client_state.pt`

### 4. `step40` 与 `step60` 的 checkpoint 字节内容不同

对 `pjctfbbs` 的 `global_step40` 和 `global_step60` 做 SHA256 对比后，至少下面这些 shard 已确认不同：

- `__0_0.distcp`
  - `step40`: `314e6d93abf35ee0712fb1b9ca928f0dea9764e3e1da2ffa0f927b315874051b`
  - `step60`: `72d63d9f7db6b164dfe3b82a066e389cf494179214cafdbba61dc5c08ea490d4`
- `__1_0.distcp`
  - `step40`: `10d4e34ef4a2d0c86e214bfc208f310e04ba6f27d149d8f152bce73236a9f6f8`
  - `step60`: `55a67f62830f47345f1bd234d378c06287f575728bb7defa51137882aba439d4`
- `__2_0.distcp`
  - `step40`: `48e6021ea2f0941d56100f7d912c05418dd9c88d406ca4579f90f08216c13624`
  - `step60`: `8af6e408a998bdea35f05a800c51ac2dcc994ccd03cfc6bd4f4bf8241e4ce0b6`

这条证据非常关键：

- 它不能证明“训练质量没问题”
- 但它已经足够证明“磁盘上保存出来的 actor 权重不是完全一样的旧文件”

因此，`pjctfbbs` 不能直接归因成“权重根本没保下来”。

## 与“eval 链路是否吃到更新权重”相关的证据

### 1. trainer 每轮 PPO 后都会调用 `update_engine_weights()`

`spmd_ppo_trainer.py` 中，`ppo_train()` 的顺序是：

1. 训练 actor
2. `self.strategy.update_engine_weights(self.actor)`
3. 返回状态
4. 外层 `fit()` 再调用 `save_logs_and_checkpoints()`

所以从调用顺序上看，eval 本应发生在 rollout/eval engine 权重同步之后。

### 2. `pjctfbbs` 日志里确实反复打印了同步成功

日志中能看到大量类似：

- `Finished update engine weights for separate local HF rollout actor {'copy_state_s': ...}`

这说明：

- 不是压根没调用同步
- 至少代码层面走到了 `hf_separate_rollout_actor` 的状态复制分支

### 3. 但这还不能完全证明 eval 实际用到的是“最新权重”

当前还缺最后一层硬证据：

- `update_engine_weights()` 的确被调用了
- checkpoint 字节内容也变了
- 但没有直接证明 eval 真正消费的是更新后的 rollout actor，而不是某个 stale object / stale state

因此，这一块目前仍然是主要怀疑点之一。

## 对当前现象的具体判断

### 已经可以排除的解释

- **不是“eval 根本没有触发”。**
- **不是“没有 checkpoint 保存能力”。**
- **不是“权重根本没落盘”。**
- **不是纯 W&B 面板画图问题。**

### 还不能排除的解释

- **eval 使用了 stale rollout/eval 模型。**
- **训练 actor 确实变了，但 greedy held-out 输出没有变化。**
- **训练 actor 在数值上有变化，但这些变化主要落在对当前 eval 指标无影响的部分。**
- **eval 复用 `experience_maker` 后，某处存在隐式缓存或状态复用。**

### 当前最值得优先怀疑的点

如果按“链路问题优先”的思路，目前最值得继续盯的点是：

1. `hf_separate_rollout_actor=True` 时，eval 实际走的对象是否就是刚同步过的 `self.inference_engine`
2. `experience_maker.make_experience_list(...)` 在 eval 模式下是否有状态复用
3. `update_engine_weights()` 虽然被调用，但是否真的覆盖了 eval 正在使用的参数对象

## 历史 run 的 checkpoint 位置

这部分保留，是为了后续继续做“train 是否真的更新”时能快速定位旧权重。

### `51q21rc5`

- W&B：`https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/51q21rc5`
- 结果目录：`results/lightrft-ursa8b-stage3-psgrpo/lightrft-ursa8b-stage3-psgrpo-ep10-kl0.001-lr1e-6-20260322_001210`
- 当前保留：
  - `_actor/global_step20`

### `y9sulzln`

- W&B：`https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/y9sulzln`
- 结果目录：`results/lightrft-ursa8b-stage3-psgrpo/lightrft-ursa8b-stage3-psgrpo-ep10-kl0.001-lr1e-6-20260322_101842`
- 历史上保存过：
  - `20/40/60/80/100/120/140`
- 因为 `max_ckpt_num=2`，当前保留：
  - `_actor/global_step120`
  - `_actor/global_step140`

### `unuz8bej`

- W&B：`https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/unuz8bej`
- 结果目录：`results/lightrft-ursa8b-stage3-psgrpo/lightrft-ursa8b-stage3-psgrpo-ep10-kl0.001-lr1e-6-20260324_190752`
- 当前保留：
  - `_actor/global_step20`
  - `_actor/global_step40`

### `pjctfbbs`

- W&B：`https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/pjctfbbs`
- 结果目录：`results/lightrft-ursa8b-stage3-psgrpo/lightrft-ursa8b-stage3-psgrpo-ep10-kl0.001-lr1e-6-20260325_114828`
- 当前保留：
  - `_actor/global_step40`
  - `_actor/global_step60`

### 对照：`da0cx5ok` / `vo45u2kd`

这两个较早 run 当前结果目录里只有 `trajectories/`，没有 `_actor/`：

- `results/lightrft-ursa8b-stage3-real-default-v13b/...20260320_235558`
- `results/lightrft-ursa8b-stage3-real-default-v14/...20260321_005344`

目前没找到它们成功保存 DCP checkpoint 的日志证据。

## 当前还差的验证

下面这些是下一步最有价值的验证项。

### 1. 直接比较 `global_step40` 和 `global_step60` 的参数差异

目标：

- 不只比较 shard 文件哈希
- 要比较若干关键层的 tensor norm / max abs diff / cosine

意义：

- 可以进一步确认“训练 actor 的参数变化到底有多大”

### 2. 用 `global_step40` 和 `global_step60` 在同一 held-out 样本上离线 decode

目标：

- 用相同 prompt、相同 greedy 配置、相同 processor
- 直接比对输出文本是否有变化

意义：

- 可以区分“权重变了但 greedy 输出没变”和“eval 可能拿到 stale 模型”

### 3. 在 eval 前打印 rollout actor / inference engine 的参数签名

目标：

- 在每次 `update_engine_weights()` 后和每次 `evaluate()` 前
- 打印固定若干层参数 checksum / norm

意义：

- 可以最直接验证 eval 实际消费的对象是否被同步过

### 4. 检查 `experience_maker` 是否存在 eval 期间的隐式缓存或状态复用

重点看：

- 生成输入是否每次都重新构造
- 奖励模型/后处理是否引用上一次状态
- 是否存在和 train path 共享的残留状态

## 当前判断的工作结论

截至本次记录，可以明确写下来的结论是：

- `pjctfbbs` 的 eval 问题是真问题，不是只在 W&B 上看起来像问题。
- 当前更像是 **eval 链路没有反映训练后的行为变化**，而不是“eval 根本没跑”。
- 同时，训练侧和 checkpoint 侧也不是完全静止的，至少 actor checkpoint 的字节内容已经变化。
- 因此，后续排查应优先围绕 **eval 是否真正使用了更新后的 rollout/eval 模型** 来做，而不是先把锅完全甩给“模型一点都没训练”。

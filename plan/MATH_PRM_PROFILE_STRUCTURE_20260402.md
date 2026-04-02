# Math PRM Profile 结构说明

本文档说明 `examples/math_prm` 主训练入口里新增的 step profile 是怎么组织的，重点回答四个问题：

- **profile 会写到哪里**
- **每个字段看什么**
- **各 section 的层级关系是什么**
- **W&B 上为什么要等 train step 结束后才看到 `profile/*`**

## 1. 当前 live run

当前按真实配置启动、且**没有手动停止**的 live run：

- 启动方式：
  - `set -a && source .env && set +a`
  - `EXPERIMENT_NAME=lightrft-ursa8b-stage3-profile-eval5-rerun EVAL_STEPS=5 ENABLE_PROFILE=1 bash examples/math_prm/run_grpo_math_prm_ursa_8b.sh`
- W&B run：
  - `lightrft-ursa8b-stage3-profile-eval5-rerun-20260402_104537`
  - `https://wandb.ai/hansbug/LightRFT-URSA8B-Stage3/runs/hkdb6m7c`
- 当前状态：
  - `running`
- 本地 save path：
  - `results/lightrft-ursa8b-stage3-profile-eval5-rerun/lightrft-ursa8b-stage3-profile-eval5-rerun-ep10-kl0.001-lr1e-6-20260402_104537`

当前已经进入：

- `Episode 1`
- `train_step = 1`
- `collect/generate_engine`

实时 partial snapshot 里，当前已经能看到：

- `collect/generate_engine = 181.05s`
- `collect/generate_prepare = 4.37s`
- `collect/generate_engine_ratio = 97.52%`
- `collect/generate_prepare_ratio = 2.35%`

这说明当前第一个 step 的绝对主耗时是 **rollout generation engine**，不是 learn / eval / checkpoint。

## 2. Profile 产物层

启用 `--enable_profile` 后，profile 会写到：

```text
<save_path>/profile/
├── step_profile.current.json
├── step_profile.latest.json
├── step_profile.rank0.jsonl
└── traces/
```

这四类产物的职责不同：

- `step_profile.current.json`
  - **实时中的快照**
  - 按秒 heartbeat 刷新
  - 进程中途被杀，也尽量能留下当前 step 的可读状态
  - 当前文件内容是 **rank0 本地 partial 视角**
- `step_profile.latest.json`
  - **最近一个完整结束 step 的聚合结果**
  - 是 rank0 写出的最终快照
- `step_profile.rank0.jsonl`
  - **完整 step 的历史序列**
  - 每个 train step 一行 JSON
  - 适合后续离线分析
- `traces/`
  - `torch.profiler` 采样 trace
  - 用来做更底层的 CPU/CUDA 事件排查

## 3. 字段层

### 3.1 `current.json`

`step_profile.current.json` 的核心字段：

- `train_step`
  - 当前正在进行的 train step
- `episode`
  - 当前 episode
- `current_elapsed_s`
  - 当前 step 已经过去了多少秒
- `active_section`
  - 现在正在跑哪一个 section
- `active_section_elapsed_s`
  - 当前这个 active section 已经跑了多久
- `last_section`
  - 最近一个刚结束的 section
- `last_elapsed_s`
  - 最近一个刚结束 section 的耗时
- `sections_local_s`
  - **rank0 本地**已经累计下来的 section 秒数
- `sections_local_ratio`
  - `sections_local_s / current_elapsed_s`

这里要特别注意：

- `current.json` 是 **partial + local**
- 它不是全局聚合结果
- 它的 ratio 分母是 **当前 step 已经过掉的时间**

### 3.2 `latest.json` / `rank0.jsonl`

完整 step 结束后，rank0 会写出聚合结果，核心字段是：

- `sections_max_s`
  - 所有 rank 对同名 section 的 **max 秒数**
- `sections_mean_s`
  - 所有 rank 对同名 section 的 **mean 秒数**
- `sections_max_ratio`
  - `sections_max_s / sections_max_s["step/total"]`
- `sections_mean_ratio`
  - `sections_mean_s / sections_mean_s["step/total"]`

这里 `max` 的意义很重要：

- 训练 step 的真实 wall time 通常由**最慢 rank**决定
- 所以对 W&B 和 step 级分析来说，`max_s` 更接近真实瓶颈

当前实现里，W&B 写的是：

- `profile/<flat_name>_s` ← 来自 `sections_max_s`
- `profile/<flat_name>_ratio` ← 来自 `sections_max_ratio`
- `profile/train_step` ← 用来对齐横轴

## 4. Section 层

profile 的 section 名字虽然是扁平字符串，但它们其实有明确的**树状层级**。斜杠 `/` 表示父子关系。

完整树如下：

```text
step/total
├── collect/total
│   ├── collect/generate
│   │   ├── collect/generate_prepare
│   │   ├── collect/generate_engine
│   │   └── collect/generate_build_samples
│   ├── collect/sp_preprocess
│   ├── collect/model_total
│   │   ├── collect/model/actor_logprob
│   │   ├── collect/model/reference_logprob
│   │   ├── collect/model/critic_forward
│   │   └── collect/model/reward_forward
│   ├── collect/sp_postprocess
│   ├── collect/process_rewards
│   └── collect/advantages
├── learn/total
│   ├── learn/sp_preprocess
│   ├── learn/micro_batch_total
│   ├── learn/actor/total
│   │   ├── learn/actor/forward
│   │   ├── learn/actor/loss
│   │   ├── learn/actor/backward
│   │   ├── learn/actor/ptx
│   │   ├── learn/actor/optimizer_step
│   │   └── learn/actor/ema
│   ├── learn/critic/total
│   │   ├── learn/critic/forward
│   │   ├── learn/critic/loss
│   │   ├── learn/critic/backward
│   │   └── learn/critic/optimizer_step
│   ├── learn/update_engine_weights
│   └── learn/save_trajectories
├── eval/total
└── checkpoint/total
```

### 4.1 `step/total`

- 整个 train step 的总 wall time
- 所有 ratio 的根分母都是它

### 4.2 `collect/*`

`collect` 是一次 rollout + experience 构建阶段：

- `collect/generate_prepare`
  - 采样参数准备
  - 多模态数据展开
  - prompt token ids 准备
- `collect/generate_engine`
  - 真正调用 rollout engine 生成
  - 当前 live run 的主瓶颈就在这里
- `collect/generate_build_samples`
  - 把 engine 输出重组为 `Samples` / `SamplesVL`
- `collect/sp_preprocess`
  - SPMD 数据预处理
- `collect/model_total`
  - 模型侧打表阶段总耗时
  - 下面再细分 actor / reference / critic / reward
- `collect/process_rewards`
  - reward shaping、过滤、超长惩罚等
- `collect/advantages`
  - advantage / return 计算

### 4.3 `learn/*`

`learn` 是 PPO 训练更新阶段：

- `learn/sp_preprocess`
  - replay buffer item 先做 SPMD preprocess
- `learn/micro_batch_total`
  - 单个 micro-batch 的整体训练耗时
- `learn/actor/*`
  - actor 前向、loss、backward、optimizer step
  - 如果开 PTX 或 EMA，也单独拆出来
- `learn/critic/*`
  - critic 前向、loss、backward、optimizer step
- `learn/update_engine_weights`
  - 把训练后的 actor 权重推回 rollout engine
- `learn/save_trajectories`
  - 轨迹保存

### 4.4 `eval/*`

- 当前实现只包一层 `eval/total`
- 它表示 runtime eval 的整体耗时
- 触发条件是 `global_step % eval_steps == 0`

### 4.5 `checkpoint/*`

- 当前实现只包一层 `checkpoint/total`
- 表示 checkpoint 保存整体耗时

## 5. 父子关系怎么理解

这里最容易误读的地方有三个。

### 5.1 父节点是“显式包围”的耗时，不一定等于子节点严格求和

比如：

- `collect/total` 显式包住整个 collect 阶段
- `collect/generate` 只是 collect 里的一个子阶段
- `collect/model_total` 只包模型前向阶段

因此：

- 父节点通常**大于等于**子节点求和
- 差值来自未单独打点的代码、同步、框架开销

### 5.2 ratio 不是树内局部占比，而是统一对 `step/total`

例如：

- `collect/generate_engine_ratio = 97%`
- `collect/generate_ratio = 99%`

这两个 ratio **不是**“相对于 `collect/generate` 的内部占比”，而都是：

- `section_s / step_total_s`

所以：

- **父子 ratio 不能直接相加**
- 它们只适合回答“这一段占整个 train step 的多少”

### 5.3 `current.json` 和 W&B 看到的不是同一层语义

- `current.json`
  - 是实时 partial、本地 rank0 视角
- `latest/jsonl`
  - 是完整 step 结束后的全局聚合
- W&B `profile/*`
  - 来自完整 step 结束后的 `sections_max_*`

所以当前 live run 会出现一种现象：

- 本地 `current.json` 已经有很多内容
- 但 W&B 还没有任何 `profile/*`

这不是异常，而是因为：

- **W&B 只在 step 结束后才写 profile**

## 6. W&B 层

W&B 上和 profile 相关的 key 组织是：

```text
profile/train_step
profile/<section>_s
profile/<section>_ratio
profile/episode
```

其中 `<section>` 会把 `/` 展平成 `_`，例如：

- `collect/generate_engine`
  - `profile/collect_generate_engine_s`
  - `profile/collect_generate_engine_ratio`
- `learn/actor/backward`
  - `profile/learn_actor_backward_s`
  - `profile/learn_actor_backward_ratio`

还有两组非 profile 指标：

- `rollout/*`
  - rollout / reward / response_length 等
- `train/*`
  - policy_loss / critic_loss / kl / ratio_mean 等

以及：

- `eval/*`
  - 只有到 `eval_steps` 命中时才会写

## 7. 实际读图建议

如果是看 **“为什么一个 step 很慢”**，优先看：

1. `profile/step_total_s`
2. `profile/collect_total_ratio`
3. `profile/learn_total_ratio`
4. `profile/eval_total_ratio`
5. `profile/checkpoint_total_ratio`

如果发现主要耗在 `collect`，继续看：

1. `profile/collect_generate_ratio`
2. `profile/collect_model_total_ratio`
3. `profile/collect_process_rewards_ratio`
4. `profile/collect_advantages_ratio`

如果主要耗在 `learn`，继续看：

1. `profile/learn_actor_total_ratio`
2. `profile/learn_critic_total_ratio`
3. `profile/learn_update_engine_weights_ratio`

## 8. 当前 live run 的直接结论

对于当前 live run `hkdb6m7c`，在第一个 step 还没结束时，`current.json` 已经非常明确地显示：

- **当前瓶颈是 `collect/generate_engine`**
- 它当前占整个 step 的 **97%+**
- `generate_prepare` 只占 **2% 左右**

也就是说，当前默认配置下：

- profile 不是“没有意义的很多小点”
- 它已经能直接回答：**时间主要花在哪一层**

当前这条 live run 还在继续跑；只要第一个完整 step 结束，W&B 上就会出现第一批 `profile/*` 指标点。

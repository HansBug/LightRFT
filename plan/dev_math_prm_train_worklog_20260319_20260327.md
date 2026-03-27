# `dev/math_prm_train` 工作日志

## 时间范围

- 2026-03-19 至 2026-03-27

## 说明

- 本文只覆盖 `/data/LightRFT` 仓库的 `dev/math_prm_train` 分支。
- 这里的“上周四”按当前日期 `2026-03-27` 计算，对应 `2026-03-19`。
- 统计口径基于 `git log --since='2026-03-19 00:00:00' dev/math_prm_train`。
- 该时间段内，`dev/math_prm_train` 共发生 5 个提交，工作主要集中在 `2026-03-20`、`2026-03-21`、`2026-03-22` 和 `2026-03-26`。

## 一、本周主线

本周 `dev/math_prm_train` 的主线，不是继续把完整工作现场直接往 PR 分支里堆，而是把 `URSA-MATH Stage 3` 在 LightRFT 里的最小可上游化提交面逐步收紧、补齐、稳定下来。

可以概括成三件事：

1. 先把 `dev/math_prm_train` 收成一个最小但完整的 Stage 3 上游分支，只保留训练主链路、URSA 运行时、自定义 reward、manifest/engine 辅助脚本，以及必要的策略与 trainer 改动。
2. 再把 `working` 分支里已经验证过的本地 `hf` rollout、separate rollout actor、runtime eval 和 W&B 指标整理能力，按“只搬必要代码、不搬辅助材料”的原则同步进来。
3. 最后把分支层面的噪音清掉，包括不再需要的 heartbeat logging 和 whitespace 问题，确保这条 PR 分支在行为上可跑、在 diff 上也足够干净。

所以，这一周结束时，`dev/math_prm_train` 的状态已经不是“从零开始接入 URSA”，而是：

- 已经具备精简后的 Stage 3 训练路径。
- 已经具备本地 `hf` rollout 的独立训练/推理装配能力。
- 已经具备 runtime eval 与分离指标上报能力。
- 已经收敛到一条更适合往 upstream PR 发送的最小分支面。

## 二、这周完成的技术工作

### 1. 2026-03-20：建立最小 upstream Stage 3 路径

对应提交：`fec2744cb18412216851ce1eefe3bc0c13d4896e`  
标题：`feature(math_prm): keep minimal upstream stage3 path`

这一提交是本周的基础盘，核心目标是把 `dev/math_prm_train` 明确收成“最小 upstream Stage 3 面”，而不是继续承载完整实验现场。

这一轮主要完成了下面几件事：

- 将 `examples/math_prm/` 收成一条自洽的 Stage 3 训练路径，补入了 `train_colocate.py`、`run_grpo_math_prm_ursa_8b.sh`、`reward_models.py`、`reward_models_utils.py`、`sitecustomize.py` 等核心文件。
- 把 `URSA` 所需的本地运行时代码整体纳入 `examples/math_prm/ursa_model/`，包括 config、processor、vision tower、projector 和主模型定义，避免当前 PR 分支依赖外部仓库运行时代码。
- 保留了最小但必要的辅助脚本面，包括 `tools/prepare_ursa_stage3_manifest.py` 和 `tools/prepare_ursa_engine_checkpoint.py`，让数据转换和 engine wrapper 这两条关键准备路径仍然在 PR 分支内可用。
- 在 `lightrft/` 主路径中同步了 Stage 3 必需的策略、trainer、actor 和 structured output 相关改动，例如 `lightrft/utils/math_prm_output.py`、`lightrft/strategy/strategy_base.py`、`lightrft/trainer/fast_exp_maker.py`、`lightrft/trainer/spmd_ppo_trainer.py` 等。
- 新增中英文 README，明确 `dev/math_prm_train` 不再是通用多模态 reward 示例目录，而是专门服务于 `URSA-MATH Stage 3` 迁移与复现的精简目录。

这一提交的意义在于，`dev/math_prm_train` 从这一步开始真正具备了“可单独审阅、可单独发送 upstream”的基础结构。

### 2. 2026-03-21：同步 Stage 3 rollout 更新，补齐 separate local HF rollout actor

对应提交：`d8590aff6ff8bdde0e76bb317b9997433ef24035`  
标题：`fix(math_prm): sync stage3 rollout updates from working branch`

这一提交把 `dev/math_prm_train_working` 上已经验证过的 Stage 3 rollout 关键改动，选择性搬进了精简 PR 分支。

这次同步的重点不在 reward，而在 rollout 链路本身：

- 为本地 `hf` rollout 增加了 `--hf_separate_rollout_actor` 选项，并打通到 `lightrft/utils/cli_args.py`、`lightrft/strategy/config.py` 和 `lightrft/strategy/strategy_base.py`。
- 在 `strategy_base.py` 里补齐了 separate rollout actor 的状态同步、sleep/wakeup、显存切换和参数复制逻辑，使训练 actor 与 rollout actor 可以分离存在，而不是永远共用同一个实例。
- 在 `examples/math_prm/train_colocate.py` 中补上 rollout actor 的构建与传递逻辑，使 `engine_type='hf'` 下可以走“训练 actor + 独立 rollout actor”的装配方式。
- 在 `examples/math_prm/run_grpo_math_prm_ursa_8b.sh` 中同步了当前可用的 launcher 设置，包括更贴近 Stage 3 的默认超参、W&B 组织配置、以及 local HF rollout 的开关与预检信息。

这一步的价值是把 `dev/math_prm_train` 的本地 `hf` rollout 路径从“能勉强跑”推进到“具备独立 rollout actor 的明确实现”，也为后面的 runtime eval 铺平了基础。

### 3. 2026-03-22：清理 W&B live heartbeat logging

对应提交：`3ff0caf3241c7af43f1ac65c6d2a01f60cb404b6`  
标题：`fix(wandb): remove live heartbeat logging`

这一提交比较小，但它解决的是分支面上的日志噪音问题，而不是训练功能问题。

本次修改主要做了两件事：

- 从 `dev/math_prm_train` 当前 PR 面里移除了 live heartbeat logging 相关的额外处理。
- 保留现有的 train/rollout/eval 正常指标记录逻辑，不再在这条精简分支里继续保留额外的 heartbeat 语义。

这一步的作用，是让 `dev/math_prm_train` 的 W&B 记录路径更聚焦于真实训练指标，而不是继续带着一层为排障服务的运行时心跳逻辑。

### 4. 2026-03-26：同步 runtime eval 路径，并把 eval 从“可调用”提升到“可观测”

对应提交：`6ae4d56121bbbb0148cb32391f751446d996c713`  
标题：`fix(math_prm): sync runtime eval updates from working branch`

这是本周后半段最关键的一次同步。它把 `working` 分支里当前 Stage 3 的 runtime eval 方案，压缩成了适合放进 `dev/math_prm_train` 的最小版本。

这一轮主要完成了下面这些工作：

- 新增 `examples/math_prm/math_prm_trainer.py`，作为 example-local trainer wrapper，把 rollout/train/eval 指标拆开整理，并将 eval 指标独立映射到 `eval/train_step` 这条 X 轴上。
- 在 `math_prm_trainer.py` 里实现 runtime eval context，显式切换 eval 时使用的 `do_sample`、`temperature`、`max_new_tokens`、`n_samples_per_prompt` 和 advantage estimator，避免训练态配置直接污染 eval 配置。
- 在 `train_colocate.py` 里加入 runtime eval 数据集的构建逻辑，支持显式 `eval_data`、`eval_split`，以及在未提供 eval 数据时，从 `prompt_data` 中切出固定大小、固定种子的 held-out 子集。
- 在 launcher `run_grpo_math_prm_ursa_8b.sh` 中同步加入 eval 相关环境变量和命令行参数，包括 `EVAL_STEPS`、`EVAL_MAX_SAMPLES`、`EVAL_HOLDOUT_SIZE`、`EVAL_HOLDOUT_SEED`、`EVAL_N_SAMPLES`、`EVAL_TEMPERATURE` 等。
- 在 `strategy_base.py` 中继续补强 separate local HF rollout actor 的 keep-on-GPU 路径、同步统计和 generation timing / structured stop 观测信息，使 rollout 与 eval 的运行状态更容易在日志中复核。
- 在 `ppo_trainer_vl.py` 中补上对 W&B summary 的主动更新，让关键 train/eval 指标在 run summary 里也能直接看到，而不是只存在于 history 曲线中。
- 同时修剪了中英文 README，把文档中的引用收敛到已经迁入 `dev/math_prm_train` 的实际文件，不再指向尚未搬运的 helper doc 或辅助脚本。

这一提交的意义是，`dev/math_prm_train` 从此不仅能跑 Stage 3 主链路，还能在最小 PR 面内完成 runtime eval 的配置、执行和指标分流。

### 5. 2026-03-26：做 whitespace-only 清理，保证分支级 diff 检查通过

对应提交：`3f5470a102a2a34ff7a536b1e92d1ccc1e233bc6`  
标题：`style(math_prm): remove trailing whitespace from stage3 files`

最后一个提交不引入行为变化，主要做的是分支卫生收尾。

处理内容包括：

- 清理 `examples/math_prm/train_colocate.py` 与 `examples/math_prm/ursa_model/` 下已纳入 PR 分支的若干文件里的 trailing whitespace。
- 保持改动为 whitespace-only，不引入额外逻辑修改。
- 让 `git diff --check` 在这条精简 Stage 3 分支上可以干净通过。

这一步虽然不改变训练行为，但对 `dev/math_prm_train` 这种准备发 upstream 的分支是必要的，因为它直接影响 reviewer 侧的 diff 可读性和基础检查结果。

## 三、当前分支状态

截至 `2026-03-27`，我对 `dev/math_prm_train` 当前状态的判断如下。

### 1. 已经明确保留下来的内容

- `examples/math_prm/` 下已经保留了当前 upstream PR 所需的最小 Stage 3 路径。
- `URSA` 训练与 reward 所需的本地运行时代码已经在分支内自洽，不再依赖完整工作分支中的辅助材料。
- `local hf rollout`、`separate rollout actor`、`runtime eval`、`compact W&B metrics` 这些当前主链路所需的能力，都已经进入 `dev/math_prm_train`。

### 2. 明确没有继续带进来的内容

- `AGENTS.md` / `CLAUDE.md`
- `plan/`
- `tmp/`
- `URSA_MIGRATION.md`
- 其它工作现场性质的临时脚本、排障脚本、性能探针脚本

也就是说，这条分支的目标已经很清楚：它是一个精简 PR 分支，而不是完整工作现场。

### 3. 当前这条分支的技术定位

从技术定位上看，`dev/math_prm_train` 现在承担的是：

- `URSA-MATH Stage 3` 在 LightRFT 中的最小复现路径
- 本地 `hf` rollout 为主的可运行训练路径
- 带 runtime eval 的最小可观测实验路径
- 面向 upstream 审阅的干净提交面

它不再承担“保存全部实验痕迹”和“承载所有工作中间产物”的职责，这部分已经被明确留在 `dev/math_prm_train_working`。

## 四、按提交列出的变更记录

| 日期 | Commit | 标题 |
| --- | --- | --- |
| 2026-03-20 | `fec2744cb184` | `feature(math_prm): keep minimal upstream stage3 path` |
| 2026-03-21 | `d8590aff6ff8` | `fix(math_prm): sync stage3 rollout updates from working branch` |
| 2026-03-22 | `3ff0caf3241c` | `fix(wandb): remove live heartbeat logging` |
| 2026-03-26 | `6ae4d56121bb` | `fix(math_prm): sync runtime eval updates from working branch` |
| 2026-03-26 | `3f5470a102a2` | `style(math_prm): remove trailing whitespace from stage3 files` |

## 五、当前还差哪里 / 问题出在哪里

结合这一周 `dev/math_prm_train` 的实际提交，以及我们前面围绕 runtime eval、W&B 面板和训练观测做过的讨论，当前可以把“还差哪里”和“问题出在哪里”拆成两部分来看。

### 1. 现在还差哪里

- 论文口径里的 `20K -> 15.3K` Stage 3 静态筛选子集，当前本机仍然没有现成可直接给 LightRFT 用的版本；现在还是用转换后的 full manifest，加 `MAX_SAMPLES=15360` 去近似训练规模。
- `runtime eval` 虽然已经补进了 `dev/math_prm_train`，但当前它仍然只是训练期 held-out eval，不是论文 benchmark eval，也还没有形成完整的“训练质量验证闭环”。
- `URSA` 在 LightRFT 里的主可运行链路，当前仍然是本地 `hf rollout`；`vllm` / `sglang` 外部 rollout engine 路径并没有在当前冻结环境下真正恢复成稳定可用方案。
- 本地 `hf` 多模态 rollout 的性能问题还在，当前健康 run 已经能跑通，但生成吞吐依然明显偏慢，这会限制后续大规模实验的迭代效率。
- 更关键的是，训练质量本身还没有稳定下来。当前分支已经把链路、指标和 runtime eval 面板补齐到了“可观测”状态，但这不等于默认 `math_prm` 主入口已经稳定收敛。

### 2. 我对“eval 完全不动”这件事的判断

如果结合我们前面讨论的上下文来看，最近那类“eval acc 看起来像一条直线”的现象，我更倾向于先判断为 **eval 链路与观测语义的问题**，而不是直接判断成“模型完全没有训练”。

原因主要有三层：

1. `dev/math_prm_train` 直到 `2026-03-26` 才同步进 runtime eval 的最小可用实现。也就是说，在这之前，这条分支并没有一个足够可信的 eval 闭环，至少没有把 eval 数据来源、eval 生成参数、eval 指标命名和 W&B 展示语义完整拆开。
2. 这次同步里，专门补了 held-out eval 数据切分、`eval/train_step` 这条独立 X 轴，以及 eval 时单独的 `do_sample` / `temperature` / `n_samples_per_prompt` 配置。这说明之前的问题并不只是“少几个日志字段”，而是 eval 本身的运行语义和展示语义都不够稳定。
3. 从我们已有的其它排查结论看，最近真正被明确定位并修掉的链路问题，主要集中在 rollout / structured output / 本地 `hf` generate 集成上；而当前还没真正闭环的，则是训练质量和训练稳定性。换句话说，训练主链路不是完全没在动，但 eval 这条观测链路此前确实不够可信。

更具体一点说，最近这类“eval 完全不动”的问题，核心更像是下面这几个点叠在一起：

- eval 数据不一定真的是稳定的 held-out 集，之前不能只靠 `eval_split` 就默认它已经成立。
- eval 不能复用训练态 rollout 的高噪声采样设置，否则曲线即便变化，也不适合拿来判断真实趋势。
- eval 指标如果没有单独的 step 语义，只和 train/rollout 混在一起看，W&B 上会出现“看起来像没动”或者“虽然有点，但不可信”的现象。

所以，按照我们前面讨论的思路，当前对这个问题的判断应该是：

- **不是先把锅甩给“模型完全没训练”。**
- **首先要承认 eval 链路此前确实存在实现与观测层面的缺口。**
- **在这部分补齐之后，如果 eval 仍然绝对不动，再去怀疑训练质量、reward 语义或泛化没有改善，才是更稳妥的顺序。**

### 3. 当前更像真正主问题的是什么

在 runtime eval 和 W&B 语义这条线被补齐之后，后续更该盯住的主问题，已经不是“有没有 eval 曲线”，而是：

- 默认 `math_prm` / `math_psgrpo` 主入口的训练质量是否稳定。
- `outcome_correct`、`reward`、`has_drop_moment` 能不能一起朝正确方向走。
- `response_length` 和 `answer_extraction_failed` 会不会再次一起恶化。
- 中段 `KL` 尖峰会不会继续导致“前段学到一点，中段又回撤”的轨迹。

也就是说，当前阶段已经不再是“完全缺观察口”，而是“观察口刚补齐，接下来要真正解决训练质量问题”。

## 六、本周总结

如果只看 `dev/math_prm_train` 这条分支，本周最重要的结果不是“又加了几个脚本”，而是这条分支已经从一个阶段性搬运目标，逐步收敛成了一个可以直接面向 upstream PR 的 Stage 3 精简提交面。

这周完成的事情本质上是：

- 把最小 Stage 3 主链路立住。
- 把本地 `hf` rollout 和 separate rollout actor 的必要能力同步进来。
- 把 runtime eval 与 W&B 指标分流能力同步进来。
- 把分支层面的噪音和格式问题清掉。

从分支治理的角度看，这意味着 `dev/math_prm_train` 已经更接近“可审、可发、可维护”的状态，而不是继续停留在“能跑但还混着完整工作现场材料”的阶段。

# URSA Rollout Engine 失败原因分析

## 1. 背景

本文记录 `URSA-8B` 在 LightRFT Stage 3 复现过程中，为什么最初尝试用 `sglang` 和 `vllm` 作为 rollout engine 都失败，最后必须先切到本地 `hf` rollout 路径，才能把训练主链路跑通。

本文关注的是：

- 当时失败到底发生在什么阶段
- 失败是训练算法问题，还是 engine 兼容性问题
- 当前 LightRFT 现有的两种外部 rollout 架构为什么都不支持 URSA
- URSA 原仓库里“vLLM 可用”到底是怎样一种可用，和当前 LightRFT 场景差在哪里

结论先行：

- 当时失败的主因不是 PPO / GRPO / reward 公式，也不是 Phase 4 还没上。
- 主因是 `sglang` / `vllm` 都需要在各自的 engine runtime 中独立识别、加载并执行 `URSA-8B` 这个自定义多模态架构，而当前 checkpoint、当前 frozen runtime、当前 engine 加载链都不满足这个要求。
- `hf` 路径之所以能支持，是因为它不再要求外部 engine 重新“认识一次 URSA”，而是直接复用 LightRFT 进程内已经成功加载好的 actor。


## 2. 当前 LightRFT rollout 架构前提

LightRFT 当前有三种 rollout 模式：

- `engine_type=vllm`
- `engine_type=sglang`
- `engine_type=hf`

在实现上，这三者不是等价的。

### 2.1 `vllm` / `sglang`

`vllm` 和 `sglang` 都走“外部 inference engine”模式。

对应代码在：

- `lightrft/strategy/strategy_base.py`

关键逻辑：

```python
if engine_type == "vllm":
    self.inference_engine = get_vllm_engine_for_rollout(args)
elif engine_type == "sglang":
    self.inference_engine = get_sglang_engine_for_rollout(args)
```

这意味着：

- rollout 依赖一个单独的 engine runtime
- engine 必须自己加载 checkpoint
- engine 必须自己识别 `config.json` 里的架构
- engine 必须自己处理多模态模型依赖、processor、vision tower、动态图模块加载

换句话说，**不是 LightRFT 主进程能 import URSA 就够了**。

### 2.2 `hf`

`hf` 路径不走独立 engine，而是直接复用训练侧已加载好的 actor。

对应代码在：

- `lightrft/strategy/strategy_base.py`

关键逻辑：

```python
elif engine_type == "hf":
    if actor is None:
        raise ValueError("engine_type='hf' requires the prepared actor to be passed in.")
    self.inference_engine = actor
```

这意味着：

- 不需要 vLLM / SGLang 再单独识别 `URSA`
- 不需要 engine worker 自己再走一遍 `AutoConfig` / `AutoModel`
- 不需要额外的 engine-side 架构注册
- 我们可以直接在仓内修 processor、kwargs、dtype、stopping 等问题

这也是为什么在当前阶段，`hf` 最终能走通，而 `sglang` / `vllm` 走不通。


## 3. 环境与 checkpoint 前提

### 3.1 URSA checkpoint 的架构标识

`URSA-8B` checkpoint 的关键标识是：

- `architectures = ["UrsaForConditionalGeneration"]`
- `model_type = "ursa"`

这在 `plan/MATH_PRM.md` 里已有确认。

这类 checkpoint 对于原生支持该架构的 runtime 没问题，但对通用 engine 来说，意味着至少需要满足下面几件事：

- `AutoConfig` 能认识 `model_type="ursa"`
- `AutoModel` / `AutoModelForVision2Seq` 能定位到 `UrsaForConditionalGeneration`
- 相关本地 Python 模块能在 engine runtime 中被导入
- 相关 vision 子模块也能被递归加载

### 3.2 当前冻结环境

当前 Docker 基线明确冻结了：

- `vllm==0.13.0`
- `sglang==0.5.6.post2`

同时计划文档也明确约束：当前阶段不允许为了临时接入 URSA，随意升级/降级这些关键包。

因此这次分析的前提是：

- 不是“理论上未来换个 runtime 也许可以”
- 而是“在当前 LightRFT 冻结环境里为什么不行”


## 4. 时间线与具体失败日志

### 4.1 第一轮：直接用 `sglang` 跑 raw URSA checkpoint，失败在 `model_type='ursa'`

最早保留下来的失败日志是：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_231456.log`

命令行上下文很清楚，原始日志开头就是：

```text
[run_grpo_math_prm_ursa_8b.sh] WANDB disabled for this run.
+ torchrun --nnodes 1 --nproc-per-node 8 --node_rank 0 --master-port 20192 --master-addr localhost \
  examples/math_prm/train_colocate.py \
  --pretrain /home/ubuntu/URSA-MATH/checkpoints/URSA-8B \
  ... \
  --rm_use_engine \
  --engine_type sglang \
  --engine_mem_util 0.6 \
  --engine_tp_size 2 \
  --enable_engine_sleep \
  ...
```

对应原始位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_231456.log:1`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_231456.log:2`

真正的 fatal error 出现在 engine 初始化阶段，下面这段是原始摘录，能看到它是从 `strategy.setup_inference_engine(...)` 一路掉进 `AutoConfig.from_pretrained(...)` 然后失败：

```text
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/server_args.py", line 979, in _handle_model_specific_adjustments
[rank0]:     hf_config = self.get_hf_config()
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/server_args.py", line 4043, in get_hf_config
[rank0]:     hf_config = get_config(
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/utils/hf_transformers_utils.py", line 262, in get_config
[rank0]:     config = AutoConfig.from_pretrained(
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 1362, in from_pretrained
[rank0]:     raise ValueError(
[rank0]: ValueError: The checkpoint you are trying to load has model type `ursa` but Transformers does not recognize this architecture. This could be because of an issue with the checkpoint, or because your version of Transformers is out of date.

[rank6]: Traceback (most recent call last):
[rank6]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 1360, in from_pretrained
[rank6]:     config_class = CONFIG_MAPPING[config_dict["model_type"]]
[rank6]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/models/auto/configuration_auto.py", line 1048, in __getitem__
[rank6]:     raise KeyError(key)
[rank6]: KeyError: 'ursa'
[rank6]: ...
[rank6]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 432, in train
[rank6]:     strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)
```

对应原始位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_231456.log:6807`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_231456.log:6816`

这轮失败说明：

- `sglang` 当时连 `config.json` 的架构解析都没过去
- 问题发生在 `setup_inference_engine(...)`
- 连第一步 rollout generation 都还没开始

所以这一轮失败和训练质量、reward 设计、PPO update 都没有关系。


### 4.2 第二轮：继续尝试 `sglang`，失败推进到 `auto_map / model module` 层

下一份关键日志是：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_233122.log`

这一轮仍然是：

```text
--engine_type sglang
```

这一轮日志里，fatal error 已经从单纯 `model_type='ursa'` 不认识，推进到了 “知道你是自定义模型，但找不到可加载模块”：

```text
  File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/eplb/expert_location.py", line 525, in from_model_config
    model_class, _ = get_model_architecture(model_config)
  File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/model_loader/utils.py", line 105, in get_model_architecture
    architectures = resolve_transformers_arch(model_config, architectures)
  File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/model_loader/utils.py", line 53, in resolve_transformers_arch
    raise ValueError(
ValueError: Cannot find model module. 'UrsaForConditionalGeneration' is not a registered model in the Transformers library (only relevant if the model is meant to be in Transformers) and 'AutoModel' is not present in the model config's 'auto_map' (relevant if the model is custom).

[2026-03-18 23:35:07] Received sigquit from a child process. It usually means the child failed.
[2026-03-18 23:35:07 TP0] Scheduler hit an exception: Traceback (most recent call last):
  File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/sglang/srt/managers/scheduler.py", line 2680, in run_scheduler_process
    scheduler = Scheduler(
```

对应位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_233122.log:6934`

这一步很关键，因为它说明问题已经从“完全不认识 `model_type=ursa`”推进到了更具体的一层：

- 即使 engine 已经能读到一些架构信息
- 它仍然找不到 `UrsaForConditionalGeneration` 对应的模型模块
- 且 checkpoint 自身也没有给出足够的 `auto_map`

这和后面新增的 wrapper 脚本说明是完全一致的。`examples/math_prm/prepare_ursa_engine_checkpoint.py` 开头直接写明：

```python
The upstream URSA checkpoints do not ship `auto_map` metadata or local model
code files, which prevents inference engines such as vLLM/SGLang from loading
the custom architecture via HuggingFace dynamic modules.
```

也就是说，第二轮失败不是偶发 bug，而是 **URSA 原始 checkpoint 与通用 engine 加载约定不匹配**。


### 4.3 为什么单纯在 LightRFT 进程里注册 URSA auto class 仍然不够

在 `examples/math_prm/train_colocate.py` 中，我们后来增加了：

- `prepare_ursa_runtime_for_inference_engines(...)`

它会做这些事：

- 把 `examples/math_prm` 放入 `sys.path` / `PYTHONPATH`
- `AutoConfig.register("ursa", UrsaConfig, exist_ok=True)`
- `AutoModelForVision2Seq.register(UrsaConfig, UrsaForConditionalGeneration, exist_ok=True)`

这一步有帮助，但它 **并不能自动解决 `sglang` / `vllm` 的外部 engine worker 问题**。

原因是：

- `sglang` / `vllm` 不是简单在当前 trainer 进程里直接调用 `model.generate()`
- 它们会拉起自己的 engine runtime / worker
- worker 自己要重新解析 checkpoint
- worker 自己要在自己的 import 环境里找到这些类

所以：

- 主进程里 `register()` 成功，不等于 engine worker 就能成功
- 对外部 engine 来说，checkpoint 元数据和可导入的本地模块仍然必须完整

这也是为什么尽管后来加了注册逻辑，`sglang` 仍然继续死在模型解析阶段。


### 4.4 第三轮：为 engine 构造 wrapper checkpoint，再试 `vllm`

为了补 checkpoint 元数据，我们新增了：

- `examples/math_prm/prepare_ursa_engine_checkpoint.py`

这个脚本的目标不是改模型权重，而是构造一个 “engine-friendly wrapper checkpoint”，它会：

1. 软链接原始权重和 tokenizer 资产
2. 软链接本地 `examples/math_prm/ursa_model/*.py`
3. 给 `config.json` / `preprocessor_config.json` / `tokenizer_config.json` 补上 `auto_map`

日志里能看到这一轮明确执行了 wrapper：

```text
[run_grpo_math_prm_ursa_8b.sh] Preparing URSA engine wrapper checkpoint at /data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready
[run_grpo_math_prm_ursa_8b.sh] Using wrapped URSA checkpoint: /data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready
```

对应日志：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:3`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:4`

同一轮命令行里已经明确切到了：

```text
--engine_type vllm
```

对应位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:5`


### 4.5 `vllm` 第一层失败：dynamic module 依赖链没有稳定带全

这轮先出现的是大批量 dynamic-module 相关警告。下面是第一段原始摘录，可以看到不是一句话，而是一整段 dynamic module 递归导入后在 cache 路径里找不到 `siglip_vit.py`：

```text
WARNING 03-18 23:46:21 [dynamic_module.py:51]   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/dynamic_module_utils.py", line 135, in get_relative_imports
WARNING 03-18 23:46:21 [dynamic_module.py:51]     with open(module_file, encoding="utf-8") as f:
WARNING 03-18 23:46:21 [dynamic_module.py:51]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
WARNING 03-18 23:46:21 [dynamic_module.py:51] FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/.cache/huggingface/modules/transformers_modules/URSA_hyphen_8B_hyphen_engine_hyphen_ready/siglip_vit.py'
WARNING 03-18 23:46:21 [dynamic_module.py:51] Unable to load modeling_ursa.UrsaForConditionalGeneration from /data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready on HF Hub.
WARNING 03-18 23:46:21 [dynamic_module.py:51] Traceback (most recent call last):
WARNING 03-18 23:46:21 [dynamic_module.py:51]   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/vllm/transformers_utils/dynamic_module.py", line 33, in try_get_class_from_dynamic_module
WARNING 03-18 23:46:21 [dynamic_module.py:51]     return get_class_from_dynamic_module(
WARNING 03-18 23:46:21 [dynamic_module.py:51]   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/dynamic_module_utils.py", line 616, in get_class_from_dynamic_module
WARNING 03-18 23:46:21 [dynamic_module.py:51]     return get_class_in_module(class_name, final_module, force_reload=force_download)
```

后面另一段原始摘录又显示，同一条链路里连 `sam.py` 也会丢：

```text
WARNING 03-18 23:46:21 [dynamic_module.py:51]     with open(module_file, encoding="utf-8") as f:
WARNING 03-18 23:46:21 [dynamic_module.py:51]          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
WARNING 03-18 23:46:21 [dynamic_module.py:51] FileNotFoundError: [Errno 2] No such file or directory: '/home/ubuntu/.cache/huggingface/modules/transformers_modules/URSA_hyphen_8B_hyphen_engine_hyphen_ready/sam.py'
WARNING 03-18 23:46:21 [dynamic_module.py:51] Unable to load modeling_ursa.UrsaForConditionalGeneration from /data/LightRFT/tmp/ursa_stage3/URSA-8B-engine-ready on HF Hub.
WARNING 03-18 23:46:21 [dynamic_module.py:51] Traceback (most recent call last):
[rank4]: Traceback (most recent call last):
[rank4]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 778, in <module>
[rank4]:     train(args)
[rank4]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 472, in train
[rank4]:     strategy.setup_inference_engine(args, engine_type=args.engine_type, actor=actor)
```

对应日志位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:6941`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:6942`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:7008`

这里有一个非常容易误判的点：

- wrapper 目录本身其实已经包含了 `siglip_vit.py` 和 `sam.py`
- 也就是说，**不是 wrapper 根本没把这些文件准备出来**
- 而是 vLLM/Transformers 的 dynamic-module 缓存加载链，在实际运行时并没有稳定把这些相对依赖带进它使用的 cache module 目录

从现象上看，更准确的描述是：

- wrapper 只解决了“checkpoint 自带元数据不足”的一部分问题
- 但 vLLM 仍然依赖一套自己的 dynamic-module 加载机制
- 这套机制在当前 URSA 多文件相对依赖链下并不稳定


### 4.6 `vllm` 第二层失败：即使 dynamic module 问题绕过去，架构本身仍不受支持

这一轮更核心的 fatal error 不是缺少某个 `.py` 文件，而是 vLLM 在构造 `ModelConfig` 时直接拒绝当前架构。下面是原始摘录：

```text
[rank5]:   File "/data/LightRFT/lightrft/strategy/vllm_utils/__init__.py", line 155, in get_vllm_engine
[rank5]:     vllm_engine = LLM(
[rank5]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/vllm/entrypoints/llm.py", line 351, in __init__
[rank5]:     self.llm_engine = LLMEngine.from_engine_args(
[rank5]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/vllm/engine/arg_utils.py", line 1189, in create_model_config
[rank5]:     return ModelConfig(
[rank5]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/pydantic/_internal/_dataclasses.py", line 121, in __init__
[rank5]:     s.__pydantic_validator__.validate_python(ArgsKwargs(args, kwargs), self_instance=s)
[rank5]: pydantic_core._pydantic_core.ValidationError: 1 validation error for ModelConfig
[rank5]:   Value error, Model architectures ['UrsaForConditionalGeneration'] are not supported for now. Supported architectures: dict_keys([...])
[rank5]:     For further information visit https://errors.pydantic.dev/2.12/v/value_error
```

对应位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:7570`
- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:7571`

随后整个 `torchrun` 以 `ChildFailedError` 退出，原始尾部如下：

```text
E0318 23:46:28.308000 926408 site-packages/torch/distributed/elastic/multiprocessing/api.py:882] failed (exitcode: 1) local_rank: 3 (pid: 926627) of binary: /home/ubuntu/miniconda3/envs/lightrft/bin/python3.12
Traceback (most recent call last):
  File "/home/ubuntu/miniconda3/envs/lightrft/bin/torchrun", line 6, in <module>
    sys.exit(main())
  ...
  File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/torch/distributed/launcher/api.py", line 293, in launch_agent
    raise ChildFailedError(
torch.distributed.elastic.multiprocessing.errors.ChildFailedError:
============================================================
examples/math_prm/train_colocate.py FAILED
------------------------------------------------------------
Root Cause (first observed failure):
```

对应位置：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260318_234347.log:7604`

这说明即使我们已经：

- 补了 `auto_map`
- 提供了本地 model code
- 让 engine 至少看到了 `UrsaForConditionalGeneration`

`vllm==0.13.0` 当前这条模型支持链仍然把 URSA 判成 unsupported architecture。

所以这一轮的结论不能写成“还差几个 import 文件”，而应该写成：

- **wrapper 只能补齐 metadata / code discovery**
- **不能让当前 vLLM 版本天然支持一个它自己不支持的架构**


## 5. 两种现有外部 rollout 架构为什么都不支持 URSA

这里的“两种现有架构”，指的是 LightRFT 当前用于 rollout 的两种外部 engine：

- `sglang`
- `vllm`

### 5.1 `sglang` 为什么不支持

`sglang` 在当前环境里不支持，原因至少有四层：

1. `URSA` 是自定义 `model_type='ursa'`

- 原始 checkpoint 不是 Transformers 原生内置模型
- 需要额外注册 `AutoConfig` / `AutoModel`

2. 原始 checkpoint 缺 `auto_map`

- `sglang` 的加载链会依赖 HF config / auto-class 元数据
- 第二轮失败已经明确指出：`AutoModel` 不在 checkpoint 的 `auto_map` 里

3. `sglang` engine worker 是独立 runtime

- LightRFT 主进程里做的 `AutoConfig.register(...)` 不足以保证 engine worker 也能拿到同样的注册状态
- 它需要 checkpoint 自身元数据和 worker import 环境都完整

4. 当前阶段甚至还没走到真正的多模态生成兼容层

- `sglang` 是在 engine 初始化时就失败了
- 所以图像处理、prompt 注入、rollout stopping 这些问题都还没轮到

因此，`sglang` 当前不是“有一点小 bug”，而是 **在模型识别和加载入口层就不成立**。


### 5.2 `vllm` 为什么不支持

`vllm` 在当前环境里不支持，原因比 `sglang` 更清楚，也更硬：

1. 它同样是独立 engine runtime

- 同样要求 checkpoint 元数据、dynamic-module、local code 都满足它的加载要求

2. 即使补了 wrapper，dynamic-module 仍然不稳定

- 日志里直接报 `siglip_vit.py` / `sam.py` 在 engine 使用的 cache 路径中找不到
- 这说明它对 URSA 这种多文件相对导入结构并不稳

3. 当前 `vllm==0.13.0` 最终明确把 `UrsaForConditionalGeneration` 判成 unsupported

- 这是最关键的结论
- 它不是配置问题，而是 runtime 的模型支持表不包含 URSA

因此，`vllm` 当前不是“只差 auto_map”，而是：

- **既有 dynamic-module 装载问题**
- **又有架构支持表本身不包含 URSA 的问题**


## 6. 为什么不能简单说“URSA 原仓库不是已经支持 vLLM 了吗”

这个问题必须单独解释，因为它最容易造成误判。

### 6.1 上游仓库确实有 vLLM 推理入口

在 `plan/MATH_PRM.md` 中，之前已记录：

- 原仓库 vLLM 路径：先执行 `/home/ubuntu/URSA-MATH/start.sh`
- 再用 `/home/ubuntu/URSA-MATH/inference/vllm_infer.py`

这条记录本身没错。

### 6.2 但上游的 vLLM 不是当前环境这套 vLLM

上游 `/home/ubuntu/URSA-MATH/start.sh` 做的第一件事就是：

```bash
pip3 uninstall -y vllm
export VLLM_COMMIT=0b8bb86bf19d68950b4d92a99350e07a26ec0d2c
pip3 install https://vllm-wheels.s3.us-west-2.amazonaws.com/${VLLM_COMMIT}/vllm-1.0.0.dev-...
```

这说明：

- 上游不是直接使用我们当前 Docker 基线里的 `vllm==0.13.0`
- 它依赖一个指定 commit 的 `vllm-1.0.0.dev` wheel
- 还会进入上游仓库自己的 `./vllm/vllm` 目录继续执行 `python_only_dev.py`

因此，上游“能跑 vLLM”的前提是：

- 换 vLLM 版本
- 很可能还带了上游自定义适配

这和当前 LightRFT 的冻结基线根本不是同一个条件。

### 6.3 上游 `vllm_infer.py` 的目标也不是 LightRFT rollout

上游 `/home/ubuntu/URSA-MATH/inference/vllm_infer.py` 做的是：

- 单独加载一个推理用 `LLM(model=model, tensor_parallel_size=1)`
- 对离线评测样本做 `llm.generate(...)`
- 输出推理结果到 jsonl

它不是下面这种场景：

- FSDP actor + rollout engine 分离
- 训练后要 `update_engine_weights(...)`
- rollout 与 reward model、trainer、KL 控制器一起工作
- 多 GPU / tensor parallel 的 RL 训练链路

也就是说，上游脚本证明的是：

- “在上游自己那套 vLLM 环境和推理脚本里，URSA 曾经有过一条可用的 inference path”

它 **不能直接推出**：

- “因此 LightRFT 当前 `vllm==0.13.0` 的 rollout engine 就应该能直接支持 URSA”


## 7. 为什么最后必须转成 `hf` 路径

在当前冻结环境下，`hf` 是当时唯一现实可行的路径，原因有三点。

### 7.1 它绕开了外部 engine 的架构识别问题

`hf` 直接复用已加载好的 actor，不需要让 `sglang` / `vllm` 再独立识别一遍 `URSA`。

这一步直接绕开了：

- `model_type='ursa'` 识别失败
- `auto_map` 不完整
- dynamic-module cache 相对导入问题
- `UrsaForConditionalGeneration` 不在 vLLM 支持表

### 7.2 它把问题降级成“仓内可修的集成问题”

切到 `hf` 后，后续遇到的问题虽然也不少，但都变成了 LightRFT 仓内可以直接修的集成问题：

1. 图像 token 对齐问题

原始摘录：

```text
[rank7]:   File "/data/LightRFT/examples/math_prm/ursa_model/modeling_ursa.py", line 258, in forward
[rank7]:     inputs_embeds, attention_mask, labels, position_ids = self._merge_input_ids_with_image_features(
[rank7]:   File "/data/LightRFT/examples/math_prm/ursa_model/modeling_ursa.py", line 187, in _merge_input_ids_with_image_features
[rank7]:     raise ValueError(
[rank7]: ValueError: The input provided to the model are wrong. The number of image tokens is 0 while the number of image given to the model is 2. This prevents correct indexing and breaks batch generation.
[rank2]: Traceback (most recent call last):
[rank2]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 784, in <module>
[rank2]:     train(args)
[rank2]:   File "/data/LightRFT/lightrft/trainer/fast_exp_maker.py", line 1225, in generate_samples
[rank2]:     all_outputs = self.strategy.gather_and_generate(
```

日志：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_001224.log:6601`

2. `generate()` 参数不兼容

原始摘录：

```text
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/generation/utils.py", line 2388, in generate
[rank0]:     self._validate_model_kwargs(model_kwargs.copy())
[rank0]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/transformers/generation/utils.py", line 1599, in _validate_model_kwargs
[rank0]:     raise ValueError(
[rank0]: ValueError: The following `model_kwargs` are not used by the model: ['image_grid_thw'] (note: typos in the generate arguments will also show up in this list)
[rank2]: Traceback (most recent call last):
[rank2]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 784, in <module>
[rank2]:     train(args)
[rank2]:   File "/data/LightRFT/lightrft/trainer/fast_exp_maker.py", line 1016, in make_experience_list
[rank2]:     samples_list = self.generate_samples(
```

日志：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_002937.log:6585`

3. 视觉分支 dtype 不匹配

原始摘录：

```text
[rank2]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/torch/nn/modules/conv.py", line 548, in forward
[rank2]:     return self._conv_forward(input, self.weight, self.bias)
[rank2]:   File "/home/ubuntu/miniconda3/envs/lightrft/lib/python3.12/site-packages/torch/nn/modules/conv.py", line 543, in _conv_forward
[rank2]:     return F.conv2d(
[rank2]: RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
[rank4]: Traceback (most recent call last):
[rank4]:   File "/data/LightRFT/examples/math_prm/train_colocate.py", line 784, in <module>
[rank4]:     train(args)
[rank4]:   File "/data/LightRFT/lightrft/trainer/fast_exp_maker.py", line 1626, in _make_experience_list_by_model
[rank4]:     output.base_action_log_probs = self.initial_model(
```

日志：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_004531.log:6638`

这些问题后来都通过仓内代码修改被解决了。

### 7.3 `hf` 最终确实把训练主链路跑通了

不是说 `hf` 只是“先绕一下”，而是它后来确实进入了真实 rollout 与训练日志阶段。

例如最新 smoke 日志：

- `/data/LightRFT/tmp/ursa_stage3/phase3_smoke/phase3_smoke_20260319_095529.log`

里面已经能看到：

- `Start VLM gather_and_generate`
- `step 0 generate length: {'min_length': 60, 'max_length': 256, ...}`
- `rollout_reward`
- `rollout_response_length`

这证明：

- URSA 不是“完全不能在 LightRFT 里 rollout”
- 它只是 **不能通过当前 `sglang` / `vllm` 外部 engine 路径稳定 rollout**
- 改成 `hf` 后，主链路是可落地的


## 8. 最终判断

当时为什么 `sglang` / `vllm` rollout 失败，可以压缩成一句话：

> 因为当前 LightRFT 的两种外部 rollout engine 都要求 engine runtime 自己识别和执行 URSA 这一自定义多模态架构，而在当前冻结环境下，SGLang 卡在 `model_type/auto_map/model module` 解析链，vLLM 则同时卡在 dynamic-module 依赖链和 architecture support 表，导致两条路径都无法稳定启动。

为什么后来必须先用 `hf` 路径，也可以压缩成一句话：

> 因为 `hf` 路径不再要求外部 engine 重新加载 URSA，而是直接复用训练侧已经成功构造好的 actor，把问题从“engine 架构支持”降级成“仓内多模态集成细节”，从而在当前环境里具备实际可修复性。


## 9. 如果未来要重新尝试 `sglang` / `vllm` 支持，需要先满足什么

### 9.1 重新尝试 `sglang` 的前提

- checkpoint 自身带完整可用的 `auto_map`
- engine worker 进程可稳定 import `UrsaConfig` / `UrsaForConditionalGeneration`
- 多模态 processor / image tower / local Python modules 在 engine runtime 中都能被递归解析
- 先做独立于 RL 训练的最小 rollout smoke，而不是直接上整条训练链

### 9.2 重新尝试 `vllm` 的前提

- 使用一个真正支持 `UrsaForConditionalGeneration` 的 vLLM 版本或自定义插件路径
- dynamic-module 对 URSA 本地多文件依赖链能稳定工作
- 明确验证该版本在当前多模态 prompt 组织方式下可以正确生成
- 再验证它能接进 LightRFT 的 `update_engine_weights(...)` 和训练循环

如果这些前提不先满足，直接回到 `sglang` / `vllm` rollout，大概率只会重复当时的失败。

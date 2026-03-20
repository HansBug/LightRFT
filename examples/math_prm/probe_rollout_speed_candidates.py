#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path("/data/LightRFT")
EXAMPLE_DIR = ROOT / "examples" / "math_prm"
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from check_hf_rollout import SYSTEM_PROMPT, load_actor
from lightrft.strategy.strategy import get_strategy
from train_colocate import load_actor_tokenizer_processor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark rollout-like URSA generate speed without modifying the training codepath."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "raw_eval_no_gc",
            "raw_train_no_gc",
            "fsdp_train_gc",
            "fsdp_train_no_gc",
            "fsdp_eval_no_gc",
        ],
        required=True,
    )
    parser.add_argument("--model-path", default="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B")
    parser.add_argument("--manifest-path", default="/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl")
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--local-samples", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def load_records(manifest_path: str, limit: int):
    records = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            images = item.get("images") or []
            if images:
                records.append((item["prompt"], images[0]))
            if len(records) >= limit:
                break
    if len(records) < limit:
        raise RuntimeError(f"Expected at least {limit} multimodal records in {manifest_path}")
    return records


def build_prompts_and_images(processor, records, local_samples: int):
    prompts = []
    images = []
    for idx in range(local_samples):
        question, image_path = records[idx % len(records)]
        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"<|image|>{question}"},
        ]
        prompts.append(processor.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True))
        images.append(Image.open(image_path).convert("RGB"))
    return prompts, images


def tensorize(processor, prompts, images, device: torch.device):
    batch = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            if torch.is_floating_point(value):
                batch[key] = value.to(device=device, dtype=torch.bfloat16)
            else:
                batch[key] = value.to(device)
    return batch


def chunk_batch(batch, start: int, end: int):
    chunk = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            chunk[key] = value[start:end]
        else:
            chunk[key] = value
    return chunk


def maybe_configure_gc(actor, enabled: bool):
    if enabled:
        actor.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        actor.gradient_checkpointing_disable()


def init_dist_if_needed():
    if dist.is_available() and not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group(backend="nccl")


def maybe_barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def world_info():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def make_fsdp_args():
    return SimpleNamespace(
        seed=42,
        max_norm=1.0,
        micro_train_batch_size=1,
        train_batch_size=8,
        bf16=True,
        zero_stage=3,
        fsdp=True,
        fsdp_cpu_offload=False,
        adam_offload=False,
        zpg=1,
        grad_accum_dtype=None,
        overlap_comm=False,
        engine_type="hf",
        engine_tp_size=1,
        local_hf_generate_max_batch_size=0,
        enable_engine_sleep=False,
        local_rank=-1,
        sp_size=1,
        actor_learning_rate=2e-6,
        critic_learning_rate=9e-6,
        adam_betas=(0.9, 0.95),
        l2=1.0e-2,
        lr_warmup_ratio=0.03,
        critic_pretrain=None,
        remote_rm_url=None,
        pretrain_data=None,
        fused_linear_logprob=False,
        reward_running_norm=False,
        reward_running_norm_minus_mean=False,
        advantages_norm=False,
        advantage_clip=0.0,
        reward_clip=0.0,
        micro_rollout_batch_size=1,
        rollout_batch_size=8,
        use_dynamic_batch=False,
        pack_max_length=0,
        ring_attn_size=1,
        ring_head_stride=1,
        ds_tensor_parallel_size=1,
        fsdp_sharding_strategy="FULL_SHARD",
        fsdp_bwd_prefetch="BACKWARD_PRE",
        fsdp_reshard_after_forward=False,
        fsdp_cpu_ram_efficient_loading=False,
        fsdp_sync_module_states=True,
        fsdp_layer_cls_to_wrap=None,
        use_liger_kernel=False,
        torch_compile=False,
        torch_compile_mode="default",
        use_wandb=None,
        use_tensorboard=None,
        wandb_run_name=None,
        wandb_project="debug",
        wandb_org=None,
        checkpoint_path="./tmp/fsdp_probe_ckpt",
    )


def summarize(args, elapsed: float, generated_tokens, peak_mem_gb: float):
    total_tokens = sum(generated_tokens)
    return {
        "mode": args.mode,
        "prompt_count": args.prompt_count,
        "local_samples": args.local_samples,
        "chunk_size": args.chunk_size,
        "max_new_tokens": args.max_new_tokens,
        "time_sec": round(elapsed, 3),
        "generated_tokens_mean": round(total_tokens / max(1, len(generated_tokens)), 3),
        "generated_tokens_max": max(generated_tokens) if generated_tokens else 0,
        "tokens_per_sec": round(total_tokens / max(elapsed, 1e-6), 3),
        "peak_mem_gb": round(peak_mem_gb, 3),
    }


def run_generate(actor, batch, chunk_size: int, args, tokenizer, device: torch.device):
    generated_tokens = []
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    maybe_barrier()
    torch.cuda.synchronize(device)
    t0 = time.time()
    for start in range(0, batch["input_ids"].shape[0], chunk_size):
        end = min(start + chunk_size, batch["input_ids"].shape[0])
        chunk = chunk_batch(batch, start, end)
        with torch.no_grad():
            _, attention_mask, _ = actor.generate(
                input_ids=chunk["input_ids"],
                attention_mask=chunk["attention_mask"],
                pixel_values=chunk.get("pixel_values"),
                image_grid_thw=chunk.get("image_grid_thw"),
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=1,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=None,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens.extend(
            attention_mask.sum(dim=1).sub(chunk["attention_mask"].sum(dim=1)).detach().cpu().tolist()
        )
    torch.cuda.synchronize(device)
    elapsed = time.time() - t0
    return elapsed, generated_tokens, torch.cuda.max_memory_allocated(device) / 1024**3


def main():
    args = parse_args()
    use_fsdp = args.mode.startswith("fsdp")
    if not use_fsdp:
        init_dist_if_needed()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    rank, world_size = world_info()
    device = torch.device(f"cuda:{local_rank}")

    records = load_records(args.manifest_path, args.prompt_count)
    train_mode = "train" in args.mode
    use_gc = args.mode.endswith("gc") or args.mode == "fsdp_train_gc"
    if args.mode in {"raw_eval_no_gc", "raw_train_no_gc", "fsdp_train_no_gc", "fsdp_eval_no_gc"}:
        use_gc = False

    if use_fsdp:
        strategy = get_strategy(make_fsdp_args())
        rank, world_size = world_info()
    else:
        strategy = None

    actor = load_actor(args.model_path, device, use_flash_attn=False)
    tokenizer, processor = load_actor_tokenizer_processor(
        model_path=args.model_path,
        model=actor.model,
        strategy=strategy if strategy is not None else SimpleNamespace(print=lambda *a, **k: None),
        use_fast=True,
    )
    maybe_configure_gc(actor, use_gc)

    if use_fsdp:
        actor = strategy.prepare_model(actor, is_training=True)
    actor.train(mode=train_mode)

    prompts, images = build_prompts_and_images(processor, records, args.local_samples)
    batch = tensorize(processor, prompts, images, device)

    elapsed, generated_tokens, peak_mem_gb = run_generate(
        actor=actor,
        batch=batch,
        chunk_size=args.chunk_size,
        args=args,
        tokenizer=tokenizer,
        device=device,
    )

    elapsed_tensor = torch.tensor([elapsed], device=device, dtype=torch.float64)
    peak_mem_tensor = torch.tensor([peak_mem_gb], device=device, dtype=torch.float64)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_mem_tensor, op=dist.ReduceOp.MAX)

    summary = summarize(args, float(elapsed_tensor.item()), generated_tokens, float(peak_mem_tensor.item()))
    if rank == 0:
        print(json.dumps(summary, ensure_ascii=False))
        if args.output_json:
            output_path = Path(args.output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Minimal validation for LightRFT local HF rollout with URSA-8B.

This script exercises the same local-HF rollout path used by LightRFT Phase 3:

1. load URSA-8B as the actor
2. load the URSA tokenizer/processor
3. setup `engine_type="hf"` via StrategyBase
4. run `gather_and_generate()` on a tiny multimodal batch
5. compare the rollout outputs against direct `actor.generate()`
6. report output-structure observations separately from rollout-engine health

Usage:

    python examples/math_prm/check_hf_rollout.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from lightrft.strategy import StrategyBase
from lightrft.strategy.fake_strategy import FakeStrategy
from lightrft.utils.math_prm_output import should_stop_math_prm_response_text
from train_colocate import load_actor_tokenizer_processor, prepare_ursa_runtime_for_inference_engines
from ursa_actor import UrsaActor
from ursa_model import UrsaForConditionalGeneration
from lightrft.strategy.strategy_base import _StructuredAnswerEosLogitsProcessor


SYSTEM_PROMPT = (
    "A conversation between the User and Assistant. "
    "The User asks a question that may require mathematical or visual reasoning, "
    "and the Assistant solves it step by step. "
    'Each step MUST begin with "Step N:" (e.g. "Step 1:", "Step 2:") on its own line. '
    'After all steps, output exactly one final answer line prefixed with "†Answer:" '
    '(e.g. "†Answer: 42"). Stop immediately after the "†Answer:" line and do not output '
    "any extra text, repeated answer markers, or additional steps."
)


@dataclass
class Sample:
    name: str
    question: str
    image_path: str


DEFAULT_SAMPLES = [
    Sample(
        name="vqa_flatness",
        question="Is the landscape flat?",
        image_path="/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/data_images/VQA2.0/images/156768.jpg",
    ),
    Sample(
        name="table_linear_eq",
        question=(
            "Please fill in the table provided below by employing the linear equation "
            "given by y = 5x + 7. The potential answers to choose from include: "
            "38, 40, 41, 37, and 39."
        ),
        image_path="/home/ubuntu/URSA-MATH/datasets/URSA-MATH/images/VarsityTutors/images1/30-1090.png",
    ),
]


class HFLocalRolloutCheckStrategy(FakeStrategy):
    def setup_inference_engine(self, args, engine_type="vllm", actor=None, tokenizer=None, processor=None):
        return StrategyBase.setup_inference_engine(
            self,
            args,
            engine_type=engine_type,
            actor=actor,
            tokenizer=tokenizer,
            processor=processor,
        )

    def engine_generate_local(
        self,
        sampling_params,
        prompt_token_ids=None,
        multi_modal_inputs=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        return StrategyBase.engine_generate_local(
            self,
            sampling_params=sampling_params,
            prompt_token_ids=prompt_token_ids,
            multi_modal_inputs=multi_modal_inputs,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
        )

    def gather_and_generate(
        self,
        sampling_params,
        all_prompt_token_ids=None,
        all_prompts=None,
        all_images=None,
        sleep_engine=True,
        images_num=None,
        all_videos=None,
        videos_num=None,
        all_images_pixel_values=None,
        all_videos_pixel_values=None,
        all_images_grid_thw=None,
        all_videos_grid_thw=None,
    ):
        return StrategyBase.gather_and_generate(
            self,
            sampling_params=sampling_params,
            all_prompt_token_ids=all_prompt_token_ids,
            all_prompts=all_prompts,
            all_images=all_images,
            sleep_engine=sleep_engine,
            images_num=images_num,
            all_videos=all_videos,
            videos_num=videos_num,
            all_images_pixel_values=all_images_pixel_values,
            all_videos_pixel_values=all_videos_pixel_values,
            all_images_grid_thw=all_images_grid_thw,
            all_videos_grid_thw=all_videos_grid_thw,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal URSA local-HF rollout validation for LightRFT.")
    parser.add_argument("--model-path", default="/home/ubuntu/URSA-MATH/checkpoints/URSA-8B")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--use-flash-attn", action="store_true", default=False)
    parser.add_argument("--output-json", type=str, default=None)
    return parser.parse_args()


def ensure_inputs_exist(samples: list[Sample], model_path: str) -> None:
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    for sample in samples:
        if not Path(sample.image_path).exists():
            raise FileNotFoundError(f"Image path does not exist: {sample.image_path}")


def ensure_single_process_group() -> tuple[bool, str | None]:
    if dist.is_initialized():
        return False, None

    fd, path = tempfile.mkstemp(prefix="lightrft_hf_rollout_", suffix=".pg")
    os.close(fd)
    init_method = f"file://{path}"
    dist.init_process_group(
        backend="nccl",
        init_method=init_method,
        rank=0,
        world_size=1,
    )
    return True, path


def build_prompt(processor, question: str) -> str:
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"<|image|>{question}"},
    ]
    return processor.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_actor(model_path: str, device: torch.device, use_flash_attn: bool) -> UrsaActor:
    attn_implementation = "flash_attention_2" if use_flash_attn else "eager"
    model = UrsaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    ).to(device)
    actor = UrsaActor(model)
    actor.eval()
    return actor


def batch_processor_inputs(processor, prompts: list[str], images: list[Image.Image], device: torch.device) -> dict[str, Any]:
    batch = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    output = {
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "pixel_values": None,
        "image_grid_thw": None,
    }
    if "pixel_values" in batch:
        output["pixel_values"] = batch["pixel_values"].to(device=device, dtype=torch.bfloat16)
    if "image_grid_thw" in batch:
        output["image_grid_thw"] = batch["image_grid_thw"].to(device)
    return output


def extract_prompt_token_ids(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> list[list[int]]:
    prompt_token_ids = []
    input_ids_cpu = input_ids.detach().cpu()
    attention_mask_cpu = attention_mask.detach().cpu()
    for row_ids, row_mask in zip(input_ids_cpu, attention_mask_cpu):
        prompt_token_ids.append(row_ids[row_mask.bool()].tolist())
    return prompt_token_ids


def direct_generate_outputs(
    actor: UrsaActor,
    tokenizer,
    model_inputs: dict[str, Any],
    sampling_params: dict[str, Any],
) -> list[list[int]]:
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_token_id
    prompt_length = int(model_inputs["input_ids"].size(1))
    logits_processor = None
    if sampling_params.get("structured_answer_stop", False):
        logits_processor = LogitsProcessorList(
            [
                _StructuredAnswerEosLogitsProcessor(
                    tokenizer,
                    prompt_length,
                    eos_token_id,
                )
            ]
        )

    generate_kwargs = {
        "input_ids": model_inputs["input_ids"],
        "attention_mask": model_inputs["attention_mask"],
        "pixel_values": model_inputs["pixel_values"],
        "image_grid_thw": model_inputs["image_grid_thw"],
        "logits_processor": logits_processor,
        "do_sample": sampling_params.get("do_sample", False),
        "max_new_tokens": sampling_params.get("max_new_tokens", 256),
        "min_new_tokens": sampling_params.get("min_new_tokens", 1),
        "repetition_penalty": sampling_params.get("repetition_penalty", 1.0),
        "no_repeat_ngram_size": sampling_params.get("no_repeat_ngram_size", 0),
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
    }
    for optional_key in ("top_k", "top_p", "temperature"):
        if optional_key in sampling_params and sampling_params[optional_key] is not None:
            generate_kwargs[optional_key] = sampling_params[optional_key]

    with torch.no_grad():
        sequences, attention_mask_out, _ = actor.generate(**generate_kwargs)

    sequences = sequences.detach().cpu()
    attention_mask_out = attention_mask_out.detach().cpu()
    direct_output_ids = []
    for row_idx in range(sequences.size(0)):
        total_length = int(attention_mask_out[row_idx].sum().item())
        total_length = max(total_length, prompt_length)
        direct_output_ids.append(sequences[row_idx, prompt_length:total_length].tolist())
    return direct_output_ids


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for URSA local-HF rollout validation.")
    if not args.device.startswith("cuda:"):
        raise ValueError("This validation script expects a CUDA device like cuda:0.")

    samples = list(DEFAULT_SAMPLES)
    ensure_inputs_exist(samples, args.model_path)

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    created_pg, pg_file = ensure_single_process_group()

    try:
        strategy = HFLocalRolloutCheckStrategy(args=None)
        # `StrategyConfig()` defaults `use_tensorboard=False`, but `GenLenAnalyser`
        # treats any non-None value as an enabled output directory.
        strategy.config.use_tensorboard = None
        strategy.genlen_analyser.plot_out_dir = None
        prepare_ursa_runtime_for_inference_engines(strategy)

        actor = load_actor(args.model_path, device, use_flash_attn=args.use_flash_attn)
        tokenizer, processor = load_actor_tokenizer_processor(
            model_path=args.model_path,
            model=actor.model,
            strategy=strategy,
            use_fast=True,
        )
        strategy.setup_inference_engine(
            args=None,
            engine_type="hf",
            actor=actor,
            tokenizer=tokenizer,
            processor=processor,
        )

        prompts = [build_prompt(processor, sample.question) for sample in samples]
        images = [Image.open(sample.image_path).convert("RGB") for sample in samples]
        model_inputs = batch_processor_inputs(processor, prompts, images, device)
        prompt_token_ids = extract_prompt_token_ids(model_inputs["input_ids"], model_inputs["attention_mask"])

        sampling_params = {
            "max_new_tokens": args.max_new_tokens,
            "min_new_tokens": 1,
            "do_sample": False,
            "top_p": 1.0,
            "top_k": None,
            "repetition_penalty": 1.05,
            "no_repeat_ngram_size": 4,
            "structured_answer_stop": True,
        }

        rollout_outputs = strategy.gather_and_generate(
            sampling_params=sampling_params,
            all_prompt_token_ids=prompt_token_ids,
            all_prompts=prompts,
            all_images=images,
            sleep_engine=False,
            images_num=[1 for _ in samples],
            all_images_pixel_values=model_inputs["pixel_values"],
            all_images_grid_thw=model_inputs["image_grid_thw"],
        )
        direct_output_ids = direct_generate_outputs(
            actor=actor,
            tokenizer=tokenizer,
            model_inputs=model_inputs,
            sampling_params=sampling_params,
        )

        decoded_outputs = tokenizer.batch_decode(
            [output.output_token_ids for output in rollout_outputs],
            skip_special_tokens=True,
        )

        summary_samples = []
        rollout_checks = {
            "engine_type_is_hf": strategy.inference_engine_type == "hf",
            "engine_reuses_actor": strategy.inference_engine is actor,
            "num_outputs_match": len(rollout_outputs) == len(samples),
            "all_non_empty": True,
            "all_match_direct_generate": True,
            "all_below_length_cap": True,
        }
        quality_checks = {
            "all_have_step_marker": True,
            "all_have_answer_marker": True,
            "all_stop_condition_satisfied": True,
        }

        for sample, rollout_output, direct_ids, decoded_text in zip(samples, rollout_outputs, direct_output_ids, decoded_outputs):
            stripped_text = decoded_text.strip()
            has_step_marker = "Step 1:" in stripped_text
            has_answer_marker = "†Answer:" in stripped_text
            stop_condition = should_stop_math_prm_response_text(stripped_text)
            below_cap = len(rollout_output.output_token_ids) < args.max_new_tokens
            tokens_match = rollout_output.output_token_ids == direct_ids

            rollout_checks["all_non_empty"] &= bool(stripped_text)
            rollout_checks["all_match_direct_generate"] &= tokens_match
            rollout_checks["all_below_length_cap"] &= below_cap
            quality_checks["all_have_step_marker"] &= has_step_marker
            quality_checks["all_have_answer_marker"] &= has_answer_marker
            quality_checks["all_stop_condition_satisfied"] &= stop_condition

            summary_samples.append(
                {
                    "name": sample.name,
                    "question": sample.question,
                    "image_path": sample.image_path,
                    "output_tokens": len(rollout_output.output_token_ids),
                    "tokens_match_direct_generate": tokens_match,
                    "has_step_marker": has_step_marker,
                    "has_answer_marker": has_answer_marker,
                    "stop_condition_satisfied": stop_condition,
                    "below_length_cap": below_cap,
                    "generated_text": stripped_text,
                }
            )

        success = all(rollout_checks.values())
        strict_structure_success = all(quality_checks.values())
        summary = {
            "success": success,
            "strict_structure_success": strict_structure_success,
            "model_path": args.model_path,
            "device": args.device,
            "engine_type": strategy.inference_engine_type,
            "system_prompt": SYSTEM_PROMPT,
            "sampling_params": sampling_params,
            "success_definition": (
                "success only validates the LightRFT local-HF rollout path itself: "
                "the script must exercise setup_inference_engine(engine_type='hf') and "
                "gather_and_generate(), and the rollout outputs must exactly match direct "
                "actor.generate() token-by-token."
            ),
            "rollout_checks": rollout_checks,
            "quality_checks": quality_checks,
            "samples": summary_samples,
            "peak_mem_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2),
        }

        payload = json.dumps(summary, ensure_ascii=False, indent=2)
        print(payload)
        if args.output_json:
            Path(args.output_json).write_text(payload + "\n", encoding="utf-8")

        return 0 if success else 1
    finally:
        if created_pg and dist.is_initialized():
            dist.destroy_process_group()
        if created_pg and pg_file is not None:
            Path(pg_file).unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())

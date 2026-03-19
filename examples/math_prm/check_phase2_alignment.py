import argparse
import json
import sys
from pathlib import Path

import regex as re
import torch
from PIL import Image


THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from reward_models import MathPRMReward
from ursa_model import UrsaForTokenClassification, UrsaProcessor


DEFAULT_MODEL_PATH = "/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
DEFAULT_IMAGE_PATH = "/home/ubuntu/URSA-MATH/figures/framework.png"
DEFAULT_QUESTION = "How many numbered training stages are shown in this diagram?"
DEFAULT_RESPONSE = (
    "Step 1: The diagram labels Stage 1 as VL Alignment.\n"
    "Step 2: The diagram labels Stage 2 as Math SFT.\n"
    "Step 3: The diagram labels Stage 3 as PRM Training and Verifying.\n"
    "†Answer: 3"
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare LightRFT MathPRMReward against the URSA reference scorer.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--image-path", default=DEFAULT_IMAGE_PATH)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--response", default=DEFAULT_RESPONSE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    return parser.parse_args()


REFERENCE_PROMPT = (
    "You are given a problem and a step-by-step solution. "
    "You need to check the correctness of each step.\nQuestion:"
)


def reference_return_score(scores: torch.Tensor, operation: str):
    if operation == "min":
        if scores.numel() == 0:
            return torch.tensor([0.0], device=scores.device)
        return torch.min(scores.view(-1))
    if operation == "avg":
        if scores.numel() == 0:
            return torch.tensor([0.0], device=scores.device)
        return torch.mean(scores.view(-1))
    raise ValueError(f"Unsupported operation: {operation}")


def reference_replace_specific_plus_minus_with_ki(text: str) -> str:
    pattern = r"Step \d+"
    matches = list(re.finditer(pattern, text))
    positions = [(match.start(), match.end()) for match in matches]

    text_list = list(text)
    insert_pos = []
    try:
        for i in range(1, len(positions)):
            for j in range(positions[i][0] - 1, positions[i - 1][1], -1):
                if text_list[j] != " " and text_list[j] != "\n":
                    insert_pos.append(j + 1)
                    break

        answer_start = text.find("†Answer:")
        for j in range(answer_start - 1, positions[-1][1], -1):
            if text_list[j] != " " and text_list[j] != "\n":
                insert_pos.append(j + 1)
                break
        for index in sorted(insert_pos, reverse=True):
            text = text[:index] + " и" + text[index:]
        return text
    except Exception:
        return text + " и"


def reference_prepare_input(question: str, response: str) -> str:
    if not question or isinstance(question, float):
        instruction = REFERENCE_PROMPT + "\n" + response
    else:
        instruction = REFERENCE_PROMPT + question + "\n" + response
    return reference_replace_specific_plus_minus_with_ki(instruction)


def reference_single_inference(
    processor: UrsaProcessor,
    model: UrsaForTokenClassification,
    cuda_device: int,
    input_prompt: str,
    image: Image.Image | None,
    system_prompt: str = "You are a helpful assistant.",
):
    conv = [{"role": "system", "content": system_prompt}] if system_prompt else []
    conv.append({"role": "user", "content": "<|image|>" + input_prompt})
    prompt = processor.apply_chat_template(conv, add_generation_prompt=True)
    raw_image = [image.convert("RGB") if image is not None else None]
    inputs = processor(prompt, raw_image, return_tensors="pt").to(cuda_device, torch.bfloat16)
    tag_id = processor.tokenizer.encode(" и", add_special_tokens=False)
    with torch.inference_mode():
        reward = model(**inputs).logits
        input_ids = inputs["input_ids"].view(-1)
        insert_values = torch.full((575,), -1, device=input_ids.device)
        input_ids = torch.cat((input_ids[:1], insert_values, input_ids[1:]))
        reward = reward.view(-1)[input_ids == tag_id[0]]
        reward = torch.sigmoid(reward).view(-1)
        min_score = reference_return_score(reward, "min")
        avg_score = reference_return_score(reward, "avg")
    return min_score, avg_score


def parse_cuda_index(device_str: str) -> int:
    if not device_str.startswith("cuda:"):
        raise ValueError("This alignment script expects a CUDA device like cuda:0.")
    return int(device_str.split(":", 1)[1])


def build_prompt_and_output(question: str, response: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>"
        f"<|im_start|>user\n<|image|>{question}<|im_end|>"
        f"<|im_start|>assistant\n{response}"
    )


def main():
    args = parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)

    processor = UrsaProcessor.from_pretrained(args.model_path)
    model = UrsaForTokenClassification.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    prepared_input_reference = reference_prepare_input(args.question, args.response)
    prepared_input_lightrft = MathPRMReward._prepare_prm_input(
        MathPRMReward.__new__(MathPRMReward),
        args.question,
        args.response,
    )

    with Image.open(args.image_path) as image:
        image = image.convert("RGB")
        min_ref, avg_ref = reference_single_inference(
            processor=processor,
            model=model,
            cuda_device=parse_cuda_index(args.device),
            input_prompt=prepared_input_reference,
            image=image,
        )

        prompt_and_output = build_prompt_and_output(args.question, args.response)
        min_reward_output = MathPRMReward(model, processor, aggregation="min")(
            sequences=None,
            attention_mask=None,
            prompt_and_output=[prompt_and_output],
            raw_images=[image],
        )
        avg_reward_output = MathPRMReward(model, processor, aggregation="avg")(
            sequences=None,
            attention_mask=None,
            prompt_and_output=[prompt_and_output],
            raw_images=[image],
        )
        last_reward_output = MathPRMReward(model, processor, aggregation="last")(
            sequences=None,
            attention_mask=None,
            prompt_and_output=[prompt_and_output],
            raw_images=[image],
        )

        def _scalar_reward(output):
            score = output["score"] if isinstance(output, dict) else output
            return score[0].item()

        min_reward = _scalar_reward(min_reward_output)
        avg_reward = _scalar_reward(avg_reward_output)
        last_reward = _scalar_reward(last_reward_output)

    result = {
        "prepared_input_match": prepared_input_reference == prepared_input_lightrft,
        "prepared_input_reference": prepared_input_reference,
        "prepared_input_lightrft": prepared_input_lightrft,
        "reference": {
            "min": float(min_ref.item()),
            "avg": float(avg_ref.item()),
        },
        "lightrft": {
            "min": float(min_reward),
            "avg": float(avg_reward),
            "last": float(last_reward),
        },
        "delta": {
            "min": abs(float(min_ref.item()) - float(min_reward)),
            "avg": abs(float(avg_ref.item()) - float(avg_reward)),
        },
        "within_tolerance": (
            abs(float(min_ref.item()) - float(min_reward)) <= args.tolerance
            and abs(float(avg_ref.item()) - float(avg_reward)) <= args.tolerance
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

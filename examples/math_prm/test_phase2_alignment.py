import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import regex as re
import torch


MATH_PRM_DIR = Path(__file__).resolve().parent
if str(MATH_PRM_DIR) not in sys.path:
    sys.path.insert(0, str(MATH_PRM_DIR))

from reward_models import MathPRMReward
from reward_models_utils import RewardModelType, load_reward_models


REFERENCE_PROMPT = (
    "You are given a problem and a step-by-step solution. "
    "You need to check the correctness of each step.\nQuestion:"
)


def reference_replace_specific_plus_minus_with_ki(text):
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


def reference_prepare_input(question, response):
    if not question or isinstance(question, float):
        instruction = REFERENCE_PROMPT + "\n" + response
    else:
        instruction = REFERENCE_PROMPT + question + "\n" + response
    return reference_replace_specific_plus_minus_with_ki(instruction)


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.pad_token_id = None
        self.eos_token = "<eos>"
        self.eos_token_id = 9
        self.padding_side = "right"

    def encode(self, text, add_special_tokens=False):
        self.last_encoded = (text, add_special_tokens)
        return [42]


class FakeBatch(dict):
    def to(self, device, dtype=None):
        converted = {}
        for key, value in self.items():
            if isinstance(value, torch.Tensor):
                target_dtype = dtype if dtype is not None and torch.is_floating_point(value) else value.dtype
                converted[key] = value.to(device=device, dtype=target_dtype)
            else:
                converted[key] = value
        return FakeBatch(converted)


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()
        self.calls = []
        self.chat_templates = []

    def apply_chat_template(self, conv, add_generation_prompt=True, tokenize=False):
        self.chat_templates.append(
            {
                "conv": conv,
                "add_generation_prompt": add_generation_prompt,
                "tokenize": tokenize,
            }
        )
        return "formatted-prompt"

    def __call__(self, text, images, return_tensors="pt"):
        self.calls.append({"text": text, "images": images, "return_tensors": return_tensors})
        return FakeBatch(
            {
                "input_ids": torch.tensor([[11, 42, 13]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            }
        )


class FakeRewardModel(torch.nn.Module):
    def __init__(self, score_logit: float = 0.8):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.score_logit = score_logit

    def forward(self, **inputs):
        input_ids = inputs["input_ids"].view(-1)
        seq_len = input_ids.shape[0] + MathPRMReward._IMAGE_PAD
        logits = torch.zeros((1, seq_len, 1), dtype=torch.float32, device=input_ids.device)
        logits[0, MathPRMReward._IMAGE_PAD + 1, 0] = self.score_logit
        return SimpleNamespace(logits=logits)


class FakeStrategy:
    def __init__(self):
        self.args = SimpleNamespace(text_only=False)
        self.messages = []

    def print(self, message):
        self.messages.append(str(message))

    def init_model_context(self):
        class _NoopContext:
            def __enter__(self_inner):
                return None

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return _NoopContext()


class FakeConfig:
    def __init__(self):
        self.pad_token_id = None


class FakeModelForProcessor:
    def __init__(self):
        self.config = FakeConfig()


def _minimal_math_prm_instance():
    reward = MathPRMReward.__new__(MathPRMReward)
    torch.nn.Module.__init__(reward)
    return reward


class Phase2AlignmentTests(unittest.TestCase):
    def test_prepare_input_matches_reference_port(self):
        reward = _minimal_math_prm_instance()
        question = "What is 2 + 2?"
        response = "Step 1: Add 2 and 2.\n†Answer: 4"

        self.assertEqual(reward._prepare_prm_input(question, response), reference_prepare_input(question, response))
        self.assertEqual(
            reward.replace_specific_plus_minus_with_ki(response),
            reference_replace_specific_plus_minus_with_ki(response),
        )

    def test_math_prm_forward_strips_vision_tokens_and_passes_images(self):
        processor = FakeProcessor()
        model = FakeRewardModel()
        reward = MathPRMReward(model, processor, aggregation="min")

        sample_image = object()
        prompt_and_output = [
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>How many stages are shown?<|im_end|>"
            "<|im_start|>assistant\nStep 1: Count the labels.\n†Answer: 3"
        ]

        scores = reward(
            sequences=None,
            attention_mask=None,
            prompt_and_output=prompt_and_output,
            raw_images=[sample_image],
        )

        self.assertEqual(tuple(scores.shape), (1,))
        self.assertEqual(processor.calls[0]["images"], [sample_image])
        user_content = processor.chat_templates[0]["conv"][1]["content"]
        self.assertTrue(user_content.startswith("<|image|>You are given a problem and a step-by-step solution."))
        self.assertIn("How many stages are shown?", user_content)
        self.assertNotIn("<|vision_start|>", user_content)
        self.assertNotIn("<|image|><|vision_start|>", user_content)

    def test_load_reward_models_uses_direct_ursa_loader_for_math_prm(self):
        import reward_models_utils as rm_utils

        strategy = FakeStrategy()
        calls = {"engine": 0, "ursa": 0, "builder_base": None}
        fake_processor = SimpleNamespace(tokenizer="ursa-tokenizer")

        def fake_load_engine(path, device):
            calls["engine"] += 1
            return "engine-base", "engine-processor"

        def fake_load_ursa(path, device):
            calls["ursa"] += 1
            return "ursa-base", fake_processor

        def fake_builder(cfg, strategy, base=None):
            calls["builder_base"] = base
            return "reward-model", "reward-tokenizer"

        with mock.patch.object(rm_utils, "_load_engine", side_effect=fake_load_engine), mock.patch.object(
            rm_utils, "_load_ursa_prm_model", side_effect=fake_load_ursa
        ), mock.patch.dict(rm_utils._BUILDERS, {RewardModelType.MATH_PRM: fake_builder}, clear=False):
            reward_models, reward_tokenizers, label_map = load_reward_models(
                raw_reward_pretrain='{"math_prm":"/tmp/ursa-rm"}',
                strategy=strategy,
                use_engine=True,
            )

        self.assertEqual(reward_models, ["reward-model"])
        self.assertEqual(reward_tokenizers, ["reward-tokenizer"])
        self.assertEqual(label_map["math_prm"], 0)
        self.assertEqual(calls["engine"], 0)
        self.assertEqual(calls["ursa"], 1)
        self.assertEqual(calls["builder_base"][0], "ursa-base")
        self.assertIs(calls["builder_base"][1], fake_processor)

    def test_load_actor_tokenizer_processor_uses_ursa_processor(self):
        import train_colocate

        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            (model_dir / "config.json").write_text(
                json.dumps({"architectures": ["UrsaForConditionalGeneration"], "model_type": "ursa"}),
                encoding="utf-8",
            )

            processor = SimpleNamespace(tokenizer=FakeTokenizer())

            class FakeUrsaProcessor:
                @classmethod
                def from_pretrained(cls, path):
                    self.assertEqual(path, str(model_dir))
                    return processor

            fake_module = ModuleType("ursa_model")
            fake_module.UrsaProcessor = FakeUrsaProcessor
            strategy = FakeStrategy()
            model = FakeModelForProcessor()

            with mock.patch.dict(sys.modules, {"ursa_model": fake_module}):
                tokenizer, loaded_processor = train_colocate.load_actor_tokenizer_processor(
                    model_path=str(model_dir),
                    model=model,
                    strategy=strategy,
                    use_fast=False,
                )

        self.assertIs(tokenizer, processor.tokenizer)
        self.assertIs(loaded_processor, processor)
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertEqual(tokenizer.pad_token, tokenizer.eos_token)
        self.assertEqual(model.config.pad_token_id, tokenizer.eos_token_id)


if __name__ == "__main__":
    unittest.main()

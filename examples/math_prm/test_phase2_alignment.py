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
from reward_models_utils import RewardModelType, load_reward_models, mix_rewards
from lightrft.models.actor_vl import ActorVL
from lightrft.utils.math_prm_output import sanitize_math_prm_response_text, should_stop_math_prm_response_text


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


class FakeMultiStepProcessor(FakeProcessor):
    def __call__(self, text, images, return_tensors="pt"):
        self.calls.append({"text": text, "images": images, "return_tensors": return_tensors})
        return FakeBatch(
            {
                "input_ids": torch.tensor([[11, 42, 13, 42, 14, 42]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
            }
        )


class FakeMultiStepRewardModel(torch.nn.Module):
    def __init__(self, step_scores):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.step_scores = torch.tensor(step_scores, dtype=torch.float32)

    def forward(self, **inputs):
        input_ids = inputs["input_ids"].view(-1)
        seq_len = input_ids.shape[0] + MathPRMReward._IMAGE_PAD
        logits = torch.zeros((1, seq_len, 1), dtype=torch.float32, device=input_ids.device)
        step_positions = [MathPRMReward._IMAGE_PAD + 1, MathPRMReward._IMAGE_PAD + 3, MathPRMReward._IMAGE_PAD + 5]
        for position, score in zip(step_positions, self.step_scores):
            logits[0, position, 0] = torch.logit(score, eps=1e-6)
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


class FakeVisionLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.bfloat16))
        self.config = SimpleNamespace(model_type="fake-vl")
        self.forward_pixel_dtype = None
        self.generate_pixel_dtype = None

    def forward(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
    ):
        self.forward_pixel_dtype = None if pixel_values is None else pixel_values.dtype
        batch_size, seq_len = input_ids.shape
        return {"logits": torch.zeros((batch_size, seq_len, 8), dtype=torch.float32, device=input_ids.device)}

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        **kwargs,
    ):
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
        }

    def generate(
        self,
        input_ids,
        attention_mask=None,
        pixel_values=None,
        image_grid_thw=None,
        pixel_values_videos=None,
        video_grid_thw=None,
        **kwargs,
    ):
        self.generate_pixel_dtype = None if pixel_values is None else pixel_values.dtype
        suffix = torch.full((input_ids.size(0), 1), 7, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat([input_ids, suffix], dim=1)


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

        self.assertTrue(isinstance(scores, torch.Tensor) or isinstance(scores, dict))
        if isinstance(scores, dict):
            self.assertEqual(tuple(scores["score"].shape), (1,))
        else:
            self.assertEqual(tuple(scores.shape), (1,))
        self.assertEqual(processor.calls[0]["images"], [sample_image])
        user_content = processor.chat_templates[0]["conv"][1]["content"]
        self.assertTrue(user_content.startswith("<|image|>You are given a problem and a step-by-step solution."))
        self.assertIn("How many stages are shown?", user_content)
        self.assertNotIn("<|vision_start|>", user_content)
        self.assertNotIn("<|image|><|vision_start|>", user_content)

    def test_math_prm_forward_keeps_legacy_tensor_output_without_phase4_args(self):
        processor = FakeProcessor()
        model = FakeRewardModel()
        reward = MathPRMReward(model, processor, aggregation="min")

        score = reward(
            sequences=None,
            attention_mask=None,
            prompt_and_output=["<|im_start|>assistant\nStep 1: Count.\n†Answer: 3"],
            raw_images=[None],
        )

        self.assertIsInstance(score, torch.Tensor)
        self.assertEqual(tuple(score.shape), (1,))

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

    def test_mix_rewards_keeps_phase3_math_prm_as_pure_model_reward(self):
        labels = ["math_prm"]
        model_scores = torch.tensor([[0.75]], dtype=torch.float32)
        label_map = {"math_prm": 0}
        solutions = ["Step 1: Count the triangles.\n†Answer: 4"]
        refs = ["4"]

        with mock.patch("torch.distributed.get_rank", return_value=0):
            reward, metrics = mix_rewards(labels, model_scores, label_map, solutions, refs)

        self.assertAlmostEqual(reward.item(), 0.75, places=6)
        self.assertAlmostEqual(metrics["model_reward"].item(), 0.75, places=6)
        self.assertAlmostEqual(metrics["rule_reward"].item(), 0.0, places=6)
        self.assertAlmostEqual(metrics["format_reward"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["final_reward"].item(), 0.75, places=6)

    def test_mix_rewards_supports_phase4_math_psgrpo_label(self):
        labels = ["math_psgrpo"]
        model_scores = torch.tensor([[1.0]], dtype=torch.float32)
        label_map = {"math_prm": 0}
        solutions = ["Step 1: Compute carefully.\n†Answer: 37"]
        refs = ["37"]

        with mock.patch("torch.distributed.get_rank", return_value=0):
            reward, metrics = mix_rewards(labels, model_scores, label_map, solutions, refs)

        self.assertAlmostEqual(reward.item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["model_reward"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["final_reward"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["format_reward"].item(), 1.0, places=6)

    def test_mix_rewards_still_applies_global_format_reward_for_non_math_labels(self):
        labels = ["general"]
        model_scores = torch.tensor([[0.25]], dtype=torch.float32)
        label_map = {"general": 0}
        solutions = ["<think>reason</think>\nfinal answer"]
        refs = [""]

        with mock.patch("torch.distributed.get_rank", return_value=0):
            reward, metrics = mix_rewards(labels, model_scores, label_map, solutions, refs)

        self.assertAlmostEqual(metrics["format_reward"].item(), 1.0, places=6)
        self.assertAlmostEqual(metrics["model_reward"].item(), 0.25, places=6)
        self.assertAlmostEqual(reward.item(), 1.25, places=6)

    def test_sanitize_math_prm_response_truncates_after_first_answer_line(self):
        response = (
            "StepStep 1: Observe the image.\n"
            "Step 2: Decide whether the terrain is flat.\n"
            "†Answer: It is a flat landscape Camp Miniwauca Camp Miniwauca Camp Miniwauca Camp Miniwauca\n"
            "Step 1: repeated garbage"
        )

        self.assertEqual(
            sanitize_math_prm_response_text(response),
            "Step 1: Observe the image.\nStep 2: Decide whether the terrain is flat.\n†Answer: It is a flat landscape",
        )

    def test_sanitize_math_prm_response_removes_second_answer_tail(self):
        response = (
            "Step 1: Compute y = 5x + 7.\n"
            "Step 2: Substitute x = 6.\n"
            "†Answer: 37\n"
            "7777777777777777777777 †Answer: 3777777777777777"
        )

        self.assertEqual(
            sanitize_math_prm_response_text(response),
            "Step 1: Compute y = 5x + 7.\nStep 2: Substitute x = 6.\n†Answer: 37",
        )

    def test_should_stop_math_prm_response_detects_short_final_answers(self):
        self.assertTrue(should_stop_math_prm_response_text("Step 1: Inspect.\n†Answer: yes"))
        self.assertTrue(should_stop_math_prm_response_text("Step 1: Compute.\n†Answer: 37"))
        self.assertFalse(should_stop_math_prm_response_text("Step 1: Compute carefully."))

    def test_math_prm_psgrpo_metrics_reward_mapping_matches_phase4(self):
        reward = _minimal_math_prm_instance()

        correct_no_drop = reward._compute_psgrpo_metrics(
            response="Step 1: Compute.\n†Answer: 37",
            reference="37",
            step_scores=torch.tensor([0.92, 0.88, 0.84], dtype=torch.float32),
        )
        self.assertAlmostEqual(correct_no_drop["outcome_correct"], 1.0, places=6)
        self.assertAlmostEqual(correct_no_drop["has_drop_moment"], 0.0, places=6)
        self.assertAlmostEqual(correct_no_drop["final_reward"], 1.0, places=6)

        correct_with_drop = reward._compute_psgrpo_metrics(
            response="Step 1: Compute.\n†Answer: 37",
            reference="37",
            step_scores=torch.tensor([0.90, 0.50, 0.48], dtype=torch.float32),
        )
        self.assertAlmostEqual(correct_with_drop["outcome_correct"], 1.0, places=6)
        self.assertAlmostEqual(correct_with_drop["has_drop_moment"], 1.0, places=6)
        self.assertGreater(correct_with_drop["max_relative_drop"], 0.3)
        self.assertAlmostEqual(correct_with_drop["final_reward"], 0.5, places=6)

        incorrect = reward._compute_psgrpo_metrics(
            response="Step 1: Compute.\n†Answer: 41",
            reference="37",
            step_scores=torch.tensor([0.90, 0.85, 0.82], dtype=torch.float32),
        )
        self.assertAlmostEqual(incorrect["outcome_correct"], 0.0, places=6)
        self.assertAlmostEqual(incorrect["final_reward"], 0.0, places=6)

    def test_math_prm_forward_returns_phase4_psgrpo_metrics(self):
        processor = FakeMultiStepProcessor()
        model = FakeMultiStepRewardModel([0.90, 0.50, 0.48])
        reward = MathPRMReward(model, processor, aggregation="min")

        sample = (
            "<|im_start|>user\nSolve for y when x = 6 in y = 5x + 7.<|im_end|>"
            "<|im_start|>assistant\n"
            "Step 1: Substitute x = 6.\n"
            "Step 2: Compute 5 * 6 = 30.\n"
            "Step 3: Add 7 to get 37.\n"
            "†Answer: 37"
        )

        result = reward(
            sequences=None,
            attention_mask=None,
            prompt_and_output=[sample],
            raw_images=[object()],
            references=["37"],
            labels=["math_psgrpo"],
        )

        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["score"].item(), 0.5, places=6)
        self.assertAlmostEqual(result["model_reward"].item(), 0.48, places=4)
        self.assertAlmostEqual(result["outcome_correct"].item(), 1.0, places=6)
        self.assertAlmostEqual(result["has_drop_moment"].item(), 1.0, places=6)
        self.assertGreater(result["max_relative_drop"].item(), 0.3)
        self.assertAlmostEqual(result["final_reward"].item(), 0.5, places=6)
        self.assertAlmostEqual(result["step_count"].item(), 3.0, places=6)

    def test_actor_vl_casts_multimodal_tensors_to_model_dtype(self):
        fake_model = FakeVisionLanguageModel()
        actor = ActorVL(pretrain_or_model=fake_model)

        sequences = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(sequences)
        pixel_values = torch.randn(1, 3, 16, 16, dtype=torch.float32)

        actor(
            sequences=sequences,
            num_actions=None,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            return_output=True,
        )
        actor.generate(
            input_ids=sequences,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            eos_token_id=9,
            pad_token_id=0,
            max_new_tokens=1,
        )

        self.assertEqual(fake_model.forward_pixel_dtype, torch.bfloat16)
        self.assertEqual(fake_model.generate_pixel_dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()

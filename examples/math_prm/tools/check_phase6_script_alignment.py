#!/usr/bin/env python3
"""Validate the Phase 6 Stage 3 launcher alignment for URSA math_prm.

This script is intentionally lightweight:
    - It does not import LightRFT training modules.
    - It only inspects example source files as text.
    - It verifies the Phase 6 checklist items that should remain stable.

Usage:
    python examples/math_prm/tools/check_phase6_script_alignment.py
    python examples/math_prm/tools/check_phase6_script_alignment.py --output-json /tmp/report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_SCRIPT_PATH = REPO_ROOT / "examples" / "math_prm" / "run_grpo_math_prm_ursa_8b.sh"
TRAIN_SCRIPT_PATH = REPO_ROOT / "examples" / "math_prm" / "train_colocate.py"


def _bash_default_pattern(name: str, *, exported: bool = False) -> re.Pattern[str]:
    prefix = r"export\s+" if exported else ""
    return re.compile(
        rf"^{prefix}{name}=\"\$\{{{name}:-(.+?)\}}\"(?:\s+#.*)?$",
        re.M,
    )


RUN_SCRIPT_ENV_PATTERNS = {
    "PATH_TO_YOUR_BASE_MODEL": _bash_default_pattern("PATH_TO_YOUR_BASE_MODEL"),
    "PATH_TO_URSA_RM": _bash_default_pattern("PATH_TO_URSA_RM"),
    "PATH_TO_YOUR_MATH_DATASET": _bash_default_pattern("PATH_TO_YOUR_MATH_DATASET"),
    "EXPECTED_REWARD_LABEL": _bash_default_pattern("EXPECTED_REWARD_LABEL"),
    "DOCKER_BASELINE": _bash_default_pattern("DOCKER_BASELINE"),
    "N_SAMPLES": _bash_default_pattern("N_SAMPLES"),
    "TBS": _bash_default_pattern("TBS"),
    "MICRO_TRAIN_BATCH_SIZE": _bash_default_pattern("MICRO_TRAIN_BATCH_SIZE"),
    "TEMPERATURE": _bash_default_pattern("TEMPERATURE"),
    "KL": _bash_default_pattern("KL"),
    "LR": _bash_default_pattern("LR"),
    "PROMPT_MAX_LEN": _bash_default_pattern("PROMPT_MAX_LEN"),
    "GENERATE_MAX_LEN": _bash_default_pattern("GENERATE_MAX_LEN"),
    "MLP_WORKER_NUM": _bash_default_pattern("MLP_WORKER_NUM", exported=True),
    "MLP_WORKER_GPU": _bash_default_pattern("MLP_WORKER_GPU", exported=True),
}


TRAIN_ARG_PATTERNS = {
    "engine_type": re.compile(r'parser\.add_argument\("--engine_type",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
    "prompt_max_len": re.compile(r'parser\.add_argument\("--prompt_max_len",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
    "generate_max_len": re.compile(r'parser\.add_argument\("--generate_max_len",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
    "train_batch_size": re.compile(r'parser\.add_argument\("--train_batch_size",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
    "n_samples_per_prompt": re.compile(r'parser\.add_argument\(\s*"--n_samples_per_prompt",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
    "actor_learning_rate": re.compile(r'parser\.add_argument\("--actor_learning_rate",.*?default=([^\s,\)]+)', re.S),
    "init_kl_coef": re.compile(r'parser\.add_argument\("--init_kl_coef",.*?default=([^\s,\)]+)', re.S),
    "images_key": re.compile(r'parser\.add_argument\("--images_key",.*?default=(".*?"|\d+(?:\.\d+)?)', re.S),
}


def _literal_or_raw(value: str) -> Any:
    value = value.strip()
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _extract_with_patterns(text: str, patterns: Dict[str, re.Pattern[str]]) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {}
    for key, pattern in patterns.items():
        match = pattern.search(text)
        if not match:
            raise RuntimeError(f"Could not find {key!r} in inspected source text.")
        extracted[key] = _literal_or_raw(match.group(1))
    return extracted


def collect_phase6_alignment() -> Dict[str, Any]:
    run_script_text = RUN_SCRIPT_PATH.read_text(encoding="utf-8")
    train_script_text = TRAIN_SCRIPT_PATH.read_text(encoding="utf-8")

    run_defaults = _extract_with_patterns(run_script_text, RUN_SCRIPT_ENV_PATTERNS)
    train_defaults = _extract_with_patterns(train_script_text, TRAIN_ARG_PATTERNS)

    world_size = int(run_defaults["MLP_WORKER_NUM"]) * int(run_defaults["MLP_WORKER_GPU"])
    micro_train_batch_size = int(run_defaults["MICRO_TRAIN_BATCH_SIZE"])
    train_batch_size = int(run_defaults["TBS"])
    grad_accum = train_batch_size // (world_size * micro_train_batch_size)

    checks = {
        "reward_label_stage3": {
            "passed": run_defaults["EXPECTED_REWARD_LABEL"] == "math_psgrpo",
            "actual": run_defaults["EXPECTED_REWARD_LABEL"],
            "expected": "math_psgrpo",
        },
        "local_model_paths": {
            "passed": (
                run_defaults["PATH_TO_YOUR_BASE_MODEL"] == "/home/ubuntu/URSA-MATH/checkpoints/URSA-8B"
                and run_defaults["PATH_TO_URSA_RM"] == "/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B"
            ),
            "actual": {
                "base_model": run_defaults["PATH_TO_YOUR_BASE_MODEL"],
                "reward_model": run_defaults["PATH_TO_URSA_RM"],
            },
            "expected": {
                "base_model": "/home/ubuntu/URSA-MATH/checkpoints/URSA-8B",
                "reward_model": "/home/ubuntu/URSA-MATH/checkpoints/URSA-RM-8B",
            },
        },
        "manifest_dataset_path": {
            "passed": run_defaults["PATH_TO_YOUR_MATH_DATASET"] == "/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl",
            "actual": run_defaults["PATH_TO_YOUR_MATH_DATASET"],
            "expected": "/data/LightRFT/tmp/ursa_stage3/mmathcot_stage3_math_psgrpo.jsonl",
        },
        "docker_baseline_documented": {
            "passed": (
                run_defaults["DOCKER_BASELINE"] == "/data/LightRFT/Dockerfile"
                and "/data/LightRFT/Dockerfile" in run_script_text
                and "installation order" in run_script_text
            ),
            "actual": run_defaults["DOCKER_BASELINE"],
            "expected": "/data/LightRFT/Dockerfile",
        },
        "no_rm_use_engine": {
            "passed": "--rm_use_engine" not in run_script_text,
            "actual": "--rm_use_engine" in run_script_text,
            "expected": False,
        },
        "multimodal_required_flags": {
            "passed": (
                '--mixed_mm_data' in run_script_text
                and '--images_key "images"' in run_script_text
                and '--freeze_prefix' in run_script_text
            ),
            "actual": {
                "mixed_mm_data": '--mixed_mm_data' in run_script_text,
                "images_key_images": '--images_key "images"' in run_script_text,
                "freeze_prefix": '--freeze_prefix' in run_script_text,
            },
            "expected": True,
        },
        "resource_smoke_guidance": {
            "passed": (
                "/home/ubuntu/URSA-MATH/examples/run_dataset_loading_example.py" in run_script_text
                and "/home/ubuntu/URSA-MATH/examples/validate_dataset_entrypoints.py" in run_script_text
                and "tools/run_phase3_smoke.sh" in run_script_text
            ),
            "actual": True,
            "expected": True,
        },
        "table14_launcher_defaults": {
            "passed": (
                int(run_defaults["N_SAMPLES"]) == 8
                and float(run_defaults["TEMPERATURE"]) == 1.0
                and float(run_defaults["KL"]) == 0.003
                and float(run_defaults["LR"]) == 2e-6
                and int(run_defaults["PROMPT_MAX_LEN"]) == 6048
                and int(run_defaults["GENERATE_MAX_LEN"]) == 3072
                and int(run_defaults["TBS"]) == 512
            ),
            "actual": {
                "n_samples_per_prompt": run_defaults["N_SAMPLES"],
                "temperature": run_defaults["TEMPERATURE"],
                "init_kl_coef": run_defaults["KL"],
                "actor_learning_rate": run_defaults["LR"],
                "prompt_max_len": run_defaults["PROMPT_MAX_LEN"],
                "generate_max_len": run_defaults["GENERATE_MAX_LEN"],
                "train_batch_size": run_defaults["TBS"],
            },
            "expected": {
                "n_samples_per_prompt": 8,
                "temperature": 1.0,
                "init_kl_coef": 0.003,
                "actor_learning_rate": 2e-6,
                "prompt_max_len": 6048,
                "generate_max_len": 3072,
                "train_batch_size": 512,
            },
        },
        "table14_train_batch_implementation": {
            "passed": world_size == 8 and micro_train_batch_size == 4 and train_batch_size == 512 and grad_accum == 16,
            "actual": {
                "world_size": world_size,
                "micro_train_batch_size": micro_train_batch_size,
                "train_batch_size": train_batch_size,
                "gradient_accumulation": grad_accum,
            },
            "expected": {
                "world_size": 8,
                "micro_train_batch_size": 4,
                "train_batch_size": 512,
                "gradient_accumulation": 16,
            },
        },
        "train_colocate_defaults_aligned": {
            "passed": (
                train_defaults["engine_type"] == "hf"
                and int(train_defaults["prompt_max_len"]) == 6048
                and int(train_defaults["generate_max_len"]) == 3072
                and int(train_defaults["train_batch_size"]) == 512
                and int(train_defaults["n_samples_per_prompt"]) == 8
                and float(train_defaults["actor_learning_rate"]) == 2e-6
                and float(train_defaults["init_kl_coef"]) == 0.003
                and train_defaults["images_key"] == "images"
            ),
            "actual": train_defaults,
            "expected": {
                "engine_type": "hf",
                "prompt_max_len": 6048,
                "generate_max_len": 3072,
                "train_batch_size": 512,
                "n_samples_per_prompt": 8,
                "actor_learning_rate": 2e-6,
                "init_kl_coef": 0.003,
                "images_key": "images",
            },
        },
    }

    return {
        "success": all(item["passed"] for item in checks.values()),
        "run_script": str(RUN_SCRIPT_PATH),
        "train_script": str(TRAIN_SCRIPT_PATH),
        "run_defaults": run_defaults,
        "train_defaults": train_defaults,
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate Phase 6 script alignment for URSA Stage 3")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write the JSON report")
    args = parser.parse_args()

    report = collect_phase6_alignment()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")

    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

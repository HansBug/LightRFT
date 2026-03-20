#!/usr/bin/env python3
"""Analyze a Phase 7 URSA Stage 3 observation run.

This script is intentionally observation-oriented:
    - It reads saved trajectories and training logs.
    - It computes the Phase 7 checklist metrics offline.
    - It can optionally run a small PRM image ablation on saved multimodal samples.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from PIL import Image

TOOLS_DIR = Path(__file__).resolve().parent
MATH_PRM_DIR = TOOLS_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[3]
for path in (TOOLS_DIR, MATH_PRM_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lightrft.trainer.image_utils import normalize_images


STEP_PATTERN = re.compile(r"(?m)^Step\s+\d+:")
ANSWER_PATTERN = re.compile(r"(?m)^†Answer:\s*.+$")
PROGRESS_PG_PATTERN = re.compile(r"pg=([-+0-9.eE]+)")
PROGRESS_KL_PATTERN = re.compile(r"kl=([-+0-9.eE]+)")
PROGRESS_RM_PATTERN = re.compile(r"rm=([-+0-9.eE]+)")


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        if not value:
            return None
        if len(value) == 1:
            return _as_float(value[0])
        scalar_values = [_as_float(v) for v in value]
        scalar_values = [v for v in scalar_values if v is not None]
        if not scalar_values:
            return None
        return float(np.mean(scalar_values))
    return None


def _safe_mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return float(np.mean(values))


def _distribution(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "variance": None,
            "std": None,
            "min": None,
            "p25": None,
            "p50": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "variance": float(arr.var()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(arr.max()),
    }


def is_valid_stage3_format(text: str) -> bool:
    if not text:
        return False
    if not STEP_PATTERN.search(text):
        return False
    answer_lines = ANSWER_PATTERN.findall(text)
    if len(answer_lines) != 1:
        return False
    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not non_empty_lines:
        return False
    return non_empty_lines[-1].startswith("†Answer:")


def parse_progress_metrics(log_text: str) -> Dict[str, Any]:
    normalized = log_text.replace("\r", "\n")
    policy_gradient_values: List[float] = []
    kl_values: List[float] = []
    reward_values: List[float] = []
    for line in normalized.splitlines():
        if "pg=" not in line or "kl=" not in line:
            continue
        pg_match = PROGRESS_PG_PATTERN.search(line)
        kl_match = PROGRESS_KL_PATTERN.search(line)
        rm_match = PROGRESS_RM_PATTERN.search(line)
        if pg_match:
            policy_gradient_values.append(float(pg_match.group(1)))
        if kl_match:
            kl_values.append(float(kl_match.group(1)))
        if rm_match:
            reward_values.append(float(rm_match.group(1)))

    error_markers = []
    for marker in ("Traceback", "CUDA out of memory", "ChildFailedError", "RuntimeError:"):
        if marker in normalized:
            error_markers.append(marker)

    return {
        "policy_gradient": _distribution(policy_gradient_values),
        "kl": _distribution(kl_values),
        "progress_reward": _distribution(reward_values),
        "error_markers": error_markers,
    }


def load_latest_trajectory_json(results_dir: Path) -> Optional[Path]:
    trajectory_dir = results_dir / "trajectories"
    if not trajectory_dir.exists():
        return None
    candidates = sorted(
        trajectory_dir.glob("trajectories_step_*.json"),
        key=lambda path: (
            int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
            path.stat().st_mtime,
        ),
    )
    return candidates[-1] if candidates else None


def summarize_trajectories(trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
    rollout_rewards: List[float] = []
    psgrpo_rewards: List[float] = []
    correctness: List[float] = []
    drop_moment: List[float] = []
    answer_extraction_failed: List[float] = []
    format_success_flags: List[float] = []
    prm_metric_missing = 0
    multimodal_samples = 0

    for traj in trajectories:
        info = traj.get("info") or {}
        reward_metrics = info.get("reward_metrics") or {}

        rollout_reward = _as_float(info.get("reward"))
        if rollout_reward is not None:
            rollout_rewards.append(rollout_reward)

        psgrpo_reward = _as_float(reward_metrics.get("final_reward"))
        if psgrpo_reward is not None:
            psgrpo_rewards.append(psgrpo_reward)

        outcome_correct = _as_float(reward_metrics.get("outcome_correct"))
        if outcome_correct is not None:
            correctness.append(outcome_correct)

        has_drop_moment = _as_float(reward_metrics.get("has_drop_moment"))
        if has_drop_moment is not None:
            drop_moment.append(has_drop_moment)

        extraction_failed = _as_float(reward_metrics.get("answer_extraction_failed"))
        if extraction_failed is not None:
            answer_extraction_failed.append(extraction_failed)

        text = (
            traj.get("pure_generated_text")
            or traj.get("generated_text")
            or traj.get("full_sequence")
            or ""
        )
        format_success_flags.append(1.0 if is_valid_stage3_format(text) else 0.0)

        if traj.get("has_images"):
            multimodal_samples += 1

        model_reward = _as_float(reward_metrics.get("model_reward"))
        if model_reward is None or math.isnan(model_reward):
            prm_metric_missing += 1

    total = len(trajectories)
    prm_inference_failure_ratio = float(prm_metric_missing / total) if total else 0.0
    return {
        "num_trajectories": total,
        "multimodal_sample_count": multimodal_samples,
        "rollout_reward": _distribution(rollout_rewards),
        "psgrpo_final_reward": _distribution(psgrpo_rewards),
        "correctness_ratio": _safe_mean(correctness),
        "drop_moment_ratio": _safe_mean(drop_moment),
        "answer_extraction_failure_ratio": _safe_mean(answer_extraction_failed),
        "format_success_ratio": _safe_mean(format_success_flags),
        "prm_inference_failure_ratio": prm_inference_failure_ratio,
    }


def scan_manifest_images(dataset_path: Path, limit: int) -> Dict[str, Any]:
    if not dataset_path.exists():
        return {
            "dataset_missing": True,
            "records_scanned": 0,
            "records_with_images": 0,
            "image_read_failures": 0,
            "image_read_failure_ratio": None,
            "failure_examples": [],
        }

    scanned = 0
    with_images = 0
    failures = 0
    failure_examples = []

    with dataset_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if limit > 0 and scanned >= limit:
                break
            line = line.strip()
            if not line:
                continue
            scanned += 1
            record = json.loads(line)
            images = record.get("images")
            if not images:
                continue
            with_images += 1
            try:
                normalize_images(images if isinstance(images, list) else [images])
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if len(failure_examples) < 5:
                    failure_examples.append(
                        {
                            "index": idx,
                            "images": images,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    failure_ratio = float(failures / with_images) if with_images else 0.0
    return {
        "records_scanned": scanned,
        "records_with_images": with_images,
        "image_read_failures": failures,
        "image_read_failure_ratio": failure_ratio,
        "failure_examples": failure_examples,
    }


def measure_multimodal_prm_impact(
    trajectories: List[Dict[str, Any]],
    trajectory_path: Path,
    rm_path: Optional[str],
    max_samples: int,
) -> Dict[str, Any]:
    if not rm_path:
        return {
            "checked_samples": 0,
            "failed_samples": 0,
            "mean_abs_delta": None,
            "details": [],
            "skipped_reason": "missing_rm_path",
        }

    candidates = [
        traj for traj in trajectories
        if traj.get("has_images") and traj.get("image_paths") and (traj.get("full_sequence") or traj.get("generated_text"))
    ]
    if not candidates:
        return {
            "checked_samples": 0,
            "failed_samples": 0,
            "mean_abs_delta": None,
            "details": [],
            "skipped_reason": "no_multimodal_trajectories",
        }

    from reward_models_utils import RewardModelConfig, RewardModelType, build_math_prm

    class _SilentStrategy:
        @staticmethod
        def print(*args, **kwargs) -> None:  # noqa: D401, ANN003
            return None

    reward_model, _ = build_math_prm(
        RewardModelConfig(RewardModelType.MATH_PRM, rm_path, use_engine=False),
        _SilentStrategy(),
    )

    details = []
    deltas = []
    failed = 0
    trajectory_root = trajectory_path.parent

    for traj in candidates[:max_samples]:
        image_paths = [
            trajectory_root / relative_path
            for relative_path in traj.get("image_paths", [])
            if relative_path
        ]
        if not image_paths:
            continue
        prompt_and_output = traj.get("full_sequence") or traj.get("generated_text") or ""
        try:
            images = [Image.open(path).convert("RGB") for path in image_paths]
            blank_images = [Image.new("RGB", image.size, color=(255, 255, 255)) for image in images]
            image_payload: Any = images if len(images) > 1 else images[0]
            blank_payload: Any = blank_images if len(blank_images) > 1 else blank_images[0]
            with_image_score = float(
                reward_model(
                    sequences=None,
                    attention_mask=None,
                    prompt_and_output=[prompt_and_output],
                    raw_images=[image_payload],
                ).item()
            )
            without_image_score = float(
                reward_model(
                    sequences=None,
                    attention_mask=None,
                    prompt_and_output=[prompt_and_output],
                    raw_images=[blank_payload],
                ).item()
            )
            abs_delta = abs(with_image_score - without_image_score)
            deltas.append(abs_delta)
            details.append(
                {
                    "sample_id": traj.get("sample_id"),
                    "with_image_score": with_image_score,
                    "without_image_score": without_image_score,
                    "abs_delta": abs_delta,
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            details.append(
                {
                    "sample_id": traj.get("sample_id"),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return {
        "checked_samples": len(deltas),
        "failed_samples": failed,
        "mean_abs_delta": float(np.mean(deltas)) if deltas else None,
        "details": details,
    }


def build_issue_list(summary: Dict[str, Any]) -> List[str]:
    issues: List[str] = []
    traj = summary.get("trajectory_metrics", {})
    log_metrics = summary.get("log_metrics", {})
    image_scan = summary.get("image_scan", {})
    multimodal_impact = summary.get("multimodal_impact", {})

    if traj.get("num_trajectories", 0) == 0 and (log_metrics.get("policy_gradient") or {}).get("count", 0) == 0:
        issues.append("本次 Phase 7 观测没有形成有效训练样本，trajectory 和训练进度指标都为空")

    format_success_ratio = traj.get("format_success_ratio")
    if format_success_ratio is not None and format_success_ratio < 0.95:
        issues.append(f"生成格式成功率偏低：format_success_ratio={format_success_ratio:.4f}")

    extraction_failure_ratio = traj.get("answer_extraction_failure_ratio")
    if extraction_failure_ratio is not None and extraction_failure_ratio > 0.05:
        issues.append(
            f"final answer 抽取失败比例偏高：answer_extraction_failure_ratio={extraction_failure_ratio:.4f}"
        )

    correctness_ratio = traj.get("correctness_ratio")
    if correctness_ratio is not None and correctness_ratio < 0.20:
        issues.append(f"correctness 偏低：correctness_ratio={correctness_ratio:.4f}")

    prm_failure_ratio = traj.get("prm_inference_failure_ratio")
    if prm_failure_ratio is not None and prm_failure_ratio > 0.0:
        issues.append(f"轨迹中存在 PRM 指标缺失：prm_inference_failure_ratio={prm_failure_ratio:.4f}")

    image_failure_ratio = image_scan.get("image_read_failure_ratio")
    if image_scan.get("dataset_missing"):
        issues.append("Phase 7 分析时找不到 manifest，dataset 缺失")
    elif image_failure_ratio is not None and image_failure_ratio > 0.0:
        issues.append(f"数据集图片读取失败比例不为 0：image_read_failure_ratio={image_failure_ratio:.4f}")

    kl_dist = (log_metrics.get("kl") or {})
    if kl_dist.get("max") is not None and kl_dist["max"] > 1.0:
        issues.append(f"KL 峰值过高：kl_max={kl_dist['max']:.4f}")

    pg_dist = (log_metrics.get("policy_gradient") or {})
    if pg_dist.get("max") is not None and max(abs(pg_dist["min"]), abs(pg_dist["max"])) > 10.0:
        issues.append(
            "policy gradient / actor loss 波动过大："
            f"min={pg_dist['min']:.4f}, max={pg_dist['max']:.4f}"
        )

    if log_metrics.get("error_markers"):
        issues.append(f"日志中出现错误标记：{', '.join(log_metrics['error_markers'])}")

    if multimodal_impact.get("failed_samples", 0) > 0:
        issues.append(
            "多模态 PRM 图像消融存在失败样本："
            f"failed_samples={multimodal_impact['failed_samples']}"
        )

    mean_abs_delta = multimodal_impact.get("mean_abs_delta")
    if multimodal_impact.get("checked_samples", 0) > 0 and mean_abs_delta is not None and mean_abs_delta == 0.0:
        issues.append("抽样多模态 PRM 图像消融未观察到分数变化，需要继续核查视觉分支是否生效")

    return issues


def analyze_phase7_run(
    *,
    results_dir: Path,
    log_path: Path,
    dataset_path: Path,
    rm_path: Optional[str],
    image_scan_limit: int,
    multimodal_check_samples: int,
) -> Dict[str, Any]:
    trajectory_path = load_latest_trajectory_json(results_dir)
    trajectories: List[Dict[str, Any]] = []
    if trajectory_path and trajectory_path.exists():
        trajectories = json.loads(trajectory_path.read_text(encoding="utf-8"))

    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""

    summary = {
        "results_dir": str(results_dir),
        "log_path": str(log_path),
        "trajectory_path": str(trajectory_path) if trajectory_path else None,
        "trajectory_metrics": summarize_trajectories(trajectories),
        "log_metrics": parse_progress_metrics(log_text),
        "image_scan": scan_manifest_images(dataset_path, image_scan_limit),
        "multimodal_impact": measure_multimodal_prm_impact(
            trajectories=trajectories,
            trajectory_path=trajectory_path or results_dir,
            rm_path=rm_path,
            max_samples=multimodal_check_samples,
        ),
    }
    summary["issues"] = build_issue_list(summary)
    summary["healthy_pass"] = len(summary["issues"]) == 0
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Phase 7 URSA Stage 3 observation run")
    parser.add_argument("--results-dir", type=str, required=True, help="Run result directory containing trajectories/")
    parser.add_argument("--log-path", type=str, required=True, help="Training log path")
    parser.add_argument("--dataset-path", type=str, required=True, help="Manifest path used by the run")
    parser.add_argument("--rm-path", type=str, default=None, help="Optional URSA-RM path for multimodal PRM impact check")
    parser.add_argument("--image-scan-limit", type=int, default=128, help="How many manifest rows to scan for image load failures")
    parser.add_argument("--multimodal-check-samples", type=int, default=2, help="How many saved multimodal samples to ablate")
    parser.add_argument("--output-json", type=str, default=None, help="Optional output JSON path")
    args = parser.parse_args()

    summary = analyze_phase7_run(
        results_dir=Path(args.results_dir),
        log_path=Path(args.log_path),
        dataset_path=Path(args.dataset_path),
        rm_path=args.rm_path,
        image_scan_limit=args.image_scan_limit,
        multimodal_check_samples=args.multimodal_check_samples,
    )

    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

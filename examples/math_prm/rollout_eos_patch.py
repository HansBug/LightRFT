"""
Math PRM rollout EOS patch — keeps the fix local to examples/math_prm/.

Background
----------
LightRFT's local-HF rollout path (``engine_generate_local`` in
``lightrft/strategy/strategy_base.py``) already installs a
``_StructuredAnswerEosLogitsProcessor`` that *nudges* logits toward EOS once
the response contains a fully-formed "†Answer: <answer>" line. That was the
intent: force HF to sample EOS next step.

In practice, on our 8-GPU FSDP rollout, this nudge fires hundreds of times
(``forced_eos_rows=291`` per batch-of-4) yet generation still runs the full
``max_new_tokens`` for basically every sample (``mean_length=511.8`` out of
512). The logits nudge → sampled token → ``EosTokenCriteria`` feedback loop
does not reliably terminate the HF sample loop under FSDP.

Fix
---
Add a ``StoppingCriteria`` that decides termination directly — HF's sample
loop calls it every step and marks a sequence finished when we return True
for that row. This bypasses the "sample eos, then notice we sampled eos"
handshake that is failing.

This module is self-contained under ``examples/math_prm/`` and is installed
from ``train_colocate.py`` via ``install_math_prm_rollout_eos_patch``. The
patch wraps ``rollout_actor.model.generate`` (the HF model's native generate)
so it only affects math_prm runs; non-structured batches come in without the
``_StructuredAnswerEosLogitsProcessor`` in ``logits_processor`` and the
patch is a pure pass-through.

No changes to ``lightrft/`` are required.
"""

from __future__ import annotations

import functools
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Union

import torch
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

from lightrft.strategy.strategy_base import _StructuredAnswerEosLogitsProcessor
from lightrft.utils.math_prm_output import MATH_PRM_ANSWER_MARKER, should_stop_math_prm_response_text


class StructuredAnswerStoppingCriteria(StoppingCriteria):
    """
    Mirrors the detection logic of ``_StructuredAnswerEosLogitsProcessor`` but
    plugs into HF's ``stopping_criteria`` API instead of the logits API.

    Key properties:

    - Exposes ``eos_token_id`` as an attribute so HF's internal
      ``has_eos_stopping_criteria`` detection (``utils.py:2735``) treats this
      criteria as EOS-equivalent, which enables the post-EOS pad-fill path
      (``utils.py:2835``). Without that attr, rows we mark done would keep
      getting non-pad filler tokens written into their slots, and
      ``process_sequences`` — which derives attention-mask from
      ``seq.ne(eos_token_id) & seq.ne(pad_token_id)`` — would still count
      those positions as real content.
    - Checks only every ``check_interval`` tokens to amortise CPU
      ``batch_decode`` cost (matching the existing LogitsProcessor cadence).
    - Done bits are *sticky*: once set, the criteria re-asserts them on
      every subsequent call, including between gated checks. This is
      critical — HF's sample loop ANDs our return into
      ``unfinished_sequences`` (``utils.py:2842``), so if we ever returned
      False for a row we had previously stopped, HF would un-stop it.
    """

    def __init__(self, tokenizer, prompt_length: int, eos_token_id: int):
        self.tokenizer = tokenizer
        self.prompt_length = int(prompt_length)
        self.eos_token_id = int(eos_token_id)
        self.check_interval = 4
        self.marker_scan_max_tokens = 192
        self.answer_tail_max_tokens = 128
        self.answer_marker_token_ids = tuple(
            int(token_id) for token_id in tokenizer.encode(MATH_PRM_ANSWER_MARKER, add_special_tokens=False)
        )
        self._marker_seen: Optional[List[bool]] = None
        self._done: Optional[torch.Tensor] = None
        self._stats: Dict[str, float] = defaultdict(float)

    def _ensure_state(self, batch_size: int, device) -> None:
        if self._marker_seen is None or len(self._marker_seen) != batch_size:
            self._marker_seen = [False] * batch_size
        if self._done is None or self._done.numel() != batch_size:
            self._done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        elif self._done.device != device:
            self._done = self._done.to(device)

    def _scan_row_for_answer_marker(self, row_token_ids: torch.Tensor) -> bool:
        marker_token_ids = self.answer_marker_token_ids
        if not marker_token_ids:
            return MATH_PRM_ANSWER_MARKER in self.tokenizer.decode(row_token_ids, skip_special_tokens=False)

        token_ids = row_token_ids.tolist()
        marker_len = len(marker_token_ids)
        if len(token_ids) < marker_len:
            return False

        search_start = max(0, len(token_ids) - self.marker_scan_max_tokens)
        token_ids = token_ids[search_start:]
        last_start = len(token_ids) - marker_len + 1
        for start_idx in range(max(last_start, 0)):
            if tuple(token_ids[start_idx:start_idx + marker_len]) == marker_token_ids:
                return True
        return False

    def _decode_rows(self, row_token_ids: torch.Tensor) -> List[str]:
        decode_t0 = time.time()
        texts = self.tokenizer.batch_decode(row_token_ids, skip_special_tokens=False)
        self._stats["decode_time_s"] += time.time() - decode_t0
        self._stats["decoded_rows"] += len(texts)
        return texts

    def get_debug_stats(self) -> Optional[Dict[str, Union[int, float]]]:
        if self._stats["calls"] <= 0:
            return None
        return {
            "calls": int(self._stats["calls"]),
            "gated_checks": int(self._stats["gated_checks"]),
            "marker_scan_rows": int(self._stats["marker_scan_rows"]),
            "marker_hits": int(self._stats["marker_hits"]),
            "answer_tail_rows": int(self._stats["answer_tail_rows"]),
            "decoded_rows": int(self._stats["decoded_rows"]),
            "stopped_rows": int(self._stats["stopped_rows"]),
            "decode_time_s": round(float(self._stats["decode_time_s"]), 4),
        }

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        self._stats["calls"] += 1
        batch_size = input_ids.size(0)
        self._ensure_state(batch_size, input_ids.device)

        if input_ids.size(1) <= self.prompt_length:
            return self._done.clone()

        generated_length = input_ids.size(1) - self.prompt_length
        # Always return the sticky done mask, even on non-gated-check steps,
        # so HF cannot flip previously-stopped rows back to unfinished.
        if generated_length % self.check_interval != 0:
            return self._done.clone()
        self._stats["gated_checks"] += 1

        unresolved_rows = [
            idx for idx in range(batch_size)
            if not self._marker_seen[idx] and not bool(self._done[idx].item())
        ]
        if unresolved_rows:
            scan_start = max(self.prompt_length, input_ids.size(1) - self.marker_scan_max_tokens)
            scan_ids = input_ids[unresolved_rows, scan_start:].detach().cpu()
            self._stats["marker_scan_rows"] += len(unresolved_rows)
            matched_row_indices = []
            matched_scan_ids = []
            for row_idx, row_token_ids in zip(unresolved_rows, scan_ids):
                if self._scan_row_for_answer_marker(row_token_ids):
                    self._marker_seen[row_idx] = True
                    matched_row_indices.append(row_idx)
                    matched_scan_ids.append(row_token_ids)

            if matched_row_indices:
                self._stats["marker_hits"] += len(matched_row_indices)
                matched_scan_ids = torch.stack(matched_scan_ids)
                scan_texts = self._decode_rows(matched_scan_ids)
                for row_idx, text in zip(matched_row_indices, scan_texts):
                    if should_stop_math_prm_response_text(text):
                        self._done[row_idx] = True

        marker_rows = [
            idx for idx in range(batch_size)
            if self._marker_seen[idx] and not bool(self._done[idx].item())
        ]
        if marker_rows:
            tail_start = max(self.prompt_length, input_ids.size(1) - self.answer_tail_max_tokens)
            tail_ids = input_ids[marker_rows, tail_start:].detach().cpu()
            self._stats["answer_tail_rows"] += len(marker_rows)
            tail_texts = self._decode_rows(tail_ids)
            for row_idx, text in zip(marker_rows, tail_texts):
                if should_stop_math_prm_response_text(text):
                    self._done[row_idx] = True

        self._stats["stopped_rows"] = int(self._done.sum().item())
        return self._done.clone()


def install_math_prm_rollout_eos_patch(rollout_actor, tokenizer, eos_token_id: int) -> None:
    """
    Wrap ``rollout_actor.model.generate`` so that whenever LightRFT's
    existing structured-answer LogitsProcessor is present in the kwargs
    (which is exactly when ``fast_exp_maker`` has marked the batch as
    ``structured_answer_stop=True``), we also inject a matching
    ``StructuredAnswerStoppingCriteria`` into the generate call.

    Idempotent: if already installed, this is a no-op.

    Pure pass-through for non-structured generate calls: the LogitsProcessor
    match is an ``isinstance`` check against LightRFT's specific class, so
    any future callers that build different processors won't accidentally
    trigger our termination logic.
    """
    model = rollout_actor.model
    if getattr(model, "_math_prm_rollout_eos_patch_installed", False):
        return

    orig_generate = model.generate

    @functools.wraps(orig_generate)
    def patched_generate(*args: Any, **kwargs: Any):
        logits_processor = kwargs.get("logits_processor")
        should_inject = False
        if logits_processor is not None:
            for proc in logits_processor:
                if isinstance(proc, _StructuredAnswerEosLogitsProcessor):
                    should_inject = True
                    break

        if should_inject:
            input_ids = kwargs.get("input_ids")
            if input_ids is None and args:
                input_ids = args[0]
            if input_ids is not None:
                prompt_length = int(input_ids.size(1))
                new_criteria = StructuredAnswerStoppingCriteria(
                    tokenizer=tokenizer,
                    prompt_length=prompt_length,
                    eos_token_id=int(eos_token_id),
                )
                existing = kwargs.get("stopping_criteria")
                if existing is None:
                    kwargs["stopping_criteria"] = StoppingCriteriaList([new_criteria])
                else:
                    # Be conservative — if caller already provided criteria,
                    # prepend ours rather than dropping theirs.
                    kwargs["stopping_criteria"] = StoppingCriteriaList([new_criteria, *existing])

        return orig_generate(*args, **kwargs)

    model.generate = patched_generate
    model._math_prm_rollout_eos_patch_installed = True

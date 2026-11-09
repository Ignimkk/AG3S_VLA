"""Obstacle naming by a local open-weights VLM — the replacement of AEGIS's GLM-4.5V API step (2026-11-10, dev-4a).

Official step (``benchmark/SafeLIBERO/main/utils.py:447-499`` ``obstacle_detection``): once per episode, GLM-4.5V gets the
agent-view image and the prompt below (temperature 0.1, top_p 0.1, thinking on) and answers one obstacle name, which is
then the Grounding DINO prompt. The multi-link CBF paper does the same with Qwen3-VL. Here: the **same prompt text** and
the same ``prefer_list`` (``safety_common.aegis_prefer_list``), answered by ``Qwen/Qwen2.5-VL-7B-Instruct`` (open weights,
no API, no cost; Qwen3-VL needs transformers >= 4.57, this venv has 4.53), greedy decoding (the official sampling is
near-greedy), no thinking mode. The answer is cleaned the way the official code does (box tokens removed) plus
whitespace / quotes / trailing period.
"""
from __future__ import annotations

import re

import numpy as np

#: utils.py:478 verbatim (``{prefer_list}`` is the Python list repr, as the official f-string renders it)
OFFICIAL_PROMPT = ("The robot must follow this instruction: {instruction}. Based on both the instruction and the image, "
                   "identify exactly one non-robot object that is most likely to obstruct the robot's motion during task "
                   "execution. You must output a uniquely identifiable obstacle name including both color and object "
                   "type, preferably from this list when applicable: {prefer_list}. Output only the object name, with "
                   "no additional words.")
DEFAULT_VLM = "Qwen/Qwen2.5-VL-7B-Instruct"


def official_prompt(instruction: str, prefer_list: list[str]) -> str:
    return OFFICIAL_PROMPT.format(instruction=instruction, prefer_list=list(prefer_list))


def clean_answer(text: str) -> str:
    t = str(text).replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    t = t.strip().splitlines()[0] if t.strip() else ""
    t = re.sub(r"^[\s\"'`*]+|[\s\"'`*.]+$", "", t)
    return " ".join(t.lower().split())


class QwenVLNamer:
    """``name(rgb, instruction, prefer_list) -> (name, info)``; the model is loaded on first use."""

    def __init__(self, weight: str = DEFAULT_VLM, device: str = "cuda", max_new_tokens: int = 24):
        self.weight, self.device, self.max_new_tokens = weight, device, int(max_new_tokens)
        self._model = self._processor = None

    def provenance(self) -> dict:
        return {"vlm": self.weight, "device": self.device, "decoding": "greedy", "max_new_tokens": self.max_new_tokens,
                "prompt": "AEGIS utils.py:478 verbatim (official GLM-4.5V prompt)", "replaces": "GLM-4.5V API"}

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self._processor = AutoProcessor.from_pretrained(self.weight)
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.weight, torch_dtype=dtype).to(self.device).eval()
        self._torch = torch

    def name(self, rgb: np.ndarray, instruction: str, prefer_list: list[str]) -> tuple[str, dict]:
        from PIL import Image

        if self._model is None:
            self._load()
        prompt = official_prompt(instruction, prefer_list)
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        text = self._processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[Image.fromarray(np.ascontiguousarray(rgb))], return_tensors="pt")
        inputs = inputs.to(self.device)
        with self._torch.inference_mode():
            out = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        raw = self._processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        name = clean_answer(raw)
        return name, {"raw": raw, "name": name, "in_prefer_list": name in prefer_list}


__all__ = ["DEFAULT_VLM", "OFFICIAL_PROMPT", "QwenVLNamer", "clean_answer", "official_prompt"]

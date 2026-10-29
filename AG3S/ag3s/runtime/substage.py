"""AG3S-GPU-STAGES M1 — opt-in exclusive timing of the parts *inside* one profiler stage.

`StageProfiler` times whole stages (`target_grounding`, `scene_reconstruction`, …). Deciding which part of a
stage to move to the GPU needs the parts, measured in the server itself: the recorded inputs do not replay
grounding exactly (the confirm's per-frame notes and the destination are not recorded), and the attention
maps scene reconstruction lifts are not recorded at all.

Off by default (`ENABLED = False`) and then every `part()` is a bare `yield`. On (`--profile-substages`),
each part's wall time is accumulated *exclusive* of the parts nested inside it; with `sync=True` the CUDA
stream is synchronized on entry and exit so asynchronous GPU work is charged to the part that launched it
(this changes the stage's own wall time — a profiling run is not a latency run). `pop()` returns this
frame's parts and clears them; the pipeline records them as `profile_ms["sub.<stage>.<part>"]`.

Not a YAML field for the same reason as `config.KDTREE_WORKERS`: it does not change any result.
"""
from __future__ import annotations

import contextlib
import time
from typing import Iterator

ENABLED: bool = False

_parts: dict[str, float] = {}
_stack: list[list] = []


def _sync() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextlib.contextmanager
def part(name: str, *, sync: bool = False) -> Iterator[None]:
    if not ENABLED:
        yield
        return
    if sync:
        _sync()
    _stack.append([name, 0.0])
    start = time.perf_counter()
    try:
        yield
    finally:
        if sync:
            _sync()
        elapsed = (time.perf_counter() - start) * 1000.0
        _, nested = _stack.pop()
        _parts[name] = _parts.get(name, 0.0) + elapsed - nested
        if _stack:
            _stack[-1][1] += elapsed


def pop() -> dict[str, float]:
    out = dict(_parts)
    _parts.clear()
    return out

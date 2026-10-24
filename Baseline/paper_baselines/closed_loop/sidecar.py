"""Filter sidecar: the ``SafetyFilter`` methods and the Grounding DINO detector in ``.venv-baselines`` (cvxpy, torch),
driven by the method server (``.venv-openpi-live``) over a pipe (``framing``: JSON + ``.npy``, no pickle).

Started by :class:`SidecarClient` as ``<.venv-baselines python> -m Baseline.paper_baselines.closed_loop.sidecar``.
stdout is the reply channel (the process moves Python's ``sys.stdout`` / fd 1 to stderr so library prints cannot
corrupt it); stderr goes to the server's log file.

Ops (request ``{"op": ..., ...}`` -> reply ``{"ok": True, ...}`` or ``{"ok": False, "error": traceback}``):

| op | request | reply |
|---|---|---|
| ``init`` | ``method``, ``kwargs``, ``detector`` (``weight``, ``device``) | ``provenance`` |
| ``reset`` | -- | -- |
| ``filter`` | ``obs`` (``wire_obs.obs_to_dict``), ``actions`` (T, 7) | ``actions``, ``intervened``, ``info``, ``ms`` |
| ``target`` | ``obs``, ``cameras``, thresholds | per camera top-1 ``box`` / ``score`` of ``obs.target_phrase``, ``ms`` |
| ``ping`` / ``close`` | -- | versions / -- |
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import threading
import time
import traceback

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
DEFAULT_DETECTOR = {"weight": "grounding-dino-tiny", "device": "cpu", "box_threshold": 0.35, "text_threshold": 0.25}


# =============================================================================================================
# client side (method server process)
class SidecarError(RuntimeError):
    pass


class SidecarClient:
    """One sidecar process. ``call(op, **payload)`` blocks until the reply (one request in flight)."""

    def __init__(self, python: str | os.PathLike | None = None, *, log_path: str | os.PathLike | None = None,
                 env: dict | None = None):
        from Baseline.paper_baselines.closed_loop import framing

        self._framing = framing
        python = str(python or PROJECT_ROOT / ".venv-baselines/bin/python")
        e = dict(os.environ if env is None else env)
        cache = PROJECT_ROOT / ".cache/paper_baselines"  # = setup/paper_baselines_env.sh
        e.setdefault("PAPER_BASELINES_CACHE", str(cache))
        e.setdefault("HF_HOME", str(cache / "huggingface"))
        e.setdefault("TORCH_HOME", str(cache / "torch"))
        e.setdefault("PAPER_BASELINES_WEIGHTS", str(cache / "weights"))
        e.setdefault("HF_HUB_OFFLINE", "1")
        e["PYTHONPATH"] = str(PROJECT_ROOT)
        for k in ("JAX_PLATFORMS", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE"):
            e.pop(k, None)
        self.log_path = str(log_path) if log_path else None
        self._log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        self.proc = subprocess.Popen([python, "-m", "Baseline.paper_baselines.closed_loop.sidecar"], cwd=str(PROJECT_ROOT),
                                     env=e, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._log)
        self.python = python
        self._lock = threading.Lock()

    def call(self, op: str, **payload):
        with self._lock:
            if self.proc.poll() is not None:
                raise SidecarError(f"sidecar exited with {self.proc.returncode} (log {self.log_path})")
            self._framing.write_frame(self.proc.stdin, {"op": op, **payload})
            try:
                reply = self._framing.read_frame(self.proc.stdout)
            except EOFError as exc:
                raise SidecarError(f"sidecar closed the pipe during {op!r} (log {self.log_path})") from exc
        if not reply.get("ok"):
            raise SidecarError(f"sidecar {op!r} failed:\n{reply.get('error')}")
        return reply

    def close(self) -> int | None:
        if self.proc.poll() is None:
            try:
                self._framing.write_frame(self.proc.stdin, {"op": "close"})
                self._framing.read_frame(self.proc.stdout)
            except Exception:  # noqa: BLE001 -- shutting down
                pass
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self._log not in (None, subprocess.DEVNULL):
            self._log.close()
        return self.proc.returncode


# =============================================================================================================
# sidecar side (.venv-baselines)
class _State:
    def __init__(self):
        self.filter = None
        self.method = None
        self.detector = None
        self.det_cfg = dict(DEFAULT_DETECTOR)

    def get_detector(self):
        if self.detector is None:
            from Baseline.paper_baselines.providers.grounding_dino import GroundingDinoDetector

            self.detector = GroundingDinoDetector(self.det_cfg["weight"], device=self.det_cfg["device"])
        return self.detector


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "numpy": np.__version__, "executable": sys.executable}
    for m in ("scipy", "cvxpy", "osqp", "torch", "transformers"):
        try:
            out[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            out[m] = None
    return out


def handle(state: _State, req: dict) -> dict:
    from Baseline.paper_baselines.closed_loop.wire_obs import obs_from_dict
    from Baseline.paper_baselines.methods.interface import get_method, jsonable, validate_filter_result

    op = req.get("op")
    if op == "ping":
        return {"versions": _versions()}
    if op == "init":
        state.det_cfg.update(req.get("detector") or {})
        kwargs = dict(req.get("kwargs") or {})
        if kwargs.get("obstacle_source") == "detector":
            kwargs["detector"] = state.get_detector()
        state.method = req.get("method")
        if not state.method:  # detector only (guidance arms: target box)
            return {"provenance": None, "versions": _versions()}
        state.filter = get_method(state.method, **kwargs)
        return {"provenance": jsonable(state.filter.provenance()), "versions": _versions()}
    if op == "reset":
        if state.filter is not None:
            state.filter.reset()
        return {}
    if op == "filter":
        obs = obs_from_dict(req["obs"])
        A = np.asarray(req["actions"], np.float64)
        t0 = time.perf_counter()
        res = state.filter.filter(obs, A)
        ms = (time.perf_counter() - t0) * 1e3
        validate_filter_result(res, A)
        return {"actions": np.asarray(res.actions, np.float64), "intervened": np.asarray(res.intervened, bool),
                "info": jsonable(res.info), "ms": ms}
    if op == "target":
        obs = obs_from_dict(req["obs"])
        det = state.get_detector()
        bt = float(req.get("box_threshold", state.det_cfg["box_threshold"]))
        tt = float(req.get("text_threshold", state.det_cfg["text_threshold"]))
        boxes, scores = {}, {}
        t0 = time.perf_counter()
        for cam in req.get("cameras") or list(obs.rgb):
            boxes[cam], scores[cam] = None, None
            if not obs.target_phrase or cam not in obs.rgb:
                continue
            b, s, _ = det.detect(obs.rgb[cam], obs.target_phrase, bt, tt)
            if len(s):
                i = int(np.argmax(s))
                boxes[cam], scores[cam] = np.asarray(b[i], np.float64), float(s[i])
        return {"boxes": boxes, "scores": scores, "phrase": obs.target_phrase, "ms": (time.perf_counter() - t0) * 1e3,
                "detector": {k: state.det_cfg[k] for k in ("weight", "device")}}
    if op == "close":
        return {}
    raise ValueError(f"unknown op {op!r}")


def main() -> int:
    from Baseline.paper_baselines.closed_loop import framing

    out = os.fdopen(os.dup(1), "wb")  # reply channel
    os.dup2(2, 1)  # anything printed to fd 1 now goes to the log
    sys.stdout = sys.stderr
    inp = sys.stdin.buffer
    state = _State()
    while True:
        try:
            req = framing.read_frame(inp)
        except EOFError:
            return 0
        try:
            reply = {"ok": True, **handle(state, req)}
        except Exception:  # noqa: BLE001 -- reported to the server, which falls back and logs
            reply = {"ok": False, "error": traceback.format_exc()}
            print(reply["error"], file=sys.stderr, flush=True)
        framing.write_frame(out, reply)
        if req.get("op") == "close":
            return 0


if __name__ == "__main__":
    raise SystemExit(main())

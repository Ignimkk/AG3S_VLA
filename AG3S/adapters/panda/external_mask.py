"""ABL-M6 (main ablation row 6, plan §9.45) -- AG3S with an external segmenter: the oracle-visible extent's code image
comes from a Grounded-SAM-2 target mask the Panda server asks for, instead of the client's simulator segmentation.

Only the *source* of the target surface changes (``--external-target-mask grounded-sam2``). Everything downstream is the
GT-mask path of ``--target-extent oracle-visible --oracle-extent-identity pipeline`` unchanged: the code image goes into
the request as ``ag3s/oracle_seg/<cam>`` (``wire.ORACLE_SEG``), ``SafePolicy`` unpacks it (``wire.unpack_oracle_segmentation``)
and AG3S lifts it with the same depth into the episode's ``OracleMemory`` (``pipeline._oracle_update``), whose samples
decide which seeds the fine target-free layer removes (``oracle_segmentation.oracle_evidence``). Contact rule, object
attachment and the TO are those of the version.

Per request (:meth:`ExternalTargetMask.apply`):

1. prompt = the noun phrase of the instruction (``request["prompt"]``), by the **baselines' rule**
   (``Baseline.paper_baselines.closed_loop.wire_obs.target_phrase_from_instruction``: e.g. "pick up the black bowl
   between the plate and the ramekin and place it on the plate" -> "black bowl"). No phrase -> no sidecar call, every
   camera not found.
2. RGB per wire camera = the policy image of that camera (``head`` <- ``observation/image``, ``left_wrist`` <-
   ``observation/wrist_image``) turned back to the depth grid (``wire_obs.policy_rgb_to_opencv``: column flip, bilinear
   224 -> depth size) -- the frame the baselines' sidecar segments, so a mask pixel is the depth pixel of the same ray.
3. the sidecar (``.venv-baselines``, ``Baseline/paper_baselines/closed_loop/sidecar.py`` op ``target_mask``) runs
   ``GroundedSAM2Provider`` (default preset: Grounding DINO base + SAM 2.1 hiera-large, top-1 box, ``per_frame``).
4. code image (:func:`code_image`): 1 = mask pixel with a depth, 2 = any other pixel with a depth, 0 = no depth.
   The robot pixels become 0 inside AG3S (``meta["robot_mask"] == "server"``: the frame's TSDF robot mask,
   ``pipeline._oracle_server_robot_codes``), so robot surface never enters the memory -- the simulator code image has
   the robot at 0 too. Not found -> no code 1 this request (the memory keeps its earlier target samples, as when the
   target is not visible).
5. record (``panda/oracle.external_mask`` of the call row): prompt, per camera found / score / mask pixels / code-1
   pixels / latency, the sidecar and total ms.

The sidecar uses the server's GPU at run time (its environment is the server's). Tests inject a client
(``client=``) and never start CUDA.
"""
from __future__ import annotations

import logging
import os
import pathlib
import time
from typing import Any, Optional

import numpy as np

#: ``--external-target-mask`` value -> sidecar provider name (``Baseline.paper_baselines.interface.BUILTIN_PROVIDERS``)
PROVIDER_NAME = {"grounded-sam2": "grounded_sam2"}
DEFAULT_MODE = "per_frame"
#: the code image's meta (``wire.ORACLE_SEG_META``): how AG3S knows it was made here (robot pixels from its own mask)
CODE_SOURCE = "external_segmenter"
ROBOT_MASK_FROM_SERVER = "server"
REQUEST_PREFIX = "ag3s/"


def prompt_from_instruction(instruction) -> Optional[str]:
    """The baselines' noun-phrase rule (``wire_obs.target_phrase_from_instruction``), the same text KNOWS / AEGIS give
    their detector. ``None`` when no rule matches."""
    from Baseline.paper_baselines.closed_loop.wire_obs import target_phrase_from_instruction

    return target_phrase_from_instruction(str(instruction or ""))


def wire_cameras() -> dict[str, str]:
    from Baseline.paper_baselines.closed_loop.wire_obs import WIRE_CAMERAS

    return dict(WIRE_CAMERAS)


def code_image(mask: Optional[np.ndarray], depth: np.ndarray) -> np.ndarray:
    """(H, W) uint8: ``CODE_TARGET`` on mask pixels with a depth, ``CODE_STATIC`` on every other pixel with a depth,
    ``CODE_NONE`` where the depth is 0 (no return). ``mask`` None = not found (no code 1)."""
    from AG3S.ag3s.runtime.oracle_segmentation import CODE_NONE, CODE_STATIC, CODE_TARGET

    d = np.asarray(depth)
    valid = np.isfinite(d.astype(np.float64)) & (d > 0)
    out = np.where(valid, np.uint8(CODE_STATIC), np.uint8(CODE_NONE)).astype(np.uint8)
    if mask is not None:
        m = np.asarray(mask, bool)
        if m.shape != d.shape:
            raise ValueError(f"external target mask {m.shape} != depth {d.shape}")
        out[m & valid] = CODE_TARGET
    return out


class ExternalTargetMask:
    """One segmenter sidecar for the server's lifetime; :meth:`apply` per request, :meth:`close` at exit."""

    def __init__(self, provider: str = "grounded-sam2", mode: Optional[str] = None, *, client=None,
                 size: str = "default", device: Optional[str] = None, log_path=None, python=None,
                 sidecar_provider: Optional[str] = None):
        # sidecar_provider: test only -- a ``module:attr`` stub in place of the registered provider (same op, same reply)
        if provider not in PROVIDER_NAME:
            raise ValueError(f"unknown external target mask {provider!r} (known: {sorted(PROVIDER_NAME)})")
        self.provider, self.mode, self.size = provider, str(mode or DEFAULT_MODE), str(size)
        self.cameras = wire_cameras()
        self._sidecar_provider = sidecar_provider
        self.log_path = None if log_path is None else str(log_path)
        if client is None:
            from Baseline.paper_baselines.closed_loop.sidecar import SidecarClient

            env = dict(os.environ)
            env.setdefault("OMP_NUM_THREADS", "1")
            client = SidecarClient(python, log_path=log_path, env=env)
        self.client = client
        self.versions = client.call("ping").get("versions")
        kwargs = {"mode": self.mode, "size": self.size, **({"device": device} if device else {})}
        self.provider_kwargs = kwargs
        self.provenance = client.call("segmenter_init", provider=self.sidecar_provider(), kwargs=kwargs).get("provenance")
        self.n_calls = 0

    @classmethod
    def from_args(cls, args, **kw) -> Optional["ExternalTargetMask"]:
        """``None`` unless ``--external-target-mask``; the sidecar log goes to ``$AG3S_RUN_DIR/logs``."""
        name = getattr(args, "external_target_mask", None)
        if name is None:
            return None
        if "log_path" not in kw:
            run = os.environ.get("AG3S_RUN_DIR")
            kw["log_path"] = None if not run else pathlib.Path(run) / "logs" / "external_mask_sidecar.log"
        return cls(name, getattr(args, "external_target_mask_mode", None), **kw)

    def sidecar_provider(self) -> str:
        return self._sidecar_provider or PROVIDER_NAME[self.provider]

    def record(self) -> dict:
        """Server config record (``panda_server_config.json``)."""
        return {"provider": self.provider, "sidecar_provider": self.sidecar_provider(), "kwargs": dict(self.provider_kwargs),
                "provenance": self.provenance, "sidecar_versions": self.versions, "sidecar_log": self.log_path,
                "prompt_rule": "Baseline wire_obs.target_phrase_from_instruction (the KNOWS / AEGIS baselines' rule)",
                "rgb": {cam: f"{key} -> wire_obs.policy_rgb_to_opencv (column flip, bilinear to the depth grid)"
                        for cam, key in self.cameras.items()},
                "code_image": "1 = mask & depth > 0, 2 = other depth > 0, 0 = no depth; robot pixels -> 0 in AG3S "
                              "(this frame's TSDF robot mask)",
                "not_found": "no code 1 this request (the memory keeps its earlier samples)"}

    def metadata(self) -> dict:
        return {"provider": self.provider, "mode": self.mode, "client_segmentation": False}

    def reset(self) -> None:
        self.client.call("reset")

    def apply(self, obs: dict) -> dict:
        """Put this request's code images and meta into ``obs`` (``ag3s/oracle_seg/<cam>``, ``ag3s/oracle_seg_meta``).

        Returns the call record. Raises when a camera has no policy RGB or the sidecar fails (a run that silently
        lost its segmenter would read as the external-segmenter row)."""
        from AG3S.ag3s.runtime.oracle_segmentation import CODE_TARGET, CODE_VERSION
        from AG3S.trajopt import wire

        t0 = time.perf_counter()
        if obs.get(REQUEST_PREFIX + "reset"):
            self.reset()
        cams = [str(c) for c in obs.get(REQUEST_PREFIX + "cameras", []) or []]
        depth = {c: np.asarray(obs[f"{REQUEST_PREFIX}depth/{c}"]) for c in cams}
        prompt = prompt_from_instruction(obs.get("prompt"))
        rec: dict[str, Any] = {"provider": self.provider, "mode": self.mode, "prompt": prompt, "cameras": {}}
        masks: dict[str, Optional[np.ndarray]] = {c: None for c in cams}
        results: dict[str, dict] = {}
        if prompt is None:
            rec["why"] = "no_prompt"
            rec["sidecar_ms"] = None
        else:
            from Baseline.paper_baselines.closed_loop.wire_obs import policy_rgb_to_opencv

            rgb = {}
            for c in cams:
                key = self.cameras.get(c)
                if key is None or key not in obs:
                    raise KeyError(f"--external-target-mask: no policy RGB for wire camera {c!r} (request key {key!r})")
                rgb[c] = policy_rgb_to_opencv(obs[key], depth[c].shape[:2])
            reply = self.client.call("target_mask", rgb=rgb, prompt=prompt)
            rec["sidecar_ms"] = float(reply.get("ms", 0.0))
            results = dict(reply.get("results") or {})
            for c in cams:
                m = (reply.get("masks") or {}).get(c)
                masks[c] = None if m is None or not np.asarray(m).any() else np.asarray(m, bool)
        for c in cams:
            lab = code_image(masks[c], depth[c])
            obs[f"{REQUEST_PREFIX}{wire.ORACLE_SEG}/{c}"] = lab
            r = results.get(c) or {}
            ex = r.get("extras") or {}
            rec["cameras"][c] = {"found": bool(r.get("found", False)), "score": r.get("score"),
                                 "mask_pixels": int(r.get("mask_pixels", 0) or 0),
                                 "code1_pixels": int(np.count_nonzero(lab == CODE_TARGET)),
                                 "latency_ms": r.get("latency_ms"),
                                 **({"why": ex["not_found_reason"]} if "not_found_reason" in ex else {}),
                                 **({"source": ex["source"]} if "source" in ex else {})}
        obs[REQUEST_PREFIX + wire.ORACLE_SEG_META] = {
            "version": CODE_VERSION, "target": prompt, "codes": {} if prompt is None else {prompt: CODE_TARGET},
            "source": CODE_SOURCE, "provider": self.provider, "robot_mask": ROBOT_MASK_FROM_SERVER}
        rec["found"] = any(v["found"] for v in rec["cameras"].values())
        rec["total_ms"] = (time.perf_counter() - t0) * 1e3
        self.n_calls += 1
        return rec

    def close(self) -> Optional[int]:
        try:
            return self.client.close()
        except Exception:  # noqa: BLE001 -- the server is stopping
            logging.exception("external target mask sidecar close failed")
            return None


__all__ = ["CODE_SOURCE", "DEFAULT_MODE", "ExternalTargetMask", "PROVIDER_NAME", "ROBOT_MASK_FROM_SERVER", "code_image",
           "prompt_from_instruction", "wire_cameras"]

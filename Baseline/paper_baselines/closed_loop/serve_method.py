"""Method server for the LIBERO / SafeLIBERO closed loop (``.venv-openpi-live``): same wire as
``AG3S.trajopt.serve_safe --robot panda`` (openpi websocket, ``libero_safe_client.py`` request), one comparison method.

```
libero_safe_client.py (.venv-libero, unchanged)  --request (policy keys + ag3s/* geometry + panda/*)-->
  MethodServer.infer
    wire_obs.scene_observation(request)                         (non-privileged SceneObservation)
    none     : PandaVLAOnly(SeededPolicy(policy)).infer(request)  = the VLA-only server, byte for byte
    filter   : same call -> nominal chunk -> sidecar(.venv-baselines).filter(obs, chunk) -> chunk
    guidance : PandaVLAOnly(SeededPolicy(GuidedPolicy(policy))) -> pi0.sample_actions(guidance_fn=method)
  <--response: actions (H, 7) EE deltas; the client executes rows 0..4 (replan 5), "panda/mode" = "method:<arm>"
```

Every call is logged to ``<run>/method_calls/calls.jsonl`` (+ per-episode ``ep<k>_seed<s>.npz``: nominal and served
chunks): intervened rows, ``|delta a|`` (``evaluate_offline.delta_stats``, the offline keys), the method's own ``h``
(filters: ``evaluate_offline.filter_step_info``; guidance: EE / hand min signed distance of the decoded path to the
method's obstacle set, nominal vs guided), perception, timings, fallbacks. ``<run>`` = ``$AG3S_RUN_DIR`` (launcher) or
``--record-dir``. A method failure never breaks the episode: the nominal chunk is served and the error logged.

Policy loading is ``serve_safe``'s Panda path (``AG3S/adapters/panda/server.py:main``): ``create_trained_policy(
get_config(--config), --checkpoint, default_prompt=...)``. ``--fake-policy constant|recorded`` replaces it on CPU.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "third_party" / "openpi" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from Baseline.paper_baselines.closed_loop import arms as A  # noqa: E402

GUIDE_TOL_RAW = 1e-4  # = evaluate_offline.GUIDE_TOL_RAW (D's intervened-row rule)
NORM_REL = "assets/physical-intelligence/libero/norm_stats.json"
DEFAULT_SCENE_KW = {"voxel": 0.02, "max_points": 1024, "workspace_radius": 0.6, "remove_table": False}
#: methods whose numpy preparation reads every obstacle point's extent: FAR padding points (1 km away,
#: ``guidance_common.geometry.FAR``) would blow up Lan-o3dp's xy cluster grid. Unpadded = the guidance function is
#: recompiled when the point count changes (cheap with --guidance-exec callback; the pi0.5 sampler is not).
NO_PADDING = {"lan_o3dp_guidance_adapted"}


def storage_root() -> pathlib.Path:
    return pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", PROJECT_ROOT.parent / "AG3S_VLA-storage"))


def default_checkpoint() -> pathlib.Path:
    return storage_root() / "checkpoints/panda/pi05_libero"


# =============================================================================================================
class CallRecorder:
    def __init__(self, root: pathlib.Path | None):
        self.root = None if root is None else pathlib.Path(root)
        if self.root is not None:
            self.root.mkdir(parents=True, exist_ok=True)
        self._jsonl = None if self.root is None else open(self.root / "calls.jsonl", "a")
        self._arrays: dict[str, list] = {}
        self._episode = None

    def start_episode(self, episode: int, policy_seed) -> None:
        self.flush()
        self._episode = (int(episode), policy_seed)
        self._arrays = {"seq": [], "nominal": [], "served": []}

    def add(self, row: dict, nominal: np.ndarray | None, served: np.ndarray) -> None:
        from Baseline.paper_baselines.methods.interface import jsonable

        if self._jsonl is not None:
            self._jsonl.write(json.dumps(jsonable(row), allow_nan=True) + "\n")
            self._jsonl.flush()
        if self._episode is not None:
            self._arrays["seq"].append(int(row.get("seq") or 0))
            self._arrays["nominal"].append(np.asarray(served if nominal is None else nominal, np.float64))
            self._arrays["served"].append(np.asarray(served, np.float64))

    def flush(self) -> None:
        if self.root is None or self._episode is None or not self._arrays.get("seq"):
            return
        k, ps = self._episode
        try:
            arrays = {key: np.stack(v) for key, v in self._arrays.items()}
        except ValueError:
            arrays = {"seq": np.asarray(self._arrays["seq"])}
        path = self.root / f"ep{k:03d}_seed{ps}.npz"
        np.savez_compressed(path, **arrays)
        self._arrays = {"seq": [], "nominal": [], "served": []}

    def close(self) -> None:
        self.flush()
        if self._jsonl is not None:
            self._jsonl.close()
            self._jsonl = None


# =============================================================================================================
class MethodServer:
    """The served object (``WebsocketPolicyServer(policy=...)``)."""

    def __init__(self, inner, arm: dict, *, method=None, sidecar=None, norm: dict | None = None, backend=None,
                 recorder: CallRecorder | None = None, scene_kwargs: dict | None = None,
                 perception_kwargs: dict | None = None, record_nominal: bool = True, target_cameras=None):
        from AG3S.adapters.panda.server import PandaVLAOnly
        from AG3S.trajopt.policy_seed import SeededPolicy

        self.arm, self.family = arm, arm["family"]
        self.inner, self.method, self.sidecar, self.norm = inner, method, sidecar, norm
        self.recorder = recorder or CallRecorder(None)
        self.scene_kwargs = {**DEFAULT_SCENE_KW, **(scene_kwargs or {})}
        self.perception_kwargs = dict(perception_kwargs or {})
        self.target_cameras = list(target_cameras) if target_cameras else None
        self.guided = None
        if self.family == "guidance":
            from Baseline.paper_baselines.closed_loop.guided import GuidedPolicy

            self.guided = GuidedPolicy(inner, backend, method, norm, record_nominal=record_nominal)
            self.vla = PandaVLAOnly(SeededPolicy(self.guided))
        else:
            self.vla = PandaVLAOnly(SeededPolicy(inner))
        self.episode = -1
        self.policy_seed = None
        self.last_gripper = None
        self.n_calls = 0
        self.n_fallback = 0

    @property
    def metadata(self):
        return self.vla.metadata

    def reset(self):
        self.vla.reset()

    def close(self):
        self.recorder.close()
        if self.sidecar is not None:
            self.sidecar.close()

    # --- one call -----------------------------------------------------------------------------------------------
    def _new_episode(self, request: dict) -> None:
        from Baseline.paper_baselines.closed_loop.fake_policy import decode_policy_seed

        self.episode += 1
        self.policy_seed = request.get("ag3s/policy_seed")
        self.last_gripper = None
        if self.family == "filter" and self.sidecar is not None:
            self.sidecar.call("reset")
        if self.family == "guidance" and hasattr(self.method, "reset"):
            self.method.reset()
        self.recorder.start_episode(self.episode, self.policy_seed)
        self.episode_info = decode_policy_seed(self.policy_seed)

    def _target_masks(self, sobs) -> tuple[dict | None, dict]:
        from Baseline.paper_baselines.closed_loop.perception import box_masks
        from Baseline.paper_baselines.closed_loop.wire_obs import obs_to_dict

        if self.arm.get("target") != "detector" or self.sidecar is None:
            return None, {"target": self.arm.get("target")}
        d = obs_to_dict(sobs)
        d["depth"], d["extras"] = {}, {}
        r = self.sidecar.call("target", obs=d, cameras=self.target_cameras or list(sobs.rgb))
        shapes = {c: np.asarray(sobs.depth[c]).shape for c in sobs.depth}
        boxes = {c: r["boxes"].get(c) for c in shapes}
        return box_masks(boxes, shapes), {"target": "detector", "phrase": r.get("phrase"), "scores": r.get("scores"),
                                          "boxes": {c: None if b is None else np.asarray(b).round(1).tolist()
                                                    for c, b in boxes.items()}, "ms": r.get("ms")}

    def infer(self, obs: dict, **kwargs) -> dict:
        from Baseline.paper_baselines.closed_loop.wire_obs import scene_observation

        t0 = time.perf_counter()
        obs = dict(obs)
        seq = obs.get("ag3s/seq")
        if obs.get("ag3s/reset") or self.episode < 0:
            self._new_episode(obs)
        if hasattr(self.inner, "set_context"):  # fake policies only (recorded chunk lookup)
            self.inner.set_context(policy_seed=self.policy_seed, seq=seq)
        row = {"episode": self.episode, **self.episode_info, "seq": seq, "arm": self.arm["arm"],
               "variant": self.arm["variant"], "family": self.family}
        nominal = None
        if self.family == "none":
            result = self.vla.infer(obs, **kwargs)
            t_pol = time.perf_counter()
            row["ms"] = {"policy": (t_pol - t0) * 1e3}
        elif self.family == "filter":
            result, nominal = self._filter_call(obs, row, seq, t0, **kwargs)
        else:
            result, nominal = self._guidance_call(obs, row, seq, t0, **kwargs)
        served = np.asarray(result["actions"])
        if served.ndim == 2 and served.shape[0]:
            self.last_gripper = float(served[min(4, served.shape[0] - 1), 6])  # last executed row (replan 5)
        row["ms"]["total"] = (time.perf_counter() - t0) * 1e3
        row["ms"]["serve"] = row["ms"]["total"] - row["ms"].get("diagnostics", 0.0)  # latency without logging metrics
        self.recorder.add(row, nominal, served)
        self.n_calls += 1
        result["panda/mode"] = f"method:{self.arm['arm']}"
        result["timing_ms"] = {k: round(float(v), 3) for k, v in row["ms"].items()}
        return result

    def _scene_obs(self, obs, seq):
        from Baseline.paper_baselines.closed_loop.wire_obs import scene_observation

        return scene_observation(obs, last_gripper_cmd=self.last_gripper, seq=seq)

    def _filter_call(self, obs, row, seq, t0, **kwargs):
        from Baseline.paper_baselines.closed_loop.perception import clusters, scene_points
        from Baseline.paper_baselines.closed_loop.wire_obs import obs_to_dict
        from Baseline.paper_baselines.methods.evaluate_offline import delta_stats, filter_step_info

        result = self.vla.infer(obs, **kwargs)
        t_pol = time.perf_counter()
        A0 = np.asarray(result["actions"])
        row["ms"] = {"policy": (t_pol - t0) * 1e3}
        try:
            sobs = self._scene_obs(obs, seq)
            if self.arm["obstacles"] == "depth":
                masks, tinfo = self._target_masks(sobs)
                sp = scene_points(sobs, target_masks=masks, **self.perception_kwargs)
                sobs.extras["obstacle_points"] = clusters(sp["obstacle_points"])
                row["perception"] = {**sp["info"], **tinfo, "clusters": {k: int(len(v)) for k, v in
                                                                         sobs.extras["obstacle_points"].items()}}
            t_perc = time.perf_counter()
            d = obs_to_dict(sobs)
            r = self.sidecar.call("filter", obs=d, actions=np.asarray(A0, np.float64))
            t_f = time.perf_counter()
            B = np.asarray(r["actions"], np.float64)
            iv = np.asarray(r["intervened"], bool)
            served = B.astype(A0.dtype)
            row.update(delta_stats(A0, B, iv))
            row["intervened"] = iv.tolist()
            row.update({k: v for k, v in filter_step_info(self.arm["method"], r["info"]).items()
                        if k in ("h_steps", "h_min", "h_neg_steps", "qp_status", "qp_fail", "emergency_stops")})
            row["filter_perception"] = r["info"].get("perception")
            row["ms"].update({"perception": (t_perc - t_pol) * 1e3, "sidecar_roundtrip": (t_f - t_perc) * 1e3,
                              "filter": r.get("ms")})
            row["fallback"] = None
            result["actions"] = served
        except Exception:  # noqa: BLE001 -- serve the nominal chunk, log why
            self.n_fallback += 1
            row["fallback"] = "nominal (method error)"
            row["error"] = traceback.format_exc()
            logging.error("method %s failed at seq %s:\n%s", self.arm["arm"], seq, row["error"])
        return result, A0

    def _guidance_call(self, obs, row, seq, t0, **kwargs):
        from Baseline.paper_baselines.closed_loop.perception import scene_points
        from Baseline.paper_baselines.methods.evaluate_offline import delta_stats
        from Baseline.paper_baselines.methods.guidance_common.evaluation import path_metrics
        from Baseline.paper_baselines.methods.guidance_common.scene import build_scene

        scene, sobs, err = None, None, None
        try:
            sobs = self._scene_obs(obs, seq)
            masks, tinfo = self._target_masks(sobs)
            sp = scene_points(sobs, target_masks=masks, **self.perception_kwargs)
            sobs.extras["obstacle_points"] = sp["obstacle_points"]
            sobs.extras["target_points"] = sp["target_points"]
            kw = {k: v for k, v in self.scene_kwargs.items()}
            scene = build_scene(sobs, obstacle_source="points", **kw)
            if self.arm["method"] not in NO_PADDING:  # fixed shapes -> one guidance compile per structure
                scene.obstacles = scene.obstacles.padded(int(kw.get("max_points", 1024)), 4)
            row["perception"] = {**sp["info"], **tinfo, "scene": scene.info}
        except Exception:  # noqa: BLE001
            err = traceback.format_exc()
        t_perc = time.perf_counter()
        if scene is None:  # no scene: unguided sample through the same path
            self.guided.set_call(sobs, None)
            saved, self.guided.method = self.guided.method, _NoGuidance()
            try:
                result = self.vla.infer(obs, **kwargs)
            finally:
                self.guided.method = saved
        else:
            self.guided.set_call(sobs, scene)
            result = self.vla.infer(obs, **kwargs)
        t_s = time.perf_counter()
        g = dict(self.guided.last)
        nominal = g.pop("nominal_actions", None)
        served = np.asarray(result["actions"])
        err = err or g.get("error")
        row["ms"] = {"perception": (t_perc - t0) * 1e3, "policy_and_guidance": (t_s - t_perc) * 1e3,
                     **{f"guided_{k}": v for k, v in (g.get("ms") or {}).items()}}
        row["guidance"] = {k: g.get(k) for k in ("guided", "identity_reason", "jaxpr_key", "n_consts", "cache", "skip")}
        row["denoise_steps"] = g.get("denoise_steps")
        t_d = time.perf_counter()
        if nominal is not None:
            nom = np.asarray(nominal, np.float64)
            iv = np.max(np.abs(served.astype(np.float64)[:, :3] - nom[:, :3]), axis=1) > GUIDE_TOL_RAW
            row.update(delta_stats(nom, served, iv))
            row["intervened"] = iv.tolist()
            if scene is not None:
                H = served.shape[0]
                mb, mg = path_metrics(nom, scene, H), path_metrics(served.astype(np.float64), scene, H)
                row["h_nominal"] = {"ee_min_sdf_m": mb["ee_min_sdf_m"], "hand_min_sdf_m": mb["hand_min_sdf_m"]}
                row["h_served"] = {"ee_min_sdf_m": mg["ee_min_sdf_m"], "hand_min_sdf_m": mg["hand_min_sdf_m"]}
                row["h_min"] = mg["hand_min_sdf_m"]
        row["ms"]["diagnostics"] = (time.perf_counter() - t_d) * 1e3  # logging only, after the chunk is final
        row["fallback"] = None if err is None else "unguided sample (guidance error)"
        if err is not None:
            self.n_fallback += 1
            row["error"] = err
            logging.error("guidance %s failed at seq %s:\n%s", self.arm["arm"], seq, err)
        return result, nominal


class _NoGuidance:
    def make_guidance_fn(self, obs, norm, *, noise=None, scene=None):
        return lambda x, t: x


# =============================================================================================================
def load_norm(checkpoint: pathlib.Path) -> dict:
    from Baseline.paper_baselines.methods.guidance_common.core import load_norm_stats

    return load_norm_stats(str(pathlib.Path(checkpoint) / NORM_REL))


def build_server(args, *, policy=None, record_root=None) -> tuple[MethodServer, dict]:
    """Everything below the openpi policy (tests pass a fake ``policy``)."""
    from Baseline.paper_baselines.methods.interface import get_method, jsonable

    arm = A.resolve(args.method, args.obstacles, args.target)
    ckpt = pathlib.Path(args.checkpoint or default_checkpoint())
    norm = load_norm(ckpt) if arm["family"] == "guidance" or args.fake_policy != "none" else None
    if policy is None:
        if args.fake_policy != "none":
            from Baseline.paper_baselines.closed_loop.fake_policy import FakePolicy

            policy = FakePolicy(norm, source=args.fake_policy, run_dir=args.fake_source_run)
        else:
            from openpi.policies import policy_config as _policy_config
            from openpi.training import config as _config

            policy = _policy_config.create_trained_policy(_config.get_config(args.config), str(ckpt),
                                                          default_prompt=args.default_prompt)
    method_kwargs = json.loads(args.method_kwargs) if args.method_kwargs else {}
    sidecar, method, backend, info = None, None, None, {"arm": arm}
    run = record_root
    need_sidecar = arm["family"] == "filter" or arm.get("target") == "detector"
    if need_sidecar:
        from Baseline.paper_baselines.closed_loop.sidecar import SidecarClient

        log = None if run is None else pathlib.Path(run) / "sidecar.log"
        env = dict(os.environ)
        if args.sidecar_device == "cpu":
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("OMP_NUM_THREADS", "1")
        sidecar = SidecarClient(args.sidecar_python, log_path=log, env=env)
        info["sidecar"] = {"python": sidecar.python, "ping": sidecar.call("ping")["versions"], "log": str(log)}
        det = {"weight": args.detector_weight, "device": args.sidecar_device}
        if arm["family"] == "filter":
            kw = dict(method_kwargs)
            if arm["obstacles"] == "detector":
                kw.setdefault("obstacle_source", "detector")
                kw.setdefault("detector_prompt", "all_objects")
                kw.setdefault("detector_device", args.sidecar_device)
                kw.setdefault("detector_weight", args.detector_weight)
            else:
                kw.setdefault("obstacle_source", "given")
            r = sidecar.call("init", method=arm["method"], kwargs=kw, detector=det)
            info["method_kwargs"], info["provenance"] = kw, r["provenance"]
        else:
            sidecar.call("init", method=None, detector=det)  # detector only (target box)
            info["target_detector"] = det
    if arm["family"] == "guidance":
        from Baseline.paper_baselines.closed_loop.guided import GuidanceSlot, OpenpiBackend, SurrogateBackend

        method = get_method(arm["method"], **method_kwargs)
        slot = GuidanceSlot()
        if args.fake_policy != "none":
            backend = SurrogateBackend(policy, slot, exec_mode=args.guidance_exec)
        else:
            backend = OpenpiBackend(policy, slot, exec_mode=args.guidance_exec)
        info["method_kwargs"], info["provenance"] = method_kwargs, jsonable(method.provenance())
        info["guidance_exec"] = args.guidance_exec
    rec_dir = None if run is None else pathlib.Path(run) / "method_calls"
    server = MethodServer(policy, arm, method=method, sidecar=sidecar, norm=norm, backend=backend,
                          recorder=CallRecorder(rec_dir), record_nominal=not args.no_record_nominal,
                          scene_kwargs=json.loads(args.scene_kwargs) if args.scene_kwargs else None)
    info.update({"config": args.config, "checkpoint": str(ckpt), "fake_policy": args.fake_policy,
                 "fake_source_run": args.fake_source_run, "record_dir": None if rec_dir is None else str(rec_dir),
                 "scene_kwargs": server.scene_kwargs, "record_nominal": not args.no_record_nominal,
                 "action_convention": "pi05_libero EE delta chunk (H=10); client executes rows 0-4 (replan 5), "
                                      "same as serve_safe --robot panda"})
    return server, info


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--method", required=True, choices=sorted(A.ARMS))
    ap.add_argument("--obstacles", choices=A.OBSTACLE_SOURCES, default=None, help="filters: detector (default) | depth")
    ap.add_argument("--target", choices=A.TARGET_SOURCES, default=None, help="target exclusion: detector (default) | none")
    ap.add_argument("--config", default="pi05_libero")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--default-prompt", default=None)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--method-kwargs", default=None, help="JSON kwargs for get_method")
    ap.add_argument("--scene-kwargs", default=None, help="JSON kwargs for guidance_common.scene.build_scene")
    ap.add_argument("--guidance-exec", choices=("callback", "inline"), default="callback")
    ap.add_argument("--no-record-nominal", action="store_true", help="guidance: skip the unguided sample per call")
    ap.add_argument("--sidecar-python", default=str(PROJECT_ROOT / ".venv-baselines/bin/python"))
    ap.add_argument("--sidecar-device", default="cpu", help="cpu | cuda (Grounding DINO in the sidecar)")
    ap.add_argument("--detector-weight", default="grounding-dino-tiny")
    ap.add_argument("--fake-policy", choices=("none", "constant", "recorded"), default="none",
                    help="CPU smoke: no checkpoint; constant descent or a recorded run's chunks")
    ap.add_argument("--fake-source-run", default=None, help="run dir of --fake-policy recorded")
    ap.add_argument("--record-dir", default=None, help="default: $AG3S_RUN_DIR")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)
    if args.fake_policy == "recorded" and not args.fake_source_run:
        raise SystemExit("--fake-policy recorded needs --fake-source-run")
    run = args.record_dir or os.environ.get("AG3S_RUN_DIR")
    server, info = build_server(args, record_root=run)
    info["port"] = args.port
    if run:
        p = pathlib.Path(run) / "logs" / "method_server_config.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x") as f:
            json.dump(info, f, indent=1, default=str)
    logging.info("method server: %s", json.dumps({"arm": info["arm"]["variant"], "port": args.port}))
    from openpi.serving import websocket_policy_server

    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.policy_seed import describe_rng_owner, describe_xla_determinism

    logging.info("%s", describe_rng_owner(server.vla._policy))
    logging.info("%s", describe_xla_determinism(os.environ.get("XLA_FLAGS")))
    ss.exit_cleanly_on_sigterm()
    try:
        websocket_policy_server.WebsocketPolicyServer(policy=server, host="0.0.0.0", port=args.port,
                                                      metadata=server.metadata).serve_forever()
    finally:
        logging.info("method server stopping: calls=%d fallbacks=%d", server.n_calls, server.n_fallback)
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

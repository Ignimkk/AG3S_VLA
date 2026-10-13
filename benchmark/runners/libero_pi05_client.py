"""pi0.5-only LIBERO client (SL0). Runs under .venv-libero (Python 3.8) -- keep 3.8 syntax.

Behaviour of observation building, image preprocessing, ``_quat2axisangle``, dummy wait,
replanning and success judgement is identical to the official OpenPI example
``third_party/openpi/examples/libero/main.py``. The differences are only in *what is recorded
and where* (run directory below ``$AG3S_STORAGE_ROOT/results/benchmark``, exclusive files) and in
the selectable subset of task/episode ids; see SL0.impl.md for the full diff list.

Heavy simulator / client imports are deferred into the functions that need them so the module
(protocol constants, ``_quat2axisangle``, run-dir rules) can be imported by CPU unit tests under
.venv-ag3s without LIBERO installed.

SL1: ``--capture-geometry`` (default off) creates the env with ``camera_depths=True`` and records, per
observation, the V11AB camera geometry of ``benchmark/adapters/libero_panda`` in
``episodes/<stem>_geom.npz``. With the flag off the code path, env arguments and every output are
those of SL0 (no geometry import, no extra file, no extra jsonl key).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import json
import logging
import math
import os
import pathlib
import sys
import time
import traceback
import uuid

import numpy as np

# --- Protocol constants: identical to the official example (checked by tests via AST). ---------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data
DEFAULT_HOST = "0.0.0.0"
DEFAULT_RESIZE_SIZE = 224
DEFAULT_REPLAN_STEPS = 5
DEFAULT_TASK_SUITE = "libero_spatial"
DEFAULT_NUM_STEPS_WAIT = 10
DEFAULT_SEED = 7
VIDEO_FPS = 10
MAX_STEPS = {
    "libero_spatial": 220,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}
STATUS_VALUES = ("success", "max_steps", "execution_error")
KST = datetime.timezone(datetime.timedelta(hours=9), "KST")

# Raw robot state recorded per observation (before the policy-side preprocessing).
RECORDED_STATE_KEYS = ("robot0_eef_pos", "robot0_eef_quat", "robot0_joint_pos", "robot0_gripper_qpos")
RECORDED_IMAGE_KEYS = ("agentview_image", "robot0_eye_in_hand_image")


# --- Run directory rules ---------------------------------------------------------------------------
def storage_root() -> pathlib.Path:
    default = pathlib.Path(__file__).resolve().parents[2].parent / "AG3S_VLA-storage"
    return pathlib.Path(os.environ.get("AG3S_STORAGE_ROOT", str(default))).expanduser().resolve()


def results_root() -> pathlib.Path:
    return storage_root() / "results" / "benchmark"


def validate_run_dir(path) -> pathlib.Path:
    """Accept only an existing run root strictly below results/benchmark that has a manifest."""
    resolved = pathlib.Path(path).expanduser().resolve()
    base = results_root().resolve()
    if resolved == base or base not in resolved.parents:
        raise ValueError("--run-dir must be below {}: {}".format(base, resolved))
    if not resolved.is_dir():
        raise ValueError("--run-dir does not exist: {}".format(resolved))
    if not (resolved / "run_manifest.json").is_file():
        raise ValueError("--run-dir has no run_manifest.json (allocate it with common.run_artifacts): {}".format(resolved))
    return resolved


def episode_stem(suite: str, task_id: int, episode_idx: int) -> str:
    return "{}_t{}_e{}".format(suite, task_id, episode_idx)


def _exclusive_publish(tmp: pathlib.Path, final: pathlib.Path) -> None:
    """Atomically publish ``tmp`` as ``final``; fail (and keep nothing) if ``final`` exists."""
    try:
        os.link(str(tmp), str(final))
    finally:
        tmp.unlink()


def _tmp_sibling(final: pathlib.Path, suffix: str) -> pathlib.Path:
    return final.parent / ".tmp-{}-{}{}".format(final.stem, uuid.uuid4().hex[:8], suffix)


def _now() -> str:
    return datetime.datetime.now(KST).isoformat()


# --- Official helpers (verbatim behaviour) ---------------------------------------------------------
def make_policy_client(host, port):
    """WebsocketClientPolicy that also works with websockets 13.x (the last release for Python 3.8).

    This checkout's openpi-client passes ``ping_timeout`` to ``websockets.sync.client.connect``;
    that keyword exists only from websockets 14 (Python >= 3.9). With 13.x it is forwarded to
    ``socket.create_connection`` and raises TypeError. The 13.x sync client sends no keepalive
    pings, so omitting the keyword is the upstream OpenPI behaviour (no ping timeout).
    Returns (client, connector name).
    """
    import inspect

    import websockets.sync.client
    from openpi_client import msgpack_numpy
    from openpi_client import websocket_client_policy as _websocket_client_policy

    base = _websocket_client_policy.WebsocketClientPolicy
    if "ping_timeout" in inspect.signature(websockets.sync.client.connect).parameters:
        return base(host, port), "openpi_client.WebsocketClientPolicy"

    class _CompatWebsocketClientPolicy(base):
        def _wait_for_server(self):
            logging.info("Waiting for server at %s...", self._uri)
            while True:
                try:
                    headers = {"Authorization": "Api-Key {}".format(self._api_key)} if self._api_key else None
                    conn = websockets.sync.client.connect(
                        self._uri, compression=None, max_size=None, additional_headers=headers
                    )
                    metadata = msgpack_numpy.unpackb(conn.recv())
                    return conn, metadata
                except ConnectionRefusedError:
                    logging.info("Still waiting for server...")
                    time.sleep(5)

    return _CompatWebsocketClientPolicy(host, port), "websockets<14 compat (no ping_timeout)"


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _get_libero_env_with_depth(task, resolution, seed):
    """``_get_libero_env`` with ``camera_depths=True`` (SL1 geometry capture); nothing else differs."""
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution,
                "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# --- Env history (shared with the scene / phase exports and the replay, SL4b) ----------------------
def make_task_env(task, seed, capture_geometry):
    """The env of one task exactly as :func:`run_tasks` builds it: ``_get_libero_env[_with_depth]`` (which
    seeds the global numpy RNG, ``env.seed`` = ``np.random.seed``) and, with geometry capture, the capture
    observable installed before the first ``env.reset()``. Returns ``(env, task_description)``."""
    make_env = _get_libero_env_with_depth if capture_geometry else _get_libero_env
    env, task_description = make_env(task, LIBERO_ENV_RESOLUTION, seed)
    if capture_geometry:
        # Read-only snapshot observable sampled with the cameras; installed before run_episode's env.reset().
        _geometry_adapter()[0].install_capture_observable(env)
    return env, task_description


def start_episode(env, initial_states, episode_idx):
    """Start of every episode of the official loop: ``env.reset()`` then ``set_init_state``. Returns the obs.

    LIBERO's reset is a hard reset that re-samples the **fixture** placement (``body_pos`` of e.g.
    ``wooden_cabinet_1`` / ``flat_stove_1``, not part of the simulator state) from the global numpy RNG, so the
    scene of episode k of a task depends on how many resets the task's env saw before (SL4b)."""
    env.reset()
    return env.set_init_state(initial_states[episode_idx])


# --- Episode --------------------------------------------------------------------------------------
class _Recorder:
    """Per-step raw data; observations have one more entry than actions (initial obs)."""

    def __init__(self):
        self.obs = {key: [] for key in RECORDED_STATE_KEYS + RECORDED_IMAGE_KEYS}
        self.action, self.reward, self.done, self.is_wait, self.chunk_index = [], [], [], [], []
        self.chunks, self.chunk_t, self.policy_state = [], [], []

    def observe(self, obs):
        # Copy before the official code mutates robot0_eef_quat in place inside _quat2axisangle.
        for key in self.obs:
            self.obs[key].append(np.array(obs[key], copy=True))

    def step(self, action, reward, done, is_wait):
        self.action.append(np.asarray(action, dtype=np.float64))
        self.reward.append(float(reward))
        self.done.append(bool(done))
        self.is_wait.append(bool(is_wait))
        self.chunk_index.append(-1 if is_wait else len(self.chunks) - 1)

    def arrays(self):
        out = {}
        for key, values in self.obs.items():
            out["obs_" + key] = np.stack(values) if values else np.zeros((0,))
        out["action"] = np.stack(self.action) if self.action else np.zeros((0, 7))
        out["reward"] = np.asarray(self.reward, dtype=np.float64)
        out["done"] = np.asarray(self.done, dtype=bool)
        out["is_wait"] = np.asarray(self.is_wait, dtype=bool)
        out["chunk_index"] = np.asarray(self.chunk_index, dtype=np.int64)
        out["action_chunks"] = np.stack(self.chunks) if self.chunks else np.zeros((0, 0, 7))
        out["chunk_t"] = np.asarray(self.chunk_t, dtype=np.int64)
        out["policy_state"] = np.stack(self.policy_state) if self.policy_state else np.zeros((0, 8))
        return out


def _geometry_adapter():
    """Import ``benchmark.adapters.libero_panda`` from this checkout (only when geometry is captured).

    The checkout root is *appended* to ``sys.path`` so it can never shadow an installed package.
    """
    root = str(pathlib.Path(__file__).resolve().parents[2])
    if root not in sys.path:
        sys.path.append(root)
    from benchmark.adapters.libero_panda import camera, wire_pack

    return camera, wire_pack


class _GeometryRecorder:
    """Per-observation camera geometry (SL1), aligned 1:1 with ``_Recorder.obs`` (``obs_index``)."""

    def __init__(self, segmentation=False, float_depth=False):
        self.camera, self.wire_pack = _geometry_adapter()
        self.segmentation, self.float_depth = bool(segmentation), bool(float_depth)
        self.cameras = [wire for _, wire in self.camera.CAMERA_TABLE]
        self.frames, self.T_world_base, self.tables, self.static = [], [], None, None
        self.sim_state, self.sim_state_at_return = [], []
        self.sim_state_image, self.qpos_image_source, self.qpos_image_residual = [], [], []

    def observe(self, env, obs):
        frames = self.camera.capture(env, obs, segmentation=self.segmentation)
        if self.tables is None:
            self.tables = self.camera.model_tables(env)
            first = frames[self.cameras[0]]
            self.static = {"depth_near_m": first["near_m"], "depth_far_m": first["far_m"]}
            for cam in self.cameras:
                self.static["fovy_deg/" + cam] = frames[cam]["fovy_deg"]
        self.frames.append(frames)
        self.T_world_base.append(frames[self.cameras[0]]["T_world_base"])
        # sim_state: simulator state at the image substep (capture observable); at_return: env.sim when
        # step returned.
        self.sim_state.append(np.asarray(frames[self.cameras[0]]["sim_state"], dtype=np.float64))
        # SL2: the kinematic state the images show (robot_state is read from it).
        self.sim_state_image.append(np.asarray(frames[self.cameras[0]]["sim_state_image"], dtype=np.float64))
        self.qpos_image_source.append(int(frames[self.cameras[0]]["qpos_image_source"]))
        self.qpos_image_residual.append(np.asarray(frames[self.cameras[0]]["qpos_image_residual_m"], np.float64))
        self.sim_state_at_return.append(np.array(env.sim.get_state().flatten(), dtype=np.float64))

    def arrays(self):
        out = {"cameras": np.array(self.cameras, dtype=str),
               "source_cameras": np.array([source for source, _ in self.camera.CAMERA_TABLE], dtype=str),
               "stamp_mode": np.array(self.camera.STAMP_MODE),
               "obs_index": np.arange(len(self.frames), dtype=np.int64),
               "T_world_base": np.stack(self.T_world_base),
               "sim_state": np.stack(self.sim_state),
               "sim_state_at_return": np.stack(self.sim_state_at_return),
               "sim_state_image": np.stack(self.sim_state_image),
               "qpos_image_source": np.asarray(self.qpos_image_source, dtype=np.int64),
               "qpos_image_residual_m": np.stack(self.qpos_image_residual),
               "depth_scale_mm": np.array(self.wire_pack.DEPTH_SCALE_MM),
               "conventions_json": np.array(json.dumps(self.camera.conventions(), ensure_ascii=False))}
        for key, value in (self.static or {}).items():
            out[key] = np.array(value, dtype=np.float64)
        out.update(self.tables or {})
        for cam in self.cameras:
            series = [frames[cam] for frames in self.frames]
            out["depth_mm/" + cam] = np.stack([self.wire_pack.depth_to_mm(f["depth_m"]) for f in series])
            for key in ("K", "T_base_cam", "T_world_cam", "robot_state"):
                out[key + "/" + cam] = np.stack([f[key] for f in series])
            out["stamp/" + cam] = np.array([f["stamp"] for f in series], dtype=np.float64)
            if self.float_depth:
                out["depth_m/" + cam] = np.stack([f["depth_m"] for f in series])
            if self.segmentation:
                out["segmentation_geom/" + cam] = np.stack([f["segmentation_geom"] for f in series])
        return out


def run_episode(env, initial_states, episode_idx, task_description, client, args, max_steps, observer=None):
    """One episode with the official loop. Returns (summary dict, recorder, replay_images).

    ``observer(env, obs)`` (SL1 geometry) is called right after every ``rec.observe(obs)``; it reads
    the env and the observation only. ``None`` = SL0 behaviour.
    """
    from openpi_client import image_tools

    rec = _Recorder()
    latencies_ms, server_timing = [], []
    error = None
    done = False
    started = _now()

    # Reset environment, set initial states (start_episode: the same two calls, shared with the exports)
    obs = start_episode(env, initial_states, episode_idx)
    action_plan = collections.deque()
    rec.observe(obs)
    if observer is not None:
        observer(env, obs)

    # Setup
    t = 0
    replay_images = []

    while t < max_steps + args.num_steps_wait:
        try:
            # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
            # and we need to wait for them to fall
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                rec.step(LIBERO_DUMMY_ACTION, reward, done, True)
                rec.observe(obs)
                if observer is not None:
                    observer(env, obs)
                t += 1
                continue

            # Get preprocessed image
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, args.resize_size, args.resize_size))
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
            )

            # Save preprocessed image for replay video
            replay_images.append(img)

            if not action_plan:
                # Finished executing previous action chunk -- compute new chunk
                # Prepare observations dict
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }

                # Query model to get action
                call_start = time.perf_counter()
                response = client.infer(element)
                latencies_ms.append((time.perf_counter() - call_start) * 1000.0)
                server_timing.append(
                    {key: _jsonable(response[key]) for key in ("server_timing", "policy_timing") if key in response}
                )
                action_chunk = response["actions"]
                rec.chunks.append(np.array(action_chunk, dtype=np.float64, copy=True))
                rec.chunk_t.append(t)
                rec.policy_state.append(np.array(element["observation/state"], dtype=np.float64, copy=True))
                assert (
                    len(action_chunk) >= args.replan_steps
                ), "We want to replan every {} steps, but policy only predicts {} steps.".format(
                    args.replan_steps, len(action_chunk)
                )
                action_plan.extend(action_chunk[: args.replan_steps])

            action = action_plan.popleft()

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            rec.step(action, reward, done, False)
            rec.observe(obs)
            if observer is not None:
                observer(env, obs)
            if done:
                break
            t += 1

        except Exception as e:  # official example swallows the exception and ends the episode
            logging.error("Caught exception: %s", e)
            error = "{}: {}\n{}".format(type(e).__name__, e, traceback.format_exc())
            break

    if error is not None:
        status = "execution_error"
    elif done:
        status = "success"
    else:
        status = "max_steps"
    env_steps = len(rec.action)
    wait_steps = int(sum(rec.is_wait))
    summary = {
        "official_success": bool(done),
        "status": status,
        "error": error,
        "env_steps_including_wait": env_steps,
        "env_steps_excluding_wait": env_steps - wait_steps,
        "wait_steps": wait_steps,
        "loop_t_final": t,
        "max_steps": max_steps,
        "num_steps_wait": args.num_steps_wait,
        "policy_calls": len(latencies_ms),
        "policy_latency_ms": latencies_ms,
        "server_timing": server_timing,
        "replay_frames": len(replay_images),
        "started_at": started,
        "ended_at": _now(),
    }
    return summary, rec, replay_images


def _save_npz(rec, final: pathlib.Path) -> None:
    tmp = _tmp_sibling(final, ".npz")
    with open(str(tmp), "xb") as stream:
        np.savez_compressed(stream, **rec.arrays())
    _exclusive_publish(tmp, final)


def _save_video(replay_images, final: pathlib.Path) -> None:
    import imageio

    tmp = _tmp_sibling(final, ".mp4")
    imageio.mimwrite(str(tmp), [np.asarray(x) for x in replay_images], fps=VIDEO_FPS)
    _exclusive_publish(tmp, final)


# --- Main -----------------------------------------------------------------------------------------
def _int_list(text):
    return [int(item) for item in str(text).replace(",", " ").split()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run-dir", required=True, help="existing run root allocated by common.run_artifacts")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--task-suite-name", default=DEFAULT_TASK_SUITE, choices=sorted(MAX_STEPS))
    parser.add_argument("--task-ids", type=_int_list, default=[0], help="comma/space separated, e.g. 0 or 0,1")
    parser.add_argument("--episode-ids", type=_int_list, default=[0], help="init-state indices per task")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resize-size", type=int, default=DEFAULT_RESIZE_SIZE)
    parser.add_argument("--replan-steps", type=int, default=DEFAULT_REPLAN_STEPS)
    parser.add_argument("--num-steps-wait", type=int, default=DEFAULT_NUM_STEPS_WAIT)
    add_geometry_arguments(parser)
    return parser


def add_geometry_arguments(parser):
    """SL1 flags (all default off; off = SL0 behaviour and outputs)."""
    parser.add_argument("--capture-geometry", action="store_true",
                        help="camera_depths=True env; per-observation depth(mm)/K/T/robot_state -> episodes/<stem>_geom.npz")
    parser.add_argument("--capture-segmentation", action="store_true",
                        help="with --capture-geometry: also render per-pixel geom ids (one extra render per camera)")
    parser.add_argument("--capture-float-depth", action="store_true",
                        help="with --capture-geometry: also store float32 metric depth (larger files)")


def check_geometry_arguments(args):
    if (args.capture_segmentation or args.capture_float_depth) and not args.capture_geometry:
        raise ValueError("--capture-segmentation/--capture-float-depth require --capture-geometry")


def require_libero_config() -> pathlib.Path:
    """Refuse before importing LIBERO, whose first import otherwise prompts via input() and creates ~/.libero."""
    config_dir = os.environ.get("LIBERO_CONFIG_PATH")
    if not config_dir or not (pathlib.Path(config_dir) / "config.yaml").is_file():
        raise RuntimeError("LIBERO_CONFIG_PATH/config.yaml missing; source scripts/setup/env.sh and run "
                           "scripts/setup/create_libero_env.sh --check")
    return pathlib.Path(config_dir) / "config.yaml"


def eval_libero(args) -> int:
    run_dir = validate_run_dir(args.run_dir)
    check_geometry_arguments(args)
    require_libero_config()
    from libero.libero import benchmark

    episodes_dir = run_dir / "episodes"
    videos_dir = run_dir / "videos"
    episodes_dir.mkdir(exist_ok=True)
    videos_dir.mkdir(exist_ok=True)
    # Exclusive: a second client invocation into the same run refuses instead of mixing rows.
    jsonl = open(str(run_dir / "episodes.jsonl"), "x")

    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info("Task suite: %s", args.task_suite_name)
    for task_id in args.task_ids:
        if not 0 <= task_id < num_tasks_in_suite:
            raise ValueError("task id {} outside suite of {} tasks".format(task_id, num_tasks_in_suite))
    max_steps = MAX_STEPS[args.task_suite_name]

    client, connector = make_policy_client(args.host, args.port)
    logging.info("Policy client connector: %s", connector)

    with jsonl:
        errors = run_tasks(args, run_dir, jsonl, task_suite, max_steps, lambda task_id, episode_idx: (client, connector, {}))
    return 1 if errors else 0


def run_tasks(args, run_dir, jsonl, task_suite, max_steps, policy_for, prepare_episode=None) -> int:
    """Task/episode loop of the official example. ``policy_for(task_id, episode_idx)`` returns
    ``(policy, connector name, extra jsonl fields)``. Returns the number of execution errors.

    ``prepare_episode(env, initial_states, task_id, episode_idx)`` (replay only, SL4b) runs right before an
    episode on the task's env -- the replay uses it to repeat the source run's resets of episodes it does not
    replay. ``None`` (the client) = the official loop unchanged."""
    episodes_dir = run_dir / "episodes"
    videos_dir = run_dir / "videos"
    total_episodes, total_successes, errors = 0, 0, 0
    for task_id in args.task_ids:
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = make_task_env(task, args.seed, args.capture_geometry)

        for episode_idx in args.episode_ids:
            if not 0 <= episode_idx < len(initial_states):
                raise ValueError("episode id {} outside {} init states".format(episode_idx, len(initial_states)))
            if prepare_episode is not None:
                prepare_episode(env, initial_states, task_id, episode_idx)
            client, connector, extra = policy_for(task_id, episode_idx)
            geometry = (_GeometryRecorder(args.capture_segmentation, args.capture_float_depth)
                        if args.capture_geometry else None)
            logging.info("\nTask: %s", task_description)
            summary, rec, replay_images = run_episode(
                env, initial_states, episode_idx, task_description, client, args, max_steps,
                observer=None if geometry is None else geometry.observe,
            )
            stem = episode_stem(args.task_suite_name, task_id, episode_idx)
            suffix = "success" if summary["official_success"] else "failure"
            npz_path = episodes_dir / (stem + ".npz")
            video_path = videos_dir / "{}_{}.mp4".format(stem, suffix)
            _save_npz(rec, npz_path)
            _save_video(replay_images, video_path)
            row = {
                "suite": args.task_suite_name,
                "task_id": task_id,
                "task_description": str(task_description),
                "episode_idx": episode_idx,
                "init_state_index": episode_idx,
                "seed": args.seed,
            }
            row.update(summary)
            row.update({
                "npz": str(npz_path.relative_to(run_dir)),
                "video": str(video_path.relative_to(run_dir)),
                "server_metadata": _jsonable(client.get_server_metadata()),
                "client_connector": connector,
            })
            if geometry is not None:
                geom_path = episodes_dir / (stem + "_geom.npz")
                _save_npz(geometry, geom_path)
                row["geometry"] = {
                    "npz": str(geom_path.relative_to(run_dir)),
                    "frames": len(geometry.frames),
                    "cameras": [{"robosuite": source, "wire": wire} for source, wire in geometry.camera.CAMERA_TABLE],
                    "segmentation": geometry.segmentation,
                    "float_depth": geometry.float_depth,
                    "env_camera_depths": True,
                }
            row.update(extra)
            jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            jsonl.flush()
            os.fsync(jsonl.fileno())

            total_episodes += 1
            total_successes += int(summary["official_success"])
            errors += int(summary["status"] == "execution_error")
            logging.info("Success: %s (status=%s)", summary["official_success"], summary["status"])
            logging.info("# episodes completed so far: %d", total_episodes)
            logging.info("# successes: %d", total_successes)
    return errors


def _jsonable(value):
    try:
        return json.loads(json.dumps(value, default=str))
    except (TypeError, ValueError):
        return str(value)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    validate_run_dir(args.run_dir)
    return eval_libero(args)


if __name__ == "__main__":
    raise SystemExit(main())

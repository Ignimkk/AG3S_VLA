"""T48a — readers for the dataset val split, the train statistics and the ep1807 recorded run.

CPU only: `pyarrow` for parquet (available in both `.venv-ag3s` and `.venv-openpi-live`), `numpy`
for the recorded run. Video decoding (`av`, `.venv-openpi-live` only) is imported lazily by
`decode_video`, which only `--stage extract` calls.

Data split = SUBTASK-b's (copies under `experiments/legacy/outputs/verify/SUBTASK-b/`):

* val units: episodes `range(1600, 1800, 4)` (`extract.py:177, 181, 260`) — 50 episodes;
* frames: `frame_index % 6 == 0` (`extract.py:198`, `--frame-stride 6` default `:246`) and
  `frame_index < n_video` (`extract.py:200`); label `phase_to_class(phase_index)`, rows with
  label < 0 dropped (`probe.py:33-39, 122`) -> 2601 frames (pick/place/home 1000/800/801);
* prompt: `meta/tasks.jsonl[task_index of row 0]` (`extract.py:178, 194`);
* simple-rule history: every parquet frame up to t (`extract.py:206`);
* train (frame-index baseline + gripper representative values): `range(0, 1600, 8)` (`extract.py:
  177, 260`), same frame stride -> 10401 frames (`kv_L4_v1.json` train.n_frames);
* recorded run ep1807: every `step_*.npz` (`extract.py:217-238`), prompt from `meta.json`, GT
  `closed_loop_gt` (`extract.py:63-70`), history = previous records (`extract.py:228-232`).
"""

from __future__ import annotations

import dataclasses
import glob
import hashlib
import json
import pathlib
from typing import Iterable, Optional

import numpy as np

from AG3S.ag3s.experiments.studies import t48_hsr_core as core

CAMS = ("cam_high", "cam_left_wrist", "cam_right_wrist")
DATASET_REL = "datasets/rby1/rby1_randomized_pick_place_16d_v1"
CKPT_REL = ("checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/"
            "rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999")
CONFIG_NAME = "pi05_rby1_randomized_pick_place_16d_lora"
EP1807_REL = "results/workspace/live_test/20260925_ep1807/run_0000"
SUBTASK_B_FEAT_REL = "results/workspace/verify/SUBTASK-b/feat"
VAL_EPISODES = tuple(range(1600, 1800, 4))
TRAIN_EPISODES = tuple(range(0, 1600, 8))
FRAME_STRIDE = 6
#: Episode of the ep1807 run (`meta.json` argv `--episode-index 1807`).
EP1807_EPISODE = 1807


@dataclasses.dataclass(frozen=True)
class Paths:
    dataset: pathlib.Path
    checkpoint: pathlib.Path
    ep1807: pathlib.Path
    subtask_b_feat: pathlib.Path

    @classmethod
    def from_storage(cls, storage: Optional[str | pathlib.Path] = None, **override) -> "Paths":
        if storage is None:
            from common.workspace import shared_storage_root

            storage = shared_storage_root()
        root = pathlib.Path(storage)
        p = dict(dataset=root / DATASET_REL, checkpoint=root / CKPT_REL, ep1807=root / EP1807_REL,
                 subtask_b_feat=root / SUBTASK_B_FEAT_REL)
        p.update({k: pathlib.Path(v) for k, v in override.items() if v is not None})
        return cls(**p)


# ----------------------------------------------------------------------------- metadata


def read_jsonl(path: pathlib.Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def task_prompts(dataset: pathlib.Path) -> dict[int, str]:
    return {int(r["task_index"]): str(r["task"]) for r in read_jsonl(dataset / "meta/tasks.jsonl")}


def episode_meta(dataset: pathlib.Path) -> dict[int, dict]:
    return {int(r["episode_index"]): r for r in read_jsonl(dataset / "meta/randomized_episodes.jsonl")}


def parquet_path(dataset: pathlib.Path, ep: int) -> pathlib.Path:
    return dataset / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"


def video_path(dataset: pathlib.Path, ep: int, cam: str) -> pathlib.Path:
    return dataset / f"videos/chunk-{ep // 1000:03d}/observation.images.{cam}/episode_{ep:06d}.mp4"


def read_episode(dataset: pathlib.Path, ep: int) -> dict[str, np.ndarray]:
    import pyarrow.parquet as pq

    t = pq.read_table(parquet_path(dataset, ep),
                      columns=["observation.state", "frame_index", "phase_index", "task_index"])
    states = np.asarray(t.column("observation.state").to_pylist(), np.float64)  # float32 -> f64
    return {"state": states, "frame_index": t.column("frame_index").to_numpy().astype(np.int64),
            "phase_index": t.column("phase_index").to_numpy().astype(np.int64),
            "task_index": t.column("task_index").to_numpy().astype(np.int64)}


def stable_key(unit_id: str) -> int:
    """Seed key for the pair RNG — independent of unit order and of Python's hash salt."""
    return int.from_bytes(hashlib.sha256(unit_id.encode()).digest()[:4], "little")


def distractor_displacement(meta: dict) -> dict[str, float]:
    """Initial -> final xyz displacement (m) of each non-target fruit (`randomized_episodes.jsonl`)."""
    ini, fin = meta["initial_fruit_poses"], meta["final_fruit_poses"]
    return {f: float(np.linalg.norm(np.asarray(fin[f][:3]) - np.asarray(ini[f][:3])))
            for f in meta["non_target_fruits"]}


# ----------------------------------------------------------------------------- units


def dataset_unit(dataset: pathlib.Path, ep: int, tasks: dict[int, str], meta: dict[int, dict], *,
                 n_video: Optional[int] = None, frame_stride: int = FRAME_STRIDE,
                 max_frames: Optional[int] = None) -> core.Unit:
    """One val episode as a `Unit`. `n_video` (when known) applies SUBTASK-b's video-length skip."""
    d = read_episode(dataset, ep)
    m = meta[ep]
    prompt = tasks[int(d["task_index"][0])]
    if prompt != m["prompt"]:
        raise ValueError(f"ep{ep}: tasks.jsonl prompt {prompt!r} != randomized_episodes {m['prompt']!r}")
    fi = d["frame_index"]
    keep = (fi % frame_stride == 0)
    if n_video is not None:
        keep &= fi < int(n_video)
    rows = np.flatnonzero(keep)
    y = core.phase_to_class(d["phase_index"][rows])
    rows, y = rows[y >= 0], y[y >= 0]
    if max_frames is not None:
        rows, y = rows[:max_frames], y[:max_frames]
    uid = f"ep_{ep:06d}"
    return core.Unit(
        unit_id=uid, set_name="val", prompt=prompt, target=str(m["target_fruit"]),
        distractors=tuple(m["non_target_fruits"]), gripper_index=core.GRIPPER_INDEX[m["used_arm"]],
        seq_states=d["state"], rows=rows, clock=fi[rows], y=y, proxy_cadence=8,
        seed_key=stable_key(uid),
        extra={"episode": ep, "phase_index": d["phase_index"][rows], "n_parquet": int(len(fi)),
               "task_type": m["task_type"], "grasp_retries": int(m["grasp_retries"]),
               "distractor_displacement_m": distractor_displacement(m)},
    )


def ep1807_records(run: pathlib.Path, *, with_images: bool = False) -> dict:
    files = sorted(glob.glob(str(run / "step_*.npz")))
    if not files:
        raise FileNotFoundError(f"no step_*.npz under {run}")
    out = {"t_step": [], "state": [], "qpos": [], "files": files}
    if with_images:
        out["images"] = {c: [] for c in CAMS}
    for f in files:
        with np.load(f) as z:
            out["t_step"].append(int(z["t_step"]))
            out["state"].append(np.asarray(z["state"], np.float64))
            out["qpos"].append(np.asarray(z["qpos"], np.float64))
            if with_images:
                for c in CAMS:
                    out["images"][c].append(np.asarray(z[f"image_{c}"]))
    out["t_step"] = np.asarray(out["t_step"], np.int64)
    out["state"] = np.stack(out["state"])
    out["qpos"] = np.stack(out["qpos"])
    if with_images:
        out["images"] = {c: np.stack(v) for c, v in out["images"].items()}
    return out


def ep1807_unit(run: pathlib.Path, meta: dict[int, dict], *, max_frames: Optional[int] = None) -> core.Unit:
    """The recorded held-out run. Distractors from `randomized_episodes.jsonl` episode 1807."""
    rmeta = json.loads((run / "meta.json").read_text())
    prompt = str(rmeta["prompt"])
    m = meta[EP1807_EPISODE]
    _, fruit = core.prompt_template(prompt)
    if fruit != m["target_fruit"]:
        raise ValueError(f"ep1807 run prompt fruit {fruit!r} != episode metadata {m['target_fruit']!r}")
    rec = ep1807_records(run)
    gt = [core.closed_loop_gt(q, s, fruit) for q, s in zip(rec["qpos"], rec["state"])]
    y = np.asarray([core.GT_MAP[g] for g in gt], np.int64)
    rows = np.arange(len(y))
    if max_frames is not None:
        rows, y = rows[:max_frames], y[:max_frames]
    q = rec["qpos"]
    disp = {f: float(np.linalg.norm(q[:, core.FRUIT_QPOS[f]:core.FRUIT_QPOS[f] + 3]
                                    - q[0, core.FRUIT_QPOS[f]:core.FRUIT_QPOS[f] + 3], axis=1).max())
            for f in m["non_target_fruits"]}
    return core.Unit(
        unit_id="ep1807_run0000", set_name="ep1807", prompt=prompt, target=fruit,
        distractors=tuple(m["non_target_fruits"]), gripper_index=core.GRIPPER_INDEX[m["used_arm"]],
        seq_states=rec["state"], rows=rows, clock=rec["t_step"][rows], y=y, proxy_cadence=1,
        seed_key=stable_key("ep1807_run0000"),
        extra={"episode": EP1807_EPISODE, "gt": np.asarray(gt)[rows], "run": str(run),
               "ctrl_hz": rmeta.get("ctrl_hz"), "open_loop_horizon": rmeta.get("open_loop_horizon"),
               "distractor_max_displacement_m": disp},
    )


# ----------------------------------------------------------------------------- train statistics


def train_stats(dataset: pathlib.Path, meta: dict[int, dict], *,
                episodes: Iterable[int] = TRAIN_EPISODES, frame_stride: int = FRAME_STRIDE) -> dict:
    """Frame-index majority (SUBTASK-b `controls.py:35-37`) and the used-arm gripper representative
    values, both from the **train** split (no evaluation frame sets a parameter).

    * open  = median used-arm gripper over sampled train frames with value >= 0.9;
    * closed = median over sampled train frames with value < 0.9 in the place phases 4-8 (holding).
    """
    clock, ys, g_open, g_closed = [], [], [], []
    by_fruit: dict[str, list] = {}
    n_ep = 0
    for ep in episodes:
        d = read_episode(dataset, ep)
        fi = d["frame_index"]
        m = fi % frame_stride == 0
        y = core.phase_to_class(d["phase_index"][m])
        clock.append(fi[m])
        ys.append(y)
        g = d["state"][m, core.GRIPPER_INDEX[meta[ep]["used_arm"]]]
        ph = d["phase_index"][m]
        g_open.append(g[g >= core.RULE_CLOSED_BELOW])
        hold = (g < core.RULE_CLOSED_BELOW) & (ph >= 4) & (ph <= 8)
        g_closed.append(g[hold])
        by_fruit.setdefault(meta[ep]["target_fruit"], []).append(g[hold])
        n_ep += 1
    clock, ys = np.concatenate(clock), np.concatenate(ys)
    g_open, g_closed = np.concatenate(g_open), np.concatenate(g_closed)
    maj = core.frame_index_majority(clock, ys)
    return {
        "episodes": n_ep, "frames": int(len(ys)), "frames_used": int((ys >= 0).sum()),
        "per_class": {c: int((ys == i).sum()) for i, c in enumerate(core.CLASSES)},
        "frame_index_majority": {str(k): v for k, v in sorted(maj.items())},
        "gripper_open_rep": float(np.median(g_open)),
        "gripper_closed_rep": float(np.median(g_closed)),
        "gripper_open_q": [float(x) for x in np.percentile(g_open, [1, 5, 50, 95, 99])],
        "gripper_closed_q": [float(x) for x in np.percentile(g_closed, [1, 5, 50, 95, 99])],
        "gripper_closed_median_by_fruit": {f: float(np.median(np.concatenate(v)))
                                            for f, v in sorted(by_fruit.items())},
        "n_open": int(len(g_open)), "n_closed": int(len(g_closed)),
        "definition": "train episodes range(0,1600,8), frame_index%6==0; open = median used-arm "
                      "gripper >= 0.9; closed = median used-arm gripper < 0.9 in phases 4-8",
    }


def majority_from_stats(stats: dict) -> dict[int, int]:
    return {int(k): int(v) for k, v in stats["frame_index_majority"].items()}


def reps_from_stats(stats: dict) -> core.GripperReps:
    return core.GripperReps(open=float(stats["gripper_open_rep"]), closed=float(stats["gripper_closed_rep"]))


def norm_stats_state(checkpoint: pathlib.Path) -> dict[str, np.ndarray]:
    """`state` q01/q99/mean/std of the checkpoint's norm stats (what `create_trained_policy` loads,
    `policy_config.py:64`)."""
    p = checkpoint / "assets/local/rby1_randomized_pick_place_16d_v1/norm_stats.json"
    st = json.loads(p.read_text())["norm_stats"]["state"]
    return {k: np.asarray(v, np.float64) for k, v in st.items()}


# ----------------------------------------------------------------------------- video (extract only)


def decode_video(path: pathlib.Path) -> list[np.ndarray]:
    """All frames as HWC uint8 RGB — SUBTASK-b `extract.py:156-158`."""
    import av

    with av.open(str(path)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]

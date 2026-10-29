"""T48b — HSR counterfactual (b): empty grasp / slip / phantom hold on re-rendered 16D records.

Does the phase probe (`kv_L4_v1`) read the *grasp state* in the image or the *arm pose*? Same
arm pose, only the target fruit (and the used hand's fingers) changed in qpos, re-rendered with the
policy-image convention, fed through the frozen prefix:

    E0rec  recorded RGB + recorded state            (reference)
    E0     re-rendered RGB + reconstructed state    parity: same prediction as E0rec?
    E1     held frame: fruit -> initial table pose, fingers -> recorded empty closure   gt pick
    E1s    held frame: fruit -> initial table pose, fingers / state untouched           gt pick
    E1z    held frame: as E1 with fingers at the joint limit (state 0) — addition       gt pick
    E2     held frame: fruit -> on the table straight below, fingers as E1              gt pick
    E3     pre-grasp approach frame: fruit -> between the fingers, fingers at the
           fruit's hold width (dataset train median)                                    gt place

    source scripts/setup/env.sh
    export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"      # once
    M=AG3S.ag3s.experiments.studies.t48b_hsr
    # CPU (MuJoCo osmesa, 1 thread): physical predicates, parity re-render, edits -> render cache
    MUJOCO_GL=osmesa OMP_NUM_THREADS=1 .venv-ag3s/bin/python -m $M --stage render
    # GPU (after CL3): prefix-only kv_L4; parity pass first, edits only when parity passes
    flock /tmp/ag3s-vla-gpu-validation.lock .venv-openpi-live/bin/python -m $M --stage extract --batch 4
    # CPU: probe + baselines + metrics
    .venv-ag3s/bin/python -m $M --stage score

Output (refuses without ``AG3S_RUN_DIR``) under ``$AG3S_RUN_DIR/verify/T48b/``:

``plan/``     summary.json (runs, membership, predicates vs closed_loop_gt, hosts, edit parameters),
              train_stats.json (cached, T48's definition)
``render/``   <run>__E0.npz, <run>__edits.npz (+ .json sidecars): images, states, poses, parity,
              exclusions; resumable by digest; render_summary.json
``feat/``     <run>__E0.npz, <run>__edits.npz: kv_L4 features + the render digest they belong to
``extract/``  timing.json, extract_meta.json, parity_gate.json
``score/``    parity.json, metrics.json, predictions.csv

Scene / predicates / edits: `t48b_scene`. Metrics: `t48b_metrics`. Reused: `t48_hsr_core`
(closed_loop_gt, baselines, episode summary), `t48_hsr_io` (dataset metadata, train statistics),
`t48_hsr_features` (GPU prefix path). Task: `AG3S/ag3s/docs/handoff/T48b.task.md`.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.studies import t48_hsr_core as core
from AG3S.ag3s.experiments.studies import t48_hsr_io as io
from AG3S.ag3s.experiments.studies import t48b_metrics as mx

STEP = "T48b"
STAGES = ("render", "extract", "score")
METHODS = ("probe", "rule", "proxy", "frame_index")
#: Bump when the render plan / cache layout changes (part of every digest).
CACHE_VERSION = "t48b-render-v1"
#: Pixel gate of the parity step: largest per-frame, per-camera MAD (gray levels) the re-render
#: may show before edits are rendered. The re-render is expected bit-identical (`t48b_scene`
#: docstring); 1.0 leaves room for a different GL / PIL build, not for a wrong convention.
PIXEL_MAD_MAX = 1.0


@dataclasses.dataclass(frozen=True)
class RunSpec:
    key: str
    rel: str          # under $AG3S_STORAGE_ROOT
    group: str        # "main" | "train_range"


#: The 16D recorded runs of the task (§data). Paths = `t47_head_selection.RECORDS` (same rollouts).
MAIN_RUNS = (
    RunSpec("ep1807", "results/workspace/live_test/20260925_ep1807/run_0000", "main"),
    RunSpec("long", "results/workspace/live_test/20260924_long16d/run_0000", "main"),
    RunSpec("t2", "results/workspace/live_test/20260925_t2/run_0000/run_0000", "main"),
    RunSpec("headfix", "results/workspace/live_test/20260925_headfix16d/records/run_0000/run_0000", "main"),
    RunSpec("ep1800", "recordings/run_16d_ep1800", "main"),
    RunSpec("ep1850", "results/workspace/live_test/20260925_t1/ep1850/run_0000", "main"),
    RunSpec("ep1900", "results/workspace/live_test/20260925_t1/ep1900/run_0000", "main"),
)
#: Opt-in (`--extra-runs train_range`): SUBTASK-b's other closed-loop runs (`extract.py:55-59`),
#: episodes inside the VLA fine-tune range. Aggregated separately, never pooled with MAIN_RUNS.
EXTRA_RUNS = (
    RunSpec("ep0", "results/workspace/live_test/20260925_train/ep0/run_0000", "train_range"),
    RunSpec("ep500", "results/workspace/live_test/20260925_train/ep500/run_0000", "train_range"),
    RunSpec("ep1200", "results/workspace/live_test/20260925_train/ep1200/run_0000", "train_range"),
)
ALL_RUNS = {r.key: r for r in MAIN_RUNS + EXTRA_RUNS}
#: `t47_head_selection.py:67-83` RecordSpec role (attention head selection; not the probe).
T47_ROLE = {"long": "selection", "t2": "selection", "ep1807": "selection", "headfix": "heldout",
            "ep1850": "heldout", "ep1900": "heldout", "ep1800": "heldout"}
#: SUBTASK-b closed-loop evaluation runs (`extract.py:55-59`).
SUBTASK_B_CLOSED = {"ep1807", "ep0", "ep500", "ep1200"}
#: VLA fine-tune episodes: `third_party/openpi/src/openpi/training/config.py:1136`
#: ``episodes=tuple(range(1600))``. Probe train episodes: `t48_hsr_io.TRAIN_EPISODES`.
VLA_TRAIN_EPISODES = range(0, 1600)


# ----------------------------------------------------------------------------- plumbing


def out_root() -> pathlib.Path:
    """``$AG3S_RUN_DIR/verify/T48b`` — refuses when the run dir is unset (no default location)."""
    from common.run_output import resolve_output_path

    run_dir = os.environ.get("AG3S_RUN_DIR")
    if not run_dir:
        raise SystemExit(
            "AG3S_RUN_DIR is not set. Allocate a run first:\n"
            '  export AG3S_RUN_DIR="$(.venv-openpi-live/bin/python -m common.run_output)"')
    root = resolve_output_path("verify/T48b", run_dir=run_dir, default_name="verify/T48b")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, pathlib.Path):
        return str(o)
    if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
    raise TypeError(type(o))


def write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, default=_json_default))
    tmp.replace(path)


def save_npz(path: pathlib.Path, compressed: bool = False, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    (np.savez_compressed if compressed else np.savez)(tmp, **arrays)
    tmp.replace(path)


def git_state() -> dict:
    root = pathlib.Path(__file__).resolve().parents[4]

    def run(*a):
        try:
            return subprocess.check_output(["git", "-C", str(root), *a], text=True, stderr=subprocess.DEVNULL)
        except (OSError, subprocess.CalledProcessError):
            return ""

    return {"head": run("rev-parse", "HEAD").strip(), "branch": run("branch", "--show-current").strip(),
            "dirty_files": [ln for ln in run("status", "--porcelain").splitlines() if ln]}


def now_kst() -> str:
    return _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=9))).isoformat(timespec="seconds")


def code_digest() -> str:
    """sha256 of the two files that define the render plan (a cache made by other code is refused)."""
    h = hashlib.sha256()
    here = pathlib.Path(__file__).resolve().parent
    for name in ("t48b_scene.py", "t48b_hsr.py"):
        h.update((here / name).read_bytes())
    return h.hexdigest()[:20]


def digest_of(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=_json_default).encode()).hexdigest()


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=STAGES)
    ap.add_argument("--runs", nargs="+", default=None, choices=sorted(ALL_RUNS),
                    help="runs to render / extract / score (default: the seven main runs [+ --extra-runs])")
    ap.add_argument("--extra-runs", nargs="*", default=[], choices=["train_range"],
                    help="add SUBTASK-b's ep0/ep500/ep1200 (VLA train range), aggregated separately")
    ap.add_argument("--families", nargs="+", default=list(mx.EDIT_FAMILIES), choices=list(mx.EDIT_FAMILIES),
                    help="edit families to render / extract")
    ap.add_argument("--max-frames", type=int, default=None, help="smoke: first N records per run only")
    ap.add_argument("--max-hosts", type=int, default=None, help="smoke: first N host records per edit family")
    ap.add_argument("--g-empty", type=float, default=None,
                    help="override the empty-closure gripper value (default: median of the recorded ones)")
    ap.add_argument("--pen-max-mm", type=float, default=None, help="override the penetration limit (mm)")
    ap.add_argument("--pixel-mad-max", type=float, default=PIXEL_MAD_MAX)
    ap.add_argument("--agreement-min", type=float, default=mx.PARITY_AGREEMENT_MIN)
    ap.add_argument("--batch", type=int, default=1, help="extract: inputs per prefix call (1 = serving path)")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--allow-busy-gpu", action="store_true")
    ap.add_argument("--parity-only", action="store_true", help="extract: E0rec / E0 only")
    ap.add_argument("--force", action="store_true", help="continue past a failed parity gate")
    ap.add_argument("--storage", default=None, help="default $AG3S_STORAGE_ROOT")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--config", default=io.CONFIG_NAME)
    ap.add_argument("--probe", default=None, help="probe asset npz (default kv_L4_v1)")
    args = ap.parse_args(argv)
    if args.batch < 1:
        ap.error("--batch must be >= 1")
    if args.warmup < 0:
        ap.error("--warmup must be >= 0")
    if args.max_frames is not None and args.max_frames < 2:
        ap.error("--max-frames must be >= 2")
    if args.max_hosts is not None and args.max_hosts < 1:
        ap.error("--max-hosts must be >= 1")
    if args.g_empty is not None and not 0.0 <= args.g_empty < 0.9:
        ap.error("--g-empty must be in [0, 0.9) (a closed hand)")
    if args.pen_max_mm is not None and args.pen_max_mm <= 0:
        ap.error("--pen-max-mm must be > 0")
    if args.parity_only and args.stage != "extract":
        ap.error("--parity-only belongs to --stage extract")
    return args


def selected_runs(args) -> list[RunSpec]:
    keys = list(args.runs) if args.runs else [r.key for r in MAIN_RUNS]
    if "train_range" in (args.extra_runs or []):
        keys += [r.key for r in EXTRA_RUNS if r.key not in keys]
    return [ALL_RUNS[k] for k in keys]


def run_groups(specs: Sequence[RunSpec]) -> dict[str, list[str]]:
    main = [s.key for s in specs if s.group == "main"]
    g: dict[str, list[str]] = {"main": main, "ep1807": [k for k in main if k == "ep1807"],
                               "main_minus_ep1807": [k for k in main if k != "ep1807"],
                               "main_scene_ep1800": [k for k in main if k in ("long", "t2", "headfix", "ep1800")]}
    tr = [s.key for s in specs if s.group == "train_range"]
    if tr:
        g["train_range"] = tr
    return {k: v for k, v in g.items() if v}


def storage_root(args) -> pathlib.Path:
    if args.storage:
        return pathlib.Path(args.storage)
    from common.workspace import shared_storage_root

    return pathlib.Path(shared_storage_root())


# ----------------------------------------------------------------------------- records


@dataclasses.dataclass
class RunData:
    spec: RunSpec
    path: pathlib.Path
    meta: dict
    files: list[str]
    t_step: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    state: np.ndarray
    episode: int
    prompt: str
    fruit: str
    hand: str
    distractors: tuple
    fingerprint: str


def episode_of(meta: dict) -> int:
    argv = list(meta.get("argv") or [])
    if "--episode-index" not in argv:
        raise ValueError("record meta.json argv has no --episode-index")
    return int(argv[argv.index("--episode-index") + 1])


def load_run_data(spec: RunSpec, root: pathlib.Path, ep_meta: dict, *, max_frames: Optional[int] = None) -> RunData:
    path = root / spec.rel
    meta = json.loads((path / "meta.json").read_text())
    files = sorted(str(p) for p in path.glob("step_*.npz"))
    if max_frames:
        files = files[:max_frames]
    if not files:
        raise FileNotFoundError(f"no step_*.npz under {path}")
    t, q, v, s = [], [], [], []
    for f in files:
        with np.load(f) as z:
            t.append(int(z["t_step"]))
            q.append(np.asarray(z["qpos"], np.float64))
            v.append(np.asarray(z["qvel"], np.float64))
            s.append(np.asarray(z["state"], np.float64))
    ep = episode_of(meta)
    em = ep_meta[ep]
    _, fruit = core.prompt_template(str(meta["prompt"]))
    if fruit != em["target_fruit"]:
        raise ValueError(f"{spec.key}: prompt fruit {fruit!r} != episode {ep} target {em['target_fruit']!r}")
    q = np.stack(q)
    h = hashlib.sha256(json.dumps(meta, sort_keys=True, default=str).encode())
    h.update(np.asarray(t, np.int64).tobytes())
    h.update(q.tobytes())
    return RunData(spec=spec, path=path, meta=meta, files=files, t_step=np.asarray(t, np.int64), qpos=q,
                   qvel=np.stack(v), state=np.stack(s), episode=ep, prompt=str(meta["prompt"]), fruit=fruit,
                   hand=str(em["used_arm"]), distractors=tuple(em["non_target_fruits"]),
                   fingerprint=h.hexdigest()[:20])


def record_images(rd: RunData, rows: Sequence[int]) -> dict[int, dict[str, np.ndarray]]:
    from AG3S.ag3s.experiments.studies.t48b_scene import CAMS

    out = {}
    for r in sorted(set(int(x) for x in rows)):
        with np.load(rd.files[r]) as z:
            out[r] = {c: np.asarray(z[f"image_{c}"]) for c in CAMS}
    return out


def membership(rd: RunData, splits: dict, episodes: dict[str, int]) -> dict:
    """Which training / selection sets the run's episode belongs to (task §data 5)."""
    ep = rd.episode
    split = next((n for n, sp in splits.items()
                  if sp["start_episode_index"] <= ep < sp["end_episode_index_exclusive"]), None)
    return {"episode": ep, "dataset_split": split,
            "vla_finetune_episode": ep in VLA_TRAIN_EPISODES,
            "probe_train_episode": ep in io.TRAIN_EPISODES,
            "subtask_b_closed_loop_eval": rd.spec.key in SUBTASK_B_CLOSED,
            "t47_head_selection_role": T47_ROLE.get(rd.spec.key),
            "same_scene_runs": sorted(k for k, e in episodes.items() if e == ep and k != rd.spec.key)}


def train_stats_cached(root: pathlib.Path, dataset: pathlib.Path, ep_meta: dict) -> dict:
    """T48's train statistics (frame-index majority, gripper hold values per fruit), cached."""
    p = root / "plan" / "train_stats.json"
    if p.exists():
        st = json.loads(p.read_text())
        if st.get("dataset") == str(dataset):
            return st
    st = io.train_stats(dataset, ep_meta)
    st["dataset"] = str(dataset)
    write_json(p, st)
    return st


# ----------------------------------------------------------------------------- render (CPU)


@dataclasses.dataclass
class Analysis:
    """Pass-1 result of one run: render poses, physical states, GT, templates."""

    rd: RunData
    q_ren: np.ndarray
    phys: list
    gt_cl: np.ndarray            # closed_loop_gt class (T48 / SUBTASK-b definition)
    gt_phys: np.ndarray          # physical-state class, -1 ambiguous
    state_recon_err: float
    held: np.ndarray
    reach: np.ndarray
    empty_values: np.ndarray
    hand_pose: np.ndarray        # (N, 7) wrist pose of the used hand (render pose)


def analyse_run(rd: RunData, model, idx, data) -> Analysis:
    import mujoco

    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    n = len(rd.files)
    q_ren = np.stack([sc.render_qpos(model, rd.qpos[i], rd.qvel[i],
                                     after_step=sc.observation_follows_step(rd.meta, i)) for i in range(n)])
    phys, hand_pose = [], np.zeros((n, 7))
    hb = idx.hand_body[rd.hand]
    for i in range(n):
        data.qpos[:] = q_ren[i]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        phys.append(sc.classify(model, data, idx, rd.fruit))
        hand_pose[i] = np.concatenate([data.xpos[hb], data.xquat[hb]])
    err = max(float(np.abs(sc.reconstruct_state(idx, rd.qpos[i]) - rd.state[i]).max()) for i in range(n))
    gt_cl = np.asarray([core.GT_MAP[core.closed_loop_gt(rd.qpos[i], rd.state[i], rd.fruit)] for i in range(n)])
    gt_phys = np.asarray([sc.PHYS_CLASS[p.state] for p in phys], np.int64)
    g = rd.state[:, sc.GRIPPER_INDEX[rd.hand]]
    return Analysis(rd=rd, q_ren=q_ren, phys=phys, gt_cl=gt_cl, gt_phys=gt_phys, state_recon_err=err,
                    held=sc.held_rows(phys, rd.hand), reach=sc.reach_rows(phys, g),
                    empty_values=sc.empty_closure_values(phys, g, rd.hand), hand_pose=hand_pose)


def choose_template(an: Analysis, pool: Sequence[Analysis], idx, g_hold: float, pen_max: float) -> tuple:
    """E3 start pose: fruit pose in the wrist frame from a clean `held` record (penetration <= limit)
    of the same fruit and hand — the same run first, else the main runs in `pool` order; within a
    run the record whose gripper is closest to `g_hold` (ties: earliest). `(rel7, "run:row")` or
    `(None, "")` (then `place_in_hand` centres on the finger pads)."""
    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    for cand in [an] + [p for p in pool if p is not an]:
        rd = cand.rd
        if rd.fruit != an.rd.fruit or rd.hand != an.rd.hand:
            continue
        rows = [int(r) for r in cand.held if cand.phys[int(r)].penetration <= pen_max]
        if not rows:
            continue
        gi = sc.GRIPPER_INDEX[rd.hand]
        r = min(rows, key=lambda k: (abs(rd.state[k, gi] - g_hold), k))
        fruit_pose = sc.free_pose(cand.q_ren[r], idx, rd.fruit)
        rel = sc.relative_pose(cand.hand_pose[r, :3], cand.hand_pose[r, 3:], fruit_pose)
        return rel, f"{rd.spec.key}:{r}"
    return None, ""


def phys_vs_closed_loop(an: Analysis) -> dict:
    """Counts of (physical state, closed_loop_gt) per run — where the two GT definitions differ."""
    out: dict[str, int] = {}
    inv = {v: k for k, v in core.GT_MAP.items()}
    for p, g in zip(an.phys, an.gt_cl):
        k = f"{p.state}|{inv[int(g)]}"
        out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def stage_render(args, root: pathlib.Path) -> int:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("LP_NUM_THREADS", "1")   # llvmpipe (osmesa) render threads: one CPU worker
    import mujoco

    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    t_all = time.time()
    sroot = storage_root(args)
    paths = io.Paths.from_storage(sroot, dataset=args.dataset)
    ep_meta = io.episode_meta(paths.dataset)
    splits = json.loads((paths.dataset / "meta/randomized_splits.json").read_text())["splits"]
    stats = train_stats_cached(root, paths.dataset, ep_meta)
    pen_max = sc.PENETRATION_MAX_M if args.pen_max_mm is None else args.pen_max_mm * 1e-3
    specs = selected_runs(args)
    sel = {s.key for s in specs}
    # Pass 1 always covers every main run (edit parameters must not depend on --runs) + extras.
    analyse_specs = list(MAIN_RUNS) + [s for s in specs if s.group != "main"]
    models: dict[str, tuple] = {}
    analyses: dict[str, Analysis] = {}
    for spec in analyse_specs:
        rd = load_run_data(spec, sroot, ep_meta, max_frames=args.max_frames)
        mkey = str(rd.meta["model_xml"])
        if mkey not in models:
            model, mpath = sc.load_record_model(rd.meta)
            models[mkey] = (model, sc.SceneIndex.from_model(model), mujoco.MjData(model), mpath)
        model, idx, data, _ = models[mkey]
        analyses[spec.key] = analyse_run(rd, model, idx, data)
        print(f"[render/pass1] {spec.key}: N={len(rd.files)} held={len(analyses[spec.key].held)} "
              f"reach={len(analyses[spec.key].reach)} empty_closure={len(analyses[spec.key].empty_values)} "
              f"state_recon_err={analyses[spec.key].state_recon_err:.3g}", flush=True)

    # ---- edit parameters
    main_an = [analyses[s.key] for s in MAIN_RUNS]
    empty = np.concatenate([a.empty_values for a in main_an])
    if args.g_empty is not None:
        g_empty, g_src = float(args.g_empty), "--g-empty"
    elif empty.size:
        g_empty, g_src = float(np.median(empty)), "median of recorded empty closures (main runs)"
    else:
        raise SystemExit("no recorded empty-closure frame in the main runs; pass --g-empty")
    g_hold = {f: float(v) for f, v in stats["gripper_closed_median_by_fruit"].items()}
    params = {
        "g_empty": g_empty, "g_empty_source": g_src,
        "g_empty_recorded": {"n": int(empty.size),
                             "quantiles_p0_p5_p50_p95_p100": [float(x) for x in np.percentile(empty, [0, 5, 50, 95, 100])]
                             if empty.size else None,
                             "per_run": {a.rd.spec.key: int(a.empty_values.size) for a in main_an}},
        "g_limit": 0.0, "g_limit_basis": "gripper_finger_x1 joint range upper limit 0 -> state 0 (fully shut)",
        "g_hold_by_fruit": g_hold,
        "g_hold_basis": "t48_hsr_io.train_stats gripper_closed_median_by_fruit (train eps range(0,1600,8), "
                        "used gripper < 0.9 in phases 4-8); E3 closes the fingers onto the placed fruit within "
                        "[g_hold - 0.15, g_hold + 0.10] (t48b_scene.place_in_hand) and feeds the fitted value",
        "penetration_max_m": pen_max, "search_mm": list(sc.EditContext.__dataclass_fields__["search_mm"].default),
        "open_at_least": sc.OPEN_AT_LEAST, "settle_eps": sc.SETTLE_EPS,
    }
    any_model = next(iter(models.values()))
    params["free_qpos_from_model"] = dict(any_model[1].free_q)
    params["crate_inner_volume"] = list(any_model[1].crate_volume)

    summary: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "storage": str(sroot),
                     "models": {k: str(v[3]) for k, v in models.items()}, "params": params, "runs": {},
                     "groups": run_groups(specs)}
    for key, an in analyses.items():
        rd = an.rd
        summary["runs"][key] = {
            "path": str(rd.path), "group": rd.spec.group, "selected": key in sel, "n_records": len(rd.files),
            "prompt": rd.prompt, "fruit": rd.fruit, "hand": rd.hand,
            "membership": membership(rd, splits, {k: a.rd.episode for k, a in analyses.items()}),
            "state_recon_max_abs": an.state_recon_err,
            "held_rows": an.held.tolist(), "reach_rows": an.reach.tolist(),
            "phys_vs_closed_loop_gt": phys_vs_closed_loop(an),
            "phys_states": [p.state for p in an.phys],
            "recorded_penetration_max_m": float(max(p.penetration for p in an.phys)),
            "empty_closure_values": an.empty_values.tolist(),
        }
    write_json(root / "plan" / "summary.json", summary)

    # ---- pass 2a: E0 parity re-render
    rdir = root / "render"
    rdir.mkdir(parents=True, exist_ok=True)
    renderers: dict[str, Any] = {}
    render_summary: dict = {"created": now_kst(), "runs": {}}
    worst_mad = 0.0
    for spec in specs:
        an = analyses[spec.key]
        model, idx, data, _ = models[str(an.rd.meta["model_xml"])]
        ren = renderers.setdefault(str(an.rd.meta["model_xml"]), sc.PolicyRenderer(model))
        info = render_e0(an, ren, idx, rdir)
        render_summary["runs"].setdefault(spec.key, {})["E0"] = info
        worst_mad = max(worst_mad, info["mad_max"])
        print(f"[render/E0] {spec.key}: {info['n']} frames, MAD max {info['mad_max']:.4f}, "
              f"max abs {info['max_abs']}, {info['render_s_median']:.3f} s/frame", flush=True)
    gate = {"pixel_mad_max_allowed": args.pixel_mad_max, "worst_frame_camera_mad": worst_mad,
            "passed": bool(worst_mad <= args.pixel_mad_max)}
    render_summary["pixel_gate"] = gate
    write_json(rdir / "render_summary.json", render_summary)
    if not gate["passed"] and not args.force:
        print(f"[render] pixel parity FAILED (worst MAD {worst_mad:.3f} > {args.pixel_mad_max}); "
              "edits not rendered (T48b.task.md: stop and report). --force to continue.", file=sys.stderr)
        return 3

    # ---- pass 2b: edits
    for spec in specs:
        an = analyses[spec.key]
        model, idx, data, _ = models[str(an.rd.meta["model_xml"])]
        ren = renderers[str(an.rd.meta["model_xml"])]
        rel, src = choose_template(an, main_an, idx, g_hold[an.rd.fruit], pen_max)
        ctx = sc.EditContext(idx=idx, fruit=an.rd.fruit, hand=an.rd.hand,
                             initial_pose=sc.free_pose(an.q_ren[0], idx, an.rd.fruit),
                             g_empty=g_empty, g_hold=g_hold[an.rd.fruit], template_rel=rel, template_src=src,
                             pen_max=pen_max)
        info = render_edits(an, ctx, model, data, ren, rdir, args.families, args.max_hosts)
        render_summary["runs"][spec.key]["edits"] = info
        print(f"[render/edits] {spec.key}: {json.dumps(info['counts'])} template={src or 'pad_midpoint'}", flush=True)
        write_json(rdir / "render_summary.json", render_summary)
    for r in renderers.values():
        r.close()
    render_summary["wall_s"] = time.time() - t_all
    write_json(rdir / "render_summary.json", render_summary)
    print(f"[render] done in {render_summary['wall_s']:.1f} s", flush=True)
    return 0


def _cached(path: pathlib.Path, digest: str) -> bool:
    if not path.exists():
        return False
    with np.load(path, allow_pickle=False) as z:
        if str(z["digest"]) == digest:
            return True
    raise SystemExit(f"{path} exists with a different digest — inputs or code changed. "
                     "Use a new AG3S_RUN_DIR; not overwriting.")


def render_e0(an: Analysis, ren, idx, rdir: pathlib.Path) -> dict:
    """Re-render every record at its render pose; parity vs the recorded RGB and state."""
    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    rd = an.rd
    dig = digest_of({"v": CACHE_VERSION, "code": code_digest(), "kind": "E0", "run": rd.spec.key,
                     "fp": rd.fingerprint})
    path = rdir / f"{rd.spec.key}__E0.npz"
    side = rdir / f"{rd.spec.key}__E0.json"
    if _cached(path, dig) and side.exists():
        print(f"[render/E0] {rd.spec.key} cached", flush=True)
        return json.loads(side.read_text())
    n = len(rd.files)
    imgs = np.zeros((n, 3, 224, 224, 3), np.uint8)
    mad, mxa, frac, secs = np.zeros((n, 3)), np.zeros((n, 3), np.int64), np.zeros((n, 3)), np.zeros(n)
    recon = np.stack([sc.reconstruct_state(idx, rd.qpos[i]) for i in range(n)])
    rec_imgs = record_images(rd, range(n))
    for i in range(n):
        t0 = time.perf_counter()
        out = ren.render(an.q_ren[i])
        secs[i] = time.perf_counter() - t0
        par = sc.image_parity(rec_imgs[i], out)
        for k, c in enumerate(sc.CAMS):
            imgs[i, k] = out[c]
            mad[i, k], mxa[i, k], frac[i, k] = par[c]["mad"], par[c]["max"], par[c]["frac_nonzero"]
    again = ren.render(an.q_ren[0])
    repeat_identical = all(np.array_equal(again[c], imgs[0, k]) for k, c in enumerate(sc.CAMS))
    save_npz(path, compressed=True, digest=np.asarray(dig), run=np.asarray(rd.spec.key), host=np.arange(n),
             t_step=rd.t_step, image=imgs, state_rec=rd.state, state_recon=recon, q_ren=an.q_ren,
             mad=mad, max_abs=mxa, frac_nonzero=frac, render_s=secs, gt_cl=an.gt_cl, gt_phys=an.gt_phys,
             phys_state=np.asarray([p.state for p in an.phys]), files=np.asarray(rd.files),
             prompt=np.asarray(rd.prompt), cams=np.asarray(sc.CAMS))
    info = {"n": n, "digest": dig, "mad_max": float(mad.max()), "mad_median_per_cam": np.median(mad, 0).tolist(),
            "max_abs": int(mxa.max()), "frac_nonzero_max": float(frac.max()),
            "state_max_abs": float(np.abs(recon - rd.state).max()),
            "render_s_median": float(np.median(secs)), "render_s_total": float(secs.sum()),
            "repeat_render_identical": bool(repeat_identical),
            "render_pose_after_step_rows": int(sum(sc.observation_follows_step(rd.meta, i) for i in range(n))),
            "convention": "render pose = mj_integratePos(qpos, qvel, -dt) when an mj_step preceded the observation "
                          "(every record but record 0 of a --start-delay 0 run, which shows qpos); 224x299 render, "
                          "reflection off, PIL bilinear -> 224x224; state from recorded qpos"}
    write_json(side, info)
    return info


def render_edits(an: Analysis, ctx, model, data, ren, rdir: pathlib.Path, families: Sequence[str],
                 max_hosts: Optional[int] = None) -> dict:
    """All edit rows of one run: E1 / E1s / E1z / E2 on `held` records, E3 on `reach` records."""
    from AG3S.ag3s.experiments.studies import t48b_scene as sc

    rd = an.rd
    plan = [(f, int(h)) for f in families for h in (an.reach if f == "E3" else an.held)[:max_hosts]]
    dig = digest_of({"v": CACHE_VERSION, "code": code_digest(), "kind": "edits", "run": rd.spec.key,
                     "fp": rd.fingerprint,
                     "plan": plan, "g_empty": ctx.g_empty, "g_hold": ctx.g_hold, "g_limit": ctx.g_limit,
                     "template": None if ctx.template_rel is None else np.round(ctx.template_rel, 12).tolist(),
                     "template_src": ctx.template_src, "pen_max": ctx.pen_max, "search": list(ctx.search_mm)})
    path = rdir / f"{rd.spec.key}__edits.npz"
    side = rdir / f"{rd.spec.key}__edits.json"
    if _cached(path, dig) and side.exists():
        print(f"[render/edits] {rd.spec.key} cached", flush=True)
        return json.loads(side.read_text())
    rows: dict[str, list] = {k: [] for k in ("family", "host", "included", "reason", "state", "q_ren",
                                             "img_index", "phys_after", "touching_after", "penetration",
                                             "gt", "info", "edit_s", "render_s")}
    images = []
    for fam, h in plan:
        t0 = time.perf_counter()
        res = sc.edit_frame(model, data, ctx, fam, rd.qpos[h], an.q_ren[h])
        t1 = time.perf_counter()
        rs = 0.0
        if res.included:
            out = ren.render(res.q_render)
            rs = time.perf_counter() - t1
            images.append(np.stack([out[c] for c in sc.CAMS]))
        rows["family"].append(fam)
        rows["host"].append(h)
        rows["included"].append(res.included)
        rows["reason"].append(res.reason)
        rows["state"].append(res.state)
        rows["q_ren"].append(res.q_render)
        rows["img_index"].append(len(images) - 1 if res.included else -1)
        rows["phys_after"].append(res.phys.state if res.phys else "")
        rows["touching_after"].append(",".join(res.phys.touching) if res.phys else "")
        rows["penetration"].append(res.phys.penetration if res.phys else np.nan)
        rows["gt"].append(sc.INTENDED[fam][1])
        rows["info"].append(json.dumps(res.info, default=_json_default))
        rows["edit_s"].append(t1 - t0)
        rows["render_s"].append(rs)
    n = len(plan)
    arr = {"family": np.asarray(rows["family"], dtype="U4"), "host": np.asarray(rows["host"], np.int64),
           "included": np.asarray(rows["included"], bool), "reason": np.asarray(rows["reason"], dtype="U32"),
           "state": np.asarray(rows["state"], np.float64).reshape(n, 16),
           "q_ren": np.asarray(rows["q_ren"], np.float64).reshape(n, -1),
           "img_index": np.asarray(rows["img_index"], np.int64), "phys_after": np.asarray(rows["phys_after"], dtype="U16"),
           "touching_after": np.asarray(rows["touching_after"]), "penetration": np.asarray(rows["penetration"], float),
           "gt": np.asarray(rows["gt"], np.int64), "info": np.asarray(rows["info"]),
           "edit_s": np.asarray(rows["edit_s"]), "render_s": np.asarray(rows["render_s"])}
    imgs = np.stack(images) if images else np.zeros((0, 3, 224, 224, 3), np.uint8)
    save_npz(path, compressed=True, digest=np.asarray(dig), run=np.asarray(rd.spec.key), image=imgs,
             t_step=rd.t_step, prompt=np.asarray(rd.prompt), cams=np.asarray(sc.CAMS), **arr)
    counts = {f: {"planned": int((arr["family"] == f).sum()), "included": int(((arr["family"] == f) & arr["included"]).sum())}
              for f in families}
    reasons: dict = {}
    for f, r, inc in zip(arr["family"], arr["reason"], arr["included"]):
        if not inc:
            reasons.setdefault(str(f), {}).setdefault(str(r), 0)
            reasons[str(f)][str(r)] += 1
    info = {"digest": dig, "counts": counts, "excluded_reasons": reasons, "template_src": ctx.template_src or None,
            "initial_pose": ctx.initial_pose.tolist(), "g_empty": ctx.g_empty, "g_hold": ctx.g_hold,
            "render_s_median": float(np.median(arr["render_s"][arr["included"]])) if arr["included"].any() else None,
            "edit_s_median": {f: float(np.median(arr["edit_s"][arr["family"] == f])) for f in families
                              if (arr["family"] == f).any()},
            "penetration_max_included_m": float(np.nanmax(arr["penetration"][arr["included"]])) if arr["included"].any() else None}
    write_json(side, info)
    return info


# ----------------------------------------------------------------------------- extract (GPU)


def _render_files(root: pathlib.Path, key: str) -> tuple[pathlib.Path, pathlib.Path]:
    return root / "render" / f"{key}__E0.npz", root / "render" / f"{key}__edits.npz"


def e0_inputs(z, rd_files: Sequence[str]) -> list[tuple[str, int, np.ndarray, dict]]:
    """[(family, host, state, images)] for E0rec (record RGB + recorded state) and E0 (re-render +
    reconstructed state), host order."""
    from AG3S.ag3s.experiments.studies.t48b_scene import CAMS

    out = []
    for i, h in enumerate(np.asarray(z["host"], np.int64)):
        with np.load(rd_files[int(h)]) as r:
            rec = {c: np.asarray(r[f"image_{c}"]) for c in CAMS}
        out.append(("E0rec", int(h), np.asarray(z["state_rec"][i]), rec))
        out.append(("E0", int(h), np.asarray(z["state_recon"][i]), {c: z["image"][i, k] for k, c in enumerate(CAMS)}))
    return out


def edit_inputs(z, families: Sequence[str]) -> list[tuple[str, int, np.ndarray, dict]]:
    from AG3S.ag3s.experiments.studies.t48b_scene import CAMS

    out = []
    for i in range(len(z["family"])):
        if not bool(z["included"][i]) or str(z["family"][i]) not in families:
            continue
        k = int(z["img_index"][i])
        out.append((str(z["family"][i]), int(z["host"][i]), np.asarray(z["state"][i]),
                    {c: z["image"][k, j] for j, c in enumerate(CAMS)}))
    return out


def _extract_rows(ex, rows, prompt: str, batch: int) -> tuple[np.ndarray, list, list]:
    feats, decoded, ntxt = [], [], []
    for i in range(0, len(rows), batch):
        xs = [ex.inputs(st, imgs, prompt) for _, _, st, imgs in rows[i:i + batch]]
        feats.append(ex.features(xs))
        decoded += [ex.decoded_prompt(x) for x in xs]
        ntxt += [int(np.asarray(x["tokenized_prompt_mask"]).sum()) for x in xs]
    f = np.concatenate(feats) if feats else np.zeros((0, 512), np.float32)
    return f, decoded, ntxt


def stage_extract(args, root: pathlib.Path) -> int:
    from AG3S.ag3s.experiments.studies.t48_hsr_counterfactual import gpu_compute_processes
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    procs = gpu_compute_processes()
    if procs and not args.allow_busy_gpu:
        raise SystemExit("other GPU compute processes are running (nvidia-smi):\n  " + "\n  ".join(procs)
                         + "\nWait for them (do not stop them) or pass --allow-busy-gpu if the lead approved.")
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_FLAGS", "--xla_gpu_enable_command_buffer=")
    sroot = storage_root(args)
    paths = io.Paths.from_storage(sroot, checkpoint=args.checkpoint)
    if not paths.checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {paths.checkpoint}")
    specs = selected_runs(args)
    for s in specs:
        for p in _render_files(root, s.key)[:1]:
            if not p.exists():
                raise SystemExit(f"{p} missing — run --stage render first")
    from AG3S.ag3s.experiments.studies.t48_hsr_features import FeatureExtractor

    t_wall = time.time()
    ex = FeatureExtractor(args.config, str(paths.checkpoint), batch=args.batch)
    probe = SubtaskProbe.load(args.probe)
    fdir = root / "feat"
    fdir.mkdir(parents=True, exist_ok=True)
    meta: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "gpu_processes_before": procs,
                  "load_s": ex.load_s, "files": {}, "batch_parity": None}
    warmed = False

    def run_file(key: str, kind: str, rows, prompt: str, rdig: str) -> Optional[np.ndarray]:
        nonlocal warmed
        out = fdir / f"{key}__{kind}.npz"
        if out.exists():
            with np.load(out, allow_pickle=False) as z:
                if str(z["render_digest"]) == rdig:
                    print(f"[extract] {key} {kind} cached", flush=True)
                    return np.asarray(z["feature"])
            raise SystemExit(f"{out} exists for another render digest — use a new AG3S_RUN_DIR")
        if not rows:
            save_npz(out, render_digest=np.asarray(rdig), family=np.zeros(0, "U5"), host=np.zeros(0, np.int64),
                     feature=np.zeros((0, 512), np.float32), decoded_prompt=np.zeros(0, "U8"),
                     n_text_tokens=np.zeros(0, np.int64), prefix_s=np.zeros(0))
            return np.zeros((0, 512), np.float32)
        if not warmed:
            first = [ex.inputs(st, imgs, prompt) for _, _, st, imgs in rows[: args.batch]]
            ex.warmup(first[0], args.warmup)
            meta["batch_parity"] = ex.batch_parity(first)
            warmed = True
        n0 = len(ex.t_prefix)
        t0 = time.time()
        f, dec, ntxt = _extract_rows(ex, rows, prompt, args.batch)
        save_npz(out, render_digest=np.asarray(rdig), family=np.asarray([r[0] for r in rows], dtype="U5"),
                 host=np.asarray([r[1] for r in rows], np.int64), feature=f.astype(np.float32),
                 decoded_prompt=np.asarray(dec), n_text_tokens=np.asarray(ntxt, np.int64),
                 prefix_s=np.asarray(ex.t_prefix[n0:], np.float64))
        meta["files"][f"{key}__{kind}"] = {"rows": len(rows), "wall_s": time.time() - t0}
        print(f"[extract] {key} {kind}: {len(rows)} rows in {time.time() - t0:.1f} s mem={ex.memory()}", flush=True)
        write_json(root / "extract" / "extract_meta.json", meta)
        return f

    # ---- pass A: parity rows of every run, then the gate
    agree_k = agree_n = 0
    per_run = {}
    for s in specs:
        p0, _ = _render_files(root, s.key)
        with np.load(p0, allow_pickle=False) as z:
            rows = e0_inputs(z, list(z["files"]))
            f = run_file(s.key, "E0", rows, str(z["prompt"]), str(z["digest"]))
        pred = probe.proba(np.asarray(f, np.float64)).argmax(1)
        fam = np.asarray([r[0] for r in rows])
        k = int((pred[fam == "E0rec"] == pred[fam == "E0"]).sum())
        per_run[s.key] = [k, int((fam == "E0").sum())]
        agree_k += k
        agree_n += int((fam == "E0").sum())
    gate = {"agreement_min": args.agreement_min, "k": agree_k, "n": agree_n,
            "agreement": agree_k / agree_n if agree_n else None, "per_run": per_run}
    gate["passed"] = bool(agree_n and gate["agreement"] >= args.agreement_min)
    write_json(root / "extract" / "parity_gate.json", gate)
    print(f"[extract] parity gate: {agree_k}/{agree_n} = {gate['agreement']} (min {args.agreement_min}) "
          f"passed={gate['passed']}", flush=True)
    stopped = False
    if args.parity_only:
        pass
    elif not gate["passed"] and not args.force:
        print("[extract] parity FAILED — edits not extracted (T48b.task.md). --force to continue.", file=sys.stderr)
        stopped = True
    else:
        for s in specs:
            _, pe = _render_files(root, s.key)
            if not pe.exists():
                print(f"[extract] {s.key}: no edits render cache, skipped", flush=True)
                continue
            with np.load(pe, allow_pickle=False) as z:
                rows = edit_inputs(z, args.families)
                run_file(s.key, "edits", rows, str(z["prompt"]), str(z["digest"]))
    timing = {"wall_s": time.time() - t_wall, **ex.timing(), "memory": ex.memory(), "gpu_processes_before": procs,
              "created": now_kst()}
    write_json(root / "extract" / "timing.json", timing)
    write_json(root / "extract" / "extract_meta.json", meta)
    print("[extract] timing", json.dumps(timing), flush=True)
    return 3 if stopped else 0


# ----------------------------------------------------------------------------- score (CPU)


@dataclasses.dataclass
class _Rows:
    """Duck type of `t48_hsr_core.VariantTable` for `baseline_predictions` (host + fed state)."""

    host: np.ndarray
    state: np.ndarray

    def __len__(self) -> int:
        return int(len(self.host))


def stage_score(args, root: pathlib.Path) -> int:
    from AG3S.ag3s.stages.subtask_probe import SubtaskProbe

    probe = SubtaskProbe.load(args.probe)
    sroot = storage_root(args)
    paths = io.Paths.from_storage(sroot, dataset=args.dataset)
    ep_meta = io.episode_meta(paths.dataset)
    stats = train_stats_cached(root, paths.dataset, ep_meta)
    majority = io.majority_from_stats(stats)
    specs = selected_runs(args)
    groups = run_groups(specs)
    cols: dict[str, list] = {k: [] for k in ("run", "family", "host", "t_step", "gt", "gt_phys", "probe", "rule",
                                             "proxy", "frame_index", "p", "included", "reason", "img_index",
                                             "fi_fallback")}
    pix_run, pix_mad, pix_max, state_err = [], [], [], {}
    excl_rows = {k: [] for k in ("run", "family", "included", "reason")}
    missing: list[str] = []
    for s in specs:
        p0, pe = _render_files(root, s.key)
        if not p0.exists():
            missing.append(f"{s.key}:render")
            continue
        rd = load_run_data(s, sroot, ep_meta, max_frames=args.max_frames)
        unit = core.Unit(unit_id=s.key, set_name=s.group, prompt=rd.prompt, target=rd.fruit,
                         distractors=rd.distractors, gripper_index=core.GRIPPER_INDEX[rd.hand],
                         seq_states=rd.state, rows=np.arange(len(rd.files)), clock=rd.t_step,
                         y=np.zeros(len(rd.files), np.int64), proxy_cadence=1, seed_key=0)
        parts = []
        with np.load(p0, allow_pickle=False) as z:
            if int(len(z["host"])) != len(rd.files):
                raise SystemExit(f"{p0}: {len(z['host'])} rows but the record has {len(rd.files)} "
                                 "(--max-frames differs from the render stage?)")
            pix_run += [s.key] * len(z["host"])
            pix_mad.append(np.asarray(z["mad"]))
            pix_max.append(np.asarray(z["max_abs"]))
            state_err[s.key] = float(np.abs(np.asarray(z["state_recon"]) - np.asarray(z["state_rec"])).max())
            host = np.asarray(z["host"], np.int64)
            for fam, st in (("E0rec", z["state_rec"]), ("E0", z["state_recon"])):
                parts.append(dict(family=np.full(len(host), fam), host=host, state=np.asarray(st),
                                  gt=np.asarray(z["gt_cl"]), gt_phys=np.asarray(z["gt_phys"]),
                                  included=np.ones(len(host), bool), reason=np.full(len(host), ""),
                                  img_index=host.copy()))
            rdig0 = str(z["digest"])
        f0 = _load_feat(root / "feat" / f"{s.key}__E0.npz", rdig0)
        feats = [_order_feats(f0, parts[0]["family"], host, "E0rec"), _order_feats(f0, parts[1]["family"], host, "E0")]
        if pe.exists():
            with np.load(pe, allow_pickle=False) as z:
                fam = np.asarray(z["family"]).astype(str)
                excl_rows["run"] += [s.key] * len(fam)
                excl_rows["family"] += fam.tolist()
                excl_rows["included"] += np.asarray(z["included"]).tolist()
                excl_rows["reason"] += np.asarray(z["reason"]).astype(str).tolist()
                inc = np.asarray(z["included"], bool) & np.isin(fam, args.families)
                hosts = np.asarray(z["host"], np.int64)[inc]
                parts.append(dict(family=fam[inc], host=hosts, state=np.asarray(z["state"])[inc],
                                  gt=np.asarray(z["gt"])[inc], gt_phys=np.asarray(z["gt"])[inc],
                                  included=np.ones(int(inc.sum()), bool), reason=np.full(int(inc.sum()), ""),
                                  img_index=np.asarray(z["img_index"])[inc]))
                rdig = str(z["digest"])
            fe = _load_feat(root / "feat" / f"{s.key}__edits.npz", rdig)
            feats.append(_order_edit_feats(fe, parts[-1]["family"], hosts))
        for part, fpart in zip(parts, feats):
            b = core.baseline_predictions(unit, _Rows(part["host"], part["state"]), majority, fallback="nearest")
            p = probe.proba(fpart) if fpart is not None else np.full((len(part["host"]), 3), np.nan)
            pred = np.where(np.isnan(p).any(1), -1, np.nan_to_num(p).argmax(1))
            n = len(part["host"])
            cols["run"] += [s.key] * n
            cols["family"] += list(part["family"])
            cols["host"] += part["host"].tolist()
            cols["t_step"] += rd.t_step[part["host"]].tolist()
            cols["gt"] += np.asarray(part["gt"]).tolist()
            cols["gt_phys"] += np.asarray(part["gt_phys"]).tolist()
            cols["probe"] += pred.tolist()
            for k in ("rule", "proxy", "frame_index"):
                cols[k] += b[k].tolist()
            cols["fi_fallback"] += b["_frame_index_fallback"].tolist()
            cols["p"] += list(p)
            cols["included"] += part["included"].tolist()
            cols["reason"] += list(part["reason"])
            cols["img_index"] += np.asarray(part["img_index"]).tolist()
    if not cols["run"]:
        raise SystemExit(f"nothing to score (missing: {missing})")
    c = {k: np.asarray(v) for k, v in cols.items() if k != "p"}
    P = np.asarray(cols["p"], float).reshape(-1, 3)
    sdir = root / "score"
    from AG3S.ag3s.experiments.studies.t48b_scene import CAMS

    parity = {"pixels": mx.parity_pixels(np.asarray(pix_run), np.concatenate(pix_mad), np.concatenate(pix_max), CAMS),
              "state_max_abs_per_run": state_err,
              "probe_agreement": mx.parity_agreement(c["run"], c["family"], c["host"], c["probe"], groups),
              "probe_p_change": mx.proba_change(c["run"], c["family"], c["host"], P),
              "agreement_min": args.agreement_min}
    ag = parity["probe_agreement"].get("main", {}).get("pooled")
    parity["passed"] = None if ag is None else bool(ag >= args.agreement_min)
    write_json(sdir / "parity.json", parity)
    print(f"[score] parity: probe agreement main={ag} passed={parity['passed']}", flush=True)
    metrics: dict = {"created": now_kst(), "git": git_state(), "args": vars(args), "groups": groups,
                     "missing": missing, "parity_passed": parity["passed"], "definitions": DEFINITIONS,
                     "exclusions": mx.exclusion_table(np.asarray(excl_rows["run"]), np.asarray(excl_rows["family"]),
                                                      np.asarray(excl_rows["included"], bool),
                                                      np.asarray(excl_rows["reason"])) if excl_rows["run"] else {},
                     "frame_index_fallback_rows": int(c["fi_fallback"].sum()), "methods": {}}
    stop = parity["passed"] is False and not args.force
    fam_mask = np.isin(c["family"], ["E0rec", "E0"]) if stop else np.ones(len(c["run"]), bool)
    for meth in METHODS:
        metrics["methods"][meth] = mx.method_metrics(c["run"][fam_mask], c["family"][fam_mask], c["host"][fam_mask],
                                                     c["gt"][fam_mask], c["gt_phys"][fam_mask], c[meth][fam_mask],
                                                     groups)
    if stop:
        metrics["note"] = "parity gate failed: only E0rec / E0 scored (T48b.task.md). --force to score edits."
    write_json(sdir / "metrics.json", metrics)
    with open(sdir / "predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "family", "host", "t_step", "gt", "gt_phys", "probe", "p_pick", "p_place", "p_home",
                    "rule", "proxy", "frame_index", "frame_index_fallback", "img_index", "render_cache"])
        for i in range(len(c["run"])):
            cache = f"render/{c['run'][i]}__{'E0' if c['family'][i] in ('E0rec', 'E0') else 'edits'}.npz"
            w.writerow([c["run"][i], c["family"][i], c["host"][i], c["t_step"][i], c["gt"][i], c["gt_phys"][i],
                        c["probe"][i], *[f"{x:.6g}" for x in P[i]], c["rule"][i], c["proxy"][i], c["frame_index"][i],
                        int(c["fi_fallback"][i]), c["img_index"][i],
                        "record" if c["family"][i] == "E0rec" else cache])
    for fam in mx.FAMILIES:
        line = []
        for meth in METHODS:
            r = metrics["methods"][meth]["families"].get(fam, {}).get("target_rate", {}).get("main", {})
            line.append(f"{meth}={_fmt(r.get('pooled'))}")
        print(f"[score] {fam:5s} target_rate(main) " + " ".join(line), flush=True)
    return 3 if stop else 0


def _load_feat(path: pathlib.Path, render_digest: str):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["render_digest"]) != render_digest:
            raise SystemExit(f"{path}: render digest differs from the render cache (re-rendered since extract?)")
        return {"family": np.asarray(z["family"]).astype(str), "host": np.asarray(z["host"], np.int64),
                "feature": np.asarray(z["feature"], np.float64)}


def _order_feats(f, family: np.ndarray, host: np.ndarray, fam: str) -> Optional[np.ndarray]:
    if f is None:
        return None
    lut = {int(h): i for i, (ff, h) in enumerate(zip(f["family"], f["host"])) if ff == fam}
    if any(int(h) not in lut for h in host):
        raise SystemExit(f"feature file lacks {fam} rows for some hosts")
    return f["feature"][[lut[int(h)] for h in host]]


def _order_edit_feats(f, family: np.ndarray, host: np.ndarray) -> Optional[np.ndarray]:
    if f is None:
        return None
    lut = {(str(ff), int(h)): i for i, (ff, h) in enumerate(zip(f["family"], f["host"]))}
    if any((str(ff), int(h)) not in lut for ff, h in zip(family, host)):
        return None   # extracted with other --families: score the edits as missing
    return f["feature"][[lut[(str(ff), int(h))] for ff, h in zip(family, host)]]


def _fmt(x) -> str:
    return "NA" if x is None else f"{x:.3f}"


DEFINITIONS = {
    "render": "render pose = mj_integratePos(recorded qpos, recorded qvel, -dt) (the kinematics mj_step left in "
              "data when the camera rendered; record 0 of a --start-delay 0 run follows mj_forward and shows "
              "qpos, t48b_scene.observation_follows_step); 224x299 render, reflection off, PIL bilinear -> 224x224 "
              "(pi05_infer.py:217-225, recording.py:28-54); state = build_obs rby1_16d from the recorded qpos "
              "(pi05_infer.py:466-497)",
    "phys_state": "contacts of mj_forward at the render pose: held = both finger bodies of a hand and no table/crate; "
                  "grasping = both fingers + table/crate; in_crate = not held, centre inside the crate volume; "
                  "on_table = table contact; other = none",
    "gt": {"E0rec/E0": "closed_loop_gt class of the host (SUBTASK-b extract.py:63-70, T48's ep1807 GT)",
           "E0 gt_phys": "physical state -> pick (on_table) / place (held) / home (in_crate); grasping/other -> excluded",
           "E1/E1s/E1z/E2": "pick", "E3": "place"},
    "hosts": {"E1/E1s/E1z/E2": "records whose target is held by the used hand",
              "E3": "records before the first used-gripper closure (< 0.9), target on_table, gripper >= 0.9"},
    "baselines": "t48_hsr_core.baseline_predictions: history = the host record's original history before t, "
                 "the variant state at t; frame_index = nearest train frame_index on the run's t_step "
                 "(T48 ep1807 convention); proxy = offline strong-rule PROXY (not the strong rule)",
    "aggregation": "pooled k/n over frames and per-run ratio median / q25 / q75, per run group",
    "changed_vs_E0": "pred != the same method's prediction on the unedited re-render (E0) of the host",
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    root = out_root()
    if args.stage == "render":
        return stage_render(args, root)
    if args.stage == "extract":
        return stage_extract(args, root)
    return stage_score(args, root)


if __name__ == "__main__":
    sys.exit(main())

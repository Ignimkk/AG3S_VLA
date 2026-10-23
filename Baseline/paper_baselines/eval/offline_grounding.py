"""Offline target-grounding evaluation of the paper baselines on recorded LIBERO frames (PAPER-BASELINES).

Inputs
  --source-run   a rollout run recorded with ``--capture-geometry --capture-segmentation``
                 (``episodes/<stem>.npz`` RGB + ``episodes/<stem>_geom.npz`` depth / K / T / segmentation_geom),
                 e.g. SL3 dev ``20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0`` (10 ep) or SL4b dev
                 ``20261013T092537_714c891415d9_sl4b-dev-pi05only-t0-9e1-4`` (40 ep).
  --scene-gt     optional ``scene_export.py`` directory (``scene_<stem>.npz``; e.g. outputs/verify/SL4/raw/scene,
                 outputs/verify/SL4b/raw/scene): frames = its exported call frames, target = its ``target_json``,
                 and the lift is checked against the exact target surface (``lift_*`` columns).
  --ats-run      optional ``perception_replay`` run: V11AB ATS target points of the same frames become the
                 pseudo-provider ``ats_v11ab`` (3D rows only; it has no 2D mask).
  --attention-run optional ``benchmark/runners/attention_audit.py`` run of the same source run (SL3 dev
                 ``20261012T202459_309e4c6da477_sl3-attention-audit-dev``, SL4b dev
                 ``20261013T094739_714c891415d9_sl4b-attention-audit-dev-e1-4``): per-call pi0.5 attention for the
                 VLA-internal providers ``localization_heads:{threshold,sam_point,sam_box}`` (``pooled_f16``, frozen
                 heads ``--lh-selection``) and ``tokencut:attention_overlap`` (V11AB cell L8H2 grid). Frames without a
                 recorded call get ``missing_reason`` rows. These two providers ignore the query text and are scored
                 once (prompt label ``no_text``).

Per frame and provider (frames = scene-GT frames, else the policy-call frames ``chunk_t``):
  RGB (OpenCV frame = raw[::-1]) -> provider masks per camera -> 2D metrics vs the GT segmentation mask ->
  ``lift.lift_mask`` (same wire depth, K, T for everyone; ``--robot-removal`` for everyone) -> 5 mm voxel keys
  (union over cameras) -> soft / hard voxel metrics on the frame's eval grid (``eval/metrics.py``).
  ``grounded_sam2:track`` is fed every recorded frame from 0 up to the last evaluated frame in time order
  (causal), but scored only on the evaluated frames.
  The back-projection of each frame and the lift-check distance of every valid depth pixel to the target surface are
  computed once per frame (``FrameGeometry``) and indexed by each arm's mask; the distance uses an exact pruning of
  the reference ``point_triangles_distance_near`` (same numbers bit for bit, ``tests/test_offline_grounding_speed.py``).

Provider failures are isolated per arm and episode: an exception of a provider (construction, ``reset``,
``predict`` or a broken mask contract) turns that arm's remaining evaluated frames of the episode into rows with
``missing_reason = "provider_error"`` (+ ``provider_error``, ``provider_error_obs_index``); the run goes on with
the next arm. ``summary.json`` lists them (``provider_errors``, per-arm ``n_provider_error``). ``--fail-fast``
restores the old stop-at-first-exception behaviour.

Outputs (``common.run_artifacts.create_run`` under AG3S_VLA-storage/results/benchmark/, or ``--out-dir`` below
``outputs/`` for smoke): ``frames.jsonl`` (one row per frame x provider-arm), ``episodes.jsonl`` (episode means),
``summary.json`` (episode-mean aggregates, pooled sums, found rates, latency, paired bootstraps), and
``provenance.json`` (provider provenance + SHA-256 of every input file). No pass/fail judgement is made here.

Run (``.venv-baselines``; GPU only under the shared lock, see README)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \\
        --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \\
        --providers gt_oracle grounded_sam grounded_sam2:per_frame grounded_sam2:track yoloe \\
                    localization_heads:threshold localization_heads:sam_point tokencut:foreground tokencut:attention_overlap \\
        --prompt-modes phrase instruction \\
        --attention-run 20261012T202459_309e4c6da477_sl3-attention-audit-dev \\
        --ats-run 20261012T225820_46a08d6ff644_sl4-perception-dev --run-id paper-baselines-sl3dev
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pathlib
import sys
import time
import traceback
from typing import Any

import numpy as np

from Baseline.paper_baselines import lift as L
from Baseline.paper_baselines.eval import metrics as M
from Baseline.paper_baselines.interface import TargetQuery, get_provider, sha256_file, validate_results, _jsonable
from Baseline.paper_baselines.providers.gt_oracle import libero_target, phrase_from_object

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
CAMERAS = ("head", "left_wrist")
RAW_RGB_KEYS = {"head": "obs_agentview_image", "left_wrist": "obs_robot0_eye_in_hand_image"}
SUPPORT_FIXTURES = ("main_table", "kitchen_table", "study_table", "floor")
LIFT_CHECK_MM = 10.0


# --- inputs ---------------------------------------------------------------------------------------------------
def resolve_run(arg: str) -> pathlib.Path:
    p = pathlib.Path(arg)
    if p.is_dir():
        return p.resolve()
    from common.run_artifacts import results_root

    q = results_root() / arg
    if not q.is_dir():
        raise FileNotFoundError(f"run not found: {arg}")
    return q.resolve()


def episode_stem(row: dict) -> str:
    return f"{row['suite']}_t{int(row['task_id'])}_e{int(row['episode_idx'])}"


def load_episodes(run: pathlib.Path) -> list[dict]:
    rows = [json.loads(line) for line in (run / "episodes.jsonl").read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        stem = episode_stem(r)
        geom = run / "episodes" / f"{stem}_geom.npz"
        npz = run / "episodes" / f"{stem}.npz"
        out.append({**r, "stem": stem, "geom_path": geom, "npz_path": npz, "has_geom": geom.exists(), "has_npz": npz.exists()})
    return out


def label_lut(target_info: dict, model_geom_bodyid, model_body_names, classes=M.CLASSES) -> np.ndarray:
    """geom id -> class index (index ``n_geom`` = segmentation -1 = background)."""
    n = len(model_geom_bodyid)
    lut = np.full(n + 1, classes.index("background"), np.int64)
    names = [str(b) for b in model_body_names]
    gb = np.asarray(model_geom_bodyid).tolist()
    for g, b in enumerate(gb):
        if names[b].startswith(L.ROBOT_BODY_PREFIXES):
            lut[g] = classes.index("robot")
        elif names[b].startswith("mount0_"):
            lut[g] = classes.index("mount")
    for name, gids in target_info["instances_geom_ids"].items():
        if name == target_info["target"]:
            c = "target"
        elif name == target_info.get("destination"):
            c = "destination"
        elif name in SUPPORT_FIXTURES:
            c = "support"
        elif name in target_info.get("fixtures", ()):
            c = "fixture"
        else:
            c = "other_object"
        lut[np.asarray(gids, np.int64)] = classes.index(c)
    return lut


def seg_labels(seg: np.ndarray, lut: np.ndarray) -> np.ndarray:
    s = np.asarray(seg, np.int64)
    return lut[np.where(s < 0, len(lut) - 1, s)]


class Episode:
    """Arrays of one recorded episode needed for evaluation (loaded once, OpenCV frame)."""

    def __init__(self, ep: dict, cameras=CAMERAS, scene_dir: pathlib.Path | None = None):
        self.meta, self.stem, self.cameras = ep, ep["stem"], tuple(cameras)
        with np.load(ep["geom_path"], allow_pickle=False) as g:
            files = set(g.files)
            missing = [f"segmentation_geom/{c}" for c in self.cameras if f"segmentation_geom/{c}" not in files]
            if missing:
                raise ValueError(f"{self.stem}: geom npz has no {missing} (record with --capture-segmentation)")
            self.obs_index = g["obs_index"].astype(np.int64)
            self.tables = {"model_geom_bodyid": g["model_geom_bodyid"], "model_body_names": g["model_body_names"]}
            self.depth_mm = {c: g[f"depth_mm/{c}"] for c in self.cameras}
            self.K = {c: g[f"K/{c}"] for c in self.cameras}
            self.T = {c: g[f"T_base_cam/{c}"] for c in self.cameras}
            self.seg = {c: g[f"segmentation_geom/{c}"] for c in self.cameras}
        with np.load(ep["npz_path"], allow_pickle=False) as z:
            self.raw = {c: z[RAW_RGB_KEYS[c]] for c in self.cameras}
            self.chunk_t = z["chunk_t"].astype(np.int64)
        self.scene = None
        if scene_dir is not None:
            from benchmark.evaluators.libero_scene_gt import SceneGT

            p = scene_dir / f"scene_{self.stem}.npz"
            if p.exists():
                self.scene = SceneGT(p)
        self.target = libero_target(ep["suite"], ep["task_id"], self.tables,
                                    self.scene.target_info if self.scene is not None else None)
        self.lut = label_lut(self.target, self.tables["model_geom_bodyid"], self.tables["model_body_names"])
        self.robot_geoms = L.robot_geom_ids(self.tables["model_geom_bodyid"], self.tables["model_body_names"])
        self.target_geoms = np.asarray(self.target["instances_geom_ids"][self.target["target"]], np.int64)
        self._target_tri = {}
        self._geometry: dict[int, FrameGeometry] = {}

    def eval_frames(self) -> list[int]:
        return [int(t) for t in (self.scene.frames if self.scene is not None else self.chunk_t)]

    def row(self, t: int) -> int:
        hit = np.flatnonzero(self.obs_index == int(t))
        if not hit.size:
            raise KeyError(f"{self.stem}: obs_index {t} not in geom npz")
        return int(hit[0])

    def frame(self, t: int) -> dict:
        r = self.row(t)
        out = {"obs_index": int(t), "geom_row": r, "rgb": {}, "depth_m": {}, "K": {}, "T": {}, "seg": {}, "labels": {}, "robot": {}}
        for c in self.cameras:
            out["rgb"][c] = L.opencv_rgb_from_raw(self.raw[c][t])
            out["depth_m"][c] = self.depth_mm[c][r].astype(np.float64) / 1000.0
            out["K"][c], out["T"][c] = self.K[c][r], self.T[c][r]
            out["seg"][c] = self.seg[c][r]
            out["labels"][c] = seg_labels(self.seg[c][r], self.lut)
            out["robot"][c] = np.isin(self.seg[c][r], self.robot_geoms)
        return out

    def target_triangles(self, t: int):
        if self.scene is None:
            return None
        if t not in self._target_tri:
            srow = self.scene.frame_row(t)
            inst = self.scene.instances()[self.scene.instance_index(self.target["target"])]
            tri = [self.scene.triangles_base(srow, g) for g in inst.visual_geoms]
            self._target_tri[t] = np.concatenate(tri) if tri else np.zeros((0, 3, 3))
        return self._target_tri[t]

    def geometry(self, fr: dict) -> "FrameGeometry":
        """Back-projection + lift-check distances of one frame (from ``fr = frame(t)``), built once per frame and
        shared by every provider arm (the per-arm work is then an index into it)."""
        t = int(fr["obs_index"])
        if t not in self._geometry:
            self._geometry[t] = FrameGeometry(fr, self.cameras, self.target_triangles(t))
        return self._geometry[t]


def query_for(ep: Episode, prompt_mode: str, phrase_map: dict) -> TargetQuery:
    name = ep.target["target"]
    phrase = phrase_map.get(name) or phrase_map.get(phrase_from_object(name)) or phrase_from_object(name)
    instruction = ep.meta.get("task_description") or ep.target.get("language") or ""
    return TargetQuery(instruction=instruction, target_phrase=phrase, prompt_mode=prompt_mode, episode_id=ep.stem)


# --- ATS ------------------------------------------------------------------------------------------------------
class ATSRun:
    """Target points of a ``perception_replay`` run (``perception/<stem>_perception.npz``)."""

    name = "ats_v11ab"

    def __init__(self, run: pathlib.Path):
        self.run = run
        self.files: dict[str, dict] = {}

    def provenance(self) -> dict:
        man = self.run / "run_manifest.json"
        return {"provider": self.name, "run": str(self.run), "run_manifest_sha256": sha256_file(man) if man.exists() else None,
                "points": "perception npz target_points (base frame) per policy call; found = has_target",
                "latency": "t_process_s (whole V11AB process_multi_debug, not grounding alone)",
                "files": {k: v["sha256"] for k, v in self.files.items()}}

    def episode(self, stem: str) -> dict | None:
        if stem not in self.files:
            p = self.run / "perception" / f"{stem}_perception.npz"
            if not p.exists():
                self.files[stem] = {"sha256": None, "data": None}
            else:
                with np.load(p, allow_pickle=False) as z:
                    d = {k: z[k] for k in ("obs_index", "has_target", "target_points", "target_points_offsets", "t_process_s")}
                self.files[stem] = {"sha256": sha256_file(p), "data": d}
        return self.files[stem]["data"]

    def points(self, stem: str, t: int):
        d = self.episode(stem)
        if d is None:
            return None
        hit = np.flatnonzero(d["obs_index"] == int(t))
        if not hit.size:
            return None
        k = int(hit[0])
        o = d["target_points_offsets"]
        return {"found": bool(d["has_target"][k]), "points": d["target_points"][o[k]:o[k + 1]].astype(np.float64),
                "latency_ms": float(d["t_process_s"][k]) * 1000.0}


# --- recorded pi0.5 attention (VLA-internal providers) ---------------------------------------------------------
class AttentionRun:
    """Per-call attention of a ``benchmark/runners/attention_audit.py`` run (``attention/<stem>_attn.npz``).

    Joined to the evaluated frame by ``obs_index`` (= the source episode's ``chunk_t[k]``); the audit's own
    ``source_run`` and ``npz_sha256`` must match the ``--source-run`` episode (checked, mismatch stops the run).
    ``aux(stem, t)`` gives, for one policy call, the inputs both VLA-internal providers read:

    * ``localization_heads``: ``attention_pooled`` = ``pooled_f16[k]`` (L, H, 768) float32,
      ``attention_blocks`` / ``attention_grid`` / ``attention_orientation = "libero_policy"``
      (= ``providers.localization_heads.aux_from_attention_npz``);
    * ``tokencut:attention_overlap``: ``attention_map[camera]`` = ``cell_grid_policy[k, i]`` (16 x 16), the V11AB
      cell (layer 8, head 2, last action token, denoise step 0) in the same ``libero_policy`` orientation.
    """

    def __init__(self, run: pathlib.Path, source: pathlib.Path):
        from Baseline.paper_baselines.providers import localization_heads as LH

        self.run, self.source, self._LH = run, source, LH
        self.index: dict[str, dict] = {}
        for line in (run / "attention_index.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                self.index[episode_stem(r)] = r
        self.files: dict[str, dict] = {}

    def check_episode(self, stem: str, npz_sha256: str) -> None:
        r = self.index.get(stem)
        if r is None:
            return
        if r.get("source_run") != self.source.name:
            raise ValueError(f"attention run {self.run.name}: {stem} comes from {r.get('source_run')!r}, "
                             f"not --source-run {self.source.name!r}")
        if r.get("npz_sha256") and r["npz_sha256"] != npz_sha256:
            raise ValueError(f"attention run {self.run.name}: {stem} npz sha256 differs from the source episode")

    def episode(self, stem: str) -> dict | None:
        if stem not in self.files:
            r = self.index.get(stem)
            if r is None:
                self.files[stem] = {"sha256": None, "data": None}
            else:
                p = self.run / r["attn_npz"]
                with np.load(p, allow_pickle=False) as z:
                    cams = [str(c) for c in z["cameras"]]
                    d = {"obs_index": z["obs_index"].astype(np.int64), "pooled_f16": z["pooled_f16"],
                         "cell_grid_policy": z["cell_grid_policy"], "cell_grid_opencv": z["cell_grid_opencv"],
                         "cameras": cams}
                self.files[stem] = {"sha256": sha256_file(p), "data": d}
        return self.files[stem]["data"]

    def call(self, stem: str, t: int) -> int | None:
        d = self.episode(stem)
        if d is None:
            return None
        hit = np.flatnonzero(d["obs_index"] == int(t))
        return int(hit[0]) if hit.size else None

    def aux(self, stem: str, t: int) -> dict | None:
        k = self.call(stem, t)
        if k is None:
            return None
        d = self.episode(stem)
        aux = self._LH.aux_from_attention_npz(d, k)
        aux["attention_map"] = {c: d["cell_grid_policy"][k, i].astype(np.float32) for i, c in enumerate(d["cameras"])}
        aux["attention_call_index"] = k
        return aux

    def describe(self, stem: str, t: int) -> dict:
        """Dry-run check of the input grids of one frame."""
        k = self.call(stem, t)
        if k is None:
            return {"attention_call": None}
        d = self.episode(stem)
        aux = self.aux(stem, t)
        pooled = aux["attention_pooled"]
        out = {"attention_call": k, "attention_pooled_shape": list(pooled.shape),
               "attention_pooled_finite": bool(np.isfinite(pooled).all())}
        for i, c in enumerate(d["cameras"]):
            lo, hi = (int(v) for v in aux["attention_blocks"][c])
            g = d["cell_grid_policy"][k, i]
            out[f"{c}_block_mass_L8H2"] = float(pooled[8, 2, lo:hi].sum())
            # cell grid (float32) vs the same cell inside pooled_f16: equal up to float16 rounding (<= 2^-11 rel.)
            out[f"{c}_cell_vs_pooled_L8H2_maxabs"] = float(np.abs(pooled[8, 2, lo:hi].reshape(g.shape) - g).max())
            out[f"{c}_cell_grid_shape"] = list(g.shape)
            out[f"{c}_cell_grid_mass"] = float(g.sum())
            # the stored OpenCV grid is the column flip of the policy grid (attention_audit.py)
            out[f"{c}_cell_opencv_is_unmirrored"] = bool(np.array_equal(d["cell_grid_opencv"][k, i], g[:, ::-1]))
        return out

    def provenance(self) -> dict:
        man = self.run / "run_manifest.json"
        return {"run": str(self.run), "run_manifest_sha256": sha256_file(man) if man.exists() else None,
                "join": "obs_index of the audit call == evaluated frame; audit source_run / npz_sha256 checked",
                "localization_heads_input": "pooled_f16[k] (L, H, 768), orientation libero_policy",
                "tokencut_attention_map": "cell_grid_policy[k, camera] = V11AB cell L8H2, orientation libero_policy",
                "files": {k: v["sha256"] for k, v in self.files.items()}}


# --- providers ------------------------------------------------------------------------------------------------
#: provider name -> constructor keyword that takes the ``:mode`` suffix of ``--providers``
MODE_KWARG = {"localization_heads": "mask_mode", "tokencut": "target_select"}
#: providers that do not read the query text: one arm (prompt label ``no_text``) instead of one per prompt mode
TEXT_FREE = ("localization_heads", "tokencut")
DEFAULT_LH_SELECTION = (PROJECT_ROOT / "Baseline/paper_baselines/providers/frozen/"
                        "localization_heads_pi05_libero_spatial_dev_init0-4_official_code.json")


def parse_arm(spec: str) -> tuple[str, dict]:
    """``name[:mode]`` -> (provider name, constructor kwargs)."""
    name, _, mode = spec.partition(":")
    kw = {MODE_KWARG.get(name, "mode"): mode} if mode else {}
    return name, kw


def needs_attention(spec: str) -> bool:
    name, kw = parse_arm(spec)
    return name == "localization_heads" or (name == "tokencut" and kw.get("target_select") == "attention_overlap")


def provider_kwargs(spec: str, size: str, device: str | None, lh_selection=None) -> tuple[str, dict]:
    """Constructor arguments of one ``--providers`` entry (sizes / devices only where the provider takes them)."""
    name, kw = parse_arm(spec)
    if name == "gt_oracle":
        return name, kw
    if name == "localization_heads":
        kw["selection"] = str(lh_selection or DEFAULT_LH_SELECTION)
        if kw.get("mask_mode", "threshold") != "threshold":
            from Baseline.paper_baselines import weights as W

            # same SAM v1 weight as grounded_sam at this --size (default sam-vit-huge, smoke sam-vit-base)
            kw["_sam_weight"] = W.preset("grounded_sam", size)["sam"]
        if device:
            kw["_sam_device"] = device
        return name, kw
    if name == "tokencut":
        if device:
            kw["device"] = device
        return name, kw
    kw["size"] = size
    if device:
        kw["device"] = device
    return name, kw


def build_providers(specs, size: str, device: str | None, lh_selection=None, errors: dict | None = None) -> dict[str, Any]:
    """``{spec: provider}``. With ``errors`` (a dict), a provider whose construction raises is left out and its error
    is stored as ``errors[spec]`` (its arms then get ``provider_error`` rows); without it the exception propagates."""
    out = {}
    for spec in specs:
        try:
            name, kw = provider_kwargs(spec, size, device, lh_selection)
            sam_weight, sam_device = kw.pop("_sam_weight", None), kw.pop("_sam_device", None)
            if sam_weight is not None:
                from Baseline.paper_baselines.providers.sam_backend import SamV1Backend

                kw["sam"] = SamV1Backend(sam_weight, device=sam_device)
            if name == "tokencut" and "device" not in kw:  # TokenCutProvider defaults to cpu; others to default_device()
                from Baseline.paper_baselines.providers.sam_backend import default_device

                kw["device"] = default_device()
            out[spec] = get_provider(name, **kw)
        except Exception as e:  # noqa: BLE001
            if errors is None:
                raise
            errors[spec] = _error_text(e)
            print(f"[offline_grounding] provider {spec} could not be built:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            _release_device_memory()
    return out


def arm_list(specs, prompt_modes) -> list[tuple[str, str]]:
    arms = []
    for spec in dict.fromkeys(specs):
        if parse_arm(spec)[0] in TEXT_FREE:  # query text unused: one arm, scored once
            arms.append((spec, "no_text"))
        else:
            arms.extend((spec, pm) for pm in prompt_modes)
    return arms


# --- provider failures (isolated per arm and episode) -----------------------------------------------------------
class ProviderFailure(Exception):
    """A provider raised (or broke the mask contract) at ``obs_index``; the arm's remaining frames of the episode
    become ``provider_error`` rows and the run goes on with the next arm."""

    def __init__(self, obs_index: int | None, cause: BaseException):
        super().__init__(f"obs {obs_index}: {_error_text(cause)}")
        self.obs_index, self.cause = obs_index, cause


def _error_text(e: BaseException) -> str:
    return f"{type(e).__name__}: {e}"[:2000]


def _release_device_memory() -> None:
    """After a failure: drop Python references, then the CUDA cache (only if torch is already loaded)."""
    import gc

    gc.collect()
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001  (a broken CUDA context must not stop the run either)
            pass


# --- lift check: distance of lifted points to the exact target surface ------------------------------------------
#: margin (m) of the exact pruning in :func:`point_triangles_distance_near_pruned`; float64 rounding of the bounds is
#: ~1e-16 m, so a pair is only dropped when its distance exceeds the kept minimum by >= 1 um.
PRUNE_MARGIN_M = 1e-6
#: a triangle with ``|ab x ac| <= SLIVER_RATIO * (longest edge)^2`` is treated as degenerate (never pruned, never used
#: as an upper bound). Above it the solved closest point has a rounding error <~ 1e-16 / SLIVER_RATIO^2 * edge
#: (<= 1e-12 m here), far below ``PRUNE_MARGIN_M``.
SLIVER_RATIO = 1e-3


def point_triangles_distance_near_pruned(p: np.ndarray, tri: np.ndarray, max_distance: float, *,
                                         margin: float = PRUNE_MARGIN_M) -> np.ndarray:
    """Same result as ``benchmark.evaluators.libero_scene_gt.point_triangles_distance_near`` (bit for bit; checked by
    ``tests/test_offline_grounding_speed.py`` on synthetic soups and recorded frames), solving far fewer pairs.

    The reference solves every (point, triangle) pair with ``|p - c_t| <= r_t + max_distance`` (centroid ``c_t``,
    circumradius ``r_t``) with ``closest_points_pairwise`` and keeps the minimum. Here, for regular triangles:
    ``ub(p) = min_t |p - c_t|`` bounds the minimum from above (a centroid lies on its triangle) and
    ``|p - c_t| - r_t`` bounds pair ``t`` from below, so a pair with ``|p - c_t| - r_t > min(ub, max_distance) + margin``
    can be neither the minimum nor ``<= max_distance`` and is not solved. Degenerate (sliver) triangles, whose solved
    closest point need not lie on the triangle, neither bound nor get pruned: their candidate pairs are exactly the
    reference's (same KD-tree query on the same points). Every kept pair is solved by the same element-wise
    ``closest_points_pairwise`` and reduced with the same ``min``; distances ``> max_distance`` are ``inf``.
    """
    from scipy.spatial import cKDTree

    from benchmark.evaluators.libero_scene_gt import closest_points_pairwise

    p = np.asarray(p, np.float64).reshape(-1, 3)
    tri = np.asarray(tri, np.float64).reshape(-1, 3, 3)
    out = np.full(len(p), np.inf)
    md = float(max_distance)
    if not len(tri) or not len(p):
        return out
    lo, hi = tri.min(1), tri.max(1)
    keep = np.all((hi >= p.min(0) - md) & (lo <= p.max(0) + md), axis=1)
    if not keep.any():
        return out
    tri = tri[keep]
    centre = tri.mean(1)
    radius = np.sqrt(((tri - centre[:, None]) ** 2).sum(-1)).max(1)
    e = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], tri[:, 2] - tri[:, 1]], 1)
    emax2 = (e ** 2).sum(-1).max(1)
    degenerate = np.linalg.norm(np.cross(e[:, 0], e[:, 1]), axis=1) <= SLIVER_RATIO * emax2
    pairs_p, pairs_t = [], []
    J = np.flatnonzero(degenerate)
    if J.size:  # the reference's candidate rule and call, restricted to these triangles
        lists = cKDTree(p).query_ball_point(centre[J], radius[J] + md + 1e-12)
        n = np.fromiter((len(x) for x in lists), np.int64, count=len(lists))
        if n.sum():
            pairs_t.append(J[np.repeat(np.arange(len(J)), n)])
            pairs_p.append(np.fromiter((i for x in lists for i in x), np.int64, count=int(n.sum())))
    R = np.flatnonzero(~degenerate)
    if R.size:
        # points with no regular centroid within max_distance + r_max have no regular triangle within max_distance
        dnn, _ = cKDTree(centre[R]).query(p, k=1, distance_upper_bound=md + float(radius[R].max()) + margin)
        act = np.flatnonzero(np.isfinite(dnn))
        pa = p[act]
        bound = np.minimum(dnn[act], md) + margin
        # radius classes keep the ball queries small (most triangles are much smaller than max_distance)
        edges = md * np.array([1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2, 1.0, np.inf])
        cls = np.searchsorted(edges, radius[R])
        for k in np.unique(cls) if act.size else ():
            Jk = R[cls == k]
            lists = cKDTree(centre[Jk]).query_ball_point(pa, bound + float(radius[Jk].max()))
            n = np.fromiter((len(x) for x in lists), np.int64, count=len(lists))
            if not n.sum():
                continue
            a = np.repeat(np.arange(len(pa)), n)
            b = Jk[np.fromiter((j for x in lists for j in x), np.int64, count=int(n.sum()))]
            ok = np.sqrt(((pa[a] - centre[b]) ** 2).sum(-1)) - radius[b] <= bound[a]
            pairs_p.append(act[a[ok]])
            pairs_t.append(b[ok])
    if not pairs_p:
        return out
    a, b = np.concatenate(pairs_p), np.concatenate(pairs_t)
    if not len(a):
        return out
    cp = closest_points_pairwise(p[a], tri[b, 0], tri[b, 1], tri[b, 2])
    np.minimum.at(out, a, np.sqrt(((cp - p[a]) ** 2).sum(-1)))
    out[out > md] = np.inf
    return out


def lift_check_from_distances(d: np.ndarray) -> dict:
    """``lift_*`` columns from the per-point distances (``inf`` beyond ``LIFT_CHECK_MM``)."""
    if not len(d):
        return {}
    within = np.isfinite(d)
    return {"lift_n_points": int(len(d)), "lift_within_10mm": float(within.mean()),
            "lift_dist_median_mm": float(np.median(d[within]) * 1000.0) if within.any() else None}


def lift_check(points: np.ndarray, tri) -> dict:
    """Distance of lifted points to the exact target surface (scene GT): independent check of the lift."""
    if tri is None or not len(points):
        return {}
    return lift_check_from_distances(point_triangles_distance_near_pruned(points, tri, LIFT_CHECK_MM / 1000.0))


class FrameGeometry:
    """One evaluated frame, shared by all provider arms: per camera the back-projection of every valid depth pixel
    (``lift.backproject_pixels``) and, computed once on first use, the lift-check distance of each of those points.

    A provider's lifted points are the rows ``points[sel]`` of the back-projection (``lift.lift_selection``), and the
    reference distance of a point does not depend on which other points are queried with it, so indexing the
    per-pixel distances gives the same ``lift_*`` values as ``lift_check(lifted points)``.
    """

    def __init__(self, fr: dict, cameras, tri):
        self.cameras, self.tri = tuple(cameras), tri
        self.shape = {c: np.asarray(fr["depth_m"][c]).shape for c in self.cameras}
        self.backprojected = {c: L.backproject_pixels(fr["depth_m"][c], fr["K"][c], fr["T"][c]) for c in self.cameras}
        self._distance: dict[str, np.ndarray] = {}

    def points(self, c: str) -> np.ndarray:
        return self.backprojected[c][0]

    def selection(self, c: str, mask, robot_removal: str, robot_pixels) -> np.ndarray:
        _, u, v = self.backprojected[c]
        return L.lift_selection(mask, u, v, self.shape[c], robot_removal=robot_removal, robot_pixels=robot_pixels)

    def pixel_distance(self, c: str) -> np.ndarray:
        if c not in self._distance:
            self._distance[c] = point_triangles_distance_near_pruned(self.points(c), self.tri, LIFT_CHECK_MM / 1000.0)
        return self._distance[c]

    def lift_check(self, selections: dict) -> dict:
        """``lift_check`` of the points ``points(c)[selections[c]]`` concatenated in camera order."""
        if self.tri is None or not sum(int(np.count_nonzero(s)) for s in selections.values()):
            return {}
        return lift_check_from_distances(np.concatenate([self.pixel_distance(c)[selections[c]] for c in selections]))


def evaluate_masks(results, fr, grid, ep: Episode, robot_removal: str, cameras) -> dict:
    row: dict[str, Any] = {"cameras": {}}
    geo = ep.geometry(fr)
    keys, sels = [], {}
    for c in cameras:
        r = results[c]
        gt = fr["labels"][c] == M.CLASSES.index("target")
        m2 = M.mask_metrics_2d(r.mask, gt)
        row["cameras"][c] = {**r.summary(), **{f"2d_{k}": v for k, v in m2.items()}}
        sels[c] = geo.selection(c, r.mask, robot_removal, fr["robot"][c])
        keys.append(L.voxel_keys(geo.points(c)[sels[c]]))
    keys = np.concatenate(keys) if keys else np.zeros((0, 3), np.int64)
    row.update({f"3d_{k}": v for k, v in M.voxel_metrics(grid, keys).items()})
    row.update(geo.lift_check(sels))
    row["found_any"] = any(results[c].found for c in cameras)
    row["latency_ms_total"] = float(sum(results[c].latency_ms for c in cameras))
    return row


def eval_grid(ep: Episode, fr: dict, cameras) -> M.EvalGrid:
    geo = ep.geometry(fr)
    return M.build_eval_grid([{"depth_m": fr["depth_m"][c], "K": fr["K"][c], "T_base_cam": fr["T"][c],
                               "labels": fr["labels"][c], "backprojected": geo.backprojected[c]} for c in cameras])


# --- main loop ------------------------------------------------------------------------------------------------
def run(args) -> pathlib.Path:
    source = resolve_run(args.source_run)
    scene_dir = pathlib.Path(args.scene_gt).resolve() if args.scene_gt else None
    ats = ATSRun(resolve_run(args.ats_run)) if args.ats_run else None
    attn = AttentionRun(resolve_run(args.attention_run), source) if args.attention_run else None
    need_attn = [s for s in args.providers if needs_attention(s)]
    if need_attn and attn is None and not args.dry_run:
        raise SystemExit(f"{need_attn} read recorded pi0.5 attention: pass --attention-run <attention_audit run>")
    phrase_map = json.loads(pathlib.Path(args.phrase_map).read_text()) if args.phrase_map else {}
    episodes = [e for e in load_episodes(source) if e["has_geom"] and e["has_npz"]]
    if args.episodes:
        episodes = [e for e in episodes if e["stem"] in set(args.episodes)]
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    if not episodes:
        raise SystemExit("no episode with both <stem>.npz and <stem>_geom.npz")
    cameras = tuple(args.cameras)
    build_errors: dict[str, str] = {}
    providers = {} if args.dry_run else build_providers(args.providers, args.size, args.device, args.lh_selection,
                                                        errors=None if args.fail_fast else build_errors)
    arms = [] if args.dry_run else arm_list(args.providers, args.prompt_modes)
    out = make_output(args, source)
    inputs: dict[str, Any] = {"source_run": str(source), "files": {}}
    man = source / "run_manifest.json"
    inputs["source_run_manifest_sha256"] = sha256_file(man) if man.exists() else None
    frames_f = (out / "frames.jsonl").open("x")
    rows = []

    def emit(row: dict) -> None:
        frames_f.write(json.dumps(_jsonable(row)) + "\n")
        frames_f.flush()
        rows.append(row)

    t0 = time.time()
    for ep_meta in episodes:
        t_ep = time.time()
        ep = Episode(ep_meta, cameras, scene_dir)
        inputs["files"][ep.stem] = {"npz": sha256_file(ep_meta["npz_path"]), "geom_npz": sha256_file(ep_meta["geom_path"]),
                                    "scene": ep.scene.sha256 if ep.scene is not None else None,
                                    "target": ep.target["target"], "destination": ep.target.get("destination")}
        if attn is not None:
            attn.check_episode(ep.stem, inputs["files"][ep.stem]["npz"])
        frames = ep.eval_frames()
        if args.limit_frames:
            frames = frames[: args.limit_frames]
        frs = {t: ep.frame(t) for t in frames}
        grids = {t: eval_grid(ep, frs[t], cameras) for t in frames}
        base = {"episode_id": ep.stem, "suite": ep_meta["suite"], "task_id": int(ep_meta["task_id"]),
                "episode_idx": int(ep_meta["episode_idx"]), "rollout_status": ep_meta.get("status"),
                "robot_removal": args.robot_removal}
        if args.dry_run:
            for t in frames:
                gt = {c: int((frs[t]["labels"][c] == M.CLASSES.index("target")).sum()) for c in cameras}
                row = {**base, "arm": "dry_run", "obs_index": t, "gt_target_pixels": gt,
                       "eval_voxels": int(len(grids[t].packed)), "target_voxel_mass": float(grids[t].target.sum())}
                if attn is not None:
                    row.update(attn.describe(ep.stem, t))
                if ats is not None:
                    a = ats.points(ep.stem, t)
                    row["ats_record"] = a is not None
                emit(row)
            continue
        for spec, pm in arms:
            for row in arm_rows(ep, spec, pm, providers.get(spec), frames, frs, grids, base, cameras=cameras,
                                phrase_map=phrase_map, attn=attn, robot_removal=args.robot_removal,
                                build_error=build_errors.get(spec), fail_fast=args.fail_fast,
                                save=(lambda sp, p_, st, t, res: save_masks(out, sp, p_, st, t, res)) if args.save_masks else None):
                emit(row)
        if ats is not None:
            for row in ats_rows(ats, ep, frames, grids, base):
                emit(row)
        print(f"[offline_grounding] {ep.stem}: {len(frames)} frames, episode {time.time() - t_ep:.1f}s, "
              f"total {time.time() - t0:.1f}s", flush=True)
    frames_f.close()
    write_summaries(out, rows, cameras, args)
    prov = {"providers": {spec: providers[spec].provenance() for spec in providers}, "provider_build_errors": build_errors,
            "ats": ats.provenance() if ats else None, "attention": attention_provenance(attn, providers),
            "inputs": inputs, "scene_gt": str(scene_dir) if scene_dir else None,
            "argv": sys.argv, "python": sys.executable,
            "settings": {k: v for k, v in vars(args).items()}, "metric_definition": M.__doc__}
    (out / "provenance.json").write_text(json.dumps(_jsonable(prov), indent=1, default=str) + "\n")
    print(f"[offline_grounding] wrote {out}")
    return out


def arm_rows(ep: Episode, spec: str, pm: str, prov, frames, frs, grids, base: dict, *, cameras, phrase_map: dict,
             attn=None, robot_removal: str = "none", build_error: str | None = None, fail_fast: bool = False, save=None):
    """Rows of one provider arm on one episode (one per evaluated frame), in time order.

    ``grounded_sam2:track`` is fed every recorded frame from 0 to the last evaluated frame (causal) and scored on the
    evaluated ones. If the provider raises (or breaks the mask contract) the remaining evaluated frames of this arm
    and episode become ``missing_reason = "provider_error"`` rows carrying the error, the provider is reset and the
    caller goes on with the next arm (``fail_fast`` re-raises instead). Errors of the evaluation itself propagate.
    """
    text_free = pm == "no_text"
    q = query_for(ep, "phrase" if text_free else pm, phrase_map)
    head = {**base, "arm": f"{spec}|{pm}", "provider": spec, "prompt_mode": pm, "prompt": None if text_free else q.prompt}
    eval_set = set(frames)
    done: set[int] = set()

    def failure_rows(t_fail, err: str):
        for t in frames:
            if t not in done and (t_fail is None or t >= t_fail):
                yield {**head, "obs_index": int(t), "missing_reason": "provider_error", "provider_error": err,
                       "provider_error_obs_index": t_fail}

    if prov is None:
        yield from failure_rows(None, f"provider not built: {build_error}")
        return
    sequential = parse_arm(spec)[1].get("mode") == "track"
    uses_attn = needs_attention(spec)
    feed = range(0, max(frames) + 1) if sequential else frames
    failure = None
    try:
        _call_provider(None, prov.reset)
        for t in feed:
            fr = frs[t] if t in frs else None
            row = {**head, "obs_index": int(t)}
            aux = {"segmentation": fr["seg"] if fr else {c: ep.seg[c][ep.row(t)] for c in cameras},
                   "target_geom_ids": ep.target_geoms.tolist()}
            if uses_attn:
                a_aux = attn.aux(ep.stem, t)
                if a_aux is None:  # no recorded policy call at this frame: scored as missing, not as not-found
                    if t in eval_set:
                        row["missing_reason"] = "no_attention_record_for_frame"
                        done.add(t)
                        yield row
                    continue
                aux.update(a_aux)
                row["attention_call"] = a_aux["attention_call_index"]
            rgb = fr["rgb"] if fr is not None else {c: L.opencv_rgb_from_raw(ep.raw[c][t]) for c in cameras}
            qt = TargetQuery(q.instruction, q.target_phrase, q.prompt_mode, q.episode_id, int(t))
            res = _call_provider(t, lambda: prov.predict(rgb, qt, aux=aux))
            _call_provider(t, lambda: validate_results(res, rgb))
            if t not in eval_set:
                continue
            row.update(evaluate_masks(res, fr, grids[t], ep, robot_removal, cameras))
            if save is not None:
                save(spec, pm, ep.stem, t, res)
            done.add(t)
            yield row
    except ProviderFailure as e:
        if fail_fast:
            raise e.cause from e
        print(f"[offline_grounding] {head['arm']} {ep.stem}: provider failed at obs {e.obs_index}; remaining frames "
              f"recorded as provider_error\n" + "".join(traceback.format_exception(e.cause)), file=sys.stderr, flush=True)
        failure = (e.obs_index, _error_text(e.cause))
        e.cause.__traceback__ = None  # the traceback's frames can hold device tensors
    if failure is not None:  # outside the except block: the exception and its frames are released by now
        try:
            prov.reset()  # drop the failed state (e.g. a track's memory bank) before the next arm
        except Exception:  # noqa: BLE001
            pass
        _release_device_memory()
        yield from failure_rows(*failure)


def _call_provider(t, fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        raise ProviderFailure(t, e) from e


def ats_rows(ats: "ATSRun", ep: Episode, frames, grids, base: dict):
    for t in frames:
        a = ats.points(ep.stem, t)
        row = {**base, "arm": "ats_v11ab|noun_attention", "provider": "ats_v11ab", "prompt_mode": "noun_attention",
               "obs_index": int(t)}
        if a is None:
            row["missing_reason"] = "no_ats_record_for_frame"
        else:
            pts = a["points"] if a["found"] else np.zeros((0, 3))
            row.update({f"3d_{k}": v for k, v in M.voxel_metrics(grids[t], L.voxel_keys(pts)).items()})
            row.update(lift_check(pts, ep.target_triangles(t)))
            row.update(found_any=a["found"], latency_ms_total=a["latency_ms"])
        yield row


def attention_provenance(attn, providers) -> dict | None:
    if attn is None:
        return None
    out = attn.provenance()
    # localization heads were selected on dev audit runs: same run here = in-sample measurement
    for spec, prov in providers.items():
        sel = getattr(prov, "selection", None)
        if isinstance(sel, dict) and "inputs" in sel:
            used = sorted({i["run"] for i in sel["inputs"]})
            out.setdefault("selection_in_sample", {})[spec] = {"in_sample": attn.run.name in used, "selection_runs": used}
    return out


def make_output(args, source: pathlib.Path) -> pathlib.Path:
    settings = {"tool": "Baseline/paper_baselines/eval/offline_grounding.py", "argv": sys.argv,
                "source_run": str(source), "providers": args.providers, "size": args.size, "dry_run": args.dry_run,
                "environment": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "HF_HOME", "TORCH_HOME", "YOLO_CONFIG_DIR")}}
    if args.out_dir:
        out = pathlib.Path(args.out_dir).resolve()
        base = (PROJECT_ROOT / "outputs").resolve()
        if not out.is_relative_to(base):
            raise ValueError(f"--out-dir must be below {base} (benchmark results go through create_run)")
        out.mkdir(parents=True, exist_ok=False)
        (out / "settings.json").write_text(json.dumps(settings, indent=1, default=str) + "\n")
        return out
    from common.run_artifacts import create_run

    return create_run(run_id=args.run_id, settings=settings)


def save_masks(out: pathlib.Path, spec: str, pm: str, stem: str, t: int, res) -> None:
    d = out / "masks" / f"{spec.replace(':', '-')}__{pm}"
    d.mkdir(parents=True, exist_ok=True)
    arrays = {f"{c}": np.packbits(r.mask, axis=None) for c, r in res.items()}
    arrays.update({f"{c}__shape": np.asarray(r.mask.shape) for c, r in res.items()})
    with open(d / f"{stem}_{t:04d}.npz", "xb") as f:
        np.savez_compressed(f, **arrays)


METRICS_3D = ("3d_iou", "3d_precision", "3d_recall", "3d_hard_iou", "3d_hard_precision", "3d_hard_recall", "lift_within_10mm")


def write_summaries(out: pathlib.Path, rows: list[dict], cameras, args) -> None:
    by_arm: dict[str, list] = {}
    for r in rows:
        by_arm.setdefault(r["arm"], []).append(r)
    summary: dict[str, Any] = {"arms": {}, "paired_3d_iou": {}, "aggregation": "frame -> episode mean -> mean over episodes",
                               "bootstrap": {"n": M.BOOTSTRAP_N, "seed": M.BOOTSTRAP_SEED}}
    with (out / "episodes.jsonl").open("x") as f:
        for arm, rs in by_arm.items():
            for e in sorted({r["episode_id"] for r in rs}):
                er = [r for r in rs if r["episode_id"] == e]
                rec = {"arm": arm, "episode_id": e, "n_frames": len(er)}
                for m in METRICS_3D:
                    v = [r[m] for r in er if r.get(m) is not None]
                    rec[m] = float(np.mean(v)) if v else None
                for c in cameras:
                    for m in ("2d_iou", "2d_precision", "2d_recall"):
                        v = [r["cameras"][c][m] for r in er if "cameras" in r and r["cameras"][c].get(m) is not None]
                        rec[f"{c}_{m}"] = float(np.mean(v)) if v else None
                f.write(json.dumps(rec) + "\n")
    for arm, rs in by_arm.items():
        if arm == "dry_run":
            d: dict[str, Any] = {"n_frames": len(rs), "n_episodes": len({r["episode_id"] for r in rs})}
            if any("attention_call" in r for r in rs):
                d["n_frames_with_attention"] = sum(r.get("attention_call") is not None for r in rs)
            if any("ats_record" in r for r in rs):
                d["n_frames_with_ats"] = sum(bool(r.get("ats_record")) for r in rs)
            summary["arms"][arm] = d
            continue
        s: dict[str, Any] = {m: M.aggregate(rs, m) for m in METRICS_3D}
        for c in cameras:
            flat = [{"episode_id": r["episode_id"], **r["cameras"][c]} for r in rs if "cameras" in r]
            for m in ("2d_iou", "2d_precision", "2d_recall"):
                s[f"{c}/{m}"] = M.aggregate(flat, m)
            s[f"{c}/found"] = M.found_rate(x["found"] for x in flat)
            s[f"{c}/latency"] = M.latency_stats(x["latency_ms"] for x in flat)
        valid = [r for r in rs if r.get("3d_iou") is not None]
        for k in ("3d_neighbor_fp", "3d_FP", "3d_TP", "3d_pred_offgrid", "3d_support_fp", "3d_robot_fp", "3d_fixture_fp"):
            s[k + "_sum"] = M.pooled_sum(valid, k)
        s["found_any"] = M.found_rate(bool(r.get("found_any")) for r in rs if "missing_reason" not in r)
        s["latency_total"] = M.latency_stats(r.get("latency_ms_total") for r in rs)
        s["n_frames"], s["n_valid_3d_frames"] = len(rs), len(valid)
        s["n_missing"] = sum("missing_reason" in r for r in rs)
        s["n_provider_error"] = sum(r.get("missing_reason") == "provider_error" for r in rs)
        summary["arms"][arm] = s
    errs: dict[tuple, dict] = {}
    for r in rows:
        if r.get("missing_reason") == "provider_error":
            e = errs.setdefault((r["arm"], r["episode_id"]), {"arm": r["arm"], "episode_id": r["episode_id"],
                                                              "obs_index": r.get("provider_error_obs_index"),
                                                              "error": r["provider_error"], "n_frames": 0})
            e["n_frames"] += 1
    summary["provider_errors"] = list(errs.values())
    arms = [a for a in by_arm if a != "dry_run"]
    for a, b in itertools.combinations(arms, 2):
        summary["paired_3d_iou"][f"{a} - {b}"] = M.paired_bootstrap(M.episode_means(by_arm[a], "3d_iou"),
                                                                    M.episode_means(by_arm[b], "3d_iou"))
    (out / "summary.json").write_text(json.dumps(_jsonable(summary), indent=1) + "\n")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-run", required=True)
    ap.add_argument("--scene-gt")
    ap.add_argument("--ats-run")
    ap.add_argument("--attention-run", help="attention_audit.py run of the same --source-run (localization_heads, "
                                            "tokencut:attention_overlap)")
    ap.add_argument("--lh-selection", help=f"frozen localization-heads json (default {DEFAULT_LH_SELECTION.name})")
    ap.add_argument("--providers", nargs="+", default=["gt_oracle"],
                    help="name[:mode]; grounded_sam2:per_frame|track, localization_heads:threshold|sam_point|sam_box, "
                         "tokencut:foreground|attention_overlap")
    ap.add_argument("--prompt-modes", nargs="+", default=["phrase"], choices=["phrase", "instruction"])
    ap.add_argument("--size", default="default", choices=["default", "smoke"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--robot-removal", default="none", choices=list(L.ROBOT_REMOVAL))
    ap.add_argument("--cameras", nargs="+", default=list(CAMERAS))
    ap.add_argument("--episodes", nargs="*")
    ap.add_argument("--limit-episodes", type=int)
    ap.add_argument("--limit-frames", type=int)
    ap.add_argument("--phrase-map", help="json {bddl object name or phrase: prompt phrase}")
    ap.add_argument("--save-masks", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="inputs + GT + eval grid only, no provider")
    ap.add_argument("--fail-fast", action="store_true",
                    help="stop the run at the first provider exception (default: record provider_error rows for that "
                         "arm and episode, go on with the next arm)")
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir", help="write here (below outputs/) instead of create_run; for smoke")
    return ap


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

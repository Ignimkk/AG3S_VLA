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

Per frame and provider (frames = scene-GT frames, else the policy-call frames ``chunk_t``):
  RGB (OpenCV frame = raw[::-1]) -> provider masks per camera -> 2D metrics vs the GT segmentation mask ->
  ``lift.lift_mask`` (same wire depth, K, T for everyone; ``--robot-removal`` for everyone) -> 5 mm voxel keys
  (union over cameras) -> soft / hard voxel metrics on the frame's eval grid (``eval/metrics.py``).
  ``grounded_sam2:track`` is fed every recorded frame from 0 up to the last evaluated frame in time order
  (causal), but scored only on the evaluated frames.

Outputs (``common.run_artifacts.create_run`` under AG3S_VLA-storage/results/benchmark/, or ``--out-dir`` below
``outputs/`` for smoke): ``frames.jsonl`` (one row per frame x provider-arm), ``episodes.jsonl`` (episode means),
``summary.json`` (episode-mean aggregates, pooled sums, found rates, latency, paired bootstraps), and
``provenance.json`` (provider provenance + SHA-256 of every input file). No pass/fail judgement is made here.

Run (``.venv-baselines``; GPU only under the shared lock, see README)::

    source Baseline/paper_baselines/setup/paper_baselines_env.sh
    .venv-baselines/bin/python -m Baseline.paper_baselines.eval.offline_grounding \\
        --source-run 20261012T163617_ce29b9be5fcb_sl3-dev-pi05only-t0-9e0 --scene-gt outputs/verify/SL4/raw/scene \\
        --providers gt_oracle grounded_sam grounded_sam2:per_frame grounded_sam2:track yoloe \\
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


# --- providers ------------------------------------------------------------------------------------------------
def parse_arm(spec: str) -> tuple[str, dict]:
    """``name[:mode]`` -> (provider name, kwargs)."""
    name, _, mode = spec.partition(":")
    kw = {"mode": mode} if mode else {}
    return name, kw


def build_providers(specs, size: str, device: str | None) -> dict[str, Any]:
    out = {}
    for spec in specs:
        name, kw = parse_arm(spec)
        if name != "gt_oracle":
            kw.update(size=size)
            if device:
                kw.update(device=device)
        out[spec] = get_provider(name, **kw)
    return out


def lift_check(points: np.ndarray, tri) -> dict:
    """Distance of lifted points to the exact target surface (scene GT): independent check of the lift."""
    if tri is None or not len(points):
        return {}
    from benchmark.evaluators.libero_scene_gt import point_triangles_distance_near

    d = point_triangles_distance_near(points, tri, LIFT_CHECK_MM / 1000.0)
    within = np.isfinite(d)
    return {"lift_n_points": int(len(points)), "lift_within_10mm": float(within.mean()),
            "lift_dist_median_mm": float(np.median(d[within]) * 1000.0) if within.any() else None}


def evaluate_masks(results, fr, grid, ep: Episode, robot_removal: str, cameras) -> dict:
    row: dict[str, Any] = {"cameras": {}}
    keys, pts_all = [], []
    for c in cameras:
        r = results[c]
        gt = fr["labels"][c] == M.CLASSES.index("target")
        m2 = M.mask_metrics_2d(r.mask, gt)
        row["cameras"][c] = {**r.summary(), **{f"2d_{k}": v for k, v in m2.items()}}
        pts = L.lift_mask(r.mask, fr["depth_m"][c], fr["K"][c], fr["T"][c], robot_removal=robot_removal,
                          robot_pixels=fr["robot"][c])
        pts_all.append(pts)
        keys.append(L.voxel_keys(pts))
    keys = np.concatenate(keys) if keys else np.zeros((0, 3), np.int64)
    row.update({f"3d_{k}": v for k, v in M.voxel_metrics(grid, keys).items()})
    pts_all = np.concatenate(pts_all) if pts_all else np.zeros((0, 3))
    row.update(lift_check(pts_all, ep.target_triangles(fr["obs_index"])))
    row["found_any"] = any(results[c].found for c in cameras)
    row["latency_ms_total"] = float(sum(results[c].latency_ms for c in cameras))
    return row


# --- main loop ------------------------------------------------------------------------------------------------
def run(args) -> pathlib.Path:
    source = resolve_run(args.source_run)
    scene_dir = pathlib.Path(args.scene_gt).resolve() if args.scene_gt else None
    ats = ATSRun(resolve_run(args.ats_run)) if args.ats_run else None
    phrase_map = json.loads(pathlib.Path(args.phrase_map).read_text()) if args.phrase_map else {}
    episodes = [e for e in load_episodes(source) if e["has_geom"] and e["has_npz"]]
    if args.episodes:
        episodes = [e for e in episodes if e["stem"] in set(args.episodes)]
    if args.limit_episodes:
        episodes = episodes[: args.limit_episodes]
    if not episodes:
        raise SystemExit("no episode with both <stem>.npz and <stem>_geom.npz")
    cameras = tuple(args.cameras)
    providers = {} if args.dry_run else build_providers(args.providers, args.size, args.device)
    arms = [(spec, pm) for spec in providers for pm in args.prompt_modes] if providers else []
    out = make_output(args, source)
    inputs: dict[str, Any] = {"source_run": str(source), "files": {}}
    man = source / "run_manifest.json"
    inputs["source_run_manifest_sha256"] = sha256_file(man) if man.exists() else None
    frames_f = (out / "frames.jsonl").open("x")
    rows = []
    t0 = time.time()
    for ep_meta in episodes:
        ep = Episode(ep_meta, cameras, scene_dir)
        inputs["files"][ep.stem] = {"npz": sha256_file(ep_meta["npz_path"]), "geom_npz": sha256_file(ep_meta["geom_path"]),
                                    "scene": ep.scene.sha256 if ep.scene is not None else None,
                                    "target": ep.target["target"], "destination": ep.target.get("destination")}
        frames = ep.eval_frames()
        if args.limit_frames:
            frames = frames[: args.limit_frames]
        eval_set = set(frames)
        grids, frs = {}, {}
        for t in frames:
            fr = ep.frame(t)
            frs[t] = fr
            grids[t] = M.build_eval_grid([{"depth_m": fr["depth_m"][c], "K": fr["K"][c], "T_base_cam": fr["T"][c],
                                           "labels": fr["labels"][c]} for c in cameras])
        base = {"episode_id": ep.stem, "suite": ep_meta["suite"], "task_id": int(ep_meta["task_id"]),
                "episode_idx": int(ep_meta["episode_idx"]), "rollout_status": ep_meta.get("status"),
                "robot_removal": args.robot_removal}
        if args.dry_run:
            for t in frames:
                gt = {c: int((frs[t]["labels"][c] == M.CLASSES.index("target")).sum()) for c in cameras}
                row = {**base, "arm": "dry_run", "obs_index": t, "gt_target_pixels": gt,
                       "eval_voxels": int(len(grids[t].packed)), "target_voxel_mass": float(grids[t].target.sum())}
                frames_f.write(json.dumps(row) + "\n")
                rows.append(row)
            continue
        for spec, pm in arms:
            prov = providers[spec]
            prov.reset()
            q = query_for(ep, pm, phrase_map)
            sequential = parse_arm(spec)[1].get("mode") == "track"
            feed = range(0, max(frames) + 1) if sequential else frames
            for t in feed:
                fr = frs[t] if t in frs else None
                rgb = fr["rgb"] if fr is not None else {c: L.opencv_rgb_from_raw(ep.raw[c][t]) for c in cameras}
                qt = TargetQuery(q.instruction, q.target_phrase, q.prompt_mode, q.episode_id, int(t))
                aux = {"segmentation": fr["seg"] if fr else {c: ep.seg[c][ep.row(t)] for c in cameras},
                       "target_geom_ids": ep.target_geoms.tolist()}
                res = prov.predict(rgb, qt, aux=aux)
                validate_results(res, rgb)
                if t not in eval_set:
                    continue
                row = {**base, "arm": f"{spec}|{pm}", "provider": spec, "prompt_mode": pm, "prompt": q.prompt,
                       "obs_index": int(t)}
                row.update(evaluate_masks(res, fr, grids[t], ep, args.robot_removal, cameras))
                if args.save_masks:
                    save_masks(out, spec, pm, ep.stem, t, res)
                frames_f.write(json.dumps(_jsonable(row)) + "\n")
                frames_f.flush()
                rows.append(row)
        if ats is not None:
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
                frames_f.write(json.dumps(_jsonable(row)) + "\n")
                rows.append(row)
        print(f"[offline_grounding] {ep.stem}: {len(frames)} frames, {time.time() - t0:.1f}s", flush=True)
    frames_f.close()
    write_summaries(out, rows, cameras, args)
    prov = {"providers": {spec: providers[spec].provenance() for spec in providers},
            "ats": ats.provenance() if ats else None, "inputs": inputs, "scene_gt": str(scene_dir) if scene_dir else None,
            "argv": sys.argv, "python": sys.executable,
            "settings": {k: v for k, v in vars(args).items()}, "metric_definition": M.__doc__}
    (out / "provenance.json").write_text(json.dumps(_jsonable(prov), indent=1, default=str) + "\n")
    print(f"[offline_grounding] wrote {out}")
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
            summary["arms"][arm] = {"n_frames": len(rs)}
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
        summary["arms"][arm] = s
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
    ap.add_argument("--providers", nargs="+", default=["gt_oracle"],
                    help="name[:mode]; grounded_sam2:per_frame / grounded_sam2:track")
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
    ap.add_argument("--run-id")
    ap.add_argument("--out-dir", help="write here (below outputs/) instead of create_run; for smoke")
    return ap


def main(argv=None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

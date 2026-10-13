"""SL4b dev calibration of the Panda ATS attention cell: the pre-registered metric, candidate selection, adoption
rule and cause split, in code (Python 3.11, numpy / scipy; CPU, read-only on runs).

The rule was fixed in ``AG3S/ag3s/docs/handoff/SL4b.task.md`` ("사전 등록", lead, 2026-10-13, commit 41a5b28a)
**before** any SL4b result existed. :data:`PREREGISTRATION_ROWS` is that table verbatim (a test checks it
against the task file); the constants below are its numbers, and :data:`OPERATIONAL_DEFINITIONS` states
how the code reads each phrase where the table is silent (taken from the SL3 / SL4 verifier definitions).
Every output of this module and every SL4b perception replay run carries :func:`preregistration_record`.

Two commands (``.venv-ag3s``, CPU; outputs are created exclusively)::

    # candidate selection: dev attention (pooled_f16 grid) -> top-16 cells + V11AB L8H2
    python -m benchmark.evaluators.ats_calibration select \\
        --source-run <init0 dev run> --attention-run <init0 audit> --source-run <init1-4 run> --attention-run <its audit> \\
        --phases <SL3 gt dir (init 0)> --phases <phase_export dir (init 1-4)> --out select.json

    # evaluation of grounding replays (perception_replay.py --grounding-only, one run per candidate + oracle)
    python -m benchmark.evaluators.ats_calibration evaluate --selection select.json \\
        --run <L8H2 replay (--attention-cell 8,2)> --run <top-1 replay> ... [--run <oracle> --run <L8H2 f16 control>] \\
        --phases <dir> [...] --out evaluate.json [--rows-out rows.jsonl]

``--phases`` directories hold one track per episode: ``gt_<stem>.npz`` (SL3 ``export_gt_dev.py``) or
``phase_<stem>.npz`` (``benchmark/adapters/libero_panda/phase_export.py``); both carry the contact / lift track
and the image-instant segmentation ``segP/<cam>`` (:mod:`benchmark.evaluators.libero_phases`). ``evaluate``
needs exactly one float32 L8H2 un-mirror run (the baseline), one ``pooled_f16`` run per selected cell, all over
the selection's episodes; any other run (oracle, L8H2 from float16, mirror) is a reference and never a
candidate. Runtime: about 2 s per 10 evaluation calls of one run (exact point-triangle distances).

Definitions (per policy call in an evaluation phase; base frame; GT = ``scene_export`` + ``libero_scene_gt``):

* phase: :mod:`benchmark.evaluators.libero_phases` (SL3 / SL4; MuJoCo gripper-target contact, lift z0 + 10 mm);
* label of an ATS target point: the instance whose visual surface is nearest if that distance is
  <= :data:`NEAR_M`; else ``robot`` if within :data:`NEAR_M` of a franka collision sphere surface; else ``none``;
* ``target_correct`` = has_target and the strict majority label is the BDDL target and the centroid error
  (ATS centroid vs. centroid of the **visible** GT target samples, SL4) is <= :data:`CENTROID_MAX_M`.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

STEP = "SL4b"

# --- pre-registration (verbatim) --------------------------------------------------------------------------
#: ``SL4b.task.md`` "사전 등록 (결과를 보기 전에 고정 — 바꾸지 않는다)" table, row by row, verbatim.
PREREGISTRATION_ROWS: tuple[tuple[str, str], ...] = (
    ("Dev 데이터", "LIBERO Spatial task 0–9 × init state 0–4 (50 episodes). init 0은 SL3 run 재사용, init 1–4는 새 "
                 "rollout(pi0.5-only, 같은 protocol)"),
    ("Phase", "SL3/SL4와 같은 정의(MuJoCo finger–target contact, lift z0+10 mm). **평가 phase = approach + grasp** "
              "(운반 단계는 attach가 없는 offline 한계로 제외)"),
    ("1차 지표 (`target_correct`)", "call별: ATS가 target을 냈고(has_target) **ATS target 점의 최근접 instance 다수결 = "
                                   "진짜 target** 이고 centroid 오차 ≤ 30 mm 이면 1. 평가 phase call 전체에 대한 비율"),
    ("2차 지표", "pooled precision(10 mm), recall median, centroid 오차 median/p95, task별 `target_correct`"),
    ("후보 cell", "V11AB L8H2 + **선별 지표**(dev 50 episode attention에서 approach+grasp call의 head+wrist target mass "
                "평균, un-mirror)의 상위 16개 cell. 진단 grid는 denoise 0, agg `last`(V11AB와 같은 집계)"),
    ("채택 조건", "후보 cell이 L8H2보다 1차 지표가 **≥ 10 %p 높고**, task별로 **10개 중 ≥ 6개 task에서 L8H2 이상**이면 "
                "\"Panda calibrated cell\" 후보로 사용자에게 제안. 여러 개면 1차 지표 최고, 동률이면 낮은 layer. 조건을 "
                "만족하는 cell이 없으면 L8H2 유지"),
    ("원인 분리", "L8H2의 오선택 call마다: attention argmax(head·wrist)가 진짜 target 위에 있었는지 → attention 원인 / "
                "grounding 원인 비율"),
    ("참고(후보 아님)", "GT target oracle seed(segmentation으로 target pixel만 attention=1) — ATS 기하 상한. privileged "
                     "input으로 표시, 선택에 사용 안 함"),
)
PREREGISTRATION_SOURCE = {"file": "AG3S/ag3s/docs/handoff/SL4b.task.md",
                          "section": "사전 등록 (결과를 보기 전에 고정 — 바꾸지 않는다)",
                          "writer": "lead (A0)", "committed_in": "41a5b28a"}
#: The adoption rule as written (the 채택 조건 row).
ADOPTION_RULE_TEXT = dict(PREREGISTRATION_ROWS)["채택 조건"]

# --- the numbers of the pre-registration --------------------------------------------------------------------
TASK_IDS: tuple[int, ...] = tuple(range(10))
DEV_INIT_STATES: tuple[int, ...] = (0, 1, 2, 3, 4)
EVAL_PHASES: tuple[str, ...] = ("approach", "grasp")
BASELINE_CELL: tuple[int, int] = (8, 2)  # V11AB L8H2 (checked against attention_policy at run time)
N_TOP_CELLS = 16
CENTROID_MAX_M = 0.030
NEAR_M = 0.010  # "pooled precision(10 mm)"; also the nearest-instance radius (SL4 #2)
ADOPT_MIN_GAIN_NUM, ADOPT_MIN_GAIN_DEN = 1, 10  # >= 10 %p, compared in integer counts
ADOPT_MIN_TASKS = 6  # of len(TASK_IDS)
GRID_DENOISE_STEP, GRID_AGGREGATION = 0, "last"
#: SL4 visibility tolerance of a GT surface sample (|recorded depth - sample camera z| in head or left_wrist).
VIS_TOL_M = 0.010

#: How the code reads the table where it is silent (all from the SL3 / SL4 verifier definitions, or stated).
OPERATIONAL_DEFINITIONS = {
    "evaluation_calls": "every policy call whose phase is approach or grasp (libero_phases); denominators of the "
                        "primary metric include calls without a target and calls flagged attention_invalid",
    "point_label": f"each ATS target point -> the scene instance (BDDL objects + fixtures + mount; visual surfaces, "
                   f"exact point-triangle distance) nearest to it if <= {NEAR_M * 1e3:.0f} mm; else 'robot' if within "
                   f"{NEAR_M * 1e3:.0f} mm of a franka collision sphere surface (61 spheres); else 'none' (SL4 #2 labels)",
    "majority": "the label with the strictly largest count over the call's ATS target points; a tie for first "
                "place is not a target majority",
    "centroid_error": f"|ATS target centroid - centroid of the visible GT target samples|; visible = |depth residual| "
                      f"<= {VIS_TOL_M * 1e3:.0f} mm in head or left_wrist (SL4 centroid_err_vis); a call with no "
                      "visible GT target sample has no centroid error and target_correct = 0",
    "precision_pooled": f"over evaluation calls with a target: ATS target points within {NEAR_M * 1e3:.0f} mm of the GT "
                        "target visual surface / all ATS target points",
    "recall": f"visible GT target samples with an ATS target point within {NEAR_M * 1e3:.0f} mm / visible GT target "
              "samples; median over calls with a target and >= 1 visible sample (also reported: no target = 0)",
    "selection_metric": "per [L, H] of the audit pooled_f16 grid (denoise 0, agg last) and per camera block (head "
                        "tokens 0-256, left_wrist 256-512): a = block / block sum; mass = sum(a * cov_target), "
                        "cov_target = AG3S.adapters.panda.attention.opencv_mask_to_policy_patches(recorded target "
                        "image-instant target segmentation segP, unmirror=True); metric = mean of mass over "
                        "(evaluation call, camera) pairs of all dev episodes; a block whose float16 sum is 0 "
                        "(underflow) counts as mass 0 -- the replay attaches such a block as it is, i.e. no attention "
                        "on the target (the SL3 #6 table left these out instead: reported as metric_nanmean_valid_only)",
    "candidate_cells": f"V11AB {BASELINE_CELL} + the {N_TOP_CELLS} other cells with the largest selection metric "
                       "(ties: lower layer, then lower head)",
    "adoption_gain": "correct_c - correct_L8H2 >= n / 10 over the same evaluation calls (= rate gain >= 10 %p), "
                     "integer arithmetic",
    "adoption_tasks": f"tasks t with correct_c(t) >= correct_L8H2(t) (same calls), at least {ADOPT_MIN_TASKS} of "
                      f"{len(TASK_IDS)}",
    "adoption_tie_break": "highest primary rate; equal -> lower layer (pre-registered); still equal -> lower head "
                          "(not pre-registered: deterministic completion)",
    "misselection": "an L8H2 evaluation call with a target whose majority label is not the GT target (category "
                    "wrong_instance: another instance, or 'none' / 'robot' / a tie; the cause split is also given per "
                    "kind)",
    "cause_split": "for each misselection: argmax over the 512 head+wrist values of the cell grid the replay used "
                   "(raw, not normalised per camera -- the fused cloud also compares raw per-camera attention by "
                   "max); on the target = the image-instant target segmentation covers part of that patch "
                   "(opencv_mask_to_policy_patches > 0, the replay's mapping) -> grounding cause, else attention "
                   "cause. Per-camera argmax flags are reported too",
    "oracle": "attention = recorded GT target segmentation (1 / 0); privileged, reported, never a candidate",
    "segmentation": "segP/<cam>: geom ids re-rendered at the image-instant state of each policy-call frame (SL3 "
                    "export_gt_dev for init 0, phase_export.py for init 1-4; the two are bit-identical on the same "
                    "episode) -- what the recorded RGB / depth / attention show. The rollout's segmentation_geom/<cam> "
                    "is rendered at the return-time state (SL3: 100-300 px per frame differ) and is only reported "
                    "(metric_recorded_segmentation)",
}


def _rows_sha256() -> str:
    return hashlib.sha256(json.dumps(PREREGISTRATION_ROWS, ensure_ascii=False).encode()).hexdigest()


def preregistration_record() -> dict:
    """JSON-able record of the pre-registration: verbatim rows, numbers, operational definitions, code hash."""
    return {
        "source": PREREGISTRATION_SOURCE,
        "rows_verbatim": [{"항목": k, "정의": v} for k, v in PREREGISTRATION_ROWS],
        "rows_sha256": _rows_sha256(),
        "adoption_rule_text": ADOPTION_RULE_TEXT,
        "constants": {"task_ids": list(TASK_IDS), "dev_init_states": list(DEV_INIT_STATES),
                      "eval_phases": list(EVAL_PHASES), "baseline_cell": list(BASELINE_CELL),
                      "n_top_cells": N_TOP_CELLS, "centroid_max_m": CENTROID_MAX_M, "near_m": NEAR_M,
                      "vis_tol_m": VIS_TOL_M, "adopt_min_gain": f"{ADOPT_MIN_GAIN_NUM}/{ADOPT_MIN_GAIN_DEN}",
                      "adopt_min_tasks": ADOPT_MIN_TASKS, "grid": {"denoise_step": GRID_DENOISE_STEP,
                                                                   "aggregation": GRID_AGGREGATION}},
        "operational_definitions": OPERATIONAL_DEFINITIONS,
        "module": "benchmark/evaluators/ats_calibration.py",
        "module_sha256": hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),
    }


def check_baseline_cell() -> dict:
    """The pre-registered L8H2 is V11AB's cell (raises otherwise)."""
    from AG3S.adapters.panda import attention as pa

    cell = pa.v11ab_cell()
    if (cell["layer"], cell["head"]) != BASELINE_CELL or cell["denoise_step"] != GRID_DENOISE_STEP \
            or cell["aggregation"] != GRID_AGGREGATION:
        raise RuntimeError(f"V11AB cell {cell} is not the pre-registered L{BASELINE_CELL[0]}H{BASELINE_CELL[1]} "
                           f"(denoise {GRID_DENOISE_STEP}, agg {GRID_AGGREGATION})")
    return cell


def cell_label(cell: Optional[Sequence[int]]) -> str:
    return "oracle" if cell is None else f"L{int(cell[0])}H{int(cell[1])}"


# --- per-call metric (pure) ---------------------------------------------------------------------------------
def label_points(inst_dist: np.ndarray, names: Sequence[str], robot_dist: Optional[np.ndarray] = None,
                 near: float = NEAR_M) -> np.ndarray:
    """Per point: nearest instance name if within ``near``, else ``robot`` (franka sphere surface within ``near``),
    else ``none``. ``inst_dist`` (n, I) exact distances (``inf`` beyond the search radius)."""
    D = np.asarray(inst_dist, np.float64).reshape(-1, len(names))
    out = np.array(["none"] * len(D), dtype=object)
    if not len(D):
        return out
    dmin, arg = D.min(1), D.argmin(1)
    hit = dmin <= near
    out[hit] = np.asarray(names, dtype=object)[arg[hit]]
    if robot_dist is not None:
        rob = (~hit) & (np.asarray(robot_dist, np.float64).reshape(-1) <= near)
        out[rob] = "robot"
    return out


def majority_label(labels: Sequence[str]) -> tuple[Optional[str], dict]:
    """``(label with the strictly largest count or None on a tie / no points, counts)``."""
    names, counts = np.unique(np.asarray(labels, dtype=object).astype(str), return_counts=True)
    table = {str(n): int(c) for n, c in zip(names, counts)}
    if not len(counts):
        return None, table
    top = counts.max()
    if (counts == top).sum() > 1:
        return None, table
    return str(names[int(np.argmax(counts))]), table


def call_metrics(*, has_target: bool, target_name: str, target_points: np.ndarray, target_centroid: np.ndarray,
                 point_labels: Sequence[str], dist_to_target: np.ndarray, gt_target_visible: np.ndarray,
                 near: float = NEAR_M, centroid_max: float = CENTROID_MAX_M) -> dict:
    """The pre-registered per-call quantities. ``dist_to_target`` (n,) exact point -> GT target surface distance;
    ``gt_target_visible`` (m, 3) visible GT target samples (base frame)."""
    tp = np.asarray(target_points, np.float64).reshape(-1, 3)
    gtv = np.asarray(gt_target_visible, np.float64).reshape(-1, 3)
    out: dict[str, Any] = {"has_target": bool(has_target), "n_target_points": int(len(tp)), "n_gt_visible": int(len(gtv)),
                           "majority": None, "label_counts": {}, "precision_hits": 0, "recall": None,
                           "centroid_err_m": None, "target_correct": 0}
    if not has_target or not len(tp):
        out["category"] = "no_target"
        if len(gtv):
            out["recall"] = 0.0
        return out
    major, counts = majority_label(point_labels)
    out["majority"], out["label_counts"] = major, counts
    out["precision_hits"] = int(np.sum(np.asarray(dist_to_target, np.float64) <= near))
    if len(gtv):
        from scipy.spatial import cKDTree

        out["recall"] = float(np.mean(cKDTree(tp).query(gtv, k=1)[0] <= near))
        out["centroid_err_m"] = float(np.linalg.norm(np.asarray(target_centroid, np.float64) - gtv.mean(0)))
    if major != target_name:
        out["category"] = "wrong_instance"
    elif out["centroid_err_m"] is None:
        out["category"] = "no_visible_gt"
    elif out["centroid_err_m"] > centroid_max:
        out["category"] = "centroid_far"
    else:
        out["category"] = "correct"
        out["target_correct"] = 1
    return out


# --- aggregation ---------------------------------------------------------------------------------------------
def _pct(values, q):
    v = np.asarray([x for x in values if x is not None], np.float64)
    return None if not v.size else float(np.percentile(v, q))


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict:
    """Primary / secondary metrics over per-call rows (each from :func:`call_metrics` + ``task``)."""
    n = len(rows)
    correct = int(sum(r["target_correct"] for r in rows))
    with_t = [r for r in rows if r["has_target"] and r["n_target_points"]]
    near_pts = int(sum(r["precision_hits"] for r in with_t))
    all_pts = int(sum(r["n_target_points"] for r in with_t))
    rec = [r["recall"] for r in with_t if r["recall"] is not None]
    rec0 = [r["recall"] for r in rows if r["recall"] is not None]
    cen = [r["centroid_err_m"] * 1e3 for r in with_t if r["centroid_err_m"] is not None]
    cats: dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    per_task = {}
    for t in sorted({int(r["task"]) for r in rows}):
        rt = [r for r in rows if int(r["task"]) == t]
        c = int(sum(r["target_correct"] for r in rt))
        per_task[str(t)] = {"correct": c, "n": len(rt), "rate": c / len(rt) if rt else None}
    majority: dict[str, int] = {}
    for r in with_t:
        key = r["majority"] if r["majority"] is not None else "(tie)"
        majority[key] = majority.get(key, 0) + 1
    return {
        "primary": {"correct": correct, "n": n, "rate": correct / n if n else None},
        "per_task": per_task,
        "precision_pooled": near_pts / all_pts if all_pts else None,
        "precision_points": {"within_near": near_pts, "total": all_pts},
        "recall_median_has_target_visible": _pct(rec, 50), "recall_n": len(rec),
        "recall_median_no_target_as_0": _pct(rec0, 50),
        "centroid_err_mm": {"n": len(cen), "median": _pct(cen, 50), "p95": _pct(cen, 95)},
        "has_target_calls": len(with_t),
        "categories": cats,
        "majority_labels": majority,
        "attention_invalid_calls": int(sum(bool(r.get("attention_invalid")) for r in rows)),
    }


# --- adoption rule (pure) ------------------------------------------------------------------------------------
def adoption_decision(results: Mapping[str, Mapping[str, Any]], baseline: str, candidates: Sequence[str]) -> dict:
    """Apply the pre-registered rule. ``results[label]`` = ``{"cell": (L, H), "primary": {"correct", "n"},
    "per_task": {t: {"correct", "n"}}}``; every candidate is compared with ``baseline`` on the same calls."""
    base = results[baseline]
    n = int(base["primary"]["n"])
    rows, eligible = [], []
    for label in candidates:
        r = results[label]
        if int(r["primary"]["n"]) != n or set(r["per_task"]) != set(base["per_task"]):
            raise ValueError(f"{label} was evaluated on different calls than {baseline}")
        diff = int(r["primary"]["correct"]) - int(base["primary"]["correct"])
        gain_ok = diff * ADOPT_MIN_GAIN_DEN >= ADOPT_MIN_GAIN_NUM * n
        tasks = {}
        for t, bt in base["per_task"].items():
            ct = r["per_task"][t]
            if int(ct["n"]) != int(bt["n"]):
                raise ValueError(f"{label} task {t}: {ct['n']} calls, {baseline} {bt['n']}")
            tasks[t] = int(ct["correct"]) >= int(bt["correct"])
        n_ge = int(sum(tasks.values()))
        tasks_ok = n_ge >= ADOPT_MIN_TASKS
        row = {"label": label, "cell": list(r["cell"]), "correct": int(r["primary"]["correct"]), "n": n,
               "rate": int(r["primary"]["correct"]) / n if n else None, "gain_correct": diff,
               "gain_pp": 100.0 * diff / n if n else None, "gain_ok": bool(gain_ok),
               "tasks_ge_baseline": n_ge, "tasks_ge_baseline_list": sorted((int(t) for t, ok in tasks.items() if ok)),
               "tasks_ok": bool(tasks_ok), "satisfies_rule": bool(gain_ok and tasks_ok)}
        rows.append(row)
        if row["satisfies_rule"]:
            eligible.append(row)
    eligible.sort(key=lambda x: (-x["correct"], x["cell"][0], x["cell"][1]))
    chosen = eligible[0] if eligible else None
    return {"rule_text": ADOPTION_RULE_TEXT, "baseline": baseline, "baseline_correct": int(base["primary"]["correct"]),
            "n": n, "candidates": rows, "satisfying": [x["label"] for x in eligible],
            "selected_by_rule": None if chosen is None else chosen["label"],
            "selected_cell": None if chosen is None else chosen["cell"],
            "outcome": ("candidate satisfies the rule" if chosen is not None else
                        "no candidate satisfies the rule (rule: keep L8H2)")}


# --- candidate selection (pure parts) ------------------------------------------------------------------------
def selection_mass(pooled: np.ndarray, covs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target mass of every cell. ``pooled`` (N, L, H, 768) attention (float16 grid), ``covs`` (N, 2, 256) target
    coverage per policy patch [head, left_wrist]. Returns ``mass (N, 2, L, H)`` (invalid block -> 0) and
    ``invalid (N, 2, L, H)`` (block sum 0 / not finite)."""
    from AG3S.adapters.panda import attention as pa

    blocks = [pa.block_for(c).tokens for c in ("head", "left_wrist")]
    N, L, H = pooled.shape[:3]
    mass = np.zeros((N, 2, L, H))
    invalid = np.zeros((N, 2, L, H), bool)
    for j, (b0, b1) in enumerate(blocks):
        P = np.asarray(pooled[..., b0:b1], np.float64)
        bs = P.sum(-1)
        bad = ~(np.isfinite(bs) & (bs > 0.0))
        m = np.einsum("nlhp,np->nlh", P, np.asarray(covs[:, j], np.float64))
        mass[:, j] = np.where(bad, 0.0, m / np.where(bad, 1.0, bs))
        invalid[:, j] = bad
    return mass, invalid


def rank_cells(metric: np.ndarray, *, exclude: Sequence[int] = BASELINE_CELL, k: int = N_TOP_CELLS) -> list[tuple[int, int]]:
    """Top-``k`` ``(L, H)`` by ``metric`` (descending; ties lower layer then lower head), ``exclude`` left out."""
    L, H = metric.shape
    cells = [(l, h) for l in range(L) for h in range(H) if (l, h) != tuple(exclude)]
    cells.sort(key=lambda c: (-float(metric[c]), c[0], c[1]))
    return cells[:k]


# --- cause split (pure) --------------------------------------------------------------------------------------
def attention_cause(grid_policy: np.ndarray, cov_policy: np.ndarray) -> dict:
    """``grid_policy`` (2, 16, 16) the cell grids the replay attached (policy orientation, before its mapping);
    ``cov_policy`` (2, 16, 16) target coverage in the same orientation. Joint argmax over both cameras."""
    g = np.asarray(grid_policy, np.float64).reshape(2, -1)
    c = np.asarray(cov_policy, np.float64).reshape(2, -1)
    flat = int(np.argmax(g))
    cam, patch = divmod(flat, g.shape[1])
    per = [bool(c[j, int(np.argmax(g[j]))] > 0) for j in range(2)]
    on = bool(c[cam, patch] > 0)
    return {"argmax_camera": ("head", "left_wrist")[cam], "argmax_patch": [patch // 16, patch % 16],
            "argmax_on_target": on, "head_argmax_on_target": per[0], "wrist_argmax_on_target": per[1],
            "target_visible": [bool(c[0].sum() > 0), bool(c[1].sum() > 0)],
            "cause": "grounding" if on else "attention"}


# --- I/O helpers ---------------------------------------------------------------------------------------------
def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def target_coverage(seg: np.ndarray, target_geoms, *, unmirror: bool = True) -> np.ndarray:
    """(16, 16) policy-orientation coverage of the target in a geom-id segmentation image (OpenCV rows)."""
    from AG3S.adapters.panda import attention as pa
    from benchmark.adapters.libero_panda import task_target as tt

    return pa.opencv_mask_to_policy_patches(tt.instance_mask(seg, target_geoms), unmirror=unmirror)


def episode_phases(phase_dirs, stem: str, frames, geom_npz_sha256: Optional[str] = None, *,
                   segmentation: bool = False):
    """``(labels, info, track)`` of one episode's policy calls ``frames`` (= ``chunk_t``, checked)."""
    from benchmark.evaluators import libero_phases as lp

    track = lp.load_track(lp.find_track(phase_dirs, stem), geom_npz_sha256=geom_npz_sha256,
                          chunk_t=[int(f) for f in frames], segmentation=segmentation)
    info, labels = lp.phase_labels(track, [int(f) for f in frames])
    return labels, {"path": track["path"], "sha256": track["sha256"], **info}, track


# --- select --------------------------------------------------------------------------------------------------
def select(source_runs, attention_runs, phase_dirs, *, task_ids=None) -> dict:
    """Selection metric over the dev attention of all pairs -> ranked cells + the candidate list."""
    from benchmark.adapters.libero_panda import task_target as tt
    from benchmark.runners import perception_replay as pr

    from benchmark.evaluators import libero_phases as lp

    v11ab = check_baseline_cell()
    sources = pr.index_source_pairs(source_runs, attention_runs, task_ids=task_ids)
    inits = pr.check_dev_split(sources)
    total = None
    total_rec = None
    total_f32_baseline = 0.0
    invalid_total = None
    sum_valid = None
    n_valid = None
    n_pairs = 0
    episodes = []
    for e in sources["episodes"]:
        dev = pathlib.Path(e["dev_run_dir"])
        with np.load(dev / e["geom_npz"]) as g:
            geom = {k: g[k] for k in ("obs_index", "model_geom_bodyid", "model_body_names",
                                      "segmentation_geom/head", "segmentation_geom/left_wrist") if k in g.files}
        with np.load(e["attn_npz"]) as a:
            if "pooled_f16" not in a.files:
                raise ValueError(f"{e['stem']}: the audit run has no pooled_f16")
            obs = np.asarray(a["obs_index"]).astype(np.int64)
            labels, pinfo, track = episode_phases(phase_dirs, e["stem"], obs, e["geom_npz_sha256"], segmentation=True)
            sel = np.flatnonzero(np.isin(labels, EVAL_PHASES))
            pooled = np.asarray(a["pooled_f16"])[sel]
            cell32 = np.asarray(a["cell_grid_policy"], np.float64)[sel]
        target_geoms = tt.scene_instances(geom, e["target"])[e["target"]["target"]]
        row_of = {int(o): i for i, o in enumerate(np.asarray(geom["obs_index"]).reshape(-1))}
        covs = np.zeros((len(sel), 2, 256))
        covs_rec = np.zeros((len(sel), 2, 256))
        for j, k in enumerate(sel):
            for c, cam in enumerate(("head", "left_wrist")):
                covs[j, c] = target_coverage(lp.segmentation_at(track, int(obs[k]), cam), target_geoms).reshape(-1)
                key = f"segmentation_geom/{cam}"
                if key in geom:
                    covs_rec[j, c] = target_coverage(geom[key][row_of[int(obs[k])]], target_geoms).reshape(-1)
        mass, invalid = selection_mass(pooled, covs)
        mass_rec, _ = selection_mass(pooled, covs_rec)
        c32 = cell32.reshape(len(sel), 2, 256)
        base32 = np.einsum("ncp,ncp->nc", c32, covs) / c32.sum(-1)
        total = mass.sum((0, 1)) if total is None else total + mass.sum((0, 1))
        total_rec = mass_rec.sum((0, 1)) if total_rec is None else total_rec + mass_rec.sum((0, 1))
        invalid_total = invalid.sum((0, 1)) if invalid_total is None else invalid_total + invalid.sum((0, 1))
        sv = np.where(invalid, 0.0, mass).sum((0, 1))
        nv = (~invalid).sum((0, 1))
        sum_valid = sv if sum_valid is None else sum_valid + sv
        n_valid = nv if n_valid is None else n_valid + nv
        total_f32_baseline += float(base32.sum())
        n_pairs += 2 * len(sel)
        episodes.append({"stem": e["stem"], "task_id": e["task_id"], "init": e.get("init_state_index"),
                         "eval_calls": int(len(sel)), "calls": e["calls"], "phases": pinfo,
                         "target": e["target"]["target"], "target_geoms": [int(x) for x in target_geoms]})
    if not n_pairs:
        raise ValueError("no evaluation-phase call in the selected episodes")
    metric = total / n_pairs
    ranked = rank_cells(metric)
    order_all = rank_cells(metric, exclude=(-1, -1), k=metric.size)
    rank_of = {c: i + 1 for i, c in enumerate(order_all)}
    cands = [{"label": cell_label(BASELINE_CELL), "cell": list(BASELINE_CELL), "role": "baseline (V11AB)",
              "metric": float(metric[BASELINE_CELL]), "rank_among_all": rank_of[BASELINE_CELL],
              "invalid_blocks": int(invalid_total[BASELINE_CELL]), "replay_arg": "--attention-cell %d,%d" % BASELINE_CELL}]
    for i, c in enumerate(ranked):
        cands.append({"label": cell_label(c), "cell": list(c), "role": f"top-{i + 1}", "metric": float(metric[c]),
                      "rank_among_all": rank_of[c], "invalid_blocks": int(invalid_total[c]),
                      "replay_arg": "--attention-cell %d,%d" % c})
    complete = sorted((int(x["task_id"]), int(x["init"])) for x in episodes) == \
        sorted((t, i) for t in TASK_IDS for i in DEV_INIT_STATES)
    return {
        "step": STEP, "command": "select", "preregistration": preregistration_record(), "v11ab_cell": v11ab,
        "inputs": {"pairs": sources["pairs"], "phase_dirs": [str(pathlib.Path(d).resolve()) for d in phase_dirs],
                   "init_states": inits, "episodes": len(episodes), "complete_dev_set_50": bool(complete)},
        "n_call_camera_pairs": int(n_pairs),
        "grid_shape": list(metric.shape),
        "metric": metric.tolist(),
        "invalid_blocks": invalid_total.tolist(),
        "metric_nanmean_valid_only": (sum_valid / np.maximum(n_valid, 1)).tolist(),
        "metric_recorded_segmentation": (total_rec / n_pairs).tolist(),
        "top_cells_recorded_segmentation": [list(c) for c in rank_cells(total_rec / n_pairs)],
        "baseline_metric_from_float32_cell": total_f32_baseline / n_pairs,
        "candidates": cands,
        "candidate_cells": [c["cell"] for c in cands],
        "episodes": episodes,
    }


# --- evaluate ------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class RunInfo:
    run_dir: pathlib.Path
    settings: dict
    label: str
    cell: Optional[tuple[int, int]]
    source: str
    mapping: str
    grid_array: Optional[str]
    grounding_only: bool

    @property
    def privileged(self) -> bool:
        return self.source == "oracle"


def read_run(run_value: str) -> RunInfo:
    from benchmark.runners import attention_audit as aa

    run_dir = aa.resolve_source_run(run_value)
    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    settings = manifest.get("settings", {})
    opts = settings.get("options", {})
    att = opts.get("attention")
    mapping = opts.get("attention_mapping", "unmirror")
    if att is None:  # SL4 runs (before SL4b): the V11AB float32 cell
        v = check_baseline_cell()
        cell, source, grid = (v["layer"], v["head"]), "vla", "cell_grid_policy"
    else:
        source = att["source"]
        cell = None if att.get("cell") is None else (int(att["cell"]["layer"]), int(att["cell"]["head"]))
        grid = att.get("grid_array")
    label = cell_label(cell)
    if source == "vla" and grid == "pooled_f16" and cell == BASELINE_CELL:
        label += "-f16"
    if mapping != "unmirror":
        label += "-" + mapping
    return RunInfo(run_dir, settings, label, cell, source, mapping, grid, bool(opts.get("grounding_only")))


def _episode_index(info: RunInfo) -> list[dict]:
    rows = [json.loads(l) for l in (info.run_dir / "perception_index.jsonl").read_text().splitlines() if l.strip()]
    sources = info.settings.get("sources", {})
    for r in rows:
        if not r.get("dev_run_dir"):
            r["dev_run_dir"] = (sources.get("dev") or {}).get("run_dir")
    return rows


class _GTCache:
    def __init__(self):
        self.scenes: dict[str, Any] = {}
        self.samples: dict[tuple, Any] = {}

    def scene(self, path: str, sha: Optional[str]):
        from benchmark.evaluators.libero_scene_gt import SceneGT

        if path not in self.scenes:
            gt = SceneGT(path)
            if sha and gt.sha256 != sha:
                raise ValueError(f"{path}: SHA-256 differs from the run's record")
            self.scenes[path] = gt
        return self.scenes[path]

    def sample(self, path: str, gt, cfg: Mapping[str, Any]):
        key = (path, float(cfg["spacing"]), int(cfg["cap"]), int(cfg["seed"]))
        if key not in self.samples:
            self.samples[key] = gt.samples(float(cfg["spacing"]), int(cfg["cap"]), int(cfg["seed"]))
        return self.samples[key]


def _ragged(P, key, k):
    o = P[key + "_offsets"]
    return P[key][o[k]:o[k + 1]]


def evaluate_run(info: RunInfo, phase_dirs, cache: _GTCache, *, with_cause: bool = False) -> list[dict]:
    """Per-call rows (evaluation phases only) of one replay run."""
    from benchmark.evaluators import libero_phases as lp

    cfg = (info.settings.get("options") or {}).get("gt_samples") or {"spacing": 0.004, "cap": 3000, "seed": 0}
    rows = []
    for ep in _episode_index(info):
        if not ep.get("gt"):
            raise ValueError(f"{info.run_dir.name}: episode {ep['stem']} has no scene GT (run with --scene-gt)")
        gt = cache.scene(ep["gt"]["path"], ep["gt"].get("sha256"))
        S = cache.sample(ep["gt"]["path"], gt, cfg)
        names = [x.name for x in gt.instances()]
        target = gt.target_info["target"]
        ti = names.index(target)
        with np.load(info.run_dir / ep["perception_npz"]) as z:
            P = {k: z[k] for k in z.files if not k.startswith(("cloud_", "slice_", "gt_esdf", "gt_tier", "sph_exact",
                                                                  "sph_inside", "sph_esdf", "sph_tier"))}
        if not (np.array_equal(S.points_local.astype(np.float32), P["gt_points_local"])
                and np.array_equal(S.instance, P["gt_point_instance"])):
            raise ValueError(f"{info.run_dir.name}/{ep['stem']}: GT samples differ from the run's")
        obs = np.asarray(P["obs_index"]).astype(np.int64)
        labels, pinfo, track = episode_phases(phase_dirs, ep["stem"], obs, ep.get("geom_npz_sha256"),
                                              segmentation=with_cause)
        grids = P.get("attention_grid_policy")
        robot_r = P.get("sph_all_radius")
        for k in range(len(obs)):
            if labels[k] not in EVAL_PHASES:
                continue
            row = gt.frame_row(int(obs[k]))
            Pb = gt.points_base(row, S)
            vis = np.zeros(len(Pb), bool)
            for cam in ("head", "left_wrist"):
                res = np.asarray(P[f"gt_depth_residual_{cam}"][k], np.float64)
                vis |= np.abs(np.nan_to_num(res, nan=1.0)) <= VIS_TOL_M
            gtv = Pb[(S.instance == ti) & vis]
            tp = _ragged(P, "target_points", k).astype(np.float64)
            has = bool(P["has_target"][k])
            if has and len(tp):
                D = gt.exact_distance_near(tp, row, max_distance=NEAR_M)
                rob = None
                if "sph_all_center" in P and robot_r is not None:
                    c = np.asarray(P["sph_all_center"][k], np.float64)
                    rob = (np.linalg.norm(tp[:, None] - c[None], axis=2) - np.asarray(robot_r)[None]).min(1)
                plab = label_points(D, names, rob)
                dt = D[:, ti]
            else:
                plab, dt = [], np.zeros(0)
            m = call_metrics(has_target=has, target_name=target, target_points=tp,
                             target_centroid=P["target_centroid"][k], point_labels=plab, dist_to_target=dt,
                             gt_target_visible=gtv)
            m.update({"run": info.label, "stem": ep["stem"], "task": int(ep["task_id"]),
                      "init": int(ep.get("episode_idx", -1)), "call_k": int(P["call_k"][k]), "obs_index": int(obs[k]),
                      "phase": str(labels[k]), "status": str(P["grounding_status"][k]),
                      "attention_invalid": bool(P["attention_invalid"][k]) if "attention_invalid" in P else False})
            if with_cause and m["category"] == "wrong_instance":
                if grids is None:  # SL4 run: the float32 V11AB grids of the audit episode
                    with np.load(ep["attn_npz"]) as a:
                        grids = np.asarray(a["cell_grid_policy"], np.float32)
                tg = gt.target_info["instances_geom_ids"][target]
                cov = np.stack([target_coverage(lp.segmentation_at(track, int(obs[k]), cam), tg,
                                                unmirror=info.mapping == "unmirror")
                                for cam in ("head", "left_wrist")])
                m["cause"] = attention_cause(grids[k], cov)
            rows.append(m)
    return rows


def majority_kind(majority: Optional[str]) -> str:
    """``other_instance`` (a scene instance that is not the target), ``none`` / ``robot`` or ``tie``."""
    if majority is None:
        return "tie"
    if majority in ("none", "robot"):
        return majority
    return "other_instance"


def cause_split(rows: Sequence[Mapping[str, Any]]) -> dict:
    wrong = [r for r in rows if r["category"] == "wrong_instance"]
    n = len(wrong)
    att = sum(1 for r in wrong if r["cause"]["cause"] == "attention")
    by_kind: dict[str, dict] = {}
    for r in wrong:
        d = by_kind.setdefault(majority_kind(r["majority"]), {"misselections": 0, "attention": 0, "grounding": 0})
        d["misselections"] += 1
        d[r["cause"]["cause"]] += 1
    by_task: dict[str, dict] = {}
    for r in wrong:
        d = by_task.setdefault(str(r["task"]), {"misselections": 0, "attention": 0, "grounding": 0, "majority": {}})
        d["misselections"] += 1
        d[r["cause"]["cause"]] += 1
        d["majority"][r["majority"] or "(tie)"] = d["majority"].get(r["majority"] or "(tie)", 0) + 1
    return {"definition": OPERATIONAL_DEFINITIONS["cause_split"], "misselections": n,
            "attention_cause": att, "grounding_cause": n - att,
            "attention_cause_frac": att / n if n else None, "grounding_cause_frac": (n - att) / n if n else None,
            "head_argmax_on_target": sum(1 for r in wrong if r["cause"]["head_argmax_on_target"]),
            "wrist_argmax_on_target": sum(1 for r in wrong if r["cause"]["wrist_argmax_on_target"]),
            # reading "argmax(head·wrist)" per camera instead (reported, not the operational definition)
            "alt_either_camera_on_target": sum(1 for r in wrong if r["cause"]["head_argmax_on_target"]
                                               or r["cause"]["wrist_argmax_on_target"]),
            "alt_both_cameras_on_target": sum(1 for r in wrong if r["cause"]["head_argmax_on_target"]
                                              and r["cause"]["wrist_argmax_on_target"]),
            "argmax_camera": {c: sum(1 for r in wrong if r["cause"]["argmax_camera"] == c) for c in ("head", "left_wrist")},
            "by_majority_kind": by_kind,
            "by_task": by_task}


def evaluate(run_values, phase_dirs, selection: Mapping[str, Any]) -> tuple[dict, list[dict]]:
    """All runs -> per-run summaries, adoption decision, cause split, oracle reference."""
    check_baseline_cell()
    infos = [read_run(v) for v in run_values]
    labels = [i.label for i in infos]
    if len(set(labels)) != len(labels):
        raise ValueError(f"two runs with the same attention label: {labels}")
    base = [i for i in infos if i.source == "vla" and i.cell == BASELINE_CELL and i.grid_array == "cell_grid_policy"
            and i.mapping == "unmirror"]
    if len(base) != 1:
        raise ValueError(f"exactly one L8H2 (float32, un-mirror) run is needed, got {[i.label for i in base]}")
    base = base[0]
    wanted = [tuple(c) for c in selection["candidate_cells"] if tuple(c) != BASELINE_CELL]
    have = {i.cell: i for i in infos if i.source == "vla" and i.grid_array == "pooled_f16" and i.mapping == "unmirror"}
    missing = [c for c in wanted if c not in have]
    if missing:
        raise ValueError(f"no replay run for selected cells {missing}")
    candidates = [have[c] for c in wanted]
    references = [i for i in infos if i is not base and i not in candidates]
    cache = _GTCache()
    all_rows: list[dict] = []
    per_run: dict[str, dict] = {}
    keyset = None
    for info in [base, *candidates, *references]:
        rows = evaluate_run(info, phase_dirs, cache, with_cause=info is base)
        keys = [(r["stem"], r["call_k"], r["obs_index"], r["phase"]) for r in rows]
        stems = {e["stem"] for e in _episode_index(info)}
        if selection.get("episodes") and stems != {e["stem"] for e in selection["episodes"]}:
            raise ValueError(f"{info.label}: episodes differ from the selection's "
                             f"({len(stems)} vs {len(selection['episodes'])})")
        if keyset is None:
            keyset = keys
        elif keys != keyset:
            raise ValueError(f"{info.label}: evaluation calls differ from {base.label}")
        summary = aggregate(rows)
        summary["by_phase"] = {p: aggregate([r for r in rows if r["phase"] == p]) for p in EVAL_PHASES}
        manifest = info.run_dir / "run_manifest.json"
        per_run[info.label] = {
            "cell": None if info.cell is None else list(info.cell), "role": (
                "baseline" if info is base else "candidate" if info in candidates else
                "reference (privileged)" if info.privileged else "reference"),
            "privileged": info.privileged, "run_dir": str(info.run_dir), "run_id": info.run_dir.name,
            "manifest_sha256": _sha256(manifest), "grid_array": info.grid_array, "mapping": info.mapping,
            "grounding_only": info.grounding_only,
            "run_preregistration_sha256": (info.settings.get("preregistration") or {}).get("rows_sha256"),
            **summary}
        all_rows.extend(rows)
    decision = adoption_decision(per_run, base.label, [c.label for c in candidates])
    pairs = sorted({(int(r["task"]), int(r["init"])) for r in all_rows if r["run"] == base.label})
    complete = {(t, i) for t, i in pairs} == {(t, i) for t in TASK_IDS for i in DEV_INIT_STATES}
    out = {
        "step": STEP, "command": "evaluate", "preregistration": preregistration_record(),
        "selection": {"candidate_cells": selection["candidate_cells"],
                      "sha256": selection.get("_sha256"), "path": selection.get("_path")},
        "phase_dirs": [str(pathlib.Path(d).resolve()) for d in phase_dirs],
        "evaluation_calls": len(keyset or []),
        "complete_dev_set_50": bool(complete),
        "tasks_with_evaluation_calls": sorted({t for t, _ in pairs}),
        "runs": per_run,
        "adoption": decision,
        "cause_split_L8H2": cause_split([r for r in all_rows if r["run"] == base.label]),
        "oracle_reference": {k: {"primary": v["primary"], "per_task": v["per_task"]} for k, v in per_run.items()
                             if v["privileged"]},
    }
    return out, all_rows


# --- CLI -------------------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("select", help="selection metric over dev attention -> top-16 + L8H2")
    s.add_argument("--source-run", action="append", required=True)
    s.add_argument("--attention-run", action="append", required=True)
    s.add_argument("--phases", action="append", required=True, help="phase_export / SL3 gt directory (repeatable)")
    s.add_argument("--out", required=True)
    e = sub.add_parser("evaluate", help="pre-registered metric, adoption rule, cause split")
    e.add_argument("--selection", required=True, help="select output JSON")
    e.add_argument("--run", action="append", required=True, help="perception replay run (repeatable)")
    e.add_argument("--phases", action="append", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--rows-out", default=None, help="per-call rows (JSON lines)")
    return parser


def _write_new(path: str, text: str) -> None:
    with open(path, "x") as stream:
        stream.write(text)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        out = select(args.source_run, args.attention_run, args.phases)
        _write_new(args.out, json.dumps(_jsonable(out), indent=1, ensure_ascii=False) + "\n")
        print(json.dumps({"candidates": [(c["label"], round(c["metric"], 4), c["rank_among_all"])
                                         for c in out["candidates"]]}, ensure_ascii=False))
        return 0
    selection = json.loads(pathlib.Path(args.selection).read_text())
    if (selection.get("preregistration") or {}).get("rows_sha256") != _rows_sha256():
        raise ValueError("the selection was made under a different pre-registration")
    selection["_sha256"], selection["_path"] = _sha256(args.selection), str(pathlib.Path(args.selection).resolve())
    out, rows = evaluate(args.run, args.phases, selection)
    _write_new(args.out, json.dumps(_jsonable(out), indent=1, ensure_ascii=False) + "\n")
    if args.rows_out:
        _write_new(args.rows_out, "".join(json.dumps(_jsonable(r), ensure_ascii=False) + "\n" for r in rows))
    print(json.dumps({"adoption": {k: out["adoption"][k] for k in ("baseline", "baseline_correct", "n", "satisfying",
                                                                   "selected_by_rule", "outcome")},
                      "primary": {k: v["primary"] for k, v in out["runs"].items()}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())

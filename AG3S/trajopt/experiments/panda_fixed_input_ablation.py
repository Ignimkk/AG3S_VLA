"""SL10 §2.1 — 기록된 Panda call 의 **입력을 고정**하고 TrajOpt 의 항을 하나씩 끄며 다시 푼다 (J1 · J2).

RB-Y1 의 `fixed_input_ablation.py` 를 Panda 로 옮긴 짝 도구다 (그 module 의 field 복원 · solver 래퍼를 import 해 쓴다).
이 도구는 서버 · 기록기 · adapter 를 **바꾸지 않는다** (읽기만 한다).

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} .venv-ag3s/bin/python \\
        -m AG3S.trajopt.experiments.panda_fixed_input_ablation \\
        --run-dir ${AG3S_STORAGE_ROOT}/results/benchmark/<run> --out ${AG3S_RUN_DIR}/j2.json

`<run>` = `--record-constraints full` 로 돈 vla_v11ab 실행 하나 (`server_constraints/run_*/chunk_*.npz`,
`episodes/<stem>_calls.{jsonl,npz}` · `<stem>_geom.npz` · `<stem>.npz`, `run_manifest.json`, `logs/panda_server_config.json`).

## call 한 개의 입력은 어디서 복원하나

| 입력 | 어디서 |
|---|---|
| reference chunk | `chunk_*.npz` 의 `reference_chunk`. 위에 더해, `calls.npz` 의 `actions_policy` · `q_now` · `T_world_base` · `prev_step_world`
  로 변환 forward (`action.delta_chunk_to_joint_chunk`) 를 다시 돌려 **같은 chunk 가 나오는지** 본다 (충실도 A) |
| `q_now` | `<stem>_geom.npz` 의 `robot_state/left_wrist[obs_index]` (요청의 `panda/robot_state` 와 같은 값; 기록기가 q_now 를 따로 싣지 않는다) |
| `prev_step_world` | `<stem>.npz` 의 `obs_robot0_eef_pos/quat[obs_index], [obs_index-1]` (클라이언트 `_request` 와 같은 식) |
| 로봇 모델 | 서버 argv (`run_manifest.json`) → `--links` · `--palm-shell` → `model.build_panda_constraint_model` (Panda adapter 가 쓰는 것과 같은 호출; Panda 는 `sphere_options` 를 적용하지 않는다) |
| TrajOpt config | 같은 argv → `serve_safe.trajopt_config_from_args` → `server._replace_horizon_limits` (서버와 같은 함수) |
| 거리장 | `esdf_coarse_*` · `esdf_*`(fine) · `esdf_free_*` → `CuroboEsdfField` |
| 접촉 정책 | 기록의 `summary.target_permission` (mask · authorized links · held depth) + Panda `AG3SConfig` 의 `ClearancePolicy` (마진) → **실제** `linearize.scene_from_constraint_set` 에 모조 constraint_set 을 먹인다 |
| context | 기록의 `exec_continuity` · `exec_feedback` · `to.metrics` (`continuity_off_by`, `deflection_rate_off_by`, `joint_track_scale`) 와 직전 call 의 chunk · 실행 action |

## 재현하지 **못하는** 것 (기록에 없다)

* **쥔 물체의 부착 구** (`SceneSnapshot.attached_*`). 기록기가 싣지 않는다. 쥠(held) call 의 replay 는 이 구멍 때문에
  충실도 검사에서 걸러진다. `--attached-spheres-json` 으로 구를 직접 줄 수 있다 (parent link 좌표계의 `[cx, cy, cz, r]`).
* 서버가 `SceneSnapshot` 에 후처리로 붙이는 것: `neighbour_contact` · `neighbour_palm_contact` · `grasp_contact` ·
  `placement_contact` · HSR6 gate · `held_capture_spheres`. 켠 판에서 이 항들이 해를 바꿨다면 충실도 검사가 잡는다.
* 라벨 층 (목적지 마진 · "가장 가까운 표면이 target 인가").
* QP 의 call 간 warm start (OSQP). call 마다 새 solver 로 푼다 (episode 를 이어 푸는 chain 은 구현하지 않았다 — impl.md).
* `manipulated.points` — 기록의 `target_points` 로 대신한다 (occluded 때 마지막 관측 형상이 아닐 수 있다).

## 충실도 (먼저 하는 것)

call 마다 넷을 잰다. 하나라도 `--fidelity-tol-mm`(기본 1 mm) 를 넘으면 그 call 은 무효다.

* A. reference 재현 — 변환 forward 를 다시 돌려 `reference_chunk` 와 손끝 위치를 비교 (q_now · prev_step 복원의 검증)
* B. 기록 clearance 재현 — 기록의 refined 위에서 `field.distance − r − esdf_margin` 을 다시 재 `clearance` 와 비교
* C. `replay` — 기록과 같은 입력으로 TrajOpt 를 다시 풀어 기록의 `refined_chunk` 와 손끝 위치를 비교
* D. 변환 C 재현 — 기록의 refined · reference 에 변환 C 를 다시 걸어 기록의 `conversion.saturated_rows` ·
  `corrected_rows` 가 같은지

기본(`--strict`)은 하나라도 무효면 **변형을 하나도 돌리지 않고** 결과 JSON 에 `"valid": false` 와 이유를 쓰고 종료 코드 2 로
끝난다. `--no-strict` 는 무효 call 만 건너뛰고 나머지를 돈다 (JSON 의 `valid` 는 여전히 무효 call 이 있으면 false).

## 변형

| 이름 | target (field) | 연속성 · 꺾임 변화율 항 | 부착 구 |
|---|---|---|---|
| `replay` | 기록대로 | 기록대로 | 기록대로 |
| `no_target` (a) | **field 에서 뺌** (target 점 근처 점유 voxel 을 지우고 EDT 로 거리장 재계산) | 〃 | 〃 |
| `no_rate_terms` (b) | 〃 | `w_continuity = w_deflection_rate = 0` | 〃 |
| `no_attached` (c) | 〃 | 〃 | 끔 |
| `no_target_no_rate_terms_no_attached` (a+b+c) | 뺌 | 끔 | 끔 |
| `action_box` | 〃 | 〃 | 〃 + action 한계 제약 (worktree `wip/sl10-actionbox` 의 옵션이 있을 때만; 없으면 `unavailable`) |

변형마다: TrajOpt − reference 편차 (rad, max · p90; 전체 창 · 실행 창), 변환 C 뒤 포화 행 · 성분 수, 계획 − 실현 가능
차이 (mm, 실행 창 행별 max · p90). 변환 C 는 `adapters/panda/action.py` 를 그대로 부른다.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import datetime
import json
import pathlib
import sys
import time
import types
from typing import Any, Callable, Optional, Sequence

import numpy as np

TOOL = "AG3S.trajopt.experiments.panda_fixed_input_ablation"

#: 쥠 상태 구간 (SL10 진단과 같은 정의: `grasp.state` 가 closing / held 면 그것, 아니면 닫기 시작을 본 적이 있나로 가른다).
PHASES = ("before", "closing", "held", "after_release")
#: J1 · J2 의 "크게 벗어난 call" 문턱 (SL10 task §1.2: reference 에서 0.1 rad 넘게).
DEV_THRESHOLD_RAD = 0.1
#: `--budget-clock` 없이 반복 수만 묶는다 (RB-Y1 도구와 같다).
NO_CLOCK_MS = 1e9
#: SL10 task §1.3: 변형 X 가 J1 대상 call 의 (편차 > 0.1 rad) 비율을 절반 아래로 줄이면 "주 계기".
MAIN_INSTRUMENT_RATIO = 0.5


# ======================================================================================
# 변형 — 켜고 끄는 항
# ======================================================================================

@dataclasses.dataclass(frozen=True)
class Variant:
    """한 변형이 **끄는 항**. 이름이 말하는 항만 바뀌고 나머지는 기록대로다."""

    name: str
    target_removed: bool = False
    rate_terms_off: bool = False
    attached_off: bool = False
    action_box: bool = False
    label: str = ""


VARIANTS: dict[str, Variant] = {v.name: v for v in (
    Variant("replay", label="기록 재현"),
    Variant("no_target", target_removed=True, label="(a) target 을 field 에서 뺌"),
    Variant("no_rate_terms", rate_terms_off=True, label="(b) 연속성 · 꺾임 변화율 항 끔"),
    Variant("no_attached", attached_off=True, label="(c) 부착 구 끔"),
    Variant("no_target_no_rate_terms_no_attached", target_removed=True, rate_terms_off=True, attached_off=True,
            label="(a+b+c)"),
    Variant("action_box", action_box=True, label="action 한계 제약 켬 (wip/sl10-actionbox)"),
)}
DEFAULT_VARIANTS = tuple(VARIANTS)

#: 변형이 건드리는 cost 가중치 (b). 이 둘 외의 cost 항은 바뀌지 않는다.
RATE_TERM_WEIGHTS = ("w_continuity", "w_deflection_rate")


def variant_cost_overrides(variant: Variant) -> dict[str, Any]:
    """`TrajOptConfig.with_overrides` 에 줄 dict. (b) 가 아니면 `{}` — 기록의 cost 그대로."""
    if not variant.rate_terms_off:
        return {}
    return {"cost": {name: 0.0 for name in RATE_TERM_WEIGHTS}}


#: 부착 구를 한 덩어리로 나르는 `SceneSnapshot` 필드. (c) 는 이 셋만 비운다.
ATTACHED_FIELDS = ("attached_points", "attached_parent_link", "attached_radii")
#: (a) 가 비우는 필드: target 을 가리키는 질의 · 마진 · 마스크. 거리장 자체는 호출자가 바꿔 넣는다.
TARGET_FIELDS = ("manipulated_points", "manipulated_spheres", "manipulated_sphere_radii", "manipulated_link_margin",
                 "target_free_mask", "target_free_exempt", "target_contact_rows")


def apply_variant_scene(scene, variant: Variant, *, field_without_target=None):
    """`replay` 의 `SceneSnapshot` 에서 변형이 끄는 필드만 바꾼 새 scene. 원본은 바뀌지 않는다.

    `field_without_target`: (a) 가 거리장으로 넣을 target 없는 장 (`field_without_target`). (a) 인데 주지 않으면 죽는다
    — 조용히 아무것도 안 하는 변형은 "target 이 원인이 아니다" 로 읽히기 때문이다.
    """
    changes: dict[str, Any] = {}
    names = {f.name for f in dataclasses.fields(scene)}
    if variant.target_removed:
        if field_without_target is None:
            raise ValueError("variant removes the target but no target-free field was given")
        changes["esdf"] = field_without_target
        changes.update({k: None for k in TARGET_FIELDS if k in names})
    if variant.attached_off:
        changes.update({k: None for k in ATTACHED_FIELDS if k in names})
    return dataclasses.replace(scene, **changes) if changes else scene


# ======================================================================================
# (a) target 을 field 에서 뺀다
# ======================================================================================

def _edt_correction(distance: np.ndarray, voxel: float, target_voxels: np.ndarray) -> np.ndarray:
    """`d_new − d_old` (>= 0). 점유(`d < 0`) 중 `target_voxels` 를 자유로 돌린 뒤 EDT 로 다시 잰다.

    두 EDT 의 **차**만 더한다 — EDT 가 표면 voxel 에서 한 voxel 만큼 어긋나는 상수 오프셋이 상쇄되어, target 에서 먼 곳은
    `d_old` 그대로 남는다.
    """
    from scipy import ndimage

    occ = distance < 0.0
    occ_wo = occ & ~target_voxels
    all_d = ndimage.distance_transform_edt(~occ) * voxel
    wo_d = ndimage.distance_transform_edt(~occ_wo) * voxel
    out = distance.astype(np.float32) + (wo_d - all_d).astype(np.float32)
    # 지워진 voxel 자신: 음수였던 값을 새 거리(자유공간)로
    out[target_voxels & occ] = wo_d[target_voxels & occ].astype(np.float32)
    return out


def _target_voxel_mask(grid, target_points: np.ndarray, radius: float) -> np.ndarray:
    """격자에서 `target_points` 중 하나로부터 `radius` 이내인 voxel (bool, `grid.shape`)."""
    from scipy import ndimage

    shape = tuple(int(v) for v in grid.shape)
    mark = np.zeros(shape, bool)
    idx = np.rint((np.asarray(target_points, np.float64) - np.asarray(grid.origin, np.float64)) / grid.voxel_size)
    idx = idx.astype(np.int64)
    ok = np.all((idx >= 0) & (idx < np.asarray(shape)), axis=1)
    idx = idx[ok]
    if not len(idx):
        return mark
    mark[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return ndimage.distance_transform_edt(~mark) * grid.voxel_size <= float(radius)


#: (a) 가 target 점에서 이만큼(+ voxel 한 칸) 안의 점유 voxel 을 지운다 (SL9 의 표면 허용 10 mm 의 두 배).
TARGET_REMOVAL_RADIUS_M = 0.02


def field_without_target(field, target_points: np.ndarray, *, radius_m: Optional[float] = None):
    """`CuroboEsdfField` 의 거친 · 미세 계층에서 target 점 `radius_m` 이내의 점유 voxel 을 지우고 거리를 다시 잰 새 필드.

    **근사다.** 서버의 target-free 계층은 target seed 를 지우고 TSDF 부호를 다시 정하는데, 여기서는 저장된 SDF 의
    음수 voxel 을 점유로 보고 EDT 를 다시 한다. target 아래의 받침면 voxel 도 `radius_m` 안이면 같이 지워진다 (서버의
    소유권 규칙은 기록되지 않는다). 결과 필드는 target-free 계층 · 라벨이 없는 단순 합성이다.
    """
    from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

    pts = np.asarray(target_points, np.float64).reshape(-1, 3)
    radius_m = TARGET_REMOVAL_RADIUS_M if radius_m is None else float(radius_m)
    if not len(pts):
        raise ValueError("no target points: nothing to remove")
    layers = []
    for layer in field.layers:
        g = layer.grid
        mask = _target_voxel_mask(g, pts, radius_m + g.voxel_size)
        d = _edt_correction(np.asarray(layer.distance_grid, np.float32), float(g.voxel_size), mask)
        layers.append(layer_from_arrays(d, g.origin, g.voxel_size))
    return CuroboEsdfField(tuple(layers), outside_distance=field.outside_distance)


# ======================================================================================
# 지표 — 편차 · 변환 C 뒤 포화 · 계획 − 실현 가능 차이
# ======================================================================================

def deviation_metrics(chunk, reference, *, execution_length: int, arm_dof: int = 7) -> dict[str, Any]:
    """TrajOpt − reference 편차 (rad). 전체 계획 창과 실행 창(앞 `execution_length` 행)을 따로.

    max · p90 은 `|chunk − reference|` 의 (행 × 관절) 원소 전체에 대한 값이다. `exceeds` = 전체 창 max > 0.1 rad
    (SL10 §1.2 의 "크게 벗어난 call").
    """
    dev = np.abs(np.asarray(chunk, np.float64)[:, :arm_dof] - np.asarray(reference, np.float64)[:, :arm_dof])
    ex = dev[:int(execution_length)]
    return {"max_rad": float(dev.max()), "p90_rad": float(np.percentile(dev, 90)),
            "exec_max_rad": float(ex.max()), "exec_p90_rad": float(np.percentile(ex, 90)),
            "per_joint_max_rad": [float(v) for v in dev.max(axis=0)],
            "exceeds": bool(dev.max() > DEV_THRESHOLD_RAD)}


def conversion_metrics(kin, conv, refined_chunk, reference_chunk, *, execution_length: int,
                       mode: str = "C", response=None) -> dict[str, Any]:
    """변환 C 를 `action.py` 그대로 걸고 지표를 잰다. `refined` · `reference` 는 서버가 와이어로 보낸 모양(float32)이다.

    * `saturated_rows` — 변환 기록의 행 목록 (한 성분이라도 ±1 로 잘린 행)
    * `saturated_components` — 출력 action 중 |a| ≥ 1 인 (행, 성분) 수. VLA 의 원본이 이미 ±1 인 성분은 빼고 센다
    * `plan_minus_realisable_mm` — 실행 창 행마다 `|p_hat − p_star|` (응답 모델이 예측한 경로 − 계획 경로, 베이스 좌표)
    """
    from AG3S.adapters.panda import action as pa

    ref = np.asarray(reference_chunk, np.float32)
    new = np.asarray(refined_chunk, np.float32)
    actions, rec = pa.joint_chunk_to_delta_chunk(kin, conv, new, ref, response=response, mode=mode)
    K = int(execution_length)
    H = min(K, actions.shape[0], new.shape[0])
    p_hat = pa.predict_ee_path(kin, conv, actions[:H], response=response)
    p_star = np.asarray([kin.fk(np.asarray(new[k, :pa.ARM_DOF], np.float64))[0] for k in range(H)])
    err_mm = 1000.0 * np.linalg.norm(p_hat - p_star, axis=1)
    pol = np.abs(np.asarray(conv.delta, np.float64)[:, :6]) >= pa.OSC_INPUT_CLIP
    out_sat = np.abs(np.asarray(actions, np.float64)[:, :6]) >= pa.OSC_INPUT_CLIP
    new_sat = out_sat & ~pol
    return {
        "saturated_rows": [int(k) for k in rec["saturated_rows"]],
        "n_saturated_rows": int(len(rec["saturated_rows"])),
        "n_saturated_rows_exec": int(sum(1 for k in rec["saturated_rows"] if k < K)),
        "corrected_rows": [int(k) for k in rec["corrected_rows"]],
        "saturated_components": int(new_sat.sum()),
        "saturated_components_exec": int(new_sat[:K].sum()),
        "saturated_component_fraction_exec": float(new_sat[:K].mean()) if new_sat[:K].size else 0.0,
        "max_abs_correction": float(rec["max_abs_correction"]),
        "plan_minus_realisable_mm_per_row": [float(v) for v in err_mm],
        "plan_minus_realisable_mm_max": float(err_mm.max()) if len(err_mm) else 0.0,
        "plan_minus_realisable_mm_p90": float(np.percentile(err_mm, 90)) if len(err_mm) else 0.0,
        "tracking_residual_mm": float(rec.get("tracking_residual_mm", 0.0)),
    }


def variant_metrics(kin, conv, chunk, reference_chunk, *, execution_length: int, mode: str = "C") -> dict[str, Any]:
    """한 변형의 결과 chunk 에 대한 지표 묶음 (편차 + 변환 C)."""
    return {"deviation": deviation_metrics(chunk, reference_chunk, execution_length=execution_length),
            "conversion": conversion_metrics(kin, conv, chunk, reference_chunk, execution_length=execution_length,
                                             mode=mode)}


def ee_mm_diff(kin, chunk_a, chunk_b, rows: Optional[int] = None) -> float:
    """두 관절 chunk 의 손끝(grip site) 위치 차이 최대값 (mm). 충실도 비교에 쓴다."""
    a, b = np.asarray(chunk_a, np.float64)[:, :7], np.asarray(chunk_b, np.float64)[:, :7]
    n = min(len(a), len(b)) if rows is None else min(int(rows), len(a), len(b))
    return float(max((np.linalg.norm(kin.fk(a[k])[0] - kin.fk(b[k])[0]) for k in range(n)), default=0.0) * 1000.0)


# ======================================================================================
# 충실도 판정과 변형 실행 (solver 를 주입받는다 — 합성 입력으로 시험할 수 있다)
# ======================================================================================

FIDELITY_KEYS = ("reference_mm", "record_clearance_mm", "replay_mm")


def fidelity_verdict(f: dict[str, Any], tol_mm: float) -> dict[str, Any]:
    """`f` = `{reference_mm, record_clearance_mm, replay_mm, conversion_reproduced}`. 재지 못한 값(None)은 무효로 센다."""
    checks = {}
    for key in FIDELITY_KEYS:
        v = f.get(key)
        checks[key] = bool(v is not None and np.isfinite(v) and v <= tol_mm)
    checks["conversion_reproduced"] = bool(f.get("conversion_reproduced"))
    bad = [k for k, ok in checks.items() if not ok]
    return {**f, "tolerance_mm": float(tol_mm), "checks": checks, "within_tolerance": not bad,
            "failed": bad}


def run_calls(calls: Sequence[Any], solve: Callable[[Any, Variant], dict], variants: Sequence[str], *,
              tol_mm: float, strict: bool = True) -> dict[str, Any]:
    """충실도(replay)를 먼저 모든 call 에서 잰 뒤 변형을 돈다.

    `calls[i]` 는 `.id` (str) 과 `.fidelity` (dict, `FIDELITY_KEYS`) 를 가진다. `solve(call, variant)` 는 그 변형의 결과
    dict 를 돌려주고, 못 돌리면 `{"unavailable": 이유}` 를 돌려준다. strict 이고 하나라도 무효면 **변형은 하나도 안 돈다**.
    """
    verdicts = {c.id: fidelity_verdict(c.fidelity, tol_mm) for c in calls}
    invalid = [c.id for c in calls if not verdicts[c.id]["within_tolerance"]]
    out: dict[str, Any] = {"valid": not invalid, "invalid_calls": invalid, "invalid_reason": None,
                           "fidelity": verdicts, "variants_run": [], "results": {}}
    if invalid:
        out["invalid_reason"] = (f"{len(invalid)} call(s) exceed the {tol_mm} mm fidelity tolerance "
                                 f"or could not be reproduced: " + ", ".join(invalid[:20]))
        if strict:
            return out
    todo = [v for v in variants if v != "replay"]
    for c in calls:
        if not verdicts[c.id]["within_tolerance"]:
            continue
        res = {"replay": solve(c, VARIANTS["replay"])}
        for name in todo:
            res[name] = solve(c, VARIANTS[name])
        out["results"][c.id] = res
    out["variants_run"] = ["replay"] + todo
    return out


def instrument_summary(results: dict[str, dict], *, variants: Sequence[str]) -> dict[str, Any]:
    """SL10 §1.3 의 판정 재료. 변형별로 (편차 > 0.1 rad) 인 call 의 비율을 `replay` 의 비율과 나란히 적는다.

    "주 계기" 는 변형 X 의 비율 < `MAIN_INSTRUMENT_RATIO` × replay 의 비율 일 때 (절반 **아래**). 여러 계기가 각각 일부만
    설명하면 비율을 그대로 적는다 — 하나로 묶지 않는다. 못 돌린(`unavailable`) call 은 분모에서 뺀다 (개수를 같이 적는다).
    """
    out: dict[str, Any] = {}
    base = [r.get("replay") for r in results.values()]
    base = [r for r in base if r and "deviation" in r]
    f0 = float(np.mean([r["deviation"]["exceeds"] for r in base])) if base else None
    for name in variants:
        rs = [r.get(name) for r in results.values()]
        ok = [r for r in rs if r and "deviation" in r]
        na = sum(1 for r in rs if not (r and "deviation" in r))
        if not ok:
            out[name] = {"n": 0, "n_unavailable": na}
            continue
        frac = float(np.mean([r["deviation"]["exceeds"] for r in ok]))
        # replay 의 비율은 같은 call 들에서 다시 센다 (변형이 일부 call 에서 unavailable 일 수 있다)
        same = [results[cid]["replay"] for cid, r in results.items() if r.get(name) and "deviation" in r[name]]
        f_same = float(np.mean([r["deviation"]["exceeds"] for r in same])) if same else None
        out[name] = {
            "n": len(ok), "n_unavailable": na, "fraction_dev_gt_0p1": frac, "replay_fraction_same_calls": f_same,
            "ratio_to_replay": (frac / f_same) if f_same else None,
            "reduces_below_half": bool(f_same and frac < MAIN_INSTRUMENT_RATIO * f_same) if name != "replay" else None,
            "dev_max_rad_median": float(np.median([r["deviation"]["max_rad"] for r in ok])),
            "dev_p90_rad_median": float(np.median([r["deviation"]["p90_rad"] for r in ok])),
            "saturated_rows_exec_mean": float(np.mean([r["conversion"]["n_saturated_rows_exec"] for r in ok])),
            "plan_minus_realisable_mm_p90_median": float(np.median(
                [r["conversion"]["plan_minus_realisable_mm_p90"] for r in ok])),
        }
    out["_replay_fraction_all"] = f0
    return out


# ======================================================================================
# 기록 읽기 — call 하나 = 클라이언트의 call 행 + 서버의 chunk npz
# ======================================================================================

def phase_of(state: Optional[str], seen_close: bool) -> str:
    """SL10 진단(`grasp_stats.py`)과 같은 구간 정의."""
    st = state or ""
    if st in ("closing", "held"):
        return st
    return "after_release" if seen_close else "before"


def rotvec_between(q1_xyzw: np.ndarray, q0_xyzw: np.ndarray) -> np.ndarray:
    """클라이언트 `_rotvec_between`: `Rot(q1) · Rot(q0)^-1` 의 회전 벡터 (world)."""
    from scipy.spatial.transform import Rotation as Rot

    return (Rot.from_quat(np.asarray(q1_xyzw, np.float64)) * Rot.from_quat(np.asarray(q0_xyzw, np.float64)).inv()).as_rotvec()


def prev_step_world(eef_pos: np.ndarray, eef_quat: np.ndarray, obs_index: int) -> np.ndarray:
    """요청의 `panda/prev_step_world` (6,) — 클라이언트 `_request` 와 같은 식. 관측이 하나뿐이면 0."""
    if obs_index < 1:
        return np.zeros(6)
    p1, p0 = np.asarray(eef_pos[obs_index], np.float64), np.asarray(eef_pos[obs_index - 1], np.float64)
    return np.concatenate([p1 - p0, rotvec_between(eef_quat[obs_index], eef_quat[obs_index - 1])])


@dataclasses.dataclass
class Call:
    """한 policy call: 클라이언트 행 + (있으면) 서버 chunk 기록. `id` = `<stem>#<seq>`."""

    stem: str
    seq: int
    index: int                    # episode 안의 call 순번 (calls.npz 의 첫 축)
    obs_index: int
    phase: str
    row: dict
    q_now: np.ndarray             # (7,)
    T_world_base: np.ndarray      # (4, 4)
    prev_step: np.ndarray         # (6,)
    actions_policy: np.ndarray    # (H, 7) float32
    joint_reference: np.ndarray   # (H, 8) float32, 와이어에서 받은 값
    joint_refined: np.ndarray     # (H, 8) float32
    actions_executed: np.ndarray  # (n, 7)
    chunk_path: Optional[pathlib.Path] = None
    #: 같은 episode 의 직전 call (seq − 1). 연속성 · 꺾임 변화율 context 를 짓는 데 쓴다.
    prev: Optional["Call"] = None

    @property
    def id(self) -> str:
        return f"{self.stem}#{self.seq}"


def load_episode_calls(episodes_dir: pathlib.Path, stem: str) -> list[Call]:
    """`<stem>_calls.jsonl` + `_calls.npz` + `_geom.npz` + `<stem>.npz` → `Call` 목록 (chunk 연결은 `attach_chunks`)."""
    from AG3S.adapters.panda import model as pm

    d = pathlib.Path(episodes_dir)
    rows = [json.loads(line) for line in (d / f"{stem}_calls.jsonl").read_text().splitlines() if line.strip()]
    calls_z = np.load(d / f"{stem}_calls.npz")
    geom = np.load(d / f"{stem}_geom.npz")
    ep = np.load(d / f"{stem}.npz")
    rs_key = "robot_state/left_wrist" if "robot_state/left_wrist" in geom.files else "robot_state/head"
    robot_state, eef_pos, eef_quat = geom[rs_key], ep["obs_robot0_eef_pos"], ep["obs_robot0_eef_quat"]
    pol, ref, red, exe, T = (calls_z[k] for k in ("actions_policy", "joint_reference", "joint_refined",
                                                 "actions_executed", "T_world_base"))
    out: list[Call] = []
    seen = False
    for i, r in enumerate(rows):
        st = (r.get("grasp") or {}).get("state")
        if st == "closing":
            seen = True
        oi = int(r["obs_index"])
        q, _ = pm.split_robot_state(robot_state[oi])
        out.append(Call(stem=stem, seq=int(r["seq"]), index=i, obs_index=oi, phase=phase_of(st, seen), row=r,
                        q_now=q, T_world_base=np.asarray(T[i], np.float64),
                        prev_step=prev_step_world(eef_pos, eef_quat, oi),
                        actions_policy=np.asarray(pol[i], np.float32), joint_reference=np.asarray(ref[i], np.float32),
                        joint_refined=np.asarray(red[i], np.float32), actions_executed=np.asarray(exe[i], np.float64),
                        prev=out[-1] if out and out[-1].seq == int(r["seq"]) - 1 else None))
    return out


def chunk_key(reference: np.ndarray) -> bytes:
    """call ↔ chunk 짝짓기 키: reference chunk 의 float32 바이트. 서버 chunk npz 와 클라이언트 `joint_reference` 가 같다."""
    return np.ascontiguousarray(np.asarray(reference, np.float32)[:, :8]).tobytes()


def attach_chunks(calls: Sequence[Call], constraints_dir: pathlib.Path) -> dict[str, Any]:
    """각 `Call.chunk_path` 를 채운다. 못 짝지은 chunk · call 수를 돌려준다 (기록이 빠진 call 은 그냥 chunk 가 없다)."""
    by_key: dict[bytes, list[Call]] = {}
    for c in calls:
        by_key.setdefault(chunk_key(c.joint_reference), []).append(c)
    matched = 0
    unmatched_chunks = []
    for f in sorted(pathlib.Path(constraints_dir).glob("chunk_*.npz")):
        with np.load(f) as z:
            key = chunk_key(z["reference_chunk"])
            t_step = int(z["t_step"])
        cands = [c for c in by_key.get(key, []) if c.chunk_path is None]
        if len(cands) > 1:
            cands = [c for c in cands if c.seq == t_step] or cands
        if not cands:
            unmatched_chunks.append(f.name)
            continue
        cands[0].chunk_path = f
        matched += 1
    return {"matched": matched, "unmatched_chunks": unmatched_chunks,
            "calls_without_chunk": int(sum(1 for c in calls if c.chunk_path is None))}


# ======================================================================================
# 서버 설정 복원
# ======================================================================================

def server_args(run_dir: pathlib.Path, ag3s_version: Optional[str] = None):
    """실행의 서버 argv → `serve_safe` parser 의 Namespace. `run_manifest.json` 의 server execution 에서 읽는다."""
    from AG3S.trajopt import serve_safe as ss

    if ag3s_version:
        from AG3S.adapters.panda import profile

        tokens = (profile.version_server_argv(ag3s_version) + profile.version_server_flags(
            ag3s_version, **({} if not profile.is_final_version(ag3s_version) else {}))
            + ["--robot", "panda"])
        return ss.build_parser().parse_args(tokens)
    manifest = json.loads((pathlib.Path(run_dir) / "run_manifest.json").read_text())
    for ex in manifest.get("executions", []):
        argv = (ex.get("settings") or {}).get("argv")
        if (ex.get("settings") or {}).get("role") == "server" and argv and "AG3S.trajopt.serve_safe" in argv:
            tokens = list(argv[argv.index("AG3S.trajopt.serve_safe") + 1:])
            return ss.build_parser().parse_args(tokens)
    raise SystemExit("run_manifest.json 에 server argv 가 없습니다 — --ag3s-version 으로 버전을 주세요")


@dataclasses.dataclass
class Runtime:
    """call 마다 같은 것 — 모델 · config · 정책 · 측정기."""

    args: Any
    robot: Any
    kin: Any
    layout: Any
    to_config: Any
    ag3s_config: Any
    policy: Any
    lin: Any
    outside_distance: float
    esdf_margin: float
    execution_length: int
    runners: dict = dataclasses.field(default_factory=dict)
    attached_spheres: Optional[dict] = None


def build_runtime(args, *, attached_spheres: Optional[dict] = None) -> Runtime:
    """서버와 **같은 호출**로 로봇 모델 · TrajOpt config · AG3S config 를 다시 짓는다 (GPU · 체크포인트 불필요)."""
    from AG3S.adapters.panda import action as pa
    from AG3S.adapters.panda import model as pm
    from AG3S.adapters.panda import profile
    from AG3S.adapters.panda import server as ps
    from AG3S.ag3s.constraints.clearance import ClearancePolicy
    from AG3S.trajopt import serve_safe as ss
    from AG3S.trajopt.linearize import CollisionLinearizer

    shell = ss.palm_shell_options(args)
    robot = pm.build_panda_constraint_model(pm.panda_constraint_link_filter(args.links),
                                            **({"palm_shell": dict(shell)} if shell else {}))
    to_config = ps._replace_horizon_limits(ss.trajopt_config_from_args(args))
    kw = profile.v11ab_build_kwargs(args)
    cfg = profile.panda_ag3s_config(profile.v11ab_ag3s_config(kw))
    policy = ClearancePolicy.from_config(cfg.contact, cfg.geometry, cfg.support_surface)
    layout = pa.panda_chunk_layout(robot.joint_names)
    lin = CollisionLinearizer(robot, layout, to_config.horizon.planned)
    unknown = getattr(cfg.esdf, "unknown_policy", "free")
    outside = float(cfg.esdf.max_distance) if unknown == "free" else -float(cfg.esdf.max_distance)
    return Runtime(args=args, robot=robot, kin=pa.PandaKinematics(robot), layout=layout, to_config=to_config,
                   ag3s_config=cfg, policy=policy, lin=lin, outside_distance=outside,
                   esdf_margin=float(to_config.collision.esdf_margin),
                   execution_length=int(to_config.horizon.execution_length), attached_spheres=attached_spheres)


# ======================================================================================
# call 하나의 입력 복원 · 충실도
# ======================================================================================

def _literal(value):
    """기록기는 metrics 의 list · dict 를 `str()` 로 적는다 (`constraint_record._scalar`)."""
    if value is None or not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None


@dataclasses.dataclass
class Prepared:
    """한 call 의 복원된 입력. 변형은 전부 이것 위에서 돈다."""

    call: Call
    id: str
    summary: dict
    z: Any
    conv: Any
    chunk: np.ndarray             # TrajOpt 에 넣는 reference (H, 8) float64 — 변환 forward 의 결과
    recorded_refined: np.ndarray  # 기록의 refined_chunk (H, 8) float32
    field: Any
    scene: Any
    certified: bool
    attached: bool
    context: dict
    sqp_iterations: int
    target_points: Optional[np.ndarray]
    scene_info: dict
    fidelity: dict
    conv_mode: str = "C"
    _field_wo: Any = None

    def recorded_reference(self) -> np.ndarray:
        """기록의 reference_chunk (float32, 와이어에서 받은 모양)."""
        return np.asarray(self.z["reference_chunk"], np.float32)


def build_context(rt: Runtime, call: Call, summary: dict) -> tuple[dict, dict]:
    """서버가 `refiner.refine(chunk, context)` 에 준 context 를 기록으로 되짚는다. `(context, 근거)`.

    키는 서버가 넣은 것만 넣는다: `previous_physical_chunk` / `previous_executed_steps` (exec_continuity 의 mode),
    `continuity_off`, `deflection_rate_off`, `deflection_history`, `joint_track_scale`, `start_at_rest`.
    """
    from AG3S.adapters.panda import action as pa

    to = summary.get("to") or {}
    metrics = to.get("metrics") or {}
    ctx: dict[str, Any] = {"t_step": call.seq}
    why: dict[str, Any] = {}
    cont = summary.get("exec_continuity") or {}
    fb = summary.get("exec_feedback") or {}
    prev_call = call.prev
    prev_refined = None
    if prev_call is not None:
        prev_refined = np.asarray(prev_call.joint_refined, np.float64)
    mode = str(cont.get("mode") or "")
    if mode == "feedback" and prev_refined is not None:
        which = cont.get("previous_chunk")
        prev = prev_refined if which == "refined" else np.asarray(prev_call.joint_reference, np.float64)
        if which in ("refined", "reference"):
            ctx["previous_physical_chunk"] = prev
            ctx["previous_executed_steps"] = int(cont.get("n_exec", fb.get("n_exec", 0)) or 0)
        why["continuity"] = f"feedback: previous {which}, n_exec={cont.get('n_exec')}"
    elif mode.startswith("assumed") and prev_refined is not None:
        ctx["previous_physical_chunk"] = prev_refined
        why["continuity"] = "assumed: previous refined, K steps"
    else:
        why["continuity"] = f"no previous chunk ({mode or 'none'})"
    off = cont.get("off_by") or metrics.get("continuity_off_by")
    if off:
        ctx["continuity_off"] = str(off)
    roff = metrics.get("deflection_rate_off_by")
    if roff:
        ctx["deflection_rate_off"] = str(roff)
    scale = _literal(metrics.get("joint_track_scale"))
    if scale:
        ctx["joint_track_scale"] = np.asarray(scale, np.float64)
    rest = summary.get("rest_start") or {}
    if rest.get("applied"):
        ctx["start_at_rest"] = True
    # 꺾임 변화율 history: `SafePolicy._ta_history` 의 식 (applied_arm 은 직전 call 의 변환 + 실행된 OSC 행으로 다시 만든다)
    if float(getattr(rt.to_config.cost, "w_deflection_rate", 0.0) or 0.0) > 0.0 and not roff:
        hist, note = _deflection_history(rt, call, fb)
        if hist is not None:
            ctx["deflection_history"] = hist
        why["deflection_history"] = note
    _ = pa
    return ctx, why


def _deflection_history(rt: Runtime, call: Call, fb: dict) -> tuple[Optional[np.ndarray], str]:
    from AG3S.adapters.panda import action as pa

    prev = call.prev
    if not fb.get("available") or prev is None:
        return None, "no exec_feedback / no previous call"
    executed = list(fb.get("executed") or [])
    n_steps = int(fb.get("n_steps", 0) or 0)
    if not executed or not executed[-1]:
        return None, "last row was a HOLD or no rows"
    osc = np.asarray(prev.actions_executed, np.float64)[:n_steps]
    if osc.shape[0] != n_steps or n_steps == 0:
        return None, f"previous call executed {prev.actions_executed.shape[0]} rows, feedback n_steps={n_steps}"
    conv_prev = pa.delta_chunk_to_joint_chunk(rt.kin, prev.actions_policy, prev.q_now, prev.T_world_base,
                                              prev_step_world=(prev.prev_step[:3], prev.prev_step[3:6]))
    arm = pa.applied_command_rows(rt.kin, conv_prev, osc)
    row = len(executed) - 1
    ref = np.asarray(prev.joint_reference, np.float64)[:, :pa.ARM_DOF]
    if row >= ref.shape[0]:
        return None, f"row {row} is outside the previous reference"
    return arm[-1] - ref[row], f"applied_arm[-1] - previous reference[{row}]"


def build_scene(rt: Runtime, call: Call, z, summary: dict, field, *, attached: bool) -> tuple[Any, dict]:
    """모조 `constraint_set` 을 **실제** `scene_from_constraint_set` 에 먹여 `SceneSnapshot` 을 만든다."""
    from AG3S.ag3s.types import ContactPolicyContext, SourceType
    from AG3S.trajopt.linearize import scene_from_constraint_set
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.trajopt.grasp_latch import GraspPhase

    names = [str(n) for n in rt.robot.sphere_link_names]
    perm = summary.get("target_permission") or {}
    grasp_state = (call.row.get("grasp") or {}).get("state")
    phase = "approach"
    if getattr(rt.args, "latch_phase", False) and grasp_state:
        try:
            phase = SafePolicy.LATCH_CONTACT_PHASE.get(GraspPhase(grasp_state), phase)
        except ValueError:
            pass
    ctx = ContactPolicyContext.make(phase, ["left"])
    from AG3S.ag3s.constraints.clearance import target_permission_context

    perm_ctx = target_permission_context(ctx, str(perm.get("policy") or "relax"), object() if attached else None)
    authorized = sorted(set(rt.policy.authorized_links(perm_ctx)) & set(names))
    info: dict[str, Any] = {"contact_phase": phase, "authorized_links_rebuilt": authorized,
                            "authorized_links_recorded": perm.get("authorized_links"),
                            "authorized_match": perm.get("authorized_links") is None
                            or sorted(perm["authorized_links"]) == authorized}
    target_points = np.asarray(z["target_points"], np.float64) if "target_points" in z.files else None
    man = None
    margin = None
    attached_obj = None
    if attached:
        spec = rt.attached_spheres
        if spec:
            parent = str(spec["parent_link"])
            sph = np.asarray(spec["spheres"], np.float64).reshape(-1, 4)
            T = np.asarray(rt.robot.link_pose(np.asarray(call.q_now, np.float64), parent), np.float64)
            base = (T[:3, :3] @ sph[:, :3].T).T + T[:3, 3]
            man = types.SimpleNamespace(points=None, sphere_centers=base, sphere_radii=sph[:, 3])
            attached_obj = types.SimpleNamespace(points=sph[:, :3], parent_link=parent, primitives=(), label="supplied")
            info["attached"] = {"source": "--attached-spheres-json", "n": int(len(sph)), "parent_link": parent}
        else:
            info["attached"] = {"source": None, "missing": "held spheres are not in the record"}
    elif target_points is not None and bool(summary.get("has_target")):
        man = types.SimpleNamespace(points=target_points, sphere_centers=None, sphere_radii=None)
    if man is not None and (not attached or attached_obj is not None):
        margin = rt.policy.margin_matrix(names, [SourceType.TARGET], context=perm_ctx, target_grounded=True)[:, 0]
    mask = perm.get("target_field_exclude")
    mask = None if mask is None else np.asarray(mask, bool)
    nonauth = None
    nm = perm.get("nonauthorized_target_margin") or {}
    if nm.get("applied"):
        nonauth = np.asarray([n not in set(rt.policy.authorized_links(perm_ctx)) for n in names], bool)
    tf = (summary.get("target_exclusion") or {})
    field.stats = {"target_free": {"exclusion_method": tf.get("exclusion_method")}} if tf.get("exclusion_method") else {}
    cs = types.SimpleNamespace(
        constraints=None, esdf=field, manipulated=man, manipulated_link_margin=margin, attached=attached_obj,
        destination_label=None, destination_margin=None, target_field_exclude=mask,
        target_field_policy=str(perm.get("policy") or ""), nonauthorized_target_rows=nonauth,
        metrics={"target_permission": perm}, support_surfaces=())
    scene = scene_from_constraint_set(cs, rt.lin.robot_radii, rt.to_config, held_capture=None)
    if attached_obj is not None and rt.attached_spheres:
        sph = np.asarray(rt.attached_spheres["spheres"], np.float64).reshape(-1, 4)
        scene = dataclasses.replace(scene, attached_radii=sph[:, 3])
    return scene, info


def _solver(rt: Runtime, cfg) -> tuple[Any, dict]:
    """같은 config 는 refiner 를 재사용한다 (call 마다 `reset`)."""
    from AG3S.trajopt.refiner import TrajOptChunkRefiner

    key = json.dumps(cfg.to_dict(), sort_keys=True, default=str)
    hit = rt.runners.get(key)
    if hit is None:
        holder: dict[str, Any] = {"value": None}
        refiner = TrajOptChunkRefiner(rt.robot, rt.layout, lambda _ctx: holder["value"], cfg)
        hit = (refiner, holder)
        rt.runners[key] = hit
    return hit


def variant_config(rt: Runtime, variant: Variant, sqp_iterations: int, extra: Optional[dict] = None):
    """서버의 `to_config` 에 변형이 끄는 cost 가중치만 바꾸고, 예산은 기록의 반복 수로 묶는다."""
    overrides: dict[str, Any] = {"sqp": {"time_budget_ms": NO_CLOCK_MS, "max_iterations": max(1, int(sqp_iterations))}}
    for section, values in {**variant_cost_overrides(variant), **(extra or {})}.items():
        overrides[section] = {**overrides.get(section, {}), **values}
    return rt.to_config.with_overrides(overrides)


def solve(rt: Runtime, prep: Prepared, cfg, scene, context: Optional[dict] = None):
    refiner, holder = _solver(rt, cfg)
    refiner.reset()
    holder["value"] = (scene, np.asarray(prep.call.q_now, np.float64), prep.certified)
    refiner.refine(prep.chunk, dict(prep.context if context is None else context))
    result = refiner.last_result
    if result is None:
        raise RuntimeError(f"refiner returned no result: {refiner.last_failure}")
    return result


def prepare_call(rt: Runtime, call: Call, *, tol_mm: float) -> tuple[Optional[Prepared], dict]:
    """입력 복원 + 충실도 A · B · C · D. 복원할 수 없으면 `(None, 이유)`."""
    from AG3S.adapters.panda import action as pa
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field

    if call.chunk_path is None:
        return None, {"skipped_reason": "no constraint record for this call"}
    z = np.load(call.chunk_path, allow_pickle=False)
    summary = json.loads(str(z["summary_json"]))
    stats = summary.get("esdf_stats") or {}
    attached = int(stats.get("n_attached_points", 0) or 0) > 0 or bool((call.row.get("grasp") or {}).get("attached"))
    recorded_refined = np.asarray(z["refined_chunk"], np.float32)
    recorded_reference = np.asarray(z["reference_chunk"], np.float32)
    fid: dict[str, Any] = {}
    # A. reference 재현
    conv = pa.delta_chunk_to_joint_chunk(rt.kin, call.actions_policy, call.q_now, call.T_world_base,
                                         prev_step_world=(call.prev_step[:3], call.prev_step[3:6]))
    fid["reference_mm"] = ee_mm_diff(rt.kin, conv.joint_chunk.astype(np.float32), recorded_reference)
    fid["reference_max_joint_rad"] = float(np.abs(conv.joint_chunk[:, :7] - recorded_reference[:, :7]).max())
    # D. 변환 C 재현 (기록의 refined · reference 위에서 기록의 conversion 행 목록과 비교)
    rec_conv = call.row.get("conversion") or {}
    conv_mode = str(rec_conv.get("mode") or "C")
    cm = conversion_metrics(rt.kin, conv, recorded_refined, recorded_reference, execution_length=rt.execution_length,
                            mode=conv_mode)
    fid["conversion_reproduced"] = bool(
        cm["saturated_rows"] == [int(k) for k in rec_conv.get("saturated_rows", [])]
        and cm["corrected_rows"] == [int(k) for k in rec_conv.get("corrected_rows", [])])
    fid["conversion_rows_rebuilt"] = {"saturated": cm["saturated_rows"], "corrected": cm["corrected_rows"]}
    fid["conversion_rows_recorded"] = {"saturated": rec_conv.get("saturated_rows"),
                                       "corrected": rec_conv.get("corrected_rows")}
    # B. 기록 clearance 재현
    field = build_field(z, outside_distance=rt.outside_distance)
    traj = rt.layout.chunk_to_trajectory(recorded_refined.astype(np.float64))[:, :rt.to_config.horizon.planned]
    centres = rt.lin.sphere_states(traj, call.q_now)[0]
    mine = (np.asarray(field.distance(centres.reshape(-1, 3)), np.float64).reshape(centres.shape[:2])
            - rt.lin.robot_radii[None, :] - rt.esdf_margin)
    if "clearance" in z.files:
        rec_c = np.asarray(z["clearance"], np.float64)
        h = min(len(rec_c), len(mine))
        fid["record_clearance_mm"] = float(np.abs(mine[:h] - rec_c[:h]).max() * 1000.0)
    else:
        fid["record_clearance_mm"] = None
    scene, info = build_scene(rt, call, z, summary, field, attached=attached)
    metrics = (summary.get("to") or {}).get("metrics") or {}
    certified = bool(metrics.get("geometry_certified", summary.get("status") == "ok"))
    context, why = build_context(rt, call, summary)
    info["context"] = why
    info["attached_in_record"] = attached
    sqp_it = int((summary.get("to") or {}).get("sqp_iterations") or (summary.get("to") or {}).get("iterations") or 1)
    prep = Prepared(call=call, id=call.id, summary=summary, z=z, conv=conv, chunk=np.asarray(conv.joint_chunk, np.float64),
                    recorded_refined=recorded_refined, field=field, scene=scene, certified=certified,
                    attached=attached, context=context, sqp_iterations=sqp_it,
                    target_points=np.asarray(z["target_points"], np.float64) if "target_points" in z.files else None,
                    scene_info=info, fidelity=fid, conv_mode=conv_mode)
    # C. replay
    result = solve(rt, prep, variant_config(rt, VARIANTS["replay"], sqp_it), scene)
    fid["replay_mm"] = ee_mm_diff(rt.kin, result.chunk, recorded_refined)
    fid["replay_max_joint_rad"] = float(np.abs(np.asarray(result.chunk)[:, :7] - recorded_refined[:, :7]).max())
    fid["replay_status"] = result.status.value
    prep.fidelity = fid
    return prep, {}


# ======================================================================================
# 변형 실행
# ======================================================================================

def detect_action_box_overrides(to_config, margin: float = 0.05) -> Optional[dict]:
    """worktree `wip/sl10-actionbox` 의 옵션이 이 checkout 의 `TrajOptConfig` 에 있나. 있으면 켜는 override, 없으면 `None`.

    이름은 아직 정해지지 않았으므로 config 섹션의 필드 중 `action_box` 가 이름에 든 bool 필드를 찾는다 (margin 필드는
    `<이름>_margin`). hook 일 뿐이다 — 옵션이 들어오면 필요한 context 키를 `action_box_context` 에 적는다.
    """
    for section in ("collision", "cost", "limits", "sqp", "reduction"):
        sec = getattr(to_config, section, None)
        if sec is None or not dataclasses.is_dataclass(sec):
            continue
        names = {f.name for f in dataclasses.fields(sec)}
        for name in sorted(names):
            if "action_box" in name and not name.endswith("margin") and isinstance(getattr(sec, name), bool):
                vals: dict[str, Any] = {name: True}
                if f"{name}_margin" in names:
                    vals[f"{name}_margin"] = float(margin)
                return {section: vals}
    return None


def action_box_context(prep: Prepared) -> dict:
    """action 한계 제약이 context 로 받는 값 (a_k · 응답 모델 상수 …). 옵션이 들어오기 전에는 비어 있다."""
    return {}


def run_variant(rt: Runtime, prep: Prepared, variant: Variant) -> dict[str, Any]:
    """한 call 에서 한 변형. 결과 dict (`deviation` · `conversion` …) 또는 `{"unavailable": 이유}`."""
    scene = prep.scene
    extra = None
    context = None
    if variant.action_box:
        extra = detect_action_box_overrides(rt.to_config)
        if extra is None:
            return {"unavailable": "TrajOptConfig has no action-box option (wip/sl10-actionbox not merged here)"}
        context = {**prep.context, **action_box_context(prep)}
    if variant.target_removed:
        if prep.attached:
            return {"unavailable": "held: the held object is carved by the attached path, not the target extent"}
        if prep.target_points is None or not len(prep.target_points):
            return {"unavailable": "no target points in the record"}
        if prep._field_wo is None:
            prep._field_wo = field_without_target(prep.field, prep.target_points)
    if variant.attached_off and not (prep.scene.attached_points is not None):
        return {"unavailable": "no attached spheres in this call's scene (not held, or held spheres not supplied)"}
    scene = apply_variant_scene(prep.scene, variant, field_without_target=prep._field_wo)
    t0 = time.perf_counter()
    result = solve(rt, prep, variant_config(rt, variant, prep.sqp_iterations, extra), scene, context)
    out = variant_metrics(rt.kin, prep.conv, np.asarray(result.chunk, np.float32), prep.recorded_reference(),
                          execution_length=rt.execution_length, mode=prep.conv_mode)
    out.update({"status": result.status.value, "wall_ms": (time.perf_counter() - t0) * 1000.0,
                "sqp_iterations": result.metrics.get("sqp_iterations"),
                "max_violation_m": float(result.max_violation),
                "returned": result.metrics.get("returned"),
                "continuity_active": result.metrics.get("continuity_active"),
                "vs_recorded_refined_mm": ee_mm_diff(rt.kin, result.chunk, prep.recorded_refined)})
    return out


# ======================================================================================
# J1 — TrajOpt 를 미는 제약의 주인
# ======================================================================================

def sphere_kind(link: str) -> str:
    """제약 sphere 의 종류: 손가락 / 손바닥 / 팔 / 부착 구."""
    link = str(link)
    if link.startswith("held"):
        return "attached"
    if "finger" in link:
        return "finger"
    if link.endswith("hand"):
        return "palm"
    if link.startswith("panda_link"):
        return "arm"
    return "other"


def select_rows(clearance: np.ndarray, link_names: Sequence[str], *, band_m: float,
                steps: Optional[int] = None) -> dict[str, Any]:
    """기록된 `clearance (H, S)` 에서 위반 (`< 0`) · margin 안 (`< band_m`) 인 (step, sphere) 를 종류별로 센다."""
    c = np.asarray(clearance, np.float64)
    if steps is not None:
        c = c[:int(steps)]
    kinds = np.asarray([sphere_kind(n) for n in link_names])
    out: dict[str, Any] = {"band_m": float(band_m), "shape": list(c.shape), "violating": {}, "within_band": {},
                           "rows": []}
    for kind in ("finger", "palm", "arm", "attached", "other"):
        sel = kinds == kind
        out["violating"][kind] = int((c[:, sel] < 0.0).sum())
        out["within_band"][kind] = int((c[:, sel] < band_m).sum())
    for k, s in zip(*np.nonzero(c < band_m)):
        out["rows"].append({"step": int(k), "sphere": int(s), "link": str(link_names[s]), "kind": str(kinds[s]),
                            "clearance_mm": float(c[k, s] * 1000.0)})
    return out


# --- MuJoCo geom 으로 voxel 의 class 를 매긴다 (모델 snapshot 이 있을 때만) ---------------------------------

GEOM_CLASSES = ("target", "support", "destination", "other", "robot", "ghost")
#: 로봇 자신의 body 접두사 — 점유 voxel 이 로봇 근처면 "robot" (task 의 다섯 class 와 별도로 센다: 잔상이 아니라 자기 몸의 잔재)
ROBOT_BODY_PREFIX = ("robot0_", "gripper0_", "mount0_")
SUPPORT_BODIES = ("world", "table")
GHOST_MM = 30.0


def geom_signed_distance(points: np.ndarray, xpos, xmat, size, gtype) -> np.ndarray:
    """`(N, G)` — 점에서 각 geom 표면까지의 (대략적) 거리 (m). 음수 = 안쪽.

    plane(0) · sphere(2) · capsule(3) · ellipsoid(4, 구로 근사) · cylinder(5) · box(6) 를 정확히/근사하고, mesh(7) 는
    `geom_size` 를 half-extent 로 하는 box 로 근사한다 (MuJoCo 가 mesh 의 geom_size 에 AABB half-extent 를 넣는다).
    """
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.empty((len(pts), len(gtype)), np.float64)
    for g in range(len(gtype)):
        R = np.asarray(xmat[g], np.float64).reshape(3, 3)
        loc = (pts - np.asarray(xpos[g], np.float64)) @ R          # = R^T (p - c) per row
        s = np.asarray(size[g], np.float64)
        t = int(gtype[g])
        if t == 0:                                                   # plane: z of local frame
            out[:, g] = loc[:, 2]
        elif t in (2, 4):                                            # sphere / ellipsoid(≈ sphere of the mean radius)
            out[:, g] = np.linalg.norm(loc, axis=1) - (s[0] if t == 2 else float(np.mean(s)))
        elif t == 3:                                                 # capsule along z
            z = np.clip(loc[:, 2], -s[1], s[1])
            out[:, g] = np.linalg.norm(loc - np.stack([np.zeros_like(z), np.zeros_like(z), z], 1), axis=1) - s[0]
        elif t == 5:                                                 # cylinder along z
            q = np.stack([np.linalg.norm(loc[:, :2], axis=1) - s[0], np.abs(loc[:, 2]) - s[1]], 1)
            out[:, g] = np.minimum(q.max(axis=1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)
        else:                                                        # box (6) and mesh (7, ≈ AABB)
            q = np.abs(loc) - s
            out[:, g] = np.minimum(q.max(axis=1), 0.0) + np.linalg.norm(np.maximum(q, 0.0), axis=1)
    return out


def classify_points(points_world: np.ndarray, snap: dict, *, target_body: Optional[str],
                    destination_body: Optional[str], ghost_m: float = GHOST_MM / 1000.0) -> dict[str, Any]:
    """world 좌표의 점들을 MuJoCo geom class 로 센다. `snap` 은 `load_geom_snapshot` 의 dict (한 프레임).

    가장 가까운 collision geom 의 body 로 가른다: target body → `target`, `world` · `table` → `support`,
    destination body → `destination`, 로봇 body → `robot`, 나머지 → `other`. 가장 가까운 geom 이 `ghost_m` 보다 멀면 `ghost`.
    """
    contype = np.asarray(snap["geom_contype"]) | np.asarray(snap["geom_conaffinity"])
    cols = np.flatnonzero(contype > 0)
    names = [str(n) for n in snap["body_names"]]
    out = {k: 0 for k in GEOM_CLASSES}
    if not len(points_world):
        return {"counts": out, "geom_of_point": []}
    d = geom_signed_distance(points_world, np.asarray(snap["geom_xpos"])[cols], np.asarray(snap["geom_xmat"])[cols],
                             np.asarray(snap["geom_size"])[cols], np.asarray(snap["geom_type"])[cols])
    nearest = d.argmin(axis=1)
    near_d = d[np.arange(len(d)), nearest]
    body = [names[int(np.asarray(snap["geom_bodyid"])[cols][j])] for j in nearest]
    labels = []
    for b, dist in zip(body, near_d):
        if dist > ghost_m:
            c = "ghost"
        elif target_body and b == target_body:
            c = "target"
        elif b in SUPPORT_BODIES:
            c = "support"
        elif destination_body and b == destination_body:
            c = "destination"
        elif b.startswith(ROBOT_BODY_PREFIX):
            c = "robot"
        else:
            c = "other"
        out[c] += 1
        labels.append((c, b))
    return {"counts": out, "geom_of_point": labels}


def load_geom_snapshot(path: pathlib.Path, obs_index: int) -> Optional[dict]:
    """`geom_snapshot_from_mujoco` 가 쓴 npz 에서 `obs_index` 프레임. 없으면 `None`."""
    z = np.load(path, allow_pickle=False)
    idx = np.flatnonzero(np.asarray(z["obs_index"]) == int(obs_index))
    if not len(idx):
        return None
    i = int(idx[0])
    snap = {k: z[k] for k in ("geom_type", "geom_size", "geom_bodyid", "geom_contype", "geom_conaffinity",
                              "body_names", "geom_names")}
    snap.update(geom_xpos=z["geom_xpos"][i], geom_xmat=z["geom_xmat"][i])
    return snap


def geom_snapshot_from_mujoco(model, sim_states: np.ndarray, obs_indices: Sequence[int], path) -> None:
    """MuJoCo `model` (LIBERO env 의 `env.sim.model._model`) 에서 `sim_states` 프레임들의 geom pose 를 npz 로 쓴다.

    `sim_states` = `<stem>_geom.npz` 의 `sim_state_image[obs_index]` (robosuite `sim.get_state().flatten()` =
    `[time, qpos, qvel]`). 모델은 기록에 없으므로 **호출자가 env 를 만들어** 넘긴다 (.venv-libero). py3.8 에서도 돈다.
    """
    import mujoco

    data = mujoco.MjData(model)
    xpos, xmat = [], []
    for s in np.asarray(sim_states, np.float64):
        data.time = float(s[0])
        data.qpos[:] = s[1:1 + model.nq]
        data.qvel[:] = s[1 + model.nq:1 + model.nq + model.nv]
        mujoco.mj_kinematics(model, data)
        xpos.append(data.geom_xpos.copy())
        xmat.append(data.geom_xmat.copy())
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(model.nbody)]
    gnames = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(model.ngeom)]
    np.savez_compressed(str(path), obs_index=np.asarray(list(obs_indices), np.int64), geom_xpos=np.asarray(xpos),
                        geom_xmat=np.asarray(xmat), geom_type=model.geom_type.copy(), geom_size=model.geom_size.copy(),
                        geom_bodyid=model.geom_bodyid.copy(), geom_contype=model.geom_contype.copy(),
                        geom_conaffinity=model.geom_conaffinity.copy(), body_names=np.asarray(names),
                        geom_names=np.asarray(gnames))


def neighbour_voxels(field, centre_base: np.ndarray, radius_m: float, *, layer: int = -1) -> np.ndarray:
    """`centre_base` 에서 `radius_m` 이내인 **점유 voxel**(`d < 0`) 의 중심들 (base 좌표, (N, 3)). 기본은 미세 계층."""
    lay = field.layers[layer]
    g = lay.grid
    d = np.asarray(lay.distance_grid)
    lo = np.floor((np.asarray(centre_base) - radius_m - g.origin) / g.voxel_size).astype(int)
    hi = np.ceil((np.asarray(centre_base) + radius_m - g.origin) / g.voxel_size).astype(int) + 1
    lo, hi = np.maximum(lo, 0), np.minimum(hi, np.asarray(d.shape))
    if np.any(hi <= lo):
        return np.zeros((0, 3))
    sub = d[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    ijk = np.argwhere(sub < 0.0) + lo
    if not len(ijk):
        return np.zeros((0, 3))
    pts = g.origin + ijk * g.voxel_size
    return pts[np.linalg.norm(pts - centre_base, axis=1) <= radius_m]


def j1_for_call(rt: Runtime, call: Call, *, band_m: float, geom_snapshot: Optional[pathlib.Path] = None,
                target_body: Optional[str] = None, destination_body: Optional[str] = None,
                max_rows: int = 64) -> dict[str, Any]:
    """한 call 의 J1: 위반 / margin 안 (step, sphere) 의 종류별 개수와, 근처 점유 voxel 의 MuJoCo geom class."""
    from AG3S.trajopt.experiments.fixed_input_ablation import build_field

    if call.chunk_path is None:
        return {"unavailable": "no constraint record for this call"}
    z = np.load(call.chunk_path, allow_pickle=False)
    summary = json.loads(str(z["summary_json"]))
    names = summary.get("sphere_link_names") or [str(n) for n in rt.robot.sphere_link_names]
    out: dict[str, Any] = {"id": call.id, "phase": call.phase,
                           "rows": select_rows(np.asarray(z["clearance"]), names, band_m=band_m)}
    out["held_spheres"] = ("unavailable: the held (attached) spheres' clearance is not in the record"
                           if (call.row.get("grasp") or {}).get("attached") else None)
    if geom_snapshot is None:
        out["voxel_classes"] = {"unavailable": "no --geom-snapshot: the MuJoCo model is not in the record"}
        return out
    snap = load_geom_snapshot(pathlib.Path(geom_snapshot), call.obs_index)
    if snap is None:
        out["voxel_classes"] = {"unavailable": f"obs_index {call.obs_index} is not in the geom snapshot"}
        return out
    field = build_field(z, outside_distance=rt.outside_distance)
    traj = rt.layout.chunk_to_trajectory(np.asarray(z["refined_chunk"], np.float64))[:, :rt.to_config.horizon.planned]
    centres = rt.lin.sphere_states(traj, call.q_now)[0]
    T = np.asarray(call.T_world_base, np.float64)
    if target_body is None and "target_centroid" in z.files:
        target_body = _nearest_free_body(snap, T @ np.append(np.asarray(z["target_centroid"], np.float64), 1.0))
    totals = {k: 0 for k in GEOM_CLASSES}
    rows = out["rows"]["rows"][:max_rows]
    dominant = {k: 0 for k in GEOM_CLASSES}
    for r in rows:
        c = centres[r["step"], r["sphere"]]
        vox = neighbour_voxels(field, c, float(rt.lin.robot_radii[r["sphere"]]) + band_m + 2 * field.layers[-1].grid.voxel_size)
        if not len(vox):
            r["voxel_class"] = None
            continue
        world = (T[:3, :3] @ vox.T).T + T[:3, 3]
        cls = classify_points(world, snap, target_body=target_body, destination_body=destination_body)
        for k, v in cls["counts"].items():
            totals[k] += v
        # 이 행의 가장 가까운 점유 voxel 의 class
        near = int(np.argmin(np.linalg.norm(vox - c, axis=1)))
        r["voxel_class"] = cls["geom_of_point"][near][0]
        dominant[r["voxel_class"]] += 1
    out["voxel_classes"] = {"target_body": target_body, "destination_body": destination_body,
                            "voxel_counts": totals, "rows_by_nearest_voxel_class": dominant,
                            "n_rows_classified": len(rows), "n_rows_total": len(out["rows"]["rows"]),
                            "ghost_mm": GHOST_MM}
    return out


def _nearest_free_body(snap: dict, centroid_world4: np.ndarray) -> Optional[str]:
    """target centroid 에 body 위치가 가장 가까운 물체 body 이름 (로봇 · 탁자 · world 제외, geom pose 평균 기준)."""
    names = [str(n) for n in snap["body_names"]]
    bid = np.asarray(snap["geom_bodyid"])
    best, best_d = None, 1e9
    for b, n in enumerate(names):
        if n in SUPPORT_BODIES or n.startswith(ROBOT_BODY_PREFIX) or n.startswith("eef"):
            continue
        sel = bid == b
        if not sel.any():
            continue
        d = float(np.linalg.norm(np.asarray(snap["geom_xpos"])[sel].mean(axis=0) - centroid_world4[:3]))
        if d < best_d:
            best, best_d = n, d
    return best


def select_calls(calls: Sequence[Call], phases: Sequence[str], *, min_dev_rad: float = DEV_THRESHOLD_RAD,
                 only_deviating: bool = True) -> list[Call]:
    """J1 · J2 대상: 구간이 `phases` 에 들고, (기본) 기록된 refined 가 reference 에서 `min_dev_rad` 넘게 벗어난 call."""
    out = []
    for c in calls:
        if c.phase not in phases or c.chunk_path is None:
            continue
        dev = float(np.abs(np.asarray(c.joint_refined, np.float64)[:, :7]
                           - np.asarray(c.joint_reference, np.float64)[:, :7]).max())
        if only_deviating and not dev > min_dev_rad:
            continue
        out.append(c)
    return out


# ======================================================================================
# main
# ======================================================================================

def _jsonable(obj):
    from AG3S.trajopt.experiments.fixed_input_ablation import _jsonable as base

    return base(obj)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="vla_v11ab 실행 디렉터리 (episodes/ · server_constraints/ · run_manifest.json)")
    ap.add_argument("--constraints", default=None, help="chunk_*.npz 디렉터리 (기본: <run-dir>/server_constraints/run_0000)")
    ap.add_argument("--out", required=True, help="결과 JSON")
    ap.add_argument("--ag3s-version", default=None, help="run_manifest 에 server argv 가 없을 때 버전 이름으로 argv 를 짓는다")
    ap.add_argument("--episodes", default="all", help="`all` 또는 episode stem 목록 (쉼표)")
    ap.add_argument("--phases", nargs="+", default=["closing", "held"], choices=PHASES,
                    help="J1 · J2 대상 구간 (기본: 쥐는 중 · 쥠). before 를 더하면 비교용으로 같이 본다")
    ap.add_argument("--calls", default=None, help="`stem#seq,...` 만 (구간 · 편차 선택 무시)")
    ap.add_argument("--all-calls", dest="only_deviating", action="store_false",
                    help="편차 문턱을 걸지 않는다 (기본: reference 에서 --min-dev-rad 넘게 벗어난 call 만)")
    ap.add_argument("--min-dev-rad", type=float, default=DEV_THRESHOLD_RAD)
    ap.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS), choices=list(VARIANTS))
    ap.add_argument("--j1-only", action="store_true", help="재풀이 없이 J1 만 (clearance 기록 분석)")
    ap.add_argument("--j1-band-mm", type=float, default=20.0, help="J1 의 margin 안 문턱 (기록 clearance < 이 값)")
    ap.add_argument("--geom-snapshot", default=None, help="geom_snapshot_from_mujoco 가 쓴 npz (voxel class 용)")
    ap.add_argument("--target-body", default=None)
    ap.add_argument("--destination-body", default=None)
    ap.add_argument("--attached-spheres-json", default=None,
                    help='{"parent_link": "panda_hand", "spheres": [[cx, cy, cz, r], ...]} (parent 좌표계). 쥠 call 의 scene 에 쓴다')
    ap.add_argument("--fidelity-tol-mm", type=float, default=1.0)
    ap.add_argument("--no-strict", dest="strict", action="store_false",
                    help="무효 call 만 건너뛰고 나머지를 돈다 (기본: 하나라도 무효면 변형을 하나도 안 돈다)")
    return ap


def _write(args, out: dict) -> None:
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(out), ensure_ascii=False, indent=1))
    print(f"[sl10-j2] wrote {path} (valid={out.get('valid')})")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    started = time.time()
    run_dir = pathlib.Path(args.run_dir)
    constraints = pathlib.Path(args.constraints) if args.constraints else run_dir / "server_constraints" / "run_0000"
    attached = json.loads(pathlib.Path(args.attached_spheres_json).read_text()) if args.attached_spheres_json else None
    sargs = server_args(run_dir, args.ag3s_version)
    rt = build_runtime(sargs, attached_spheres=attached)
    cfg_path = run_dir / "logs" / "panda_server_config.json"
    server_cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    n_rec = ((server_cfg.get("ag3s") or {}).get("n_constraint_spheres"))
    if n_rec is not None and int(n_rec) != int(rt.robot.n_spheres):
        raise SystemExit(f"다시 지은 제약 모델의 구 {rt.robot.n_spheres} 개 ≠ panda_server_config 의 {n_rec} 개")
    episodes_dir = run_dir / "episodes"
    stems = sorted(p.name[:-len("_calls.jsonl")] for p in episodes_dir.glob("*_calls.jsonl"))
    if args.episodes != "all":
        want = set(args.episodes.split(","))
        stems = [s for s in stems if s in want]
    calls: list[Call] = []
    for stem in stems:
        calls.extend(load_episode_calls(episodes_dir, stem))
    pairing = attach_chunks(calls, constraints)
    if args.calls:
        wanted = set(args.calls.split(","))
        targets = [c for c in calls if c.id in wanted and c.chunk_path is not None]
    else:
        targets = select_calls(calls, args.phases, min_dev_rad=args.min_dev_rad, only_deviating=args.only_deviating)
    out: dict[str, Any] = {
        "tool": TOOL, "created": datetime.datetime.now().isoformat(timespec="seconds"), "argv": sys.argv,
        "inputs": {"run_dir": str(run_dir), "constraints": str(constraints), "episodes": stems},
        "pairing": pairing, "n_calls": len(calls), "targets": [c.id for c in targets],
        "target_phases": {c.id: c.phase for c in targets}, "dev_threshold_rad": args.min_dev_rad,
        "robot": {"n_constraint_spheres": int(rt.robot.n_spheres), "links": args.links if hasattr(args, "links") else None,
                  "sphere_options": None, "note": "Panda applies no constraint_sphere_options (adapter)"},
        "not_reproduced": ["held (attached) spheres unless --attached-spheres-json", "label layer / destination margin",
                           "neighbour_contact / neighbour_palm_contact / grasp_contact / placement_contact / HSR6 gates",
                           "held_capture_spheres", "QP warm start across calls", "manipulated.points (target_points used)"],
    }
    # J1 (재풀이 없이, 기록만으로)
    out["j1"] = {c.id: j1_for_call(rt, c, band_m=args.j1_band_mm / 1000.0, geom_snapshot=args.geom_snapshot,
                                   target_body=args.target_body, destination_body=args.destination_body)
                 for c in targets}
    if args.j1_only:
        out.update(valid=True, j2="skipped (--j1-only)", elapsed_s=round(time.time() - started, 1))
        _write(args, out)
        return 0
    # J2
    preps: list[Prepared] = []
    unprepared: dict[str, Any] = {}
    for c in targets:
        p, why = prepare_call(rt, c, tol_mm=args.fidelity_tol_mm)
        if p is None:
            unprepared[c.id] = why
            continue
        preps.append(p)
        print(f"[sl10-j2] {c.id} {c.phase}: " + ", ".join(f"{k}={p.fidelity.get(k):.3f}" for k in FIDELITY_KEYS
                                                         if p.fidelity.get(k) is not None), flush=True)
    res = run_calls(preps, lambda p, v: run_variant(rt, p, v), args.variants, tol_mm=args.fidelity_tol_mm,
                    strict=args.strict)
    out.update({k: res[k] for k in ("valid", "invalid_calls", "invalid_reason", "fidelity", "variants_run", "results")})
    out["unprepared"] = unprepared
    out["scene_info"] = {p.id: p.scene_info for p in preps}
    out["instrument_summary"] = instrument_summary(res["results"], variants=[v for v in res["variants_run"]]) if res["results"] else {}
    out["elapsed_s"] = round(time.time() - started, 1)
    _write(args, out)
    return 0 if res["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

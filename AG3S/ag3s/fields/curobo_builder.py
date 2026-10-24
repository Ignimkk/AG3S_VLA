"""cuRobo `Mapper` 를 `EsdfBuilder` 와 **같은 서명**으로 감싼다.

    from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder
    builder = CuroboFieldBuilder(config)          # config = AG3SConfig.esdf
    field = builder.update(depth_cameras, target_points=..., attached_points=...,
                           labelled_points=..., static_geometry=...)

같은 서명인 것이 요점이다. `pipeline._build_esdf` 는 구현 한 줄만 갈아타고, 배선과 정책은
그대로 남는다 — `AG3S_REVIEW_LOG.md` 의 모듈 처분표가 적은 대로 E1(조작 대상을 필드에서
파내면 손끝뿐 아니라 전신에게 사라진다)의 접촉 권한, G2~G4 의 검증 플래그는 어느 필드
구현을 쓰든 필요하다.

## 무엇을 cuRobo 에 넘기고 무엇을 우리가 들고 있나

| 책임 | 어디서 |
|---|---|
| depth 적분, coarse/fine ESDF | cuRobo `Mapper` (실측 적분 3.8 ms + ESDF 0.44 ms) |
| 잔상 감쇠 (F20 — 사라진 물체의 잔상이 8 프레임 뒤에도 남는다) | cuRobo native (`MapperCfg.decay_factor`/`frustum_decay_factor`). **기본 끔** |
| 라벨 층 | 우리 — `site_index` 조회 (아래) |
| 쥔 물체 (A2) | 우리 — **seed 에서 제외** (아래). 격자를 지우지 않는다 |
| 해석적 정적 기하 (N2) | 우리 — 질의 시점 `min(복셀, 해석적)`. cuRobo `update_static_obstacles` 는 **격자 안만** 찍고, 우리 실측은 중요 기하(선반 y=-1550, 벽 ±3 m)가 격자 밖이라 이득 0.0 mm 였다 |
| 지지면 파내기 | **안 한다** — live 경로는 `exclude_support_surfaces=False` 라 발동하지 않는다. 주어지면 조용히 무시하지 않고 예외를 던진다 |

## 라벨 층은 조회 한 번이다

cuRobo 의 PBA+ 는 복셀마다 **그 거리를 만든 표면 복셀**을 `site_index` 에 남긴다
(dense int32, ESDF 격자와 같은 shape, 포장 `(z<<20)|(y<<10)|x`). 실측으로 확인했다 —
유효 100 %, `|site 까지 기하 거리 - |ESDF||` 중앙 0.10 mm · 최대 0.49 mm (float16 양자화).
그래서 "이 점에서 가장 가까운 표면이 무엇인가" 는 추가 알고리즘이 아니라 조회다.

## 쥔 물체는 지우지 않고 seed 에서 뺀다

`Mapper.clear_region` 은 **파괴적이고 보수적인 AABB** 라 쓰지 않는다. 누적 TSDF 를 지우므로
침식(있는 것을 잊는 것 = 위험한 방향)을 만들고, 게다가 잔상은 물체의 *옛* 자리에 있어 현재
AABB 로는 닿지도 않는다.

대신 seed 와 propagate 사이에 끼어들고, **그 뒤에 부호를 고친다.**

    integrator._seed_esdf_impl(origin, vs)            # site_index 에 표면 복셀을 심는다
    site_index[쥔 물체 복셀] = -1                       # (1) 크기를 다음 표면까지로
    integrator._propagate_and_distance_impl(...)      # PBA 가 남은 표면에서 전파한다
    values[순수 복셀] = abs(values[순수 복셀])           # (2) 부호를 양수로

**(2) 가 없으면 (1) 은 상황을 더 나쁘게 만든다.** cuRobo 커널이 부호를 seed 가 아니라
**질의 복셀 자신의 TSDF** 에서 가져오기 때문이다 (`builder_esdf.py:455-489`):

    tsdf_sdf = lookup_combined_sdf_at_esdf_coords(tsdf, esdf_x, esdf_y, esdf_z, ...)
    if tsdf_sdf < 0.0: edt_dist = -edt_dist

seed 를 지우면 크기만 "다음 표면까지" 로 커지고 부호는 여전히 그 자리에 남아 있는 쥔 물체의
TSDF 가 정한다 — 즉 *더 먼 거리에 음수 부호*가 붙는다. 실측으로 잡혔다: 쥔 물체 질의점
184 개에서 음수가 38 → **59** 개로 늘고 최소가 −12.93 → **−33.07 mm** 가 됐다.

**순수 복셀만 고친다** (2026-09-22 판정 B). 순수 = 쥔 물체 말고는 아무것도 없는 복셀이고,
판정은 추가 데이터 없이 된다 — seed 제외 **뒤**의 거리가 복셀 반 칸보다 크면 그 자리에 다른
표면이 없다는 뜻이다. 반 칸 안이면 무언가 있으므로 부호를 **그대로 둔다**.

그래서 환경 표면과 같은 복셀을 쓰는 자리에서는 관통이 **그대로 보고된다** — 숨길 수 있는 양이
경계값이 아니라 **구조적으로 0** 이다. 대가는 그 복셀들에서 쥔 물체가 자기 자신에게 남는
잔여 행이고, 실측상 운반 구간(들어올림 100 mm 이상)에서 **0** 이며 담기 국면에서만
20 mm 계층 24 개 · 5 mm 계층 17 개까지 나온다.

파내기(legacy `carve`)와의 차이는 **TSDF 나 second-nearest 가 아니다** — `carve` 는 매 프레임
다시 계산되는 occupancy 배열을 건드리므로 그 둘을 이미 만족한다. 차이는 cuRobo 의 부호 경로가
occupancy 를 거치지 않는다는 것뿐이고, 그래서 (2) 가 필요하다.

쥔 물체 자체는 로봇 쪽으로 편입되는데 그것은 **점 기반**이다 (E3 가 `sphere_states` 에
반지름 0 질의점으로 붙였다). cuRobo 의 attached-object(구 근사)를 쓰지 않는 이유는 F19 —
관측 점군에 구 하나를 맞추면 r = 57.4 mm 로 담는 구간 여유가 중앙 13.2 mm 나빠지고 프레임
21 에서 없는 충돌을 보고한다.

**`_site_index` 와 `_seed_esdf_impl` 은 cuRobo integrator 의 private 이다.** commit `78fd485`
에 핀되어 있고, `_assert_curobo_surface()` 가 시작할 때 존재를 확인해 버전이 바뀌면 조용히
다른 동작을 하지 않고 **거기서 죽는다**.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Optional, Sequence

import numpy as np

from AG3S.ag3s.config import EsdfConfig
from AG3S.ag3s.fields.curobo_field import CuroboEsdfField
from AG3S.ag3s.fields.device_field import layer_from_device
from AG3S.ag3s.fields.esdf import CameraDepth, VoxelGrid, default_bounds
from AG3S.ag3s.fields.observation import probe_from_cameras
from AG3S.ag3s.fields.provenance import FieldProvenance
from AG3S.ag3s.types import TARGET_LABEL

__all__ = ["CuroboFieldBuilder", "TargetBall", "OwnedEvidence", "OWNED_CLASSES", "ATTACHED_LABEL", "unpack_site_linear",
           "CoarseFreeSpec", "CUROBO_SEED_BAND_VOXELS", "coarse_exclusion_params", "COARSE_SIGN_FLOORS",
           "FineWindowRequest", "FineWindow", "place_fine_window", "WINDOW_BASES",
           "WINDOW_PLACEMENTS", "NONTARGET_EVIDENCES", "nontarget_surface_evidence",
           "OracleVisibleEvidence", "oracle_candidates", "oracle_visible_split", "oracle_cell_keys",
           "SurfaceProtectEvidence", "surface_protect_split",   # IFG-CL2
           "oracle_band_masks",
           "TargetLayerRequest", "target_owned", "target_pixel_mask", "roi_boxes", "in_boxes"]   # TASK-DYNAMIC-MAP

#: 쥔 물체에 붙이는 내부 라벨. 호출자의 `labelled_points` 이름과 겹치면 안 되므로 밑줄로 감싼다.
ATTACHED_LABEL = "__attached__"


def _same_cuda_device(a, b) -> bool:
    """`a` and `b` name the same CUDA device (``"cuda"`` without an index = the current one)."""
    import torch

    a, b = torch.device(a), torch.device(b)
    if a.type != "cuda" or b.type != "cuda":
        return a == b
    cur = torch.cuda.current_device()
    return (cur if a.index is None else a.index) == (cur if b.index is None else b.index)


def unpack_site_linear(site_index, shape: Sequence[int]):
    """packed `site_index` -> 그 site 복셀의 **선형 인덱스**. `-1` 은 그대로 `-1`.

    포장은 `(z<<20) | (y<<10) | x` 이고 축마다 10 비트다 (`utils_quantization.pack_site_coords`).
    선형화는 `(x*ny + y)*nz + z` — numpy 의 C 순서이고 `feature_tensor` 의 `(nx, ny, nz)` 와
    같다 (X 가 가장 느리고 Z 가 가장 빠르다는 cuRobo 주석).

    torch tensor 가 오면 **그 device 에서** 같은 정수 연산을 하고 int64 tensor 를 돌려준다 (T38 B3 —
    격자를 host 로 내리지 않는다). numpy 면 예전 그대로 numpy.
    """
    nx, ny, nz = (int(v) for v in shape)
    if not isinstance(site_index, np.ndarray) and hasattr(site_index, "device"):
        import torch

        packed = site_index.to(torch.int64)
        x = packed & 0x3FF
        y = (packed >> 10) & 0x3FF
        z = (packed >> 20) & 0x3FF
        lin = (x * ny + y) * nz + z
        return torch.where(packed < 0, torch.full_like(lin, -1), lin)
    packed = np.asarray(site_index, np.int64)
    x = packed & 0x3FF
    y = (packed >> 10) & 0x3FF
    z = (packed >> 20) & 0x3FF
    lin = (x * ny + y) * nz + z
    return np.where(packed < 0, -1, lin)


@dataclasses.dataclass(frozen=True)
class TargetBall:
    """`target_free` 계층에서 **기하 범위로** 지울 공. 테이블 면 위만 지운다 (T13).

    **왜 점이 아니라 공인가.** 점으로 지우는 길(`_exclude_seed_points`, 쥔 물체가 쓰는 쪽)은
    `target_free` 에서 **한 voxel도 지우지 못했다.** 실측 (`chunk_00014.npz`, `t=104`):

    | | voxel |
    |---|---|
    | 사과 표면 (중심 45 mm 안) | **2,203** |
    | 두 계층(`esdf` · `esdf_free`)이 다른 voxel | 578 |
    | 그중 사과 45 mm 안 | **0** |

    손끝 질의점에서 두 계층이 `17.319 mm` 로 **같았다.** 원인 셋이 겹쳤다 — grounding 표본점이
    112 개뿐이고(그중 seed 72 개), 제외가 팽창하지 않고(그 결정에는 이유가 있다,
    `_exclude_seed_points` 참고), **거리장이 성긴 제거에 꿈쩍하지 않는다** (2,203 중 72 를 지워도
    지워진 자리마다 5 mm 옆에 이웃 seed 가 살아 있다).

    쥔 물체 경로는 멀쩡하다 — 그때 주는 점들이 물체 **전체**를 덮기 때문이다. 같은 함수를 두
    용도로 쓴 것이 문제였고, 그래서 이름으로 갈랐다.

    `z_min` 은 **지원면 위로만 지우기 위한 것**이다. 사과 밑면은 테이블에 닿아 있어 그 아래까지
    지우면 **테이블에 구멍**이 난다 — 팔이 테이블을 통과하고 아무도 막지 않는다. 그래서 이 값은
    추측하지 않고 AG3S 가 뽑은 평면에서 온다. 못 얻으면 호출자가 **공을 만들지 않는다.**
    """

    centre: np.ndarray
    radius: float
    #: 이 z **이상**만 지운다. 지원면(테이블 상판) + 여유.
    z_min: float
    #: 진단용 — 무엇을 재서 이 값이 나왔나. 기록과 시작 로그에 그대로 실린다.
    provenance: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "centre",
                           np.asarray(self.centre, np.float64).reshape(3))
        if not float(self.radius) > 0.0:
            raise ValueError(f"TargetBall.radius 는 양수여야 합니다: {self.radius}")

    def summary(self) -> dict:
        return {"centre_m": [float(v) for v in self.centre],
                "radius_mm": round(float(self.radius) * 1000.0, 3),
                "z_min_m": float(self.z_min),
                **{k: v for k, v in self.provenance.items()}}


@dataclasses.dataclass(frozen=True)
class OwnedEvidence:
    """T44 I1 — `ball_owned` 가 공 안 seed 의 **주인**을 가르는 재료.

    공(`TargetBall`)은 그대로 **외곽 상한**이다. 공 안 seed voxel 중심 `c` 마다
    `d_t` = 가장 가까운 target 점, `d_o` = 가장 가까운 non-target 관측점까지의 거리를 재서

    | 조건 (순서대로) | 분류 | 처리 |
    |---|---|---|
    | `min(d_t, d_o) > max_dist_voxels·vs` | unowned | `unowned` 대로 (`remove` · `keep`) |
    | `|d_t − d_o| ≤ tie_voxels·vs` | ambiguous | `ambiguous` 대로 (`keep` 기본 = **보존** · `remove` = T46 F2) · 기록 |
    | `d_t < d_o` | target | 제거 |
    | 그 밖 | non-target | **보존** |

    non-target 점이 없으면 (`None` · 빈 배열) `d_o = ∞` 다 — 공 밖으로 넓히는 fallback 은 없다.

    | 필드 | 무엇 |
    |---|---|
    | `target_points` | target 점 (`TargetGeometry.points` — pipeline 의 `target_free_points`) |
    | `nontarget_points` | fused cloud − target `point_indices` − support (pipeline `_nontarget_points`; T46 F1 `surface` 면 `nontarget_surface_evidence` 로 더 뺀 것) |
    | `ambiguous` | (T46 F2) `keep` (기본, T44 I1 그대로) · `remove` — ambiguous seed 도 지운다 |
    """

    target_points: np.ndarray
    nontarget_points: Optional[np.ndarray] = None
    tie_voxels: float = 1.0
    max_dist_voxels: float = 2.0
    unowned: str = "remove"
    #: T46 F2 — `"keep"` (기본) · `"remove"`. 미세 target-free 계층만 이 값을 넘긴다 (거친 계층은 기본 keep).
    ambiguous: str = "keep"

    def __post_init__(self) -> None:
        tp = np.asarray(self.target_points, np.float64).reshape(-1, 3)
        object.__setattr__(self, "target_points", tp)
        nt = self.nontarget_points
        object.__setattr__(self, "nontarget_points",
                           np.zeros((0, 3)) if nt is None
                           else np.asarray(nt, np.float64).reshape(-1, 3))
        if self.unowned not in ("remove", "keep"):
            raise ValueError(f"unowned 는 'remove' 또는 'keep' 이어야 합니다: {self.unowned!r}")
        if self.ambiguous not in ("keep", "remove"):
            raise ValueError(f"ambiguous 는 'keep' 또는 'remove' 이어야 합니다: {self.ambiguous!r}")
        if float(self.tie_voxels) < 0.0 or not float(self.max_dist_voxels) > 0.0:
            raise ValueError("tie_voxels ≥ 0, max_dist_voxels > 0 이어야 합니다")


#: cuRobo 가 ESDF voxel 을 표면 seed 로 심는 띠 — `|tsdf(중심)| ≤ 0.9 · vs` (`builder_esdf.py:302-305`,
#: `T44E2b.verify.json` code_state). **계층마다 m 로는 다르다** (5 mm → 4.5 mm, 20 mm → 18 mm): 한 seed
#: voxel 의 중심에서 그 seed 를 만든 표면까지가 이 거리 안이다. T44c 의 거친 계층 제외가 이 값으로
#: "이 voxel 을 심을 수 있었던 증거" 의 범위를 정한다.
CUROBO_SEED_BAND_VOXELS = 0.9
#: T44c A6 — `CoarseFreeSpec.sign_floor` 의 선택지 (`EsdfConfig.target_free_coarse_sign_floor` 와 같은 목록).
COARSE_SIGN_FLOORS = ("guard", "plane")
#: T44e A8 — `CoarseFreeSpec.support` 의 선택지 (`EsdfConfig.target_free_coarse_support` 와 같은 목록).
COARSE_SUPPORT_MODES = ("keep", "exclude", "exclude_target")
#: T44e A8b — 지원면 seed 를 실제로 지우는 선택지 (근거 = 지원면 inlier 점 · 평면이 필요하다).
COARSE_SUPPORT_REMOVING = ("exclude", "exclude_target")
#: T44e A8b — `"exclude_target"` 발자국의 팽창 (거친 voxel 단위, 지원면 위 평면 거리). 거친 계층은 trilinear 로
#: 읽힌다 — 한 query 는 축마다 한 칸 안의 voxel 중심 8 개를 읽고, 그 중심은 query 에서 평면 방향으로 최대
#: `√2 · vs` (대각) 떨어져 있다. 그래서 발자국 (공 원판 · 관측점) 안의 어떤 query 가 읽는 지원면 voxel 은 모두
#: 발자국에서 `√2 · vs` 안이다: target 발자국은 그 voxel 까지 지우고 (공 아래 query 가 테이블 seed 를 읽지 않게),
#: non-target 발자국은 그 voxel 까지 남긴다 (이웃 과일 아래 query 가 keep 과 같은 seed 를 읽게). 둘이 같은 팽창이다.
COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS = float(np.sqrt(2.0))
#: T44e A8b — 관측점이 "지원면 위 물체" 의 발자국 증거가 되는 최소 높이 (m, 평면 법선 방향) 의 기본값.
#: `SupportSurfaceConfig.distance_threshold` (RANSAC inlier 문턱, 8 mm) 와 같다 — 그 안의 점은 같은 규칙으로는 테이블과
#: 구별되지 않는다. pipeline 은 활성 config 값을 `support_seed_evidence["inlier_threshold_m"]` 로 넘긴다.
COARSE_SUPPORT_FOOTPRINT_MIN_HEIGHT_M = 0.008
#: T44e A8 — 지원면 seed 판정의 여유 (거친 voxel 단위, seed 띠 `0.9 · vs` 에 더한다). 두 거리에 쓴다:
#: (1) seed 중심에서 지원면 평면까지 — 테이블의 TSDF 영교차는 맞춘 평면보다 몇 mm 위에 서므로
#: (`linearize._obstacle_rows` docstring: 20 mm 계층 중앙 8.7 mm) 띠만으로는 위쪽 seed 를 놓친다;
#: (2) seed 중심에서 가장 가까운 그 면의 inlier 점까지 — 점구름은 voxel 로 성겨서 (5 mm) 발 밑 점이 반 칸 옆에 있을 수 있다.
COARSE_SUPPORT_REACH_EXTRA_VOXELS = 0.5


@dataclasses.dataclass(frozen=True)
class CoarseFreeSpec:
    """T44c A5 — target 없는 **거친** 계층 (`coarse_no_target`) 의 제외 재료. **전부 m 단위다.**

    미세 계층의 제외 (`TargetBall` + `target_exclusion`) 와 **같은 identity · 같은 의미**를 거친 voxel 에
    적용하기 위한 것이다. voxel 단위 값 (`owned_*_voxels`, `attached_sign_threshold_voxels`,
    `target_ball_pad_voxels`) 은 **미세 voxel 로 정의된 것**이므로 여기서 m 로 바꿔 싣는다
    (`coarse_exclusion_params`). 거친 voxel 로 다시 나누는 것은 `_coarse_ball_exclusion` 한 곳이다.

    | 필드 | 무엇 |
    |---|---|
    | `owned` | `ball_owned` 이면 그 증거 (target 점 · non-target 점). `None` = `ball` |
    | `nontarget_points` | 공유 판정의 non-target 관측점 (`ball` 에서도 쓴다). 없으면 공유 판정 없음 |
    | `tie_m` · `max_dist_m` | `ball_owned` 의 tie · unowned 거리 (미세 voxel × 값) |
    | `sign_threshold_m` | 부호 교정의 순수 판정 임계 (`attached_sign_threshold_voxels` × 미세 voxel) |
    | `support_plane_z` · `support_pad_m` | 공의 `z_min` 을 만든 지원면 z 와 여유 (`TargetBall.provenance`) |
    | `sign_floor` | (T44c A6) 부호 교정의 아래 한계. `"plane"` (기본, A7) = 공의 `z_min` (미세 계층과 같은 m 값 = 지원면 + 여유) · `"guard"` = seed 제거 guard 와 같은 높이 (A5, 비교용) |
    | `target_points` | (T44c A6) target 점 — `"plane"` 띠에서 "target 쪽" 을 가르는 `d_t`. 없으면 (`ball`) 공유 판정만 |
    | `support` · `support_points` · `support_plane` | (T44e A8) `"exclude"` 면 공 아래 지원면의 seed 를 지운다 (`_coarse_support_exclusion`) — 그 면의 inlier 점 · 평면이 근거. 공의 seed guard 는 공의 `z_min` 으로 내려간다 (`coarse_exclusion_params`) |
    | `support="exclude_target"` · `support_min_height_m` | (T44e A8b) 지원면 seed 중 **target 발자국 안 · non-target 발자국 밖** 의 것만 지운다 (`_coarse_support_footprints`). `support_min_height_m` = 관측점이 발자국 증거가 되는 평면 위 최소 높이 |

    **`sign_floor` (T44c A6).** 거친 계층의 seed 제거 guard (지원면 + max(여유, 거친 voxel) = +20 mm) 는
    테이블 seed 를 지키려는 것이다. A5 는 부호 교정도 같은 높이에서 끊어, 그 아래의 target 바닥 내부
    (seed 가 아닌 음수 voxel) 가 장애물로 남았다 (`T44c.verify.json`: target 내부 음수 11,180 / 67,361,
    그중 11,057 이 지원면 + 20 mm 아래). `"plane"` 은 **seed 는 그대로 두고** (테이블 seed 보존) 부호
    교정만 미세 계층과 같은 m 높이 (공의 `z_min`) 까지 내린다 — 그 띠에서도 공유 영역은 빼고, target
    쪽 (non-target 관측이 더 가깝지 않은) non-seed voxel 만.
    """

    owned: Optional["OwnedEvidence"] = None
    nontarget_points: Optional[np.ndarray] = None
    tie_m: float = 0.0
    max_dist_m: float = 0.0
    sign_threshold_m: float = 0.0
    support_plane_z: Optional[float] = None
    support_pad_m: float = 0.0
    fine_voxel_m: float = 0.0
    sign_floor: str = "plane"           # T44c A7: 기본 plane (EsdfConfig 와 같다)
    target_points: Optional[np.ndarray] = None
    #: T44e A8 — `"keep"` (기본, A5–A7 그대로) · `"exclude"` (지원면 seed 를 이 계층에서 지운다,
    #: `_coarse_support_exclusion`). `"exclude"` 는 아래 두 근거가 있어야 한다.
    support: str = "keep"
    #: T44e A8 — 공 아래 지원면의 **inlier 점** (`SupportSurface.point_indices` 의 cloud 점) `(P, 3)`.
    support_points: Optional[np.ndarray] = None
    #: T44e A8 — 그 면 `[n_x, n_y, n_z, offset]` (단위 법선, 자유쪽 `n·p ≥ offset`).
    support_plane: Optional[np.ndarray] = None
    #: T44e A8b — `"exclude_target"` 의 발자국 증거 최소 높이 (m). 기본 = RANSAC inlier 문턱 기본값.
    support_min_height_m: float = COARSE_SUPPORT_FOOTPRINT_MIN_HEIGHT_M

    def __post_init__(self) -> None:
        nt = self.nontarget_points
        object.__setattr__(self, "nontarget_points",
                           np.zeros((0, 3)) if nt is None
                           else np.asarray(nt, np.float64).reshape(-1, 3))
        tp = self.target_points
        object.__setattr__(self, "target_points",
                           None if tp is None else np.asarray(tp, np.float64).reshape(-1, 3))
        if self.sign_floor not in COARSE_SIGN_FLOORS:
            raise ValueError(f"sign_floor 는 {COARSE_SIGN_FLOORS} 중 하나여야 합니다: "
                             f"{self.sign_floor!r}")
        if self.support not in COARSE_SUPPORT_MODES:
            raise ValueError(f"support 는 {COARSE_SUPPORT_MODES} 중 하나여야 합니다: {self.support!r}")
        sp = self.support_points
        object.__setattr__(self, "support_points",
                           None if sp is None else np.asarray(sp, np.float64).reshape(-1, 3))
        pl = self.support_plane
        object.__setattr__(self, "support_plane",
                           None if pl is None else np.asarray(pl, np.float64).reshape(4))
        if self.support in COARSE_SUPPORT_REMOVING and (self.support_points is None
                                                        or self.support_plane is None):
            raise ValueError(f"support={self.support!r} 는 support_points 와 support_plane 이 있어야 합니다 — "
                             "근거 없이 지원면 seed 를 지우지 않는다 (`_coarse_free_spec` 이 근거가 없으면 keep 으로 둔다)")


def coarse_support_spec_fields(mode: str, evidence: Optional[dict]) -> dict:
    """T44e A8/A8b — `CoarseFreeSpec` 의 `support*` 인자 (builder `_coarse_support_fields` · harness 가 같이 쓴다).

    `mode` 가 지우는 선택지 (`COARSE_SUPPORT_REMOVING`) 가 아니거나 근거 (`points` · `plane`) 가 없으면 `{}` (= keep —
    근거 없이 지우지 않는 쪽으로 닫힌다). `exclude_target` 이면 `evidence["inlier_threshold_m"]` (있으면) 을
    `support_min_height_m` 로 싣는다.
    """
    mode = str(mode)
    if mode not in COARSE_SUPPORT_REMOVING:
        return {}
    if not evidence or evidence.get("points") is None or evidence.get("plane") is None:
        return {}
    out = {"support": mode,
           "support_points": np.asarray(evidence["points"], np.float64).reshape(-1, 3),
           "support_plane": np.asarray(evidence["plane"], np.float64).reshape(4)}
    if mode == "exclude_target" and evidence.get("inlier_threshold_m") is not None:
        out["support_min_height_m"] = float(evidence["inlier_threshold_m"])
    return out


def coarse_exclusion_params(ball: "TargetBall", voxel_size: float, spec: CoarseFreeSpec) -> dict:
    """미세 voxel 로 정의된 제외 파라미터 → **이 계층 voxel (`voxel_size`)** 에서 쓸 값 (T44c A5).

    순수 함수다 (CPU 단위 검증이 이것을 본다). 미세 voxel 에 넣으면 미세 계층이 지금 쓰는 값이 그대로
    나온다 — 같은 규칙을 해상도만 바꿔 적용한다는 것을 그렇게 보인다.

    | 값 | 식 | 왜 |
    |---|---|---|
    | `band_m` | `0.9 · vs` | 이 계층에서 한 seed voxel 을 심을 수 있는 표면까지의 거리 (`CUROBO_SEED_BAND_VOXELS`) |
    | `z_min` | `max(ball.z_min, 지원면 z + max(pad_m, vs))` | 지원면이 이 계층의 voxel 을 심을 수 있는 높이 (`0.9 · vs` 안) 위만 지운다 — 미세 (5 mm) 에서는 `z + 5 mm` 로 지금 값과 같고, 거친 (20 mm) 에서는 `z + 20 mm` 다. 테이블 seed 를 지우면 테이블에 구멍이 난다 (T13) |
    | `tie_voxels` | `tie_m / vs` | tie 는 두 m 거리의 비교라 해상도와 무관하다 — m 그대로 |
    | `max_dist_voxels` | `max(max_dist_m, band_m) / vs` | unowned = "이 seed 를 설명하는 관측점이 없다". 이 계층에서는 `band_m` 안의 점이 seed 를 설명한다 — `max_dist_m` 이 띠보다 작으면 (d=3: 15 mm < 18 mm) 표면에서 15–18 mm 떨어진 거친 seed 중심이 증거가 있는데도 unowned 가 된다. 그래서 띠까지 넓힌다 |
    | (T44e A8, `support="exclude"`) `z_min` | `ball.z_min` (guard 를 올리지 않는다) · `support_reach_m = band + 0.5 · vs` | 테이블 seed 는 지원면 규칙이 지우므로 guard 가 지킬 것이 없다 |
    | `sign_threshold_voxels` | `sign_threshold_m / vs` | 순수 판정 "지운 뒤 `|d|` 가 이 m 보다 크면 다른 표면이 없다" 를 m 로 유지한다. 거친 계층에서는 7.5 mm < 20 mm 라 "seed 자신 (값 0) 만 보호" 와 같다 — 표면이 7.5 mm 안이면 그 voxel 이 seed 이기 때문이다. 공유 증거가 있는 voxel 은 아예 교정 집합에서 뺀다 (`_coarse_ball_exclusion`) |
    """
    vs = float(voxel_size)
    band = CUROBO_SEED_BAND_VOXELS * vs
    z_min = float(ball.z_min)
    # T44e A8b: `exclude_target` 도 같은 guard 규칙 — 공 원판은 target 발자국 안이라 공 안의 테이블 seed 는 지원면
    # 규칙이 지우거나 (non-target 발자국 밖) 보호 집합으로 남긴다 (`_coarse_ball_exclusion` 의 `protect`).
    exclude_support = getattr(spec, "support", "keep") in COARSE_SUPPORT_REMOVING
    if (not exclude_support and spec.support_plane_z is not None
            and np.isfinite(float(spec.support_plane_z))):
        # T44e A8: `exclude` 에서는 guard 를 올리지 않는다 — guard 는 테이블 seed 를 지키려는 것이고 (T13),
        # 그 seed 는 `exclude` 에서 어차피 지원면 규칙이 지운다. 공 안 seed 제거 높이 = 공의 `z_min` (미세와 같은 m).
        z_min = max(z_min, float(spec.support_plane_z) + max(float(spec.support_pad_m), vs))
    out = {"voxel_m": vs, "band_m": band, "z_min_m": z_min,
           "tie_voxels": float(spec.tie_m) / vs,
           "max_dist_voxels": max(float(spec.max_dist_m), band) / vs,
           "max_dist_m_effective": max(float(spec.max_dist_m), band),
           "sign_threshold_voxels": float(spec.sign_threshold_m) / vs}
    if spec.sign_floor == "plane":
        # T44c A6: 부호 교정의 아래 한계 = 공의 `z_min` (미세 계층이 쓰는 그 m 값). seed guard 는 그대로.
        out["sign_z_min_m"] = min(float(ball.z_min), z_min)
    if exclude_support:
        # T44e A8 — 키는 exclude 일 때만 (keep 의 params 는 A7 그대로).
        out["support"] = str(spec.support)
        out["support_reach_m"] = band + COARSE_SUPPORT_REACH_EXTRA_VOXELS * vs
        if spec.support == "exclude_target":
            # T44e A8b — 발자국 팽창 (trilinear 의 평면 방향 대각 도달) · 증거 최소 높이.
            out["footprint_dilation_m"] = COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS * vs
            out["footprint_min_height_m"] = float(spec.support_min_height_m)
    return out


#: `ball_owned` 분류 이름 — `_ball_owned_split` 의 키, 기록의 ijk 키 접미와 같다.
OWNED_CLASSES = ("target", "nontarget", "ambiguous", "unowned")


def seen_through(points: np.ndarray, cameras: Sequence[Any], margin: float, *,
                 depth_min: float = 0.0, depth_max: float = float("inf"),
                 footprint_voxel: Optional[float] = None, footprint_pad_px: int = 1,
                 footprint_robot: str = "block") -> tuple[np.ndarray, dict]:
    """AG3S-DYNAMIC-FIELD D1 — which `points` (N, 3, base frame) a depth ray of this frame passed through.

    A point is seen through when, in at least one camera, it projects inside the image in front of the
    camera and the observed z-depth of that pixel is more than `margin` beyond the point's own depth. The
    camera then saw *past* it, so nothing opaque is there. Pixels under `robot_mask`, zero / non-finite
    depth and depth outside `[depth_min, depth_max]` decide nothing (unobserved stays as it was) — so a
    fruit hidden behind the hand is never cleared. Pinhole z-depth, `T_base_cam` camera→base, as
    `stages.reconstruction.backproject`. Nearest pixel; pure numpy (the CPU tests read it).

    AG3S-TASK-DYNAMIC-MAP (`footprint_voxel`, off = None): instead of the nearest pixel, every pixel of the point's
    projected voxel footprint — a square of radius ``ceil(0.5 · fx · footprint_voxel / z) + footprint_pad_px`` px —
    must be valid (in range, not robot, inside the image) and see more than `margin` past the point. At an object's
    silhouette the nearest pixel can already see the background; the footprint keeps such edge voxels (thin objects).
    `footprint_robot`: ``block`` (a robot pixel anywhere in the footprint decides nothing) or ``ignore`` (robot pixels of the
    footprint do not count, but the nearest pixel itself must not be robot and must see past — what the hand hides is still
    never cleared).

    Returns `(mask (N,) bool, {camera name: n seen through})`.
    """
    p = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(p), bool)
    per = {}
    for cam in cameras:
        if footprint_voxel is not None:
            hit = _seen_through_footprint(p, cam, margin, depth_min, depth_max, float(footprint_voxel),
                                          int(footprint_pad_px), robot=str(footprint_robot))
            out[hit] = True
            per[str(getattr(cam, "name", "camera"))] = int(len(hit))
            continue
        d = np.asarray(cam.depth, np.float64)
        h, w = d.shape[:2]
        K = np.asarray(cam.camera_intrinsics, np.float64)
        T = np.asarray(cam.T_base_cam, np.float64)
        pc = (p - T[:3, 3][None, :]) @ T[:3, :3]          # base -> camera
        z = pc[:, 2]
        front = z > 1e-6
        u = np.full(len(p), -1, np.int64)
        v = np.full(len(p), -1, np.int64)
        u[front] = np.rint(K[0, 0] * pc[front, 0] / z[front] + K[0, 2]).astype(np.int64)
        v[front] = np.rint(K[1, 1] * pc[front, 1] / z[front] + K[1, 2]).astype(np.int64)
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.flatnonzero(inside)
        obs = d[v[idx], u[idx]]
        ok = np.isfinite(obs) & (obs > 0.0) & (obs >= depth_min) & (obs <= depth_max)
        mask = getattr(cam, "robot_mask", None)
        if mask is not None:
            ok &= ~np.asarray(mask, bool)[v[idx], u[idx]]
        hit = idx[ok & (obs - z[idx] > float(margin))]
        out[hit] = True
        per[str(getattr(cam, "name", "camera"))] = int(len(hit))
    return out, per


#: the largest footprint radius (px) `_seen_through_footprint` evaluates — a voxel nearer than that is not carved
FOOTPRINT_MAX_PX = 8


def _seen_through_footprint(p, cam, margin, depth_min, depth_max, voxel, pad_px, robot: str = "block") -> np.ndarray:
    """Indices of `p` that one camera saw through over the whole projected voxel footprint (`seen_through`)."""
    from scipy.ndimage import minimum_filter

    d = np.asarray(cam.depth, np.float64)
    h, w = d.shape[:2]
    K = np.asarray(cam.camera_intrinsics, np.float64)
    T = np.asarray(cam.T_base_cam, np.float64)
    good = np.isfinite(d) & (d > 0.0) & (d >= depth_min) & (d <= depth_max)
    mask = getattr(cam, "robot_mask", None)
    rmask = None if mask is None else np.asarray(mask, bool)
    if rmask is not None:
        good &= ~rmask
    d0 = np.where(good, d, 0.0)          # an undecided pixel anywhere in the footprint → min 0 → not seen through
    dc = d0                              # the nearest pixel: robot / invalid → 0 (decides nothing)
    if robot == "ignore" and rmask is not None:
        d0 = np.where(rmask, np.inf, d0)  # robot pixels of the footprint do not count
    pc = (p - T[:3, 3][None, :]) @ T[:3, :3]
    z = pc[:, 2]
    front = z > 1e-6
    u = np.full(len(p), -1, np.int64)
    v = np.full(len(p), -1, np.int64)
    u[front] = np.rint(K[0, 0] * pc[front, 0] / z[front] + K[0, 2]).astype(np.int64)
    v[front] = np.rint(K[1, 1] * pc[front, 1] / z[front] + K[1, 2]).astype(np.int64)
    idx = np.flatnonzero(front & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    if not len(idx):
        return idx
    r = np.ceil(0.5 * max(K[0, 0], K[1, 1]) * voxel / z[idx]).astype(np.int64) + int(pad_px)
    obs = np.zeros(len(idx))
    for rv in np.unique(r):
        if rv > FOOTPRINT_MAX_PX:
            continue                      # too near (footprint too large to judge) → leave it
        dm = minimum_filter(d0, size=2 * int(rv) + 1, mode="constant", cval=0.0)
        sel = r == rv
        obs[sel] = dm[v[idx[sel]], u[idx[sel]]]
    centre = dc[v[idx], u[idx]]
    return idx[(obs > 0.0) & np.isfinite(obs) & (obs - z[idx] > float(margin))
               & (centre > 0.0) & (centre - z[idx] > float(margin))]


# -- AG3S-ROI-CARVE-GPU: `seen_through` on the GPU ------------------------------------------------------
def gpu_cameras(cameras: Sequence[Any], device, depth_min: float, depth_max: float) -> list[dict]:
    """The cameras of this frame as float64 tensors on `device` — uploaded once per update, read by every TSDF's judgement.

    `good` = finite, positive and in `[depth_min, depth_max]` (the pixels that may decide), robot pixels not removed yet.
    """
    import torch

    out = []
    for cam in cameras:
        d = torch.as_tensor(np.asarray(cam.depth, np.float64), device=device)
        mask = getattr(cam, "robot_mask", None)
        K = np.asarray(cam.camera_intrinsics, np.float64)
        T = np.asarray(cam.T_base_cam, np.float64)
        out.append({"name": str(getattr(cam, "name", "camera")), "d": d,
                    "good": torch.isfinite(d) & (d > 0.0) & (d >= depth_min) & (d <= depth_max),
                    "robot": None if mask is None else torch.as_tensor(np.asarray(mask, bool), device=device),
                    "K": K, "R": torch.as_tensor(T[:3, :3], device=device), "t": torch.as_tensor(T[:3, 3], device=device)})
    return out


def _project_gpu(p, cam: dict):
    """`seen_through`'s pinhole projection on (N, 3) float64 `p`: `(z, u, v, inside)` (u, v = -1 behind the camera)."""
    import torch

    q = p - cam["t"][None, :]
    R = cam["R"]
    pc = q[:, 0:1] * R[0][None, :] + q[:, 1:2] * R[1][None, :] + q[:, 2:3] * R[2][None, :]   # base -> camera
    z = pc[:, 2]
    K = cam["K"]
    h, w = cam["d"].shape[:2]
    front = z > 1e-6
    zf = torch.where(front, z, torch.ones_like(z))
    u = torch.where(front, torch.round(float(K[0, 0]) * pc[:, 0] / zf + float(K[0, 2])), torch.full_like(z, -1.0)).long()
    v = torch.where(front, torch.round(float(K[1, 1]) * pc[:, 1] / zf + float(K[1, 2])), torch.full_like(z, -1.0)).long()
    return z, u, v, front & (u >= 0) & (u < w) & (v >= 0) & (v < h)


def _min_filter_gpu(img, radius: int):
    """`scipy.ndimage.minimum_filter(img, size=2 r + 1, mode="constant", cval=0)` of a 2-D float64 tensor."""
    import torch.nn.functional as F

    x = F.pad(img[None, None], (radius, radius, radius, radius), value=0.0)
    return -F.max_pool2d(-x, kernel_size=2 * radius + 1, stride=1)[0, 0]


def seen_through_gpu(points, cams: list[dict], margin: float, *, footprint_voxel: Optional[float] = None,
                     footprint_pad_px: int = 1, footprint_robot: str = "block"):
    """`seen_through` with the points (N, 3 float64 tensor) and the cameras (`gpu_cameras`) on the GPU — the same rule.

    Returns `(mask (N,) bool tensor, {camera name: n seen through as a 0-d tensor})` — nothing is read back here.
    """
    import torch

    out = torch.zeros(len(points), dtype=torch.bool, device=points.device)
    per = {}
    for cam in cams:
        z, u, v, inside = _project_gpu(points, cam)
        idx = torch.nonzero(inside, as_tuple=True)[0]
        ui, vi, zi = u[idx], v[idx], z[idx]
        good, rmask = cam["good"], cam["robot"]
        if footprint_voxel is None:
            obs = cam["d"][vi, ui]
            ok = good[vi, ui]
            if rmask is not None:
                ok = ok & ~rmask[vi, ui]
            hit = idx[ok & (obs - zi > float(margin))]
        else:
            if rmask is not None:
                good = good & ~rmask
            d0 = torch.where(good, cam["d"], torch.zeros_like(cam["d"]))   # undecided → min 0 → not seen through
            dc = d0
            if footprint_robot == "ignore" and rmask is not None:
                d0 = torch.where(rmask, torch.full_like(d0, float("inf")), d0)
            K = cam["K"]
            r = torch.ceil(0.5 * max(K[0, 0], K[1, 1]) * float(footprint_voxel) / zi).long() + int(footprint_pad_px)
            obs = torch.zeros_like(zi)
            for rv in torch.unique(r).tolist():
                if rv > FOOTPRINT_MAX_PX:
                    continue
                sel = r == rv
                obs[sel] = _min_filter_gpu(d0, int(rv))[vi[sel], ui[sel]]
            centre = dc[vi, ui]
            hit = idx[(obs > 0.0) & torch.isfinite(obs) & (obs - zi > float(margin))
                      & (centre > 0.0) & (centre - zi > float(margin))]
        out[hit] = True
        per[cam["name"]] = torch.as_tensor(hit.numel(), device=points.device)
    return out, per


def inside_robot_spheres(points: np.ndarray, spheres: dict) -> np.ndarray:
    """AG3S-DYNAMIC-FIELD D2 — which `points` (N, 3) lie inside the robot's (shrunk) spheres this frame.

    `spheres = {"centres": (S, 3), "radii": (S,), "z_min": (S,), "destination": (D, 3) | None, "dest_pad": m}`
    (pipeline `_robot_free_spheres`). A point counts when it is inside a sphere **and** not below that sphere's
    floor (`z_min`, the support plane under it + pad: never a hole in the table) and not within `dest_pad` of
    a destination point (the crate is never eroded). Pure numpy / scipy (the CPU tests read it).
    """
    from scipy.spatial import cKDTree

    p = np.asarray(points, np.float64).reshape(-1, 3)
    c = np.asarray(spheres["centres"], np.float64).reshape(-1, 3)
    r = np.asarray(spheres["radii"], np.float64).reshape(-1)
    zmin = np.asarray(spheres["z_min"], np.float64).reshape(-1)
    out = np.zeros(len(p), bool)
    if not len(p) or not len(c):
        return out
    k = min(16, len(c))
    d, i = cKDTree(c).query(p, k=k, distance_upper_bound=float(r.max()) + 1e-9)
    d, i = d.reshape(len(p), -1), i.reshape(len(p), -1)
    valid = i < len(c)
    ii = np.where(valid, i, 0)
    hit = valid & (d <= r[ii]) & (p[:, 2][:, None] >= zmin[ii])
    out = hit.any(axis=1)
    dest = spheres.get("destination")
    if out.any() and dest is not None and len(dest):
        dd, _ = cKDTree(np.asarray(dest, np.float64).reshape(-1, 3)).query(
            p[out], distance_upper_bound=float(spheres.get("dest_pad", 0.0)))
        idx = np.flatnonzero(out)
        out[idx[np.isfinite(dd)]] = False
    return out


@dataclasses.dataclass
class TargetLayerRequest:
    """AG3S-TASK-DYNAMIC-MAP ① — what the target-only TSDF needs this frame (pipeline `_target_layer_request`).

    `target_points` = this frame's manipulated points (the observed or followed geometry). A pixel or voxel is
    the target's when its nearest evidence point is a target point within `tol` and it is at or above `z_floor`
    (support plane + one fine voxel: never the table) — `target_owned`. `nontarget_points` = the observed
    non-target cloud (`AG3S._nontarget_points`), the competing evidence; `None` = only `tol` and `z_floor` decide.
    `centre` = the object's centre this frame (surface-sphere fit, else the point mean) — the move test of ①-b.
    `held` = the object is attached: the target TSDF is reset and unused (the held object is the robot's).
    """

    target_points: Optional[np.ndarray] = None
    nontarget_points: Optional[np.ndarray] = None
    tol: float = 0.010
    z_floor: Optional[float] = None
    centre: Optional[np.ndarray] = None
    held: bool = False

    def __post_init__(self) -> None:
        for name in ("target_points", "nontarget_points"):
            p = getattr(self, name)
            if p is not None:
                p = np.asarray(p, np.float64).reshape(-1, 3)
                object.__setattr__(self, name, p if len(p) else None)
        if not (np.isfinite(float(self.tol)) and float(self.tol) > 0.0):
            raise ValueError(f"tol 은 양수여야 합니다: {self.tol}")
        if self.centre is None and self.target_points is not None:
            object.__setattr__(self, "centre", self.target_points.mean(axis=0))

    @property
    def active(self) -> bool:
        return not self.held and self.target_points is not None


def target_owned(points: np.ndarray, target_points: Optional[np.ndarray],
                 nontarget_points: Optional[np.ndarray], tol, z_floor: Optional[float]) -> np.ndarray:
    """AG3S-TASK-DYNAMIC-MAP ① — which `points` (N, 3) belong to the target.

    A point is the target's when a target point is within `tol` (scalar or per point (N,)), its nearest
    target point is strictly closer than its nearest non-target point (the `ball_owned` ownership rule,
    T44 I1; a tie stays an obstacle), and it is at or above `z_floor`. Pure numpy / scipy (the CPU tests read it).
    """
    from scipy.spatial import cKDTree

    p = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(p), bool)
    if not len(p) or target_points is None or not len(target_points):
        return out
    tol = np.broadcast_to(np.asarray(tol, np.float64), (len(p),))
    dt, _ = cKDTree(np.asarray(target_points, np.float64).reshape(-1, 3)).query(
        p, distance_upper_bound=float(tol.max()) + 1e-12)
    cand = np.isfinite(dt) & (dt <= tol)
    if z_floor is not None:
        cand &= p[:, 2] >= float(z_floor)
    if cand.any() and nontarget_points is not None and len(nontarget_points):
        idx = np.flatnonzero(cand)
        dn, _ = cKDTree(np.asarray(nontarget_points, np.float64).reshape(-1, 3)).query(
            p[idx], distance_upper_bound=float(dt[idx].max()) + 1e-12)
        out[idx[dt[idx] < dn]] = True
        return out
    return cand


def target_pixel_mask(depth: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray,
                      request: TargetLayerRequest, *, depth_min: float = 0.0,
                      depth_max: float = float("inf")) -> np.ndarray:
    """AG3S-TASK-DYNAMIC-MAP ① — the `(H, W)` bool mask of this camera's pixels that go to the target TSDF.

    `depth` is the robot-masked z-depth the TSDF integrates (masked pixels are 0 and decide nothing). Each
    valid pixel is back-projected (`seen_through`'s pinhole convention, pixel ``u ↔ fx·x/z + cx``) and kept
    when `target_owned`. Pixels outside the target points' box ± `tol` are not tested.
    """
    d = np.asarray(depth, np.float64)
    mask = np.zeros(d.shape[:2], bool)
    tp = request.target_points
    if tp is None:
        return mask
    valid = np.isfinite(d) & (d > 0.0) & (d >= depth_min) & (d <= depth_max)
    v, u = np.nonzero(valid)
    if not len(v):
        return mask
    K = np.asarray(K, np.float64)
    T = np.asarray(T_base_cam, np.float64)
    z = d[v, u]
    pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
    pb = pc @ T[:3, :3].T + T[:3, 3][None, :]
    lo = tp.min(axis=0) - float(request.tol)
    hi = tp.max(axis=0) + float(request.tol)
    box = np.all((pb >= lo) & (pb <= hi), axis=1)
    if not box.any():
        return mask
    own = target_owned(pb[box], tp, request.nontarget_points, float(request.tol), request.z_floor)
    sel = np.flatnonzero(box)[own]
    mask[v[sel], u[sel]] = True
    return mask


def roi_boxes(point_sets: Sequence[Optional[np.ndarray]], pad: float,
              boxes: Sequence[tuple] = ()) -> list[tuple[np.ndarray, np.ndarray]]:
    """AG3S-TASK-DYNAMIC-MAP ② — the region of interest as AABBs: each point set's box ± `pad`, each given box ± `pad`."""
    out = []
    for p in point_sets:
        if p is None:
            continue
        p = np.asarray(p, np.float64).reshape(-1, 3)
        if len(p):
            out.append((p.min(axis=0) - float(pad), p.max(axis=0) + float(pad)))
    for lo, hi in boxes:
        out.append((np.asarray(lo, np.float64).reshape(3) - float(pad),
                    np.asarray(hi, np.float64).reshape(3) + float(pad)))
    return out


def in_boxes(points: np.ndarray, boxes: Sequence[tuple]) -> np.ndarray:
    """Which `points` (N, 3) lie inside any `(lo, hi)` box."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(p), bool)
    for lo, hi in boxes:
        out |= np.all((p >= np.asarray(lo)) & (p <= np.asarray(hi)), axis=1)
    return out


def _nearest(points: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """`queries` 마다 가장 가까운 `points` 까지 거리. 점이 없으면 `inf`."""
    if not len(points) or not len(queries):
        return np.full(len(queries), np.inf)
    from scipy.spatial import cKDTree

    d, _ = cKDTree(points).query(queries, k=1)
    return np.asarray(d, np.float64)


def _ball_owned_split(centres: np.ndarray, is_seed: np.ndarray, ball: "TargetBall",
                      owned: OwnedEvidence, voxel_size: float) -> dict:
    """공 안 voxel (`centres`, `is_seed`) 을 가른다 — 순수 numpy (CPU 단위 검증이 이것을 본다).

    돌려주는 것 (전부 `centres` 와 같은 길이의 bool, `d_t`·`d_o` 는 float):

    - `target` · `nontarget` · `ambiguous` · `unowned` — **seed 에 대한** 분류 (`OwnedEvidence` 의 표)
    - `removed` — 지울 seed = target ∪ (unowned if `remove`) ∪ (ambiguous if `owned.ambiguous == "remove"`, T46 F2)
    - `sign` — 부호 교정 후보 = `removed` ∪ **non-seed 중 target 쪽** (`d_t < d_o − tie`)

    **왜 non-seed 를 sign 에 넣나.** `ball` 의 부호 교정 집합은 공 안 voxel 전부다 (`_clear_seeds`
    가 seed 아닌 cand 까지 돌려준다) — target 내부 (seed 아님, TSDF < 0) 의 음수 부호를 그것이
    고친다. `removed` 만 고치면 target 내부가 "다음 표면까지의 음수 거리" 로 남는다. 반대로 공 안
    전부를 고치면 이웃 물체 내부의 음수를 양수로 바꾼다 (관통을 숨긴다). 그래서 non-seed 는 거리
    비교로 **target 쪽이 확실한 것**만 (tie 밖) 넣는다. `_force_sign_on_pure_voxels` 의 순수 판정
    (`|d| > 1.5 vs`) 은 그대로 그 위에 걸린다 — 보존된 seed 곁의 voxel 은 손대지 않는다.

    non-target 점은 공 중심에서 `r + R + vs` 안의 것만 쓴다 (`R` = 공 안 점에서 target 점까지의
    상한 + tie). 그 밖의 점은 `d_o ≥ R ≥ d_t + tie` 라 분류를 바꾸지 못한다 — 결과는 전체 점과 같다.
    """
    vs = float(voxel_size)
    c = np.asarray(centres, np.float64).reshape(-1, 3)
    seed = np.asarray(is_seed, bool).reshape(-1)
    tp = owned.target_points
    r = float(ball.radius)
    t_reach = float(np.max(np.linalg.norm(tp - ball.centre[None, :], axis=1))) if len(tp) else 0.0
    tie = float(owned.tie_voxels) * vs
    reach = r + (r + t_reach + tie) + vs
    nt = owned.nontarget_points
    if len(nt):
        nt = nt[np.linalg.norm(nt - ball.centre[None, :], axis=1) <= reach]
    d_t = _nearest(tp, c)
    d_o = _nearest(nt, c)
    max_d = float(owned.max_dist_voxels) * vs
    unowned = np.minimum(d_t, d_o) > max_d
    ambiguous = ~unowned & (np.abs(d_t - d_o) <= tie)
    decided = ~unowned & ~ambiguous
    tgt = decided & (d_t < d_o)
    ntg = decided & ~tgt
    removed = seed & (tgt | (unowned & (owned.unowned == "remove")))
    if owned.ambiguous == "remove":
        # T46 F2 — ambiguous seed 도 지운다 (`sign` 은 `removed` 를 담으므로 그 부호 교정도 함께).
        removed = removed | (seed & ambiguous)
    sign = removed | (~seed & (d_t < d_o - tie))
    return {"target": seed & tgt, "nontarget": seed & ntg, "ambiguous": seed & ambiguous,
            "unowned": seed & unowned, "removed": removed, "sign": sign,
            "d_t": d_t, "d_o": d_o, "n_nontarget_near": int(len(nt))}


#: T46 F1 — `esdf.nontarget_evidence` 의 선택지 (config 의 목록을 그대로 쓴다).
NONTARGET_EVIDENCES = EsdfConfig.NONTARGET_EVIDENCES


def nontarget_surface_evidence(nontarget_points, target_points, ball: "TargetBall", *,
                               shell_m: float, tie_m: float = 0.0) -> tuple[np.ndarray, dict]:
    """T46 F1 `"surface"` — non-target 근거 점 중 **target 자신의 표면** 인 것 `(drop (N,) bool, info)`.

    `ball_owned` 의 non-target 근거 (`AG3S._nontarget_points`) 는 grasp 중 `proximity` (target 점에서 미세 voxel
    5 mm 안을 뺀다) 로 떨어진다 — target 기하가 옛 frame 의 anchor 라 `point_indices` 가 이 cloud 의 것이 아니다.
    fused cloud 의 점 간격 (fusion voxel 10 mm) 이 그 반경보다 커서 target 표면 점이 근거로 남고 (`T45TF.verify.json`
    D-TF: 667 중 280), 그 점이 target seed 를 ambiguous / non-target 으로 보존시킨다. 여기서는 **GT 없이 기하만으로**
    그 점을 고른다:

    | 단계 | 조건 | 근거 |
    |---|---|---|
    | 문 (gate) | 공의 중심이 표면 구 맞춤에서 왔다 (`provenance["centre_method"] == "surface_fit"`) | 맞춤이 물러난 공 (`centroid`) 은 target 점이 한 볼록 물체가 아니다 — 이웃 과일이 섞인 기하 (D-TF F3: 섞인 6/6 chunk 가 물러남, 안 섞인 11/11 이 받아들임). 그때는 아무것도 더 빼지 않는다 |
    | 후보 | 공 안 (`|p − c| ≤ 공 반지름`, `z ≥ z_min`) **이고** `|p − c| ≤ 맞춘 반지름 + shell_m` | 맞춘 구 안의 관측 점은 target 이다 — 물체는 서로 파고들지 않으므로 이웃은 맞춘 구에 접촉 띠로만 들어온다. `shell_m` = cloud 양자화 · 비구형 (배) 여유 |
    | 이웃 보호 | 후보 `p` 에서 가장 가까운 **target 점** 거리 `d_t` 와 가장 가까운 **후보 밖 근거 점** 거리 `d_c` 가 `d_t + tie_m ≤ d_c` 인 것만 뺀다 | 접촉 띠에 든 이웃 점은 제 몸의 (껍질 밖) 점이 가깝다. seed 의 `ball_owned` 규칙 (`d_t` vs `d_o`, tie 안 = ambiguous = 보존) 을 근거 점에 같은 tie (`owned_tie_voxels`) 로 건다 — 애매한 근거 점은 근거로 남는다 |

    `info` = `{"method", "shell_mm", "tie_mm", "applied", "why"?, "fit_radius_mm", "limit_mm", "n_candidates",
    "n_protected", "n_dropped"}`. 순수 numpy — pipeline 과 offline replay (`AG3S.trajopt.experiments.target_free_replay`) 가 같이 쓴다.
    맞춘 반지름은 `provenance["fit"]["radius_mm"]` (기록과 같은 1 µm 반올림 값) 을 읽는다 — 서버와 replay 가 같은 수를 쓴다.
    """
    nt = np.asarray(nontarget_points, np.float64).reshape(-1, 3)
    tp = np.asarray(target_points, np.float64).reshape(-1, 3)
    drop = np.zeros(len(nt), bool)
    info: dict = {"method": "surface", "shell_mm": round(float(shell_m) * 1000.0, 3),
                  "tie_mm": round(float(tie_m) * 1000.0, 3),
                  "applied": False, "n_candidates": 0, "n_protected": 0, "n_dropped": 0}
    prov = dict(getattr(ball, "provenance", {}) or {})
    fit = prov.get("fit") or {}
    if prov.get("centre_method") != "surface_fit" or fit.get("radius_mm") is None:
        reason = fit.get("reason") if isinstance(fit, dict) else None
        info["why"] = "no_surface_fit" + (f" ({reason})" if reason else "")
        return drop, info
    if not len(tp):
        info["why"] = "no_target_points"
        return drop, info
    r_fit = float(fit["radius_mm"]) / 1000.0
    limit = min(float(ball.radius), r_fit + float(shell_m))
    info.update({"applied": True, "fit_radius_mm": round(r_fit * 1000.0, 3),
                 "limit_mm": round(limit * 1000.0, 3)})
    if not len(nt):
        return drop, info
    centre = np.asarray(ball.centre, np.float64).reshape(3)
    r = np.linalg.norm(nt - centre[None, :], axis=1)
    cand = (r <= limit) & (nt[:, 2] >= float(ball.z_min))
    info["n_candidates"] = int(cand.sum())
    if not cand.any():
        return drop, info
    q = nt[cand]
    d_t = _nearest(tp, q)
    d_c = _nearest(nt[~cand], q)
    own = d_t + float(tie_m) <= d_c
    drop[np.flatnonzero(cand)[own]] = True
    info["n_protected"] = int((~own).sum())
    info["n_dropped"] = int(own.sum())
    return drop, info


@dataclasses.dataclass(frozen=True)
class OracleVisibleEvidence:
    """T46e A13 — `esdf.target_extent == "oracle_visible"` 의 재료 (시뮬레이터 GT, 진단 상한).

    | 필드 | 무엇 |
    |---|---|
    | `target_points` | 지울 물체의 **보인** 표면 표본 (`runtime.oracle_segmentation.OracleMemory.evidence`) |
    | `other_points` | 그 밖의 모든 표본 (이웃 과일 · table · crate · 바닥) — 로봇은 없다 (code 0) |
    | `max_dist_voxels` | seed 후보 = target 표본에서 이 거리 (voxel) 안 (`esdf.oracle_max_dist_voxels`) |
    | `code` | 지울 물체의 code (`band_masks` 의 bit) |
    | `band_cell` · `band_keys` · `band_masks` | 이 episode 의 **truncation 띠** — 물체 pixel 의 광선을 따라 보인 면 뒤 truncation 까지 지나간 칸 (`band_cell` m 격자 key, 정렬) 과 그 칸을 지나간 물체 code 의 bit mask (`OracleMemory.band_lookup`). 비어 있으면 띠 길을 쓰지 않는다 |
    | `surface_protect_voxels` | 띠 길은 다른 표본에서 이 거리 (voxel) 안의 seed 를 가져가지 않는다 (관측된 이웃 · table 표면 보호) |
    | `z_floor` | 이 z (m) 아래는 어느 길로도 지우지 않는다 — 지지면 상판 (`TargetBall.z_min` 과 같은 값; None = 없음) |
    | `info` | 기록에 그대로 실린다 |
    """

    target_points: np.ndarray
    other_points: Optional[np.ndarray] = None
    max_dist_voxels: float = 6.0
    info: dict = dataclasses.field(default_factory=dict)
    code: int = 1
    band_cell: float = 0.0
    band_keys: Optional[np.ndarray] = None
    band_masks: Optional[np.ndarray] = None
    surface_protect_voxels: float = 1.0
    z_floor: Optional[float] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_points",
                           np.asarray(self.target_points, np.float64).reshape(-1, 3))
        op = self.other_points
        object.__setattr__(self, "other_points",
                           np.zeros((0, 3)) if op is None else np.asarray(op, np.float64).reshape(-1, 3))
        bk = np.zeros(0, np.int64) if self.band_keys is None else np.asarray(self.band_keys, np.int64).reshape(-1)
        bm = np.zeros(0, np.uint32) if self.band_masks is None else np.asarray(self.band_masks, np.uint32).reshape(-1)
        if len(bk) != len(bm):
            raise ValueError("band_keys 와 band_masks 의 길이가 다릅니다")
        object.__setattr__(self, "band_keys", bk)
        object.__setattr__(self, "band_masks", bm)
        if not float(self.max_dist_voxels) > 0.0:
            raise ValueError(f"max_dist_voxels 는 양수여야 합니다: {self.max_dist_voxels}")
        if not (float(self.surface_protect_voxels) >= 0.0 and float(self.band_cell) >= 0.0):
            raise ValueError("surface_protect_voxels ≥ 0, band_cell ≥ 0")
        if len(bk) and not float(self.band_cell) > 0.0:
            raise ValueError("band_keys 가 있으면 band_cell > 0 이어야 합니다")
        if not 0 <= int(self.code) < 32:
            raise ValueError(f"code 는 0..31 (band mask 의 bit): {self.code}")


def oracle_cell_keys(points: np.ndarray, cell: float) -> np.ndarray:
    """T46e — `floor(p / cell)` 칸의 int64 key (`runtime.oracle_segmentation` 의 표본 · 띠와 같은 규약, 축마다 ±2²⁰ 칸)."""
    c = np.floor(np.asarray(points, np.float64).reshape(-1, 3) / float(cell)).astype(np.int64) + (1 << 20)
    return (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]


def oracle_band_masks(points: np.ndarray, keys: np.ndarray, masks: np.ndarray, cell: float) -> np.ndarray:
    """T46e — 점마다 그 칸의 띠 bit mask (uint32, 띠 밖이면 0). `keys` 는 정렬돼 있어야 한다."""
    p = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros(len(p), np.uint32)
    if not len(p) or not len(keys):
        return out
    k = oracle_cell_keys(p, cell)
    i = np.clip(np.searchsorted(keys, k), 0, len(keys) - 1)
    hit = keys[i] == k
    out[hit] = masks[i[hit]]
    return out


def oracle_candidates(target_points: np.ndarray, origin: np.ndarray, voxel_size: float, shape,
                      max_dist: float) -> Optional[np.ndarray]:
    """T46e — 격자 안에서 중심이 target 표본에서 `max_dist` 안인 voxel `(K, 3)` int64 (meshgrid `ij` 순서), 없으면 None."""
    tp = np.asarray(target_points, np.float64).reshape(-1, 3)
    if not len(tp):
        return None
    vs = float(voxel_size)
    nx, ny, nz = (int(v) for v in shape)
    org = np.asarray(origin, np.float64).reshape(3)
    lo = np.floor((tp.min(axis=0) - max_dist - org) / vs + 0.5).astype(np.int64)
    hi = np.ceil((tp.max(axis=0) + max_dist - org) / vs + 0.5).astype(np.int64)
    lo = np.maximum(lo, 0)
    hi = np.minimum(hi + 1, np.array([nx, ny, nz]))
    if np.any(hi <= lo):
        return None
    ax = [np.arange(int(lo[i]), int(hi[i]), dtype=np.int64) for i in range(3)]
    grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
    centres = org[None, :] + grid.astype(np.float64) * vs
    cand = grid[_nearest(tp, centres) <= float(max_dist)]
    return cand if len(cand) else None


def oracle_visible_split(centres: np.ndarray, is_seed: np.ndarray, oracle: OracleVisibleEvidence,
                         voxel_size: float) -> dict:
    """T46e — 후보 voxel 을 GT 로 가른다 (순수 numpy; server · offline replay · 단위 검증이 같이 쓴다).

    seed 가 **그 물체의 pixel 에서 왔다** 를 두 길로 판정한다. `d_t` = 가장 가까운 그 물체 표본, `d_o` = 가장 가까운
    다른 표본 (없으면 ∞), `z_floor` 아래는 어느 길로도 지우지 않는다 (지지면 상판 — `ball` · `ball_owned` 의 `z_min` 과 같다):

    | 길 | 조건 (seed) | 무엇을 잡나 |
    |---|---|---|
    | 표면 | `d_t ≤ max_dist` ∧ `d_t < d_o` | 보인 표면의 seed — 이 episode 에 본 모든 표본 중 그 물체가 가장 가깝다 |
    | 띠 | 그 칸을 지나간 띠가 **그 물체 것뿐** (`band_masks`) ∧ `d_o > surface_protect·vs` ∧ `d_t ≤ max_dist` | 보인 면 **뒤** truncation 띠의 seed — TSDF 가 그 물체 pixel 의 광선으로 만든 음수 띠의 경계 (물체 밑 · 안쪽의 그림자 seed). 표면 표본에서 멀어 표면 길이 못 잡는다 (실측 C1a_ep54_s541 seq 16: 425 개, 앞 frame 광선의 것이 많아 이 요청의 카메라만으로는 설명되지 않는다 — 그래서 띠도 episode 동안 모은다) |
    | 그 밖 | | 보존 (이웃 · table · crate 표면, 두 물체의 띠가 겹친 칸, 동률 — 보수 쪽) |

    부호 교정 후보 (`sign`) = 지운 seed ∪ 표면 길의 non-seed (그 물체 안쪽). 띠 길은 non-seed 의 부호를 바꾸지 않는다 —
    띠는 보인 면 뒤 truncation 까지라 다른 물체 안쪽에 닿을 수 있어서다. tie 는 없다 (code 가 GT 다; 같은 거리 = 보존).
    다른 표본은 그 물체 표본 AABB + `2·max_dist + vs` 안의 것만 쓴다 — 그 밖은 결과를 바꾸지 못한다.
    """
    vs = float(voxel_size)
    c = np.asarray(centres, np.float64).reshape(-1, 3)
    seed = np.asarray(is_seed, bool).reshape(-1)
    tp = oracle.target_points
    max_d = float(oracle.max_dist_voxels) * vs
    op = oracle.other_points
    if len(op) and len(tp):
        lo = tp.min(axis=0) - (2.0 * max_d + vs)
        hi = tp.max(axis=0) + (2.0 * max_d + vs)
        op = op[np.all((op >= lo) & (op <= hi), axis=1)]
    d_t = _nearest(tp, c)
    d_o = _nearest(op, c)
    above = (np.ones(len(c), bool) if oracle.z_floor is None
             else c[:, 2] >= float(oracle.z_floor))
    surface = (d_t <= max_d) & (d_t < d_o) & above
    band = np.zeros(len(c), bool)
    if len(oracle.band_keys):
        m = oracle_band_masks(c, oracle.band_keys, oracle.band_masks, oracle.band_cell)
        bit = np.uint32(1) << np.uint32(int(oracle.code))
        band = ((m == bit) & seed & (d_t <= max_d) & above
                & (d_o > float(oracle.surface_protect_voxels) * vs) & ~surface)
    removed = seed & (surface | band)
    return {"removed": removed, "kept_other": seed & ~removed, "sign": surface | removed,
            "surface": seed & surface, "band": band,
            "d_t": d_t, "d_o": d_o, "n_other_near": int(len(op))}


# --- IFG-CL2 — begin: `esdf.target_extent == "attention_surface_hybrid"` (ball_owned + attention-surface protection)
@dataclasses.dataclass(frozen=True)
class SurfaceProtectEvidence:
    """IFG-CL2 — what protects seeds of the `ball_owned` fine target-free tier (`runtime.attention_surface_hybrid`).

    | 필드 | 무엇 |
    |---|---|
    | `target_points` | 이번 frame 의 attention-surface 선택 영역 graph node 중심 (5 mm voxel centroid) |
    | `other_points` | 선택 영역 **밖** 이고 지지면 (graph RANSAC) 이 **아닌** node 중심 — 이웃 과일 · crate · 그 밖의 관측 표면 |
    | `info` | 기록에 그대로 실린다 |

    지우는 범위는 `ball_owned` (공 + cloud 근거) 그대로다. 이 근거는 **보호에만** 쓴다 (`surface_protect_split`).
    """

    target_points: np.ndarray
    other_points: Optional[np.ndarray] = None
    info: dict = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_points", np.asarray(self.target_points, np.float64).reshape(-1, 3))
        op = self.other_points
        object.__setattr__(self, "other_points",
                           np.zeros((0, 3)) if op is None else np.asarray(op, np.float64).reshape(-1, 3))
        if not len(self.target_points):
            raise ValueError("SurfaceProtectEvidence 에 선택 영역 표본이 없습니다 (그 frame 은 보호 없이 짓는다)")


def surface_protect_split(centres: np.ndarray, is_seed: np.ndarray, ball: "TargetBall", owned: OwnedEvidence,
                          protect: SurfaceProtectEvidence, voxel_size: float) -> dict:
    """IFG-CL2 — `ball_owned` 분할 (`_ball_owned_split`, cloud 근거) 에 attention-surface 보호를 건다. 순수 numpy.

    같은 공 안 voxel 을 **두 번** 가른다 — 같은 규칙 (`OwnedEvidence` 의 tie · max_dist · unowned · ambiguous) 으로:

    | 분할 | target 근거 | non-target 근거 |
    |---|---|---|
    | `est` (= C1aN, `estimated`) | manipulated 기하 점 (`owned.target_points`) | cloud − target − 지지면 (`owned.nontarget_points`) |
    | `srf` (보호) | 선택 영역 node 중심 (`protect.target_points`) | 선택 밖 · 비지지면 node 중심 (`protect.other_points`) |

    지우는 seed = `est.removed ∧ srf.removed` — 지우는 범위는 `est` 를 넘지 않는다. `protected` = `est` 는 지우려 했는데
    `srf` 가 non-target · ambiguous (보존 쪽) 으로 본 seed. `srf` 에서 두 근거 모두 `max_dist` 밖 (unowned) 인 seed — target 의
    **안 보인 쪽** · 그림자 seed — 는 `unowned = remove` 이면 보호하지 않는다 (`est` 대로 지운다; SURF 가 실패한 부분).
    부호 교정 후보 = `est.sign ∧ srf.sign` (두 분할 모두 target 쪽이라 본 non-seed 와 지운 seed).

    돌려주는 것: `est` 의 키 전부 (분류 · `d_t` · `d_o`) 에 `removed` · `sign` 을 위 값으로 바꾸고 `protected` · `srf` 를 더한 것.
    """
    est = _ball_owned_split(centres, is_seed, ball, owned, voxel_size)
    srf_owned = dataclasses.replace(owned, target_points=protect.target_points,
                                    nontarget_points=protect.other_points)
    srf = _ball_owned_split(centres, is_seed, ball, srf_owned, voxel_size)
    removed = est["removed"] & srf["removed"]
    return {**est, "removed": removed, "sign": est["sign"] & srf["sign"],
            "protected": est["removed"] & ~srf["removed"], "srf": srf}
# --- IFG-CL2 — end


#: `stats["window"]["basis"]` 의 값. 창이 **무엇을 근거로** 놓였나 (T21).
#:
#: | 값 | 뜻 |
#: |---|---|
#: | `hand_swept` | 권한 link 구의 swept AABB (지금 자세 + 실행 구간) ∪ manipulated AABB 가 창에 들어갔다 |
#: | `manipulated` | 손 정보가 없거나 (`q_now`·모델·권한 link 없음) 옛 규칙(`target_centroid`)이다 — 물체 중심 |
#: | `overflow` | 들어가지 않았다. 지금 손 → 실행 구간 → 물체 순으로 담고 **넘친 것을 기록**했다 |
WINDOW_BASES = ("hand_swept", "manipulated", "overflow")
#: `esdf.fine_window_placement` 의 선택지 — config 의 목록을 그대로 쓴다 (두 곳에 박지 않는다).
WINDOW_PLACEMENTS = EsdfConfig.FINE_WINDOW_PLACEMENTS


@dataclasses.dataclass(frozen=True)
class FineWindowRequest:
    """pipeline → builder: 미세 창을 **어디에** 놓을지 정하는 재료 (T21, 지침 §5.1).

    배치 자체는 builder 가 한다 — 창의 크기(격자 shape × 미세 복셀)를 아는 쪽이 builder 이기 때문이다.
    pipeline 은 "권한 있는 link 의 구가 지금 어디 있고 실행 창 동안 어디를 지나가는가" 와 "제외할
    물체가 어디 있는가" 를 넘긴다. **둘은 다른 결정이다** — 창의 배치는 손이 정하고, 제외 기하
    (`TargetBall`)는 manipulated 물체만 정한다. 창을 넓히는 것과 제외 영역을 넓히는 것을 섞지 않는다
    (지침 §5.3-3).

    | 필드 | 무엇 |
    |---|---|
    | `hand_centres` | `(T, S, 3)` — 권한 link 구 중심. **step 0 = `q_now`**, 그 뒤가 실행 구간 (우선순위 순서) |
    | `hand_radii` | `(S,)` 구 반지름 |
    | `hand_pad` | 구마다 반지름 **밖에** 더할 여유 (m) = 충돌 마진 + 보간·기울기 여유 |
    | `manipulated_points` | 제외할 물체의 점군 (T20 manipulated 기하). 없으면 `None` |
    | `manipulated_pad` | 물체 AABB 에 더할 보간 여유 (m) |
    | `placement` | `hand_swept` (기본) · `target_centroid` (T21 이전 규칙 — 물체 점 평균) |
    | `hand_source` | 기록용 — `q_now` / `q_now+path` / `none` |
    | `links` | 기록용 — 권한 link 이름 |
    """

    hand_centres: Optional[np.ndarray] = None
    hand_radii: Optional[np.ndarray] = None
    hand_pad: float = 0.0
    manipulated_points: Optional[np.ndarray] = None
    manipulated_pad: float = 0.0
    placement: str = "hand_swept"
    hand_source: str = "none"
    links: tuple = ()

    def __post_init__(self) -> None:
        if self.placement not in WINDOW_PLACEMENTS:
            raise ValueError(f"placement 는 {WINDOW_PLACEMENTS} 중 하나여야 합니다: "
                             f"{self.placement!r}")
        hc = self.hand_centres
        if hc is not None:
            hc = np.asarray(hc, np.float64)
            if hc.ndim == 2:
                hc = hc[None]
            if hc.ndim != 3 or hc.shape[-1] != 3:
                raise ValueError(f"hand_centres 는 (T, S, 3) 이어야 합니다: {hc.shape}")
            radii = np.asarray(self.hand_radii if self.hand_radii is not None
                               else np.zeros(hc.shape[1]), np.float64).reshape(-1)
            if radii.shape[0] != hc.shape[1]:
                raise ValueError(f"hand_radii {radii.shape[0]} 개 ≠ 구 {hc.shape[1]} 개")
            if hc.shape[0] == 0 or hc.shape[1] == 0:
                hc, radii = None, None
            object.__setattr__(self, "hand_centres", hc)
            object.__setattr__(self, "hand_radii", radii)
        mp = self.manipulated_points
        if mp is not None:
            mp = np.asarray(mp, np.float64).reshape(-1, 3)
            object.__setattr__(self, "manipulated_points", mp if len(mp) else None)
        if float(self.hand_pad) < 0.0 or float(self.manipulated_pad) < 0.0:
            raise ValueError("pad 는 0 이상이어야 합니다")

    @property
    def has_hand(self) -> bool:
        return self.hand_centres is not None

    def step_boxes(self) -> list[tuple[np.ndarray, np.ndarray]]:
        """step 마다 `(lo, hi)` — 구 중심 ± (반지름 + `hand_pad`). step 0 = 지금 자세."""
        if self.hand_centres is None:
            return []
        grow = (self.hand_radii + float(self.hand_pad))[None, :, None]
        lo = (self.hand_centres - grow).min(axis=1)
        hi = (self.hand_centres + grow).max(axis=1)
        return [(lo[t], hi[t]) for t in range(lo.shape[0])]

    def manipulated_box(self) -> Optional[tuple[np.ndarray, np.ndarray]]:
        if self.manipulated_points is None:
            return None
        p = self.manipulated_points
        return (p.min(axis=0) - float(self.manipulated_pad),
                p.max(axis=0) + float(self.manipulated_pad))


@dataclasses.dataclass(frozen=True)
class FineWindow:
    """놓인 미세 창 한 개. `lower`·`upper` 는 **보간 격자 경계**다 — 첫/마지막 복셀 중심.

    `CuroboEsdfField._covers` 와 같은 경계를 쓴다. 두 곳이 다르면 "창 안" 의 뜻이 갈라진다.
    """

    centre: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    voxel_size: float
    shape: tuple
    basis: str
    placement: str
    #: 넘친 것 — `None` 이면 요구한 것을 다 담았다. `hand_now` · `hand_path` · `manipulated`.
    overflow: Optional[str] = None
    n_steps: int = 0
    #: 패딩한 step AABB 가 창 안에 **통째로** 든 step 수 (앞에서부터 연속일 필요는 없다).
    n_steps_covered: int = 0
    manipulated_covered: Optional[bool] = None
    required_lower: Optional[np.ndarray] = None
    required_upper: Optional[np.ndarray] = None
    hand_source: str = "none"
    links: tuple = ()

    def contains(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, np.float64).reshape(-1, 3)
        return np.all((p >= self.lower) & (p <= self.upper), axis=1)

    def contains_box(self, lo: np.ndarray, hi: np.ndarray) -> bool:
        tol = 1e-9
        return bool(np.all(lo >= self.lower - tol) and np.all(hi <= self.upper + tol))

    def summary(self) -> dict:
        def _v(x):
            return None if x is None else [round(float(v), 6) for v in np.asarray(x).reshape(-1)]
        return {
            "placement": self.placement,
            "basis": self.basis,
            "overflow": self.overflow,
            "centre_m": _v(self.centre),
            "lower_m": _v(self.lower),
            "upper_m": _v(self.upper),
            "extent_m": _v(np.asarray(self.upper) - np.asarray(self.lower)),
            "voxel_size": float(self.voxel_size),
            "shape": [int(v) for v in self.shape],
            "required_lower_m": _v(self.required_lower),
            "required_upper_m": _v(self.required_upper),
            "hand_source": self.hand_source,
            "links": [str(n) for n in self.links],
            "n_steps": int(self.n_steps),
            "n_steps_covered": int(self.n_steps_covered),
            "manipulated_covered": self.manipulated_covered,
        }


def _union(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (np.min([b[0] for b in boxes], axis=0), np.max([b[1] for b in boxes], axis=0))


def _cover_1d(s: float, length: float, lo: float, hi: float) -> float:
    """창 `[s, s+length]` 가 구간 `[lo, hi]` 를 덮는 길이."""
    return max(0.0, min(s + length, hi) - max(s, lo))


def _slide_axis(must_lo: float, must_hi: float, length: float, wants) -> float:
    """`[must_lo, must_hi]` 를 **반드시** 담으면서 `wants` (우선순위 순 구간 목록)를 사전식으로 가장
    많이 덮는 창의 시작점. 덮는 길이는 `s` 에 대해 조각별 선형이므로 꺾이는 점만 보면 된다."""
    s_min, s_max = must_hi - length, must_lo
    mid = 0.5 * (must_lo + must_hi) - 0.5 * length
    cands = {s_min, s_max, mid}
    for lo, hi in wants:
        cands.update((lo, hi - length, 0.5 * (lo + hi) - 0.5 * length))
    best, best_key = None, None
    for s in sorted(cands):
        s = min(max(s, s_min), s_max)
        key = tuple(round(_cover_1d(s, length, lo, hi), 12) for lo, hi in wants) \
            + (-abs(s - mid),)
        if best_key is None or key > best_key:
            best, best_key = s, key
    return float(best)


def place_fine_window(request: FineWindowRequest, *, shape: Sequence[int],
                      voxel_size: float) -> Optional[FineWindow]:
    """미세 창 하나를 놓는다 (T21). **규칙은 하나다:**

        창 = 손 swept AABB ∪ manipulated AABB 가 들어가면 그것 (가운데 정렬)
             안 들어가면 **지금 손 → 실행 구간 (step 순) → 물체** 순으로 담고, 넘친 것을 기록

    손 swept AABB 는 권한 link 구마다 `중심 ± (반지름 + hand_pad)` 의 합집합이다. `hand_pad` 는
    충돌 마진 + 보간·기울기 여유(복셀 몇 칸)이고 pipeline 이 config 에서 정한다.

    창의 보간 범위는 축마다 `(n − 1) × voxel` 이다 (첫/마지막 복셀 **중심** 사이 — `_covers` 와 같은
    경계). cuRobo 의 ESDF 격자 shape 이 계층마다 같으므로 미세 창의 크기는 **고정**이다 (128³ ×
    5 mm = 0.635 m 보간 범위) — 크기를 늘리는 대신 **어디에** 놓을지를 정하는 것이 이 함수다.

    손 정보가 없으면 (`q_now`·모델·권한 link 중 하나라도 없으면) 물체 점 평균에 놓는다 —
    `basis="manipulated"`. 물체도 없으면 `None` (미세 계층을 만들지 않는다 — T21 이전과 같다).
    `placement="target_centroid"` 는 T21 이전 규칙 그대로다 (물체 점 평균, 손 무시).
    """
    shape = tuple(int(v) for v in shape)
    vs = float(voxel_size)
    length = (np.asarray(shape, np.float64) - 1.0) * vs
    steps = request.step_boxes()
    manip = request.manipulated_box()
    tol = 1e-9

    def fits(box) -> bool:
        return box is not None and bool(np.all(box[1] - box[0] <= length + tol))

    def make(centre, basis, overflow, required):
        centre = np.asarray(centre, np.float64).reshape(3)
        lower = centre - 0.5 * length
        upper = centre + 0.5 * length
        win = FineWindow(centre=centre, lower=lower, upper=upper, voxel_size=vs, shape=shape,
                         basis=basis, placement=request.placement, overflow=overflow,
                         n_steps=len(steps),
                         required_lower=None if required is None else required[0],
                         required_upper=None if required is None else required[1],
                         hand_source=request.hand_source, links=tuple(request.links))
        covered = sum(1 for lo, hi in steps if win.contains_box(lo, hi))
        m_cov = None if manip is None else win.contains_box(*manip)
        object.__setattr__(win, "n_steps_covered", int(covered))
        object.__setattr__(win, "manipulated_covered", m_cov)
        return win

    if request.placement == "target_centroid" or not steps:
        if request.manipulated_points is None:
            return None
        return make(request.manipulated_points.mean(axis=0), "manipulated", None, manip)

    hand_all = _union(steps)
    required = _union([hand_all, manip])
    if fits(required):
        return make(0.5 * (required[0] + required[1]), "hand_swept", None, required)

    # 넘친다. 지금 손 → 실행 구간을 step 순으로, 들어가는 데까지 담는다.
    k = 0
    for t in range(1, len(steps) + 1):
        if fits(_union(steps[:t])):
            k = t
        else:
            break
    if k == 0:
        # 지금 자세의 손 하나도 안 들어간다 — 손 가운데에 놓고 넘친 것을 기록한다.
        lo, hi = steps[0]
        return make(0.5 * (lo + hi), "overflow", "hand_now", required)
    must = _union(steps[:k])
    overflow = "hand_path" if k < len(steps) else "manipulated"
    wants = [hand_all] + ([manip] if manip is not None else [])
    start = np.array([
        _slide_axis(float(must[0][i]), float(must[1][i]), float(length[i]),
                    [(float(w[0][i]), float(w[1][i])) for w in wants])
        for i in range(3)])
    return make(start + 0.5 * length, "overflow", overflow, required)


@dataclasses.dataclass
class _Tier:
    """한 계층의 산출물. 전부 **복사된** 배열이다 (함정 4).

    T38 B3: `values` · `site_linear` 는 **GPU tensor** 다 (cuRobo 버퍼의 device 사본). 격자 전체를
    host 로 내리지 않는다 — 필드는 `DeviceEsdfField` 로 질의점에서만 답한다. 테스트·오프라인
    호출자가 numpy 로 만든 `_Tier` 도 그대로 받는다 (`_labels_from_sites` 가 둘 다 안다).
    """
    name: str
    values: Any
    site_linear: Any
    origin: np.ndarray
    voxel_size: float
    extra: dict = dataclasses.field(default_factory=dict)


class _SeedIndex:
    """라벨 씨앗 점들의 KD-tree — `update` 당 **한 번** 짓고 계층마다 쓴다 (T38 B3).

    예전 `_labels_from_sites` 는 계층마다 같은 씨앗 점으로 `cKDTree` 를 다시 지었다. 트리는 점이
    같으면 같고 질의는 결정적이므로 하나를 나눠 써도 답은 같다.

    **씨앗 상자 밖의 site 는 묻지 않는다.** 씨앗은 작은 물체(target · 쥔 것)인데 site 는 작업공간
    전체의 표면이라, 질의의 대부분이 "스냅 반경 밖" 이라는 답을 받으려고 트리를 끝까지 내려갔다
    (T38 B3 cProfile: 계층당 ≈ 14 ms). 씨앗 AABB 를 `snap · (1 + 1e-6)` 만큼 부풀린 상자 밖의 점은
    어느 축에서 모든 씨앗과 그만큼 떨어져 있으므로 유클리드 거리가 `snap` 보다 크다 — 그 점의
    `hit` 은 묻지 않아도 False 다. 상자 안의 점은 **같은 트리 · 같은 `query(k=1)`** 로 묻고
    (scipy 는 질의점마다 독립적이다), 그래서 `hit` · `owner` 가 전부 묻는 것과 같다.
    """

    #: 상자 여유의 상대값. 부동소수 비교의 반올림 (≈ 1e-16 상대) 보다 열 자릿수 크다.
    BOX_MARGIN_REL = 1e-6

    def __init__(self, pts: np.ndarray, owner: np.ndarray):
        self.pts = pts
        self.owner = owner
        self._tree = None
        self._lo = pts.min(axis=0) if len(pts) else None
        self._hi = pts.max(axis=0) if len(pts) else None

    @classmethod
    def build(cls, seeds: dict, label_names: tuple) -> Optional["_SeedIndex"]:
        if not seeds:
            return None
        pts = np.concatenate([seeds[n] for n in label_names], axis=0)
        owner = np.concatenate([np.full(len(seeds[n]), i, np.int32)
                                for i, n in enumerate(label_names)])
        return cls(pts, owner)

    def snap(self, centres: np.ndarray, snap: float, *,
             box_filter: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """`(hit (M,) bool, owner[near[hit]] int32)` — 예전 식 그대로 (`dist <= snap`).

        `box_filter=False` 면 모든 점을 묻는다 (예전 그대로 — 동치 테스트의 기준).
        """
        if self._tree is None:
            from scipy.spatial import cKDTree

            self._tree = cKDTree(self.pts)
        centres = np.asarray(centres, np.float64).reshape(-1, 3)
        if not box_filter:
            dist, near = self._tree.query(centres, k=1)
            hit = dist <= snap
            return hit, self.owner[near[hit]]
        margin = float(snap) * (1.0 + self.BOX_MARGIN_REL)
        cand = np.flatnonzero(np.all((centres >= self._lo - margin)
                                     & (centres <= self._hi + margin), axis=1))
        hit = np.zeros(len(centres), bool)
        if not cand.size:
            return hit, self.owner[:0]
        dist, near = self._tree.query(centres[cand], k=1)
        ok = dist <= snap
        hit[cand[ok]] = True
        return hit, self.owner[near[ok]]       # cand 는 오름차순 — `used[hit]` 와 같은 순서


class CuroboFieldBuilder:
    """프레임마다 cuRobo 로 적분하고 2계층 ESDF 를 낸다. `EsdfBuilder` 와 교체 가능하다.

    같은 인스턴스를 프레임마다 재사용해야 누적 TSDF 가 의미를 갖는다 — `EsdfBuilder` 가
    격자를 한 번만 할당하는 것과 같은 이유다.
    """

    def __init__(self, config, *, bounds=None, image_hw: Optional[tuple] = None):
        self.config = config
        self._mapper = None
        #: T32 H4-fix: a second `Mapper` whose TSDF uses the fine truncation
        #: (`EsdfConfig.fine_truncation`) — the fine and target-free tiers are seeded from it, the
        #: coarse tier from `_mapper` as before. `None` when the two truncations agree (one TSDF).
        self._fine_mapper = None
        #: AG3S-TASK-DYNAMIC-MAP ①: the target-only TSDF (fine truncation), made on the first frame
        #: that routes target pixels to it. `_target_anchor` = the object centre when it was last reset
        #: (①-b: a move beyond `esdf.target_layer_reset_m` resets it); `None` = empty.
        self._target_mapper = None
        self._target_anchor: Optional[np.ndarray] = None
        self._image_hw = image_hw
        self._frames = 0
        self._bounds = self._resolve_bounds(config, bounds)
        self._label_names: tuple[str, ...] = ()

    # -- 설정 -------------------------------------------------------------------------
    @staticmethod
    def _resolve_bounds(config, bounds):
        if bounds is not None:
            return (np.asarray(bounds[0], float), np.asarray(bounds[1], float))
        if getattr(config, "bounds_lower", None) is not None:
            return (np.asarray(config.bounds_lower, float),
                    np.asarray(config.bounds_upper, float))
        return default_bounds()

    @property
    def coarse_voxel(self) -> float:
        return float(self.config.voxel_size)

    @property
    def fine_voxel(self) -> Optional[float]:
        """미세 계층 복셀. `None`/0 이면 단일 계층으로 돈다."""
        v = getattr(self.config, "fine_voxel_size", None)
        return None if not v else float(v)

    def _assert_curobo_surface(self, integrator) -> None:
        """우리가 의지하는 private 표면이 아직 있는지 확인한다.

        없으면 여기서 죽는다. 조용히 `compute_esdf` 로 되돌아가면 쥔 물체가 필드에 남아
        **어떤 해로도 못 푸는 제약 행**을 만들고(E3), 그것은 로그가 "조용한 실패" 로 부르는
        바로 그 방식이다.
        """
        missing = [n for n in ("_site_index", "_seed_esdf_impl",
                               "_propagate_and_distance_impl", "_esdf_voxel_size",
                               "_last_esdf_origin", "get_voxel_grid")
                   if not hasattr(integrator, n)]
        if missing:
            raise RuntimeError(
                "cuRobo integrator 에서 기대한 표면이 없습니다: " + ", ".join(missing) +
                ". 이 builder 는 seed 와 propagate 사이에 끼어들어 쥔 물체를 seed 에서 빼고 "
                "(A2), site_index 로 라벨 층을 만듭니다. 검증한 commit 은 78fd485 입니다 — "
                "cuRobo 를 올렸다면 그 두 곳을 다시 맞춰야 합니다.")

    # -- Mapper 를 늦게 만든다 (첫 프레임의 depth 크기가 필요하다) ----------------------
    @property
    def coarse_truncation(self) -> float:
        """TSDF truncation (m) of the TSDF the coarse tier is built from — `EsdfConfig.truncation`."""
        return float(self.config.truncation)

    @property
    def fine_truncation(self) -> float:
        """TSDF truncation (m) the fine tiers are built from (T32 H4-fix). Equals the coarse one
        when `esdf.fine_truncation_m` is `None` or there is no fine tier."""
        ft = getattr(self.config, "fine_truncation", None)
        return self.coarse_truncation if ft is None or not self.fine_voxel else float(ft)

    @property
    def separate_fine_tsdf(self) -> bool:
        """Two TSDFs (coarse truncation / fine truncation) instead of one (T32 H4-fix).

        The coarse tier stays on the 60 mm TSDF on purpose: in `T32.h4.verify.json` a shared 30 mm
        TSDF took the coarse band of the one-side-seen crate wall to **0** (chunks 16/20/30/36 px),
        and 20 mm took both walls to 0 — a coarse 20 mm voxel needs the long truncation to keep a
        negative interior at all.
        """
        return bool(self.fine_voxel) and abs(self.fine_truncation - self.coarse_truncation) > 1e-9

    def _ensure_mapper(self, cameras: Sequence[CameraDepth]):
        if self._mapper is not None:
            return self._mapper
        self._mapper = self._make_mapper(cameras, self.coarse_truncation)
        if self.separate_fine_tsdf:
            self._fine_mapper = self._make_mapper(cameras, self.fine_truncation)
        return self._mapper

    def _make_mapper(self, cameras: Sequence[CameraDepth], truncation: float):
        import torch
        from curobo.perception import Mapper, MapperCfg

        cfg = self.config
        lower, upper = self._bounds
        centre = 0.5 * (np.asarray(lower, float) + np.asarray(upper, float))
        extent = np.asarray(upper, float) - np.asarray(lower, float)
        h, w = (self._image_hw if self._image_hw is not None
                else np.asarray(cameras[0].depth).shape[:2])
        dev = getattr(cfg, "device", None) or "cuda:0"
        mcfg = MapperCfg(
            extent_meters_xyz=tuple(float(v) for v in extent),
            voxel_size=float(getattr(cfg, "tsdf_voxel_size", 0.0) or cfg.voxel_size),
            esdf_voxel_size=self.coarse_voxel,
            grid_center=torch.tensor([float(v) for v in centre], device=dev,
                                     dtype=torch.float32),
            truncation_distance=float(truncation),
            depth_minimum_distance=float(cfg.depth_min),
            depth_maximum_distance=float(cfg.depth_max),
            image_height=int(h), image_width=int(w),
            # 감쇠는 **기본 끔**이다 — numpy 경로와 같은 조건에서 두 backend 를 대조하기 위해.
            # 켤 때 값은 F20 에서 실측해 정했다 (a_t 0.99 · a_f 0.8).
            decay_factor=float(getattr(cfg, "time_decay", 1.0)),
            frustum_decay_factor=float(getattr(cfg, "frustum_decay", 1.0)),
            device=dev)
        mapper = Mapper(mcfg)
        self._assert_curobo_surface(mapper.integrator)
        return mapper

    def reset(self) -> None:
        """누적 TSDF 와 라벨 이름을 버린다. T0 이 요구하는 reset 시점 기록의 대상이다."""
        if self._mapper is not None:
            self._mapper.reset()
        if self._fine_mapper is not None:
            self._fine_mapper.reset()
        if self._target_mapper is not None:
            self._target_mapper.reset()
        self._target_anchor = None
        self._frames = 0
        self._label_names = ()

    # -- 한 프레임 --------------------------------------------------------------------
    def update(self, cameras: Sequence[CameraDepth], *,
               target_points: Optional[np.ndarray] = None,
               exclude_target: bool = False,
               target_free_points: Optional[np.ndarray] = None,
               target_free_label: Optional[str] = None,
               target_free_ball: Optional["TargetBall"] = None,
               support_points: Optional[np.ndarray] = None,
               attached_points: Optional[np.ndarray] = None,
               labelled_points: Optional[dict] = None,
               static_geometry: Optional[Sequence[Any]] = None,
               observed_at: Optional[float] = None,
               frame_id: str = "", frame_index: int = -1,
               fine_window: Optional["FineWindowRequest"] = None,
               held_free_points: Optional[np.ndarray] = None,
               nontarget_points: Optional[np.ndarray] = None,
               support_seed_evidence: Optional[dict] = None,
               owned_nontarget_points: Optional[np.ndarray] = None,
               oracle_evidence: Optional[OracleVisibleEvidence] = None,
               surface_protect: Optional[SurfaceProtectEvidence] = None,
               robot_free_spheres: Optional[dict] = None,
               target_layer: Optional[TargetLayerRequest] = None,
               roi_points: Optional[Sequence[Optional[np.ndarray]]] = None) -> CuroboEsdfField:
        """`EsdfBuilder.update` 와 같은 계약. 돌려주는 것은 `CuroboEsdfField` 다.

        `exclude_target` 는 **지원하지 않는다.** E1 이 그것을 폐기했다 — 필드는 익명이라
        아직 안 쥔 target 을 파내면 손끝뿐 아니라 전신에게 사라진다. 지금 코드는 항상
        `False` 로 부르고, `True` 로 오면 조용히 무시하지 않고 예외를 던진다.

        `target_free_points` 는 그것과 **다른 것**이다. 필드에서 파내는 것이 아니라, 그 점들의
        seed 를 지운 **미세 계층 한 겹을 따로** 만들어 `field.target_free_layers` 에 실어
        보낸다. 본 계층(`layers`)은 한 복셀도 바뀌지 않으므로 권한 없는 질의점은 예전과 같은
        답을 받는다 — 익명으로 파내는 것과 이름으로 되묻는 것의 차이가 정확히 E1 이다.
        `None`(기본)이면 그 계층을 만들지 않고 비용도 0 이다.

        `fine_window` (T21) 는 미세 창(과 target 없는 창)을 **어디에** 놓을지의 재료다 —
        `place_fine_window` 가 권한 link 의 swept AABB 와 manipulated AABB 로 놓는다. `None` 이면
        T21 이전 규칙(`target_points` 평균)이다. 어느 쪽이든 결과는 `stats["window"]` 에 실린다.

        `nontarget_points` (T44 I1) 는 `esdf.target_exclusion == "ball_owned"` 이고 공이 올 때만
        읽는다 — 공 안 seed 중 그 점들에 더 가까운 것을 보존한다 (`OwnedEvidence`). `ball` (기본)
        에서는 무시한다 (값 비트 동일). 어느 쪽이든 공 제외의 trace 가 `stats["target_free"]`
        (수) 와 `field.exclusion_trace` (ijk) 에 실린다.

        `owned_nontarget_points` (T46 F1) 는 `ball_owned` **미세** target-free 계층만 읽는 non-target 근거다
        (`esdf.nontarget_evidence == "surface"` 일 때 pipeline 이 `nontarget_points` 에서 target 표면 점을 더 뺀 것).
        `None` (기본) 이면 미세 계층도 `nontarget_points` 를 쓴다 (비트 동일). 거친 계층은 언제나 `nontarget_points` 다.
        `esdf.owned_ambiguous` (T46 F2) 도 미세 계층의 `OwnedEvidence` 에만 실린다.

        `oracle_evidence` (T46e A13, `esdf.target_extent == "oracle_visible"`) 가 오면 **미세** target-free 계층의 제외를
        공 · ownership 대신 `oracle_visible_split` (GT 표본) 으로 한다 (`_oracle_exclusion`). 공은 기록에만 남고, 거친
        target-free 계층은 그대로 공 · `target_exclusion` 이다. `None` (기본) 이면 예전 호출과 같다 (비트 동일).

        `esdf.target_free_coarse` (T44c A5) 가 켜져 있고 공이 오면 target 을 지운 **거친** 계층
        (`coarse_no_target`) 을 한 겹 더 만들어 `field.target_free_coarse_layer` 에 싣는다. 그 계층은
        `nontarget_points` 를 방식과 무관하게 읽는다 (공유 voxel 보존). 수는 `stats["target_free"]["coarse"]`,
        ijk 는 `field.coarse_exclusion_trace`.

        `target_layer` (AG3S-TASK-DYNAMIC-MAP ①, `esdf.target_layer` 일 때만) 가 오면 target pixel 을 정적 TSDF 대신
        target 전용 TSDF 에 적분하고, 본 계층 = `min(정적, target)`, target 없는 계층 = 정적 계층 그대로다 (공 · ownership
        제외를 하지 않는다). `roi_points` (②, `esdf.roi_carve` 일 때만) 는 관심 영역의 점 집합들이다 — 그 상자 ± `roi_pad` 와
        `fine_window` 의 손 경로 상자 ± `roi_pad` 안에서만 보이는 빈 공간을 지운다. 둘 다 `None` (기본) 이면 예전 호출과 같다.

        `support_seed_evidence` (T44e A8) 는 `esdf.target_free_coarse_support` 가 `"exclude"` · (A8b) `"exclude_target"` 일 때만 읽는다 —
        공 아래 지원면의 `{"points": (P, 3) inlier 점, "plane": [n, offset]}` (pipeline `_support_seed_evidence`).
        없으면 지원면 seed 를 지우지 않고 (keep 과 같음) 그 사실을 `stats["target_free"]["coarse"]["support"]` 에 남긴다.
        """
        # **인자 검증을 torch 앞에 둔다.** 이 계약은 CUDA 를 필요로 하지 않으므로, torch 가 없는
        # 프로세스(`.venv-ag3s`)의 테스트가 이것을 실제로 받아 볼 수 있어야 한다 (T7b 의
        # `announce_diagnostic_scope` 를 함수로 뺀 것과 같은 이유).
        self._check_target_free_request(target_free_points, target_points)
        # AG3S-TASK-DYNAMIC-MAP: the arguments come only with their config switch — never silently ignored.
        if target_layer is not None and not bool(getattr(self.config, "target_layer", False)):
            raise ValueError("target_layer 요청이 왔는데 esdf.target_layer 가 꺼져 있습니다")
        if roi_points is not None and not bool(getattr(self.config, "roi_carve", False)):
            raise ValueError("roi_points 가 왔는데 esdf.roi_carve 가 꺼져 있습니다")
        if surface_protect is not None:   # IFG-CL2 — protection of the `ball_owned` fine split only
            if oracle_evidence is not None:
                raise ValueError("surface_protect 와 oracle_evidence 는 함께 올 수 없습니다 (미세 분할이 둘 중 하나다)")
            if target_free_ball is None or str(getattr(self.config, "target_exclusion", "ball")) != "ball_owned":
                raise ValueError("surface_protect 는 ball_owned 미세 분할의 보호다 — 공과 "
                                 "esdf.target_exclusion = 'ball_owned' 가 필요합니다")

        import torch

        if exclude_target:
            raise ValueError(
                "exclude_target 은 E1(조작 대상을 필드에서 파내면 손끝뿐 아니라 전신에게 "
                "사라진다)로 폐기됐습니다. 접촉 권한은 to_adapter 의 manipulated_link_margin "
                "이 질의 쪽에서 처리합니다")
        if support_points is not None and len(support_points):
            raise NotImplementedError(
                "cuRobo backend 는 지지면 파내기를 하지 않습니다. live 경로는 "
                "exclude_support_surfaces=False 라 이 인자가 오지 않습니다. 필요해지면 "
                "점군의 uv 를 써서 depth 영상 공간에서 미리 마스킹하십시오 (로봇 self-filter "
                "와 같은 기전). 조용히 무시하면 지지면이 필드에 남아 평면 행과 다른 여유거리를 "
                "요구하게 됩니다 (F15)")
        if not cameras:
            # `_build_esdf` 가 이 경우를 먼저 걸러 노트를 남긴다. 여기까지 왔다면 호출부가
            # 바뀐 것이므로 조용히 빈 필드를 내지 않고 죽는다.
            raise ValueError("cuRobo backend 에 카메라 depth 가 하나도 오지 않았습니다")

        cfg = self.config
        mapper = self._ensure_mapper(cameras)
        itg = mapper.integrator
        dev = mapper._device if hasattr(mapper, "_device") else "cuda:0"

        # T38 B3: depth 는 카메라마다 **한 번** GPU 로 올리고 두 mapper 가 같은 관측을 쓴다
        # (예전에는 mapper 마다 다시 올렸다 — 청크당 6 × 1.2 MB). cuRobo 의 적분은 관측 tensor 를
        # 읽기만 한다 (`CameraProjectIntegrator.integrate` 는 `wp.from_torch` 로 읽는다; 테스트가
        # 적분 전후 바이트와 두 업로드 대비 TSDF 비트 동일을 확인한다).
        prepared = self._observations(cameras, dev)
        # AG3S-TASK-DYNAMIC-MAP ①: target pixels go to the target TSDF only (off: `prepared` as is).
        tl_stats = None
        target_prepared = None
        t_tl = time.monotonic()
        if target_layer is not None:
            prepared, target_prepared, tl_stats = self._target_layer_route(prepared, target_layer, cameras)
        tl_active = target_layer is not None and target_layer.active
        per_camera = self._integrate(mapper, prepared)
        # T32 H4-fix: the fine tiers read their own TSDF (fine truncation), integrated from the
        # same depth. `None` = one TSDF for all tiers (the pre-T32 build).
        itg_fine = None
        if self._fine_mapper is not None:
            self._integrate(self._fine_mapper, prepared, contribution=False)
            itg_fine = self._fine_mapper.integrator
        if target_prepared is not None:
            self._integrate(self._target_mapper, target_prepared, contribution=False)
        del prepared, target_prepared
        # T32 H3: the held object's old traces are set **free** in the TSDF(s) — after this frame's
        # integration, before the tiers are seeded (see `_free_tsdf_points`).
        freed = {}
        if held_free_points is not None and len(held_free_points):
            fp = np.asarray(held_free_points, np.float64).reshape(-1, 3)
            freed["coarse_tsdf"] = self._free_tsdf_points(itg, fp, self.coarse_truncation)
            if itg_fine is not None:
                freed["fine_tsdf"] = self._free_tsdf_points(itg_fine, fp, self.fine_truncation)
        # AG3S-DYNAMIC-FIELD — both off by default (no key in the stats then: bit-identical).
        dynamic = {}
        if robot_free_spheres is not None and len(robot_free_spheres.get("centres", ())):
            # D2: occupied voxels inside the robot body now (shrunk spheres, pipeline `_robot_free_spheres`).
            dynamic["robot_free"] = {"n_spheres": int(len(robot_free_spheres["centres"])),
                                     "coarse_tsdf": self._free_inside_robot(itg, robot_free_spheres,
                                                                            self.coarse_truncation)}
            if itg_fine is not None:
                dynamic["robot_free"]["fine_tsdf"] = self._free_inside_robot(itg_fine, robot_free_spheres,
                                                                             self.fine_truncation)
        if bool(getattr(cfg, "free_space_carve", False)):
            # D1: occupied voxels a depth ray of this frame passed through.
            margin_cfg = getattr(cfg, "free_space_carve_margin", None)
            dynamic["carve"] = {"coarse_tsdf": self._carve_seen_through(
                itg, cameras, self.coarse_truncation, margin_cfg)}
            if itg_fine is not None:
                dynamic["carve"]["fine_tsdf"] = self._carve_seen_through(
                    itg_fine, cameras, self.fine_truncation, margin_cfg)
        # AG3S-TASK-DYNAMIC-MAP ①: what the static TSDFs hold of the target (integrated before it was routed) → free.
        if tl_active:
            tl_stats["n_static_freed"] = {
                "coarse_tsdf": self._free_target_in_static(itg, target_layer, self.coarse_truncation)}
            if itg_fine is not None:
                tl_stats["n_static_freed"]["fine_tsdf"] = self._free_target_in_static(
                    itg_fine, target_layer, self.fine_truncation)
        # AG3S-TASK-DYNAMIC-MAP ②: visible free space → free, inside the region of interest only.
        roi_stats = None
        if roi_points is not None:
            t_roi = time.monotonic()
            pad = float(getattr(cfg, "roi_pad", 0.10))
            path = []
            if fine_window is not None:
                path = list(fine_window.step_boxes())
                mb = fine_window.manipulated_box()
                if mb is not None:
                    path.append(mb)
            path_box = ([] if not path else
                        [(np.min([b[0] for b in path], axis=0), np.max([b[1] for b in path], axis=0))])
            boxes = roi_boxes(list(roi_points), pad, path_box)
            roi_stats = {"pad_m": pad,
                         "boxes": [[[round(float(v), 4) for v in lo], [round(float(v), 4) for v in hi]]
                                   for lo, hi in boxes]}
            if boxes:
                # AG3S-ROI-CARVE-GPU: the depth goes up once for every TSDF's judgement (`cpu`: None, the numpy judgement)
                gcams = (None if str(getattr(cfg, "roi_carve_device", "cpu")) == "cpu" else
                         gpu_cameras(cameras, dev, float(cfg.depth_min), float(cfg.depth_max)))
                roi_stats["coarse_tsdf"] = self._roi_carve(itg, cameras, boxes, self.coarse_truncation, gcams)
                if itg_fine is not None:
                    roi_stats["fine_tsdf"] = self._roi_carve(itg_fine, cameras, boxes, self.fine_truncation, gcams)
                if self._target_mapper is not None and self._target_anchor is not None:
                    roi_stats["target_tsdf"] = self._roi_carve(self._target_mapper.integrator, cameras, boxes,
                                                               self.fine_truncation, gcams)
            roi_stats["ms"] = round((time.monotonic() - t_roi) * 1e3, 3)

        # 미세 계층 창의 배치 (T21, 지침 §5.1). **창의 배치는 손이 정하고, 제외할 물체는
        # manipulated 가 정한다** — 둘을 한 점(attention centroid)에 묶지 않는다. T14 seq 12–21 에서
        # 창이 target 을 따라 crate 로 가자 손가락 질의점이 창 안에 든 chunk 가 0/8 이었고, T17
        # 접근 구간의 QP 행은 전부 20 mm coarse 계층이 답했다 (T24).
        #
        # `fine_window` 가 없으면 T21 이전 규칙 — target 점 평균 — 이다 (직접 호출자, legacy 비교).
        tgt = (None if target_points is None
               else np.asarray(target_points, np.float64).reshape(-1, 3))
        request = fine_window if fine_window is not None else FineWindowRequest(
            manipulated_points=tgt, placement="target_centroid")
        window = None
        if self.fine_voxel:
            # 격자 shape 은 cuRobo 가 정한다 (`esdf_grid_shape`, 계층마다 같다) — 그래서 창의
            # **크기**는 고정이고 이 함수가 정하는 것은 위치뿐이다.
            window = place_fine_window(request, shape=tuple(int(v) for v in itg._site_index.shape),
                                       voxel_size=self.fine_voxel)
        self._fine_window = window
        self._fine_centre = None if window is None else window.centre

        # 라벨 씨앗. 쥔 물체는 내부 라벨로 함께 들고 간다 — seed 제외에도, 진단에도 쓴다.
        seeds: dict[str, np.ndarray] = {}
        if labelled_points:
            for name, pts in labelled_points.items():
                p = np.asarray(pts, np.float64).reshape(-1, 3)
                if len(p):
                    seeds[str(name)] = p
        attached = (None if attached_points is None
                    else np.asarray(attached_points, np.float64).reshape(-1, 3))
        if attached is not None and len(attached):
            seeds[ATTACHED_LABEL] = attached
        label_names = tuple(seeds)

        tiers = self._build_tiers(itg, attached, dev, torch, itg_fine=itg_fine)
        # AG3S-TASK-DYNAMIC-MAP ①: the target TSDF's tiers on the same grids (its own integrator — no shared buffer).
        target_tiers: list[_Tier] = []
        if tl_active and self._target_anchor is not None:
            titg = self._target_mapper.integrator
            target_tiers = self._build_tiers(titg, None, dev, torch, itg_fine=titg)

        # **target 없는 미세 계층.** 본 계층을 다 만든 **뒤에** 따로 만든다 — 같은 버퍼
        # (`_site_index`, `feature_tensor`)를 재사용하므로 (함정 4) 사이에 끼면 본 계층의 값이
        # 바뀐다. 미세 계층만 만드는 이유는 비용이다 (권한 있는 link 은 손 근처에 있다).
        free_tiers: list[_Tier] = []
        free_ms = 0.0
        free_points = (None if target_free_points is None
                       else np.asarray(target_free_points, np.float64).reshape(-1, 3))
        if tl_active and free_points is not None and len(free_points):
            # AG3S-TASK-DYNAMIC-MAP ①: the static fine tier **is** the target-free tier — the target never
            # entered it, so nothing is excluded (no ball, no ownership split). Copied before the composition.
            free_tiers = [_Tier(name="fine_no_target", values=t.values.clone(), site_linear=None,
                                origin=t.origin, voxel_size=t.voxel_size, extra={"target_layer": True})
                          for t in tiers if t.name == "fine"]
        elif free_points is not None and len(free_points):
            # 위의 `_check_target_free_request` 가 미세 계층을 만들 수 있음을 이미 보장한다 —
            # 조용히 0 겹으로 지나가는 길이 없다.
            t0 = time.monotonic()
            # **공이 오면 점은 안 쓴다** (T13). 점으로는 한 voxel도 안 지워졌다 (`TargetBall`).
            # 쥔 물체 점은 공과 **함께** 지운다 — 그쪽은 점이 물체 전체를 덮으므로 맞는 길이다.
            held = (None if attached is None or not len(attached)
                    else np.asarray(attached, np.float64).reshape(-1, 3))
            # T46e: oracle 도 점 제외를 쓰지 않는다 (GT 표본이 범위를 정한다).
            remove = held if (target_free_ball is not None or oracle_evidence is not None) else (
                free_points if held is None else np.vstack([held, free_points]))
            owned = None
            owned_fine = None
            if (target_free_ball is not None
                    and str(getattr(cfg, "target_exclusion", "ball")) == "ball_owned"):
                owned = OwnedEvidence(
                    target_points=free_points, nontarget_points=nontarget_points,
                    tie_voxels=float(getattr(cfg, "owned_tie_voxels", 1.0)),
                    max_dist_voxels=float(getattr(cfg, "owned_max_dist_voxels", 2.0)),
                    unowned=str(getattr(cfg, "owned_unowned", "remove")))
                owned_fine = owned
                # T46 F1 · F2 — 미세 계층에만. 기본 (근거 그대로 · keep) 이면 같은 객체 (비트 동일).
                amb = str(getattr(cfg, "owned_ambiguous", "keep"))
                if owned_nontarget_points is not None or amb != "keep":
                    owned_fine = dataclasses.replace(
                        owned, ambiguous=amb,
                        **({} if owned_nontarget_points is None
                           else {"nontarget_points": owned_nontarget_points}))
            free_tiers = self._build_tiers(itg, remove, dev, torch, only="fine",
                                           tier_name="fine_no_target",
                                           ball=target_free_ball, itg_fine=itg_fine,
                                           # T46e: oracle 이 오면 미세 계층의 분할은 그것이다 (owned 는 거친 계층만).
                                           owned=None if oracle_evidence is not None else owned_fine,
                                           **({} if oracle_evidence is None
                                              else {"oracle": oracle_evidence}),
                                           # IFG-CL2: only with `target_extent == "attention_surface_hybrid"`.
                                           **({} if surface_protect is None
                                              else {"surface_protect": surface_protect}))
            free_ms = (time.monotonic() - t0) * 1e3
            self._announce_target_free_cost(free_ms, len(free_points), free_tiers,
                                            ball=target_free_ball)

        # **T44c A5 — target 을 지운 거친 계층.** `esdf.target_free_coarse` 일 때만, 그리고 **공이 있을
        # 때만** 만든다 — 점 제외로 물러난 프레임 (`mode == "points"`) 은 target 을 못 지우므로 (T13)
        # 그 거친 계층은 target 이 그대로 든 main coarse 와 같고, 그것을 합성에 넣는 것이 금지된
        # 일이다 (사용자 판정 1-6). 미세 target-free 계층 **뒤에** 만든다 (같은 버퍼 — 함정 4).
        coarse_free_tier = None
        coarse_stats: dict = {"mode": "off"}
        if bool(getattr(cfg, "target_free_coarse", False)) and free_tiers and tl_active:
            # AG3S-TASK-DYNAMIC-MAP ①: the static coarse tier, as the fine one above.
            coarse_free_tier = next((_Tier(name="coarse_no_target", values=t.values.clone(), site_linear=None,
                                           origin=t.origin, voxel_size=t.voxel_size,
                                           extra={"target_layer": True})
                                     for t in tiers if t.name == "coarse"), None)
            coarse_stats = {"mode": "target_layer", "name": "coarse_no_target"}
        elif bool(getattr(cfg, "target_free_coarse", False)) and free_tiers:
            if target_free_ball is None:
                coarse_stats = {"mode": "skipped_no_ball",
                                "why": "target-free 미세 계층이 점 제외로 물러났다 — 거친 계층은 "
                                       "점으로 target 을 못 지운다 (T13)"}
            else:
                t0c = time.monotonic()
                spec = self._coarse_free_spec(target_free_ball, owned, nontarget_points,
                                              target_points=free_points,
                                              support_evidence=support_seed_evidence)
                built = self._build_tiers(itg, held, dev, torch, only="coarse",
                                          tier_name="coarse_no_target", ball=target_free_ball,
                                          itg_fine=itg_fine, coarse_free=spec)
                coarse_free_tier = built[0] if built else None
                c_ms = (time.monotonic() - t0c) * 1e3
                ex = dict((coarse_free_tier.extra.get("exclusion") or {})
                          if coarse_free_tier is not None else {})
                req_support = str(getattr(cfg, "target_free_coarse_support", "keep"))
                if req_support in COARSE_SUPPORT_REMOVING and "support" not in ex:
                    # T44e A8: exclude 를 요청했는데 근거 (지원면 inlier 점 · 평면) 가 없어 keep 으로 지었다.
                    ex["support"] = {"mode": req_support, "applied": False,
                                     "why": "no support evidence (support_seed_evidence)"}
                coarse_stats = {"mode": ex.get("exclusion_method"), "build_ms": round(c_ms, 3),
                                "name": "coarse_no_target",
                                **{k: v for k, v in ex.items() if k != "exclusion_method"},
                                "sign": {k: coarse_free_tier.extra.get(k) for k in
                                         ("n_attached_voxels", "n_sign_forced",
                                          "n_shared_left_alone", "threshold_voxels",
                                          "threshold_m")} if coarse_free_tier is not None else None}
                if not getattr(self, "_announced_coarse_free", False):
                    self._announced_coarse_free = True
                    logging.getLogger(__name__).warning(
                        "!!! T44c TARGET-FREE COARSE TIER IS ON (esdf.target_free_coarse) !!! "
                        "coarse_no_target 첫 프레임 %.2f ms · %s · sign_floor %s · 공 안 seed %s 중 제거 %s "
                        "· 공유 보존 %s",
                        c_ms, coarse_stats.get("mode"),
                        getattr(cfg, "target_free_coarse_sign_floor", "plane"),
                        coarse_stats.get("n_seeds_in_ball"),
                        coarse_stats.get("n_removed"), coarse_stats.get("n_shared_kept"))

        # AG3S-TASK-DYNAMIC-MAP ①: the main tiers (links without contact authority) see static ∪ target =
        # min(static, target) — E1: the not-yet-held target is never gone for the whole body.
        target_closer: dict[str, Any] = {}
        for st, tt in zip(tiers, target_tiers):
            if (st.name != tt.name or tuple(st.values.shape) != tuple(tt.values.shape)
                    or not np.allclose(st.origin, tt.origin, atol=1e-9)):
                raise RuntimeError(f"target TSDF 계층 {tt.name} 이 정적 계층 {st.name} 과 격자가 다릅니다 — "
                                   "합성하지 않고 멈춥니다")
            closer = tt.values < st.values
            st.values = torch.minimum(st.values, tt.values)
            target_closer[st.name] = closer
        if tl_stats is not None:
            tl_stats["n_target_closer"] = {k: int(v.sum()) for k, v in target_closer.items()}
            tl_stats["ms"] = round((time.monotonic() - t_tl) * 1e3, 3)

        layers = []
        # 씨앗 KD-tree 는 **update 당 한 번** 짓는다 (T38 B3). 예전에는 계층마다 같은 점으로
        # 다시 지었다 — 트리는 점이 같으면 같으므로 답도 같다.
        seed_index = _SeedIndex.build(seeds, label_names)
        for tier in tiers:
            grid = VoxelGrid(origin=tier.origin,
                             shape=tuple(int(v) for v in tier.values.shape),
                             voxel_size=tier.voxel_size)
            label_grid = self._labels_from_sites(tier, grid, seeds, label_names,
                                                 seed_index=seed_index)
            closer = target_closer.get(tier.name)
            if closer is not None and label_grid is not None and TARGET_LABEL in label_names:
                # ①: where the target TSDF answers, the nearest surface is the target (the static sites lack it).
                idx = label_names.index(TARGET_LABEL)
                if isinstance(label_grid, np.ndarray):
                    label_grid[closer.detach().cpu().numpy()] = idx
                else:
                    label_grid[closer] = idx
            # 격자는 GPU 에 남는다 — 필드는 질의점에서만 답한다 (`DeviceEsdfField`).
            layer = layer_from_device(tier.values, tier.origin, tier.voxel_size,
                                      labels=label_grid, label_names=label_names)
            layers.append(layer)
            tier.site_linear = None   # 라벨을 만들었으면 site 격자는 더 쓰지 않는다 (GPU 메모리)
        # target 없는 계층에는 **라벨을 달지 않는다.** 라벨은 "가장 가까운 표면이 무엇인가" 인데
        # 이 계층에는 그 판정의 주인공(target)이 없다. 라벨이 필요한 질문은 본 계층이 답한다.
        free_layers = tuple(layer_from_device(t.values, t.origin, t.voxel_size)
                            for t in free_tiers)
        for t in free_tiers:
            t.site_linear = None
        coarse_free_layer = None
        if coarse_free_tier is not None:
            # 라벨 없음 — target 없는 미세 계층과 같은 이유.
            coarse_free_layer = layer_from_device(coarse_free_tier.values, coarse_free_tier.origin,
                                                  coarse_free_tier.voxel_size)
            coarse_free_tier.site_linear = None

        # 격자 밖은 격자 안의 UNKNOWN 과 같은 것이므로 같은 정책을 따른다 (E4).
        outside = (cfg.max_distance if cfg.unknown_policy == "free"
                   else -cfg.max_distance)
        self._frames += 1
        self._label_names = label_names

        # **관측 판정은 필드와 함께 나간다** (2026-09-22 판정). block-sparse TSDF 가 답할 수
        # 없는 "이 부피를 봤는가" 를 원본 depth 로 직접 재고, 인증은 전역 비율이 아니라
        # **로봇 구와 쥔 물체가 실제로 지나가는 곳**의 관측 여부로 한다.
        probe = probe_from_cameras(cameras, depth_min=cfg.depth_min,
                                   depth_max=cfg.depth_max,
                                   truncation=float(cfg.truncation))

        stats = self._stats(tiers, label_names, per_camera,
                            n_attached=0 if attached is None else int(len(attached)),
                            n_static=len(static_geometry or ()))
        # T32 H3: TSDF voxels set free this frame (held-object traces), per TSDF.
        stats["held_free"] = {"n_points": (0 if held_free_points is None
                                           else int(len(held_free_points))),
                              "n_voxels_freed": freed}
        if dynamic:   # AG3S-DYNAMIC-FIELD D1 · D2 (only when one is on)
            stats["dynamic"] = dynamic
        if tl_stats is not None:   # AG3S-TASK-DYNAMIC-MAP ① (only when on)
            stats["target_layer"] = tl_stats
        if roi_stats is not None:   # AG3S-TASK-DYNAMIC-MAP ② (only when on)
            stats["roi_carve"] = roi_stats
        # **창이 어디에, 왜 놓였나** (T21 §3). 미세 계층이 없으면 `basis=None` 으로 싣는다 —
        # 키가 프레임마다 있어야 "창이 없었다" 와 "기록이 없다" 가 구별된다. 권한 link 질의점의
        # 계층별 수는 필드가 다 만들어진 뒤 pipeline 이 더한다 (`authorized_query`).
        stats["window"] = (window.summary() if window is not None
                           else {"placement": request.placement, "basis": None,
                                 "overflow": None, "hand_source": request.hand_source})
        # **target 없는 계층은 기록에 남는다.** 그 계층이 있었는지 없었는지 모르는 기록은
        # 정책이 실제로 걸린 프레임인지 알 방법이 없다 — 키를 정책이 꺼져 있을 때도 싣는다.
        stats["target_free"] = {
            "n_layers": len(free_tiers),
            "n_target_points": 0 if free_points is None else int(len(free_points)),
            "build_ms": round(float(free_ms), 3),
            "label": target_free_label,
            # **무엇으로 지웠는지 기록에 남긴다** (T13). `"points"` 로 남은 프레임은 사과가
            # 안 지워진 프레임이다 — 그 구별이 없으면 두 실행을 나란히 읽을 수 없다.
            "mode": ("target_layer" if tl_active and free_tiers   # AG3S-TASK-DYNAMIC-MAP ①
                     else "oracle_visible" if oracle_evidence is not None and free_tiers   # T46e
                     else "ball" if target_free_ball is not None
                     else ("points" if free_points is not None and len(free_points) else "none")),
            "ball": None if target_free_ball is None else target_free_ball.summary(),
            "n_ball_seeds_excluded": sum(
                int(t.extra.get("n_ball_seeds_excluded", 0)) for t in free_tiers),
            # T44 I0: 공 제외의 분해 (`_ball_exclusion`). `ball` 에서는 n_removed == n_seeds_in_ball
            # 이고 나머지는 0; 공이 없는 프레임은 method None · 전부 0.
            **self._exclusion_counts(free_tiers),
            "tiers": [{"name": t.name, "voxel_size": t.voxel_size,
                       "shape": [int(v) for v in t.values.shape],
                       "n_seeds_excluded": int(t.extra.get("n_attached_seeds_excluded", 0))}
                      for t in free_tiers],
            # T44c A5: target 을 지운 거친 계층 (`mode`: off · skipped_no_ball · ball · ball_owned).
            "coarse": coarse_stats,
        }
        if probe is not None:
            lower, upper = self._bounds
            # 정보용이다 — 인증에 쓰지 않는다. `stride` 를 함께 싣는 것이 추정값임을 밝히는 것.
            stats["frustum_observation"] = probe.observed_volume(
                lower, upper, voxel_size=self.coarse_voxel, stride=2)
        field = CuroboEsdfField(tuple(layers), outside_distance=outside, stats=stats,
                                target_free_layers=free_layers,
                                target_label=(target_free_label if free_layers else None),
                                target_free_coarse_layer=coarse_free_layer)
        field.label_names = label_names
        field.static_shapes = tuple(static_geometry or ())
        field.observation = probe
        field.attached_query_points = (None if attached is None or not len(attached)
                                       else attached)
        # T44 I0: target-free 계층에서 **실제로** 지운 seed 의 ijk (+ 그 계층의 origin/voxel) —
        # `--record-constraints` 가 chunk npz 에 싣는다. 공이 없으면 None.
        field.exclusion_trace = self._exclusion_trace(free_tiers)
        # T44c A5: 거친 target-free 계층의 trace (지운 seed · 공유로 보존한 seed 의 ijk). 없으면 None.
        field.coarse_exclusion_trace = (None if coarse_free_tier is None
                                        else self._exclusion_trace([coarse_free_tier]))
        # 출처는 **필드와 함께 나간다.** 거리값만 보면 이것이 방금 만들어진 것인지 낡은
        # 것인지 알 방법이 없다 — 2026-09-18 의 조용한 정지가 그래서 14 프레임 동안 안 보였다.
        field.provenance = FieldProvenance(
            sequence=self._frames, backend="curobo",
            observed_at=observed_at, built_at=time.monotonic(),
            frame_id=str(frame_id), frame_index=int(frame_index),
            cameras=tuple(str(c.name) for c in cameras),
            tiers=tuple({"voxel_size_m": t.voxel_size,
                         "shape": [int(v) for v in t.values.shape],
                         "origin_m": [float(v) for v in t.origin]} for t in tiers))
        return field

    # -- 조각들 ------------------------------------------------------------------------
    def _coarse_free_spec(self, ball: "TargetBall", owned: Optional["OwnedEvidence"],
                          nontarget_points, target_points=None,
                          support_evidence: Optional[dict] = None) -> CoarseFreeSpec:
        """T44c — 미세 voxel 로 정의된 설정값을 **m 로** 바꿔 `CoarseFreeSpec` 에 싣는다.

        미세 계층의 identity · 방식과 같은 것: `owned` (= 미세 계층이 쓴 `OwnedEvidence`, `ball` 이면
        None) · 같은 공. m 환산의 기준 voxel 은 그 값들이 정의된 **미세 voxel** (`fine_voxel_size`) 이다.
        """
        cfg = self.config
        fine = float(self.fine_voxel or cfg.voxel_size)
        prov = dict(getattr(ball, "provenance", {}) or {})
        plane = prov.get("support_plane_z_m")
        pad_mm = prov.get("pad_mm")
        pad_m = (float(pad_mm) / 1000.0 if pad_mm is not None
                 else float(getattr(cfg, "target_ball_pad_voxels", 1.0)) * fine)
        return CoarseFreeSpec(
            owned=owned,
            nontarget_points=nontarget_points,
            tie_m=float(getattr(cfg, "owned_tie_voxels", 1.0)) * fine,
            max_dist_m=float(getattr(cfg, "owned_max_dist_voxels", 2.0)) * fine,
            sign_threshold_m=float(getattr(cfg, "attached_sign_threshold_voxels", 1.5)) * fine,
            support_plane_z=None if plane is None else float(plane),
            support_pad_m=pad_m, fine_voxel_m=fine,
            # T44c A6: 부호 교정의 아래 한계와 그 띠의 target 쪽 판정 재료.
            sign_floor=str(getattr(cfg, "target_free_coarse_sign_floor", "plane")),
            target_points=target_points,
            # T44e A8: exclude 는 근거가 있을 때만 (없으면 keep — 지우지 않는 쪽으로 닫힌다).
            **self._coarse_support_fields(support_evidence))

    def _coarse_support_fields(self, evidence: Optional[dict]) -> dict:
        """T44e A8 — `CoarseFreeSpec` 의 `support*` 인자. config 가 `keep` 이거나 근거가 없으면 `{}` (= keep).

        T44e A8b: `"exclude_target"` 도 같은 근거를 쓰고, 발자국 증거 최소 높이를 `evidence["inlier_threshold_m"]`
        (pipeline 이 활성 `support_surface.distance_threshold` 를 싣는다) 에서 읽는다 — 없으면 기본 8 mm.
        """
        mode = str(getattr(self.config, "target_free_coarse_support", "keep"))
        return coarse_support_spec_fields(mode, evidence)

    def _check_target_free_request(self, target_free_points, target_points) -> None:
        """target 없는 계층을 **실제로** 만들 수 있는지 미리 본다. 못 만들면 여기서 죽는다.

        조용히 0 겹으로 지나가면 사과를 뺐다고 믿은 채 예전과 같은 필드로 돌고, 그 실행은
        기록만 보면 정책이 켜진 실행과 구별되지 않는다.
        """
        if target_free_points is None:
            return
        free = np.asarray(target_free_points, np.float64).reshape(-1, 3)
        if not len(free):
            return
        if not self.fine_voxel:
            raise ValueError(
                "target_free_points 가 왔는데 미세 계층이 없습니다 "
                f"(esdf.fine_voxel_size={getattr(self.config, 'fine_voxel_size', None)!r}). "
                "target 없는 계층은 미세 계층 한 겹으로 만들어지므로, 이 설정에서는 정책이 "
                "아무 일도 하지 않습니다 — fine_voxel_size 를 주거나 "
                "constraint.target_field_policy 를 'relax' 로 두십시오")
        has_target = (target_points is not None
                      and len(np.asarray(target_points, np.float64).reshape(-1, 3)) > 0)
        if not has_target:
            raise ValueError(
                "target_free_points 가 왔는데 target_points 가 비었습니다. target 없는 계층은 "
                "manipulated 물체를 빼는 계층이므로 그 물체가 없으면 뺄 것이 없습니다 — 두 인자는 "
                "같은 프레임의 같은 manipulated 기하에서 와야 합니다")

    def _observations(self, cameras, dev) -> list:
        """카메라마다 `(cam, host depth, CameraObservation)` — **GPU 업로드는 여기서 한 번** (T38 B3).

        두 mapper (coarse truncation / fine truncation) 가 같은 관측을 적분한다. 예전에는
        `_integrate` 가 mapper 마다 depth · rgb · intrinsics · pose 를 다시 올렸다.
        """
        import torch
        from curobo._src.types.camera import CameraObservation
        from curobo._src.types.pose import Pose

        out = []
        rgb = {}
        for cam in cameras:
            depth = np.asarray(cam.depth, np.float32)
            if cam.robot_mask is not None:
                # 로봇 픽셀을 0 으로 둔다 = 그 광선은 자유가 아니라 **미관측**이다. raw depth 를
                # 넣으면 로봇이 자기 몸을 장애물로 본다 (C1 — 구 120 중 105 가 자기 충돌).
                depth = np.where(np.asarray(cam.robot_mask, bool), np.float32(0.0), depth)
            h, w = depth.shape
            if (h, w) not in rgb:
                # geometry-only 적분의 0 RGB (cuRobo `Mapper.integrate` docstring 의 권장). 읽기만 한다.
                rgb[(h, w)] = torch.zeros((1, h, w, 3), device=dev, dtype=torch.uint8)
            # T38 B4: scene reconstruction 이 이 depth 를 이미 GPU 에 올려 마스크까지 적용해 두었으면
            # (같은 값 — `DepthRobotMask.masked_depth_f32_device`) 그것을 쓴다. 아니면 예전처럼 올린다.
            dd = getattr(cam, "device_depth", None)
            if (dd is not None and tuple(dd.shape) == (h, w) and dd.dtype == torch.float32
                    and _same_cuda_device(dd.device, dev)):
                depth_image = dd.contiguous()[None]
            else:
                depth_image = torch.as_tensor(depth, device=dev, dtype=torch.float32)[None]
            obs = CameraObservation(
                name=str(cam.name),
                depth_image=depth_image,
                rgb_image=rgb[(h, w)],
                intrinsics=torch.as_tensor(np.asarray(cam.camera_intrinsics, np.float32),
                                           device=dev, dtype=torch.float32)[None],
                pose=Pose.from_matrix(torch.as_tensor(
                    np.asarray(cam.T_base_cam, np.float32), device=dev,
                    dtype=torch.float32)),
                depth_to_meter=1.0)
            out.append((cam, depth, obs))
        return out

    def _integrate(self, mapper, prepared, *, contribution: bool = True) -> dict:
        """`_observations` 가 만든 관측을 이 mapper 에 적분한다. 카메라 순서는 예전과 같다.

        `contribution` 이면 카메라별 G3 기여(`_camera_contribution`)를 센다 — 같은 depth 라
        mapper 마다 다시 셀 필요가 없다 (예전에는 fine mapper 몫을 세고 버렸다).
        """
        per_camera = {}
        for cam, depth, obs in prepared:
            mapper.integrate(obs)
            if contribution:
                per_camera[str(cam.name)] = self._camera_contribution(cam, depth)
        return per_camera

    def _camera_contribution(self, cam, depth: np.ndarray) -> dict:
        """G3 가 읽을 `n_updated`. **legacy 와 같은 이름, 다른 정의다.**

        legacy 는 그 카메라가 쓴 복셀 수를 세는데 cuRobo 는 그것을 내놓지 않는다
        (`get_stats()` 의 `active_blocks` 는 *새로* 할당된 블록이라, 두 번째 프레임부터 같은
        부피를 다시 봐도 0 이 되어 살아 있는 카메라를 죽은 것으로 만든다).

        대신 G3 가 **잡으려던 셋**을 직접 센다 — 죽은 드라이버(유효 픽셀 0), depth 범위 밖
        (범위 내 0), 틀린 외부 파라미터(격자 안 0). 마지막 것은 복셀 수보다 오히려 강하다.
        비용을 위해 격자 안 판정은 8 픽셀 간격으로 훑고 그 비율을 곱한다.
        """
        cfg = self.config
        h, w = depth.shape
        valid = depth > 0
        in_range = valid & (depth >= float(cfg.depth_min)) & (depth <= float(cfg.depth_max))
        n_in_range = int(in_range.sum())
        n_in_grid = 0
        if n_in_range:
            step = 8
            sub = in_range[::step, ::step]
            vv, uu = np.nonzero(sub)
            if vv.size:
                z = depth[::step, ::step][vv, uu].astype(np.float64)
                K = np.asarray(cam.camera_intrinsics, np.float64)
                x = (uu * step - K[0, 2]) / K[0, 0] * z
                y = (vv * step - K[1, 2]) / K[1, 1] * z
                cam_pts = np.stack([x, y, z], axis=1)
                T = np.asarray(cam.T_base_cam, np.float64)
                base = cam_pts @ T[:3, :3].T + T[:3, 3]
                lower, upper = self._bounds
                inside = np.all((base >= lower) & (base <= upper), axis=1)
                # 부표본의 '격자 안' 비율을 전체 범위 내 픽셀 수에 곱한다.
                n_in_grid = int(round(float(inside.mean()) * n_in_range))
        return {
            "n_valid_px": int(valid.sum()),
            "n_in_range_px": n_in_range,
            # G3 이 읽는 이름. 0 이면 이 카메라가 이 프레임에 아무것도 기여하지 않았다.
            "n_updated": n_in_grid,
            "n_updated_definition": "back-projected depth pixels inside the workspace bounds "
                                    "(8 px stride, scaled). legacy counts written voxels.",
            "n_masked_px": (0 if cam.robot_mask is None
                            else int(np.asarray(cam.robot_mask, bool).sum())),
        }

    def _tier_specs(self, target_centre: Optional[np.ndarray], *, only: Optional[str] = None,
                    tier_name: Optional[str] = None):
        """어느 계층을 만들 것인가. `only="fine"` 이면 미세 계층 **하나만** 낸다.

        `only` 를 둔 이유는 target 없는 계층의 비용이다 — 거친 계층까지 두 겹 만들면 프레임당
        비용이 두 배가 되고, 권한 있는 link 은 손 근처(=미세 창 안)에 있으므로 얻는 것이 없다.
        """
        if only is None or only == "coarse":
            # T44c: `only="coarse"` 에 이름을 주면 (`coarse_no_target`) 그 이름으로. 거친 계층의 표지는
            # 이름이 아니라 `origin_override is None` 이다 (`_build_tiers`).
            yield ((tier_name or "coarse") if only == "coarse" else "coarse"), None, self.coarse_voxel
        if only == "coarse":
            return
        fine = self.fine_voxel
        if fine and target_centre is not None:
            yield (tier_name or "fine"), np.asarray(target_centre, np.float64), fine

    def _build_tiers(self, itg_coarse, attached, dev, torch, *, only: Optional[str] = None,
                     tier_name: Optional[str] = None,
                     ball: Optional["TargetBall"] = None,
                     itg_fine=None,
                     owned: Optional["OwnedEvidence"] = None,
                     coarse_free: Optional[CoarseFreeSpec] = None,
                     oracle: Optional[OracleVisibleEvidence] = None,
                     surface_protect: Optional[SurfaceProtectEvidence] = None) -> list[_Tier]:
        """계층마다 seed -> (제외) -> propagate -> 복사.

        `Mapper.compute_esdf` 를 부르지 않는다 — 그쪽은 CUDA graph 로 seed·propagate·distance
        를 한 덩어리로 실행해서 사이에 끼어들 자리가 없다. 대신 `compute_esdf` 가 하는 부수
        효과(`_esdf_voxel_size` 와 `_last_esdf_origin` 갱신)를 여기서 같이 한다.

        **제외 경로가 둘이고, 용도가 다르다** (T13 에서 갈랐다).

        | 인자 | 어떻게 지우나 | 누구 |
        |---|---|---|
        | `attached` | 점이 차지한 복셀 (`_exclude_seed_points`) | 쥔 물체 — 점이 물체 전체를 덮는다 |
        | `ball` | 공 안, **테이블 위만** (`_exclude_ball_seeds`) | target — 표본점 112 개로는 못 지웠다 |

        둘을 같은 함수로 쓰다가 `target_free` 가 **한 voxel도 안 지워졌다** (`TargetBall` 의 표).
        `ball` 은 `target_free` 계층에서만 오므로 **주 계층은 한 복셀도 안 바뀐다.**

        부호 교정은 **지운 복셀 집합 전체**에 걸린다 — 두 경로의 인덱스를 합쳐 넘긴다.

        `coarse_free` (T44c) 가 오면 공 제외는 `_coarse_ball_exclusion` (m 단위 규칙 + 공유 voxel 보존) 이
        하고 부호 교정 임계도 m 로 바꾼 값을 쓴다. `only="coarse"` 와 함께만 쓴다.
        """
        centre = getattr(self, "_fine_centre", None)  # update() 가 정한다
        tiers: list[_Tier] = []
        for name, origin_override, vs in self._tier_specs(centre, only=only,
                                                          tier_name=tier_name):
            # T32 H4-fix: fine tiers from the fine-truncation TSDF when there is one. 거친 계층의 표지는
            # `origin_override is None` 이다 (T44c — `coarse_no_target` 도 거친 TSDF 에서).
            itg = itg_coarse if (origin_override is None or itg_fine is None) else itg_fine
            itg._esdf_voxel_size.copy_(
                torch.tensor([float(vs)], device=dev, dtype=torch.float32))
            origin = (itg._origin if origin_override is None
                      else torch.as_tensor(origin_override, device=dev, dtype=torch.float32))
            itg._last_esdf_origin.copy_(origin)

            itg._seed_esdf_impl(origin, itg._esdf_voxel_size)

            shape = tuple(int(v) for v in itg._site_index.shape)
            org = self._origin_of(itg, shape, vs)
            n_excluded = 0
            att_idx = None
            if attached is not None and len(attached):
                n_excluded, att_idx = self._exclude_seed_points(
                    itg, attached, org, vs, shape)
            n_ball = 0
            excl = None
            sign_thr_voxels = float(getattr(self.config, "attached_sign_threshold_voxels", 1.5))
            if ball is not None and coarse_free is not None:
                # T44c A5: 같은 공 · 같은 방식을 **m 로** 이 계층 voxel 에 적용 + 공유 voxel 보존.
                n_ball, ball_idx, excl = self._coarse_ball_exclusion(itg, ball, org, vs, shape,
                                                                     coarse_free)
                sign_thr_voxels = excl["params"]["sign_threshold_voxels"]
                if ball_idx is not None:
                    att_idx = (ball_idx if att_idx is None
                               else np.unique(np.vstack([att_idx, ball_idx]), axis=0))
            elif oracle is not None:
                # T46e A13: 미세 target-free 계층의 제외를 GT 표본으로 (공 · ownership 대신). 공은 기록에만.
                n_ball, ball_idx, excl = self._oracle_exclusion(itg, oracle, org, vs, shape)
                if ball_idx is not None:
                    att_idx = (ball_idx if att_idx is None
                               else np.unique(np.vstack([att_idx, ball_idx]), axis=0))
            elif ball is not None:
                # T44 I0/I1: `ball` 은 예전 `_exclude_ball_seeds` 와 같은 cand · 같은 순서로 지운다
                # (값 비트 동일). 달라지는 것은 trace (지운 seed 의 ijk) 를 함께 돌려주는 것뿐이다.
                n_ball, ball_idx, excl = self._ball_exclusion(itg, ball, org, vs, shape,
                                                              owned=owned,
                                                              # IFG-CL2 (None = the call as before)
                                                              **({} if surface_protect is None
                                                                 else {"protect": surface_protect}))
                # 부호 교정은 **지운 복셀 전체**에 걸려야 한다. 한쪽만 넘기면 나머지 복셀의
                # 음수 부호가 그대로 남고, 그것이 `_force_sign_on_pure_voxels` 가 막는 실패다.
                if ball_idx is not None:
                    att_idx = (ball_idx if att_idx is None
                               else np.unique(np.vstack([att_idx, ball_idx]), axis=0))

            itg._propagate_and_distance_impl(origin, itg._esdf_voxel_size)
            vg = itg.get_voxel_grid()
            # 함정 4 — `feature_tensor` 와 `_site_index` 는 **둘 다** 재사용 버퍼다
            # (실측: 두 계층의 site_index 가 다르다). 다음 계층 전에 복사한다.
            # T38 B3: 복사는 **device 안에서** 한다. 예전에는 여기서 계층마다 128³ 값 + 128³ site 를
            # host 로 내렸다 (청크당 37–50 MB D2H) — 필드는 이제 GPU 에서 질의점에만 답한다.
            values = vg.feature_tensor.detach().to(torch.float32).clone(
                memory_format=torch.contiguous_format)
            site_linear = unpack_site_linear(itg._site_index.detach(), tuple(values.shape))
            sign = self._force_sign_on_pure_voxels(
                values, att_idx, vs, threshold_voxels=sign_thr_voxels)
            tier = _Tier(name=name, values=values, site_linear=site_linear,
                         origin=org, voxel_size=float(vs))
            tier.extra["tsdf_truncation_m"] = (self.coarse_truncation if itg is itg_coarse
                                               else self.fine_truncation)
            tier.extra["n_attached_seeds_excluded"] = n_excluded
            tier.extra["n_ball_seeds_excluded"] = n_ball
            if ball is not None:
                tier.extra["ball"] = ball.summary()
            if excl is not None:
                # T44 I0: counts → stats, ijk 배열 → `field.exclusion_trace` (기록만 읽는다).
                tier.extra["exclusion"] = excl["counts"]
                tier.extra["exclusion_trace"] = excl["trace"]
            tier.extra.update(sign)
            tiers.append(tier)
        return tiers

    @staticmethod
    def _exclusion_counts(free_tiers) -> dict:
        """`stats["target_free"]` 에 펼칠 T44 I0 키. 계층이 여럿이면 수를 더한다 (지금은 하나)."""
        keys = ("n_seeds_in_ball", "n_removed", "n_kept_nontarget", "n_ambiguous",
                "n_unowned_removed", "n_unowned_kept")
        out: dict = {"exclusion_method": None, **{k: 0 for k in keys}}
        for t in free_tiers:
            ex = t.extra.get("exclusion")
            if not ex:
                continue
            out["exclusion_method"] = ex.get("exclusion_method")
            for k in keys:
                out[k] += int(ex.get(k, 0))
            for k in ("n_target_owned", "n_sign_candidates", "n_sign_candidates_nonseed",
                      "n_ambiguous_removed"):   # T46 F2 (remove 일 때만 있다)
                if k in ex:
                    out[k] = out.get(k, 0) + int(ex[k])
            if "owned" in ex:
                out["owned"] = dict(ex["owned"])
            if "oracle" in ex:   # T46e A13 (oracle_visible 일 때만 있다)
                out["oracle_split"] = dict(ex["oracle"])
            if "surface_protect" in ex:   # IFG-CL2 (attention_surface_hybrid 이고 보호가 걸린 frame 만)
                out["surface_protect"] = dict(ex["surface_protect"])
            out["classify_ms"] = round(out.get("classify_ms", 0.0) + float(ex.get("classify_ms", 0.0)), 3)
        return out

    @staticmethod
    def _exclusion_trace(free_tiers) -> Optional[dict]:
        """`field.exclusion_trace` — 마지막 (유일한) 공 제외 계층의 ijk 배열 + origin/voxel."""
        for t in reversed(free_tiers):
            tr = t.extra.get("exclusion_trace")
            if tr is None:
                continue
            ex = t.extra.get("exclusion") or {}
            return {"tier": t.name, "method": ex.get("exclusion_method"),
                    "origin": np.asarray(t.origin, np.float64).copy(),
                    "voxel_size": float(t.voxel_size),
                    **{k: np.asarray(v, np.int64).reshape(-1, 3) for k, v in tr.items()}}
        return None

    def _announce_target_free_cost(self, ms: float, n_points: int, tiers, *,
                                   ball: Optional["TargetBall"] = None) -> None:
        """**첫 프레임에 비용을 크게 찍는다.** 그 뒤로는 stats 에만 남는다.

        시작 로그에 ms 가 없으면 "한 겹 더 만든다" 가 프레임 예산에 무엇을 했는지 아무도
        모른다 — 그리고 이 계층은 조용히 켜져 있으면 안 되는 종류의 것이다 (안전 계층을 한
        겹 비껴가는 길이므로).
        """
        if getattr(self, "_announced_target_free", False):
            return
        self._announced_target_free = True
        logging.getLogger(__name__).warning(
            "!!! TARGET-FREE ESDF LAYER IS ON (constraint.target_field_policy) !!!\n"
            "    미세 계층 %d 겹을 target 점 %d 개의 seed 를 지우고 한 번 더 만든다 — "
            "첫 프레임 %.2f ms%s.\n"
            "    본 계층(coarse+fine)은 한 복셀도 바뀌지 않는다. 이 계층을 읽을 권한이 있는 "
            "질의점만 사과를 통과할 수 있고, table·crate 는 이 계층에도 그대로 있다.",
            len(tiers), int(n_points), float(ms),
            "" if not tiers else f" ({tiers[0].voxel_size * 1000:.0f} mm 복셀, "
                                 f"seed {tiers[0].extra.get('n_attached_seeds_excluded', 0)} 개 제외)")
        # **무엇으로 지웠는지, 어떤 수를 재서 그렇게 정했는지 크게 적는다** (T13). 점으로 지운
        # 판은 사과를 한 voxel도 못 지웠으므로, 이 한 줄이 두 판을 가르는 유일한 표지다.
        log = logging.getLogger(__name__)
        if ball is None:
            log.warning(
                "    제외 방식 = **점** (표본점 %d 개). T13 실측에서 이 방식은 사과를 한 voxel도 "
                "지우지 못했다 (표면 2,203 중 0). 공(중심+반지름)을 주려면 pipeline 이 "
                "target_free_ball 을 실어야 한다 — 지원면 z 를 못 얻으면 여기로 물러난다.",
                int(n_points))
        else:
            log.warning("    제외 방식 = **공** %s · seed %d 개 제외 (테이블 면 z 위만).",
                        ball.summary(),
                        sum(int(t.extra.get("n_ball_seeds_excluded", 0)) for t in tiers))
            ex = self._exclusion_counts(tiers)
            if ex.get("exclusion_method") == "ball_owned":
                log.warning(
                    "    T44 I1 target_exclusion = **ball_owned** (%s) — 공 안 seed %d 중 제거 %d · "
                    "non-target 보존 %d · ambiguous 보존 %d · unowned 제거 %d / 보존 %d.",
                    ex.get("owned"), ex["n_seeds_in_ball"], ex["n_removed"],
                    ex["n_kept_nontarget"],
                    ex["n_ambiguous"] - int(ex.get("n_ambiguous_removed", 0)),   # T46 F2
                    ex["n_unowned_removed"], ex["n_unowned_kept"])
                if ex.get("n_ambiguous_removed") is not None:
                    log.warning("    T46 F2 owned_ambiguous = **remove** — ambiguous seed %d 개도 지웠다 "
                                "(제거 수에 포함).", int(ex["n_ambiguous_removed"]))

    @staticmethod
    def occupied_voxels(itg, *, band_voxels: float = CUROBO_SEED_BAND_VOXELS, surface_only: bool = False,
                        boxes: Optional[Sequence[tuple]] = None, with_sdf: bool = False):
        """Observed occupied voxels of one TSDF: `(block_pool_idx, local_idx, centres (N, 3) base frame)`.

        Occupied = weight > 0 and `tsdf ≤ band_voxels · vs` (the band cuRobo seeds the ESDF from, and
        everything behind it); `surface_only` keeps `|tsdf| ≤ band_voxels · vs`. Indexing is the inverse of
        `_free_tsdf_points`'s. Empty arrays when nothing is allocated. (AG3S-DYNAMIC-FIELD: D1 and the replay.)

        AG3S-TASK-DYNAMIC-MAP: `boxes` (`(lo, hi)` AABBs) keeps only the blocks that overlap one before any
        voxel is read (the voxels are then filtered to the boxes too); `with_sdf` appends the voxels' TSDF
        value (m) as a fourth array. Both off = the call as before.
        """
        import torch

        empty = (np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros((0, 3)))
        if with_sdf:
            empty = empty + (np.zeros(0),)
        ts = itg.tsdf
        data = ts.data if hasattr(ts, "data") else ts
        bd = data.block_data
        n_alloc = int(data.num_allocated.detach().cpu().reshape(-1)[0])
        if n_alloc <= 0 or bd.dim() != 3 or bd.shape[1] <= 1:
            return empty
        bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
        D, H, W = (int(v) for v in itg.grid_shape)
        vs = float(itg.voxel_size)
        origin = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
        coords = data.block_coords.detach().cpu().numpy().reshape(-1, 3)[:n_alloc].astype(np.int64)
        slots = data.block_to_hash_slot.detach().cpu().numpy()[:n_alloc]
        live = np.flatnonzero(slots >= 0)
        grid = np.array([W, H, D], np.int64)
        offsets = ((grid + bs - 1) // bs) // 2
        if boxes is not None and live.size:
            # the block's AABB (voxel faces) against each box — no voxel of a disjoint block is read
            blo = origin + ((coords[live] + offsets) * bs - 0.5 * grid).astype(np.float64) * vs
            bhi = blo + bs * vs
            keep = np.zeros(len(live), bool)
            for lo, hi in boxes:
                keep |= np.all((bhi >= np.asarray(lo)) & (blo <= np.asarray(hi)), axis=1)
            live = live[keep]
        if not live.size:
            return empty
        t_live = torch.as_tensor(live, device=bd.device, dtype=torch.long)
        w = bd[t_live, :, 1].float()
        s = bd[t_live, :, 0].float()
        band = float(band_voxels) * vs * w
        occ = (w > 0) & ((s.abs() <= band) if surface_only else (s <= band))
        nz = torch.nonzero(occ, as_tuple=True)
        bi, li = (t.detach().cpu().numpy() for t in nz)
        if not len(bi):
            return empty
        loc = np.stack([li % bs, (li // bs) % bs, li // (bs * bs)], axis=1)
        g = (coords[live[bi]] + offsets) * bs + loc
        centres = origin + (g.astype(np.float64) + 0.5 - 0.5 * grid) * vs
        out = (live[bi], li, centres)
        if with_sdf:
            out = out + ((s[nz] / w[nz]).detach().cpu().numpy().astype(np.float64),)
        if boxes is not None:
            inside = in_boxes(centres, boxes)
            out = tuple(a[inside] for a in out)
        return out

    # -- AG3S-TASK-DYNAMIC-MAP ---------------------------------------------------------
    @staticmethod
    def _masked_observation(obs, mask, *, keep: bool):
        """A copy of a cuRobo `CameraObservation` whose depth keeps only (`keep`) or drops the `(H, W)` mask pixels.

        Dropped pixels are 0 = unobserved, the robot-mask convention. The original tensor is not written.
        """
        import torch
        from curobo._src.types.camera import CameraObservation

        d = obs.depth_image
        m = torch.as_tensor(np.asarray(mask, bool), device=d.device)[None]
        depth = torch.where(m if keep else ~m, d, torch.zeros_like(d))
        return CameraObservation(name=obs.name, depth_image=depth, rgb_image=obs.rgb_image,
                                 intrinsics=obs.intrinsics, pose=obs.pose, depth_to_meter=1.0)

    def _target_layer_route(self, prepared, request: "TargetLayerRequest", cameras):
        """① — split this frame's observations: target pixels → the target TSDF, the rest → the static TSDFs.

        Returns `(static_prepared, target_prepared | None, stats)`. Also does ①-b: resets the target TSDF when
        it is empty, when the object is held (then nothing is routed), or when the object's centre moved more
        than `esdf.target_layer_reset_m` from where it was at the last reset.
        """
        cfg = self.config
        stats: dict = {"held": bool(request.held), "tol_m": round(float(request.tol), 6),
                       "z_floor": None if request.z_floor is None else round(float(request.z_floor), 6)}
        if not request.active:
            if self._target_anchor is not None and self._target_mapper is not None:
                self._target_mapper.reset()
                stats["reset"] = "held" if request.held else "no_target"
            self._target_anchor = None
            stats.update(n_target_px=0, active=False)
            return prepared, None, stats
        if self._target_mapper is None:
            self._target_mapper = self._make_mapper(cameras, self.fine_truncation)
        centre = np.asarray(request.centre, np.float64).reshape(3)
        reset = None
        if self._target_anchor is None:
            reset = "first"
        else:
            shift = float(np.linalg.norm(centre - self._target_anchor))
            stats["centre_shift_mm"] = round(shift * 1e3, 3)
            if shift > float(getattr(cfg, "target_layer_reset_m", 0.010)):
                reset = "moved"
        if reset is not None:
            self._target_mapper.reset()
            self._target_anchor = centre.copy()
        stats["reset"] = reset
        static, target = [], []
        n_px = {}
        for cam, depth, obs in prepared:
            m = target_pixel_mask(depth, cam.camera_intrinsics, cam.T_base_cam, request,
                                  depth_min=float(cfg.depth_min), depth_max=float(cfg.depth_max))
            n_px[str(cam.name)] = int(m.sum())
            if m.any():
                static.append((cam, depth, self._masked_observation(obs, m, keep=False)))
                target.append((cam, depth, self._masked_observation(obs, m, keep=True)))
            else:
                static.append((cam, depth, obs))
        stats.update(active=True, n_target_px=int(sum(n_px.values())), per_camera_px=n_px)
        return static, (target or None), stats

    def _free_target_in_static(self, itg, request: "TargetLayerRequest", truncation: float) -> int:
        """① — set free the static TSDF voxels that are the target's (`target_owned`).

        Clears what was integrated before the target was routed (frames before it was confirmed). Band voxels
        count within `tol` + half a voxel diagonal of a target point; voxels behind a surface (tsdf < 0) within
        this TSDF's truncation — without them the inside of the object stays negative and makes a new zero
        crossing (a shell). Ownership against the non-target evidence and `z_floor` as for pixels.
        """
        tp = request.target_points
        vs = float(itg.voxel_size)
        reach = max(float(request.tol) + 0.5 * np.sqrt(3.0) * vs, float(truncation))
        box = [(tp.min(axis=0) - reach, tp.max(axis=0) + reach)]
        pool, li, centres, sdf = self.occupied_voxels(itg, boxes=box, with_sdf=True)
        if not len(pool):
            return 0
        tol = np.where(sdf < 0.0, float(truncation), float(request.tol) + 0.5 * np.sqrt(3.0) * vs)
        own = target_owned(centres, tp, request.nontarget_points, tol, request.z_floor)
        self._set_free(itg, pool[own], li[own], truncation)
        return int(own.sum())

    @staticmethod
    def occupied_voxels_gpu(itg, *, boxes: Sequence[tuple], band_voxels: float = CUROBO_SEED_BAND_VOXELS):
        """`occupied_voxels(itg, boxes=boxes)` kept on the GPU: `(block_pool_idx, local_idx, centres (N, 3) float64)` tensors.

        AG3S-ROI-CARVE-GPU. The same blocks, voxels and centres (the same float64 arithmetic, element by element); only the
        allocated-block count is read back.
        """
        import torch

        ts = itg.tsdf
        data = ts.data if hasattr(ts, "data") else ts
        bd = data.block_data
        dev = bd.device
        empty = (torch.zeros(0, dtype=torch.long, device=dev), torch.zeros(0, dtype=torch.long, device=dev),
                 torch.zeros((0, 3), dtype=torch.float64, device=dev))
        n_alloc = int(data.num_allocated.detach().reshape(-1)[0].item())
        if n_alloc <= 0 or bd.dim() != 3 or bd.shape[1] <= 1:
            return empty
        bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
        D, H, W = (int(v) for v in itg.grid_shape)
        vs = float(itg.voxel_size)
        origin = itg.origin.detach().reshape(-1)[:3].to(dev).double()
        coords = data.block_coords.detach().reshape(-1, 3)[:n_alloc].long()
        live = torch.nonzero(data.block_to_hash_slot.detach()[:n_alloc] >= 0, as_tuple=True)[0]
        grid = torch.tensor([W, H, D], dtype=torch.long, device=dev)
        offsets = ((grid + bs - 1) // bs) // 2
        lo_hi = [(torch.as_tensor(np.asarray(lo, np.float64), device=dev), torch.as_tensor(np.asarray(hi, np.float64), device=dev))
                 for lo, hi in boxes]
        blo = origin + (((coords[live] + offsets) * bs).double() - 0.5 * grid.double()) * vs
        bhi = blo + bs * vs
        keep = torch.zeros(len(live), dtype=torch.bool, device=dev)
        for lo, hi in lo_hi:
            keep |= torch.all((bhi >= lo) & (blo <= hi), dim=1)
        live = live[keep]
        w = bd[live, :, 1].float()
        s = bd[live, :, 0].float()
        occ = (w > 0) & (s <= float(band_voxels) * vs * w)
        bi, li = torch.nonzero(occ, as_tuple=True)
        loc = torch.stack([li % bs, (li // bs) % bs, li // (bs * bs)], dim=1)
        g = (coords[live[bi]] + offsets) * bs + loc
        centres = origin + (g.double() + 0.5 - 0.5 * grid.double()) * vs
        inside = torch.zeros(len(centres), dtype=torch.bool, device=dev)
        for lo, hi in lo_hi:
            inside |= torch.all((centres >= lo) & (centres <= hi), dim=1)
        return live[bi][inside], li[inside], centres[inside]

    def _roi_judge_gpu(self, itg, cams: list[dict], boxes, truncation: float):
        """② on the GPU: `(pool, local, hit, {camera: n}, margin)` tensors for this TSDF (`gpu_cameras` of this frame)."""
        pool, li, centres = self.occupied_voxels_gpu(itg, boxes=boxes)
        vs = float(itg.voxel_size)
        margin_cfg = getattr(self.config, "free_space_carve_margin", None)
        m = float(margin_cfg) if margin_cfg is not None else float(truncation) + vs
        cfg = self.config
        fp = bool(getattr(cfg, "roi_carve_footprint", False))
        hit, per = seen_through_gpu(centres, cams, m,
                                    **({} if not fp else {"footprint_voxel": vs,
                                                          "footprint_pad_px": int(getattr(cfg, "roi_carve_footprint_pad_px", 1)),
                                                          "footprint_robot": str(getattr(cfg, "roi_carve_footprint_robot", "block"))}))
        return pool, li, hit, per, m

    def _roi_carve(self, itg, cameras, boxes, truncation: float, gpu_cams: Optional[list] = None) -> dict:
        """② — `_carve_seen_through` restricted to the occupied voxels inside the region of interest `boxes`.

        `gpu_cams` (`gpu_cameras`, AG3S-ROI-CARVE-GPU, `esdf.roi_carve_device = "cuda"`): the same judgement on the GPU.
        """
        if gpu_cams is not None:
            import torch

            pool, li, hit, per, m = self._roi_judge_gpu(itg, gpu_cams, boxes, truncation)
            self._set_free(itg, pool[hit], li[hit], truncation)
            counts = torch.stack([torch.as_tensor(len(pool), device=hit.device), hit.sum()]
                                 + [per[k] for k in per]).tolist()   # the one read-back
            fp = bool(getattr(self.config, "roi_carve_footprint", False))
            if not counts[0]:
                return {"n_roi_occupied": 0, "n_freed": 0}
            return {"n_roi_occupied": int(counts[0]), "n_freed": int(counts[1]), "margin_m": round(m, 6),
                    "per_camera": {k: int(c) for k, c in zip(per, counts[2:])}, **({"footprint": True} if fp else {}),
                    "device": "cuda"}
        pool, li, centres = self.occupied_voxels(itg, boxes=boxes)
        if not len(pool):
            return {"n_roi_occupied": 0, "n_freed": 0}
        vs = float(itg.voxel_size)
        margin_cfg = getattr(self.config, "free_space_carve_margin", None)
        m = float(margin_cfg) if margin_cfg is not None else float(truncation) + vs
        cfg = self.config
        fp = bool(getattr(cfg, "roi_carve_footprint", False))
        hit, per = seen_through(centres, cameras, m, depth_min=float(cfg.depth_min),
                                depth_max=float(cfg.depth_max),
                                **({} if not fp else {"footprint_voxel": vs,
                                                      "footprint_pad_px": int(getattr(cfg, "roi_carve_footprint_pad_px", 1)),
                                                      "footprint_robot": str(getattr(cfg, "roi_carve_footprint_robot", "block"))}))
        self._set_free(itg, pool[hit], li[hit], truncation)
        return {"n_roi_occupied": int(len(pool)), "n_freed": int(hit.sum()), "margin_m": round(m, 6),
                "per_camera": per, **({"footprint": True} if fp else {})}

    def _carve_seen_through(self, itg, cameras, truncation: float, margin: Optional[float]) -> dict:
        """AG3S-DYNAMIC-FIELD D1 — set **free** every occupied voxel of this TSDF that a depth ray passed through.

        `occupied_voxels` → `seen_through` per voxel centre with margin = `margin` or this TSDF's truncation +
        one voxel (the band integration itself already rewrites) → written as `_free_tsdf_points` writes
        (`+truncation`, weight kept). Returns `{n_occupied, n_freed, margin_m, per_camera}`.
        """
        pool, li, centres = self.occupied_voxels(itg)
        if not len(pool):
            return {"n_occupied": 0, "n_freed": 0, "per_camera": {}}
        vs = float(itg.voxel_size)
        m = float(margin) if margin is not None else float(truncation) + vs
        cfg = self.config
        hit, per = seen_through(centres, cameras, m, depth_min=float(cfg.depth_min),
                                depth_max=float(cfg.depth_max))
        self._set_free(itg, pool[hit], li[hit], truncation)
        return {"n_occupied": int(len(pool)), "n_freed": int(hit.sum()), "margin_m": round(m, 6),
                "per_camera": per}

    def _free_inside_robot(self, itg, spheres: dict, truncation: float) -> dict:
        """AG3S-DYNAMIC-FIELD D2 — set **free** the occupied voxels of this TSDF inside the robot's shrunk spheres."""
        pool, li, centres = self.occupied_voxels(itg)
        if not len(pool):
            return {"n_occupied": 0, "n_freed": 0}
        hit = inside_robot_spheres(centres, spheres)
        self._set_free(itg, pool[hit], li[hit], truncation)
        return {"n_occupied": int(len(pool)), "n_freed": int(hit.sum())}

    @staticmethod
    def _set_free(itg, pool: np.ndarray, li: np.ndarray, truncation: float) -> None:
        """Write voxels `(block pool index, local index)` as `_free_tsdf_points` does: `+truncation`, weight kept."""
        if not len(pool):
            return
        import torch

        ts = itg.tsdf
        bd = (ts.data if hasattr(ts, "data") else ts).block_data
        tb = torch.as_tensor(pool, device=bd.device, dtype=torch.long)
        tl = torch.as_tensor(li, device=bd.device, dtype=torch.long)
        bd[tb, tl, 0] = (bd[tb, tl, 1].float() * float(truncation)).to(bd.dtype)

    @staticmethod
    def _free_tsdf_points(itg, points: np.ndarray, truncation: float) -> int:
        """Set the TSDF voxels containing `points` to **free** (`+truncation`, weight kept). T32 H3.

        Why: once the held object's pixels are masked (H1) nothing new of it is integrated, but what
        was integrated **before** attach stays — its place on the table and the first lifted frames
        (measured: E3b ep1807 r1, the table spot keeps 1,400–1,540 of 1,852 fine voxels occupied
        26 requests after the lift; a masked ray is unobserved, so nothing ever clears it). The
        pipeline hands the voxels only the held object can occupy (`AG3S._held_free_points`).

        Per voxel, not `Mapper.clear_region` (whole 8³ blocks inside an AABB — erases the table
        under the object and everything else in those blocks). The weight is kept, so the voxel is
        "observed free" and later observations average against it; unallocated voxels are skipped
        (nothing to clear). Indexing = cuRobo `storage.py` `_feature_node_indices_at_centers`:
        centred block keys + `ceil(grid/BS)//2`, voxel `(p − origin)/vs + grid/2`, `z·BS² + y·BS + x`.
        Returns the number of voxels freed.
        """
        import torch

        ts = itg.tsdf
        data = ts.data if hasattr(ts, "data") else ts
        bd = data.block_data
        n_alloc = int(data.num_allocated.detach().cpu().reshape(-1)[0])
        if n_alloc <= 0 or bd.dim() != 3 or bd.shape[1] <= 1:
            return 0
        bs = int(round(bd.shape[1] ** (1.0 / 3.0)))
        D, H, W = (int(v) for v in itg.grid_shape)
        vs = float(itg.voxel_size)
        origin = itg.origin.detach().cpu().numpy().reshape(-1)[:3].astype(np.float64)
        coords = data.block_coords.detach().cpu().numpy().reshape(-1, 3)[:n_alloc].astype(np.int64)
        slots = data.block_to_hash_slot.detach().cpu().numpy()[:n_alloc]
        live = np.flatnonzero(slots >= 0)
        if not live.size:
            return 0
        grid = np.array([W, H, D], np.int64)
        offsets = ((grid + bs - 1) // bs) // 2
        g = np.floor((np.asarray(points, np.float64) - origin) / vs + 0.5 * grid).astype(np.int64)
        g = np.unique(g, axis=0)
        key = g // bs - offsets
        loc = g % bs

        def pack(k):
            k = k + 4096                            # hash keys are 13-bit signed per axis
            return (k[:, 0] << 26) | (k[:, 1] << 13) | k[:, 2]

        live_keys = pack(coords[live])
        order = np.argsort(live_keys)
        sorted_keys = live_keys[order]
        want = pack(key)
        pos = np.searchsorted(sorted_keys, want)
        pos = np.clip(pos, 0, len(sorted_keys) - 1)
        hit = sorted_keys[pos] == want
        if not hit.any():
            return 0
        pool = live[order[pos[hit]]]
        li = loc[hit, 2] * bs * bs + loc[hit, 1] * bs + loc[hit, 0]
        dev = bd.device
        t_pool = torch.as_tensor(pool, device=dev, dtype=torch.long)
        t_li = torch.as_tensor(li, device=dev, dtype=torch.long)
        w = bd[t_pool, t_li, 1].float()
        observed = w > 0
        if not bool(observed.any()):
            return 0
        bd[t_pool[observed], t_li[observed], 0] = (w[observed] * float(truncation)).to(bd.dtype)
        return int(observed.sum().item())

    @staticmethod
    def _origin_of(itg, shape, voxel_size: float) -> np.ndarray:
        """첫 복셀 **중심**. `_last_esdf_origin` 은 격자 **중심**이다 (함정 1)."""
        centre = itg._last_esdf_origin.detach().cpu().numpy().reshape(-1)[:3].astype(float)
        dims = np.asarray(shape, float) * float(voxel_size)
        return centre - 0.5 * dims + 0.5 * float(voxel_size)

    @staticmethod
    def _force_sign_on_pure_voxels(values: np.ndarray, att_idx, voxel_size: float,
                                   *, threshold_voxels: float = 1.5) -> dict:
        """쥔 물체 **단독** 복셀의 부호를 양수로. 환경 표면도 든 복셀은 그대로 둔다.

        순수 판정에 추가 데이터가 필요 없다 — seed 제외 **뒤**의 `|d|` 가 복셀 반 칸보다 크면
        그 자리에 다른 표면이 없다는 뜻이다 (있으면 PBA 가 **보존된 이웃 seed** 에서 전파해
        거리가 반 칸 안으로 나온다). 반 칸 안이면 무언가 있으므로 **손대지 않는다.**

        **이 판정이 성립하려면 제외가 팽창하지 않아야 한다** — 이웃 환경 seed 를 지우면
        판정할 근거가 사라진다 (`_exclude_attached_seeds` 의 주석 참고).

        한 복셀 안에 쥔 물체와 환경 표면이 **함께** 있으면 익명 필드로는 가를 수 없다 (F13 의
        교훈). 그때는 이 판정이 "공유" 로 보고 부호를 그대로 두므로 **보수적 쪽**으로 닫힌다.

        임계가 보수적인 쪽으로 기운다: 무언가 가까우면 언제나 "그대로 둔다" 를 고른다. 그래서
        숨길 수 있는 관통이 경계값이 아니라 **구조적으로 0** 이다 (2026-09-22 판정 B).
        """
        if att_idx is None or not len(att_idx):
            return {"n_attached_voxels": 0, "n_sign_forced": 0,
                    "n_shared_left_alone": 0,
                    "threshold_voxels": float(threshold_voxels)}
        thr = float(threshold_voxels) * float(voxel_size)
        if not isinstance(values, np.ndarray) and hasattr(values, "device"):
            # T38 B3: GPU 격자 그대로 고친다. 판정 산술은 numpy 와 같다 — float32 `|d|` 와
            # **float32 로 반올림한 임계** 의 비교 (numpy 는 float32 배열 대 Python float 비교에서
            # 스칼라를 float32 로 바꾼다; 그래서 임계를 명시적으로 float32 tensor 로 둔다).
            import torch

            t = torch.as_tensor(np.asarray(att_idx, np.int64), device=values.device)
            i, j, k = t[:, 0], t[:, 1], t[:, 2]
            d = values[i, j, k]
            thr32 = torch.tensor(np.float32(thr), dtype=torch.float32, device=values.device)
            pure = d.abs() > thr32
            values[i[pure], j[pure], k[pure]] = d[pure].abs()
            n_pure = int(pure.sum().item())
            return {"n_attached_voxels": int(len(att_idx)),
                    "n_sign_forced": n_pure,
                    "n_shared_left_alone": int(len(att_idx)) - n_pure,
                    "threshold_voxels": float(threshold_voxels),
                    "threshold_m": thr}
        i, j, k = att_idx[:, 0], att_idx[:, 1], att_idx[:, 2]
        d = values[i, j, k]
        pure = np.abs(d) > thr
        values[i[pure], j[pure], k[pure]] = np.abs(d[pure])
        return {"n_attached_voxels": int(len(att_idx)),
                "n_sign_forced": int(pure.sum()),
                "n_shared_left_alone": int((~pure).sum()),
                "threshold_voxels": float(threshold_voxels),
                "threshold_m": thr}

    def _exclude_seed_points(self, itg, points: np.ndarray, origin: np.ndarray,
                             voxel_size: float, shape):
        """**점이 실제로 차지한** 복셀의 seed 를 지운다. 쥔 물체(attached)가 쓰는 길이다.

        T13 에서 이름을 바꿨다 (`_exclude_attached_seeds` → 이것). 이름이 용도를 말해야 하는
        이유가 이 STEP 의 요지다 — 같은 함수를 `target_free` 에도 썼다가 **한 voxel도 지우지
        못했다** (`TargetBall` docstring 의 표). 점이 물체 **전체**를 덮는 쥔 물체에는 이 길이
        맞고, 표본점 112 개뿐인 target 에는 맞지 않는다. target 은 `_exclude_ball_seeds` 를 쓴다.

        지우는 것은 `site_index` 의 seed 뿐이고 TSDF 는 건드리지 않는다. 그래서 다음 프레임에
        물체를 놓으면 그 표면이 그대로 살아 있고, 쥔 물체 **뒤에** 있는 다른 장애물의 거리도
        유지된다 — `clear_region` 으로 지웠을 때 잃는 것이 그 둘이다.
        """
        import torch

        # **팽창하지 않는다.** legacy `carve` 의 `attached_dilate_voxels` 는 쥔 물체를
        # 확실히 없애려는 것이었는데, 여기서 팽창하면 **이웃 환경 표면의 seed 까지 지운다.**
        # 그러면 순수 판정(아래 `_force_sign_on_pure_voxels`)이 무력해진다 — 그 자리에 다른
        # 표면이 있는지를 "seed 제외 뒤의 거리" 로 보는데, 그 표면의 seed 를 내가 지웠으니
        # 언제나 "순수" 로 나온다. 실측으로 잡혔다: 팽창 1 에서 공유 복셀이 **0 개**로
        # 보고돼 판정 B 가 A 로 퇴화했다.
        #
        # 환경 증거를 지우는 것은 **위험한 방향**이므로 물체가 실제로 차지한 복셀만 건드린다.
        idx = np.floor((points - origin) / float(voxel_size) + 0.5).astype(np.int64)
        cand = np.unique(idx, axis=0)
        return self._clear_seeds(itg, cand, shape)

    def _exclude_ball_seeds(self, itg, ball: "TargetBall", origin: np.ndarray,
                            voxel_size: float, shape):
        """**공 안의** 복셀 seed 를 지운다 — `z >= ball.z_min` 인 것만. target 이 쓰는 길이다.

        점이 아니라 범위로 지우는 이유는 `TargetBall` 의 표에 있다. 여기서 중요한 것은 두 가지다.

        **(1) 테이블에 구멍을 내지 않는다.** `z_min` 아래 복셀은 손대지 않으므로 지원면 상판의
        seed 가 그대로 남는다. 사과 밑면이 테이블에 닿아 있어 공을 그냥 지우면 그 접촉면의
        table seed 까지 사라지고, 그러면 팔이 테이블을 통과해도 아무도 막지 않는다.

        **(2) 주 계층은 한 복셀도 안 바뀐다.** 이 함수는 `target_free` 계층을 만들 때만 불린다
        (`_build_tiers(..., ball=...)`). 그래서 권한 없는 link 의 보호는 그대로다 — 그것이
        E1 과 다른 점이고, 이 변경이 `_exclude_seed_points` 의 경고를 키우면서도 안전한 이유다.

        범위를 **복셀 중심**으로 판정한다 (`floor(...+0.5)` 와 같은 규약). 중심이 공 안이면 그
        복셀의 seed 를 지운다 — 표면 복셀이 공 껍질에 걸쳐 있을 때 반 칸의 차이로 남는 것을 줄인다.
        """
        cand = self._ball_candidates(ball, origin, voxel_size, shape)
        if cand is None:
            return 0, None
        return self._clear_seeds(itg, cand, shape)

    @staticmethod
    def _ball_candidates(ball: "TargetBall", origin: np.ndarray, voxel_size: float,
                         shape) -> Optional[np.ndarray]:
        """공 안 (`z ≥ z_min`) 복셀 인덱스 `(K, 3)` int64, 없으면 `None`. 순서는 meshgrid `ij` 순서다
        (T13 그대로 — `_exclude_ball_seeds` 와 `_ball_exclusion` 이 같은 배열을 쓴다)."""
        nx, ny, nz = (int(v) for v in shape)
        r = float(ball.radius)
        lo = np.floor((ball.centre - r - origin) / float(voxel_size) + 0.5).astype(np.int64)
        hi = np.ceil((ball.centre + r - origin) / float(voxel_size) + 0.5).astype(np.int64)
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi + 1, np.array([nx, ny, nz]))
        if np.any(hi <= lo):
            return None
        # AABB 안만 본다 — 격자 전체를 훑으면 2 백만 복셀이고 프레임 예산에 들어오지 않는다.
        ax = [np.arange(int(lo[i]), int(hi[i]), dtype=np.int64) for i in range(3)]
        grid = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        centres = origin[None, :] + grid.astype(np.float64) * float(voxel_size)
        inside = np.linalg.norm(centres - ball.centre[None, :], axis=1) <= r
        inside &= centres[:, 2] >= float(ball.z_min)   # **테이블 위만**
        cand = grid[inside]
        return cand if len(cand) else None

    @staticmethod
    def _seed_mask(itg, cand: np.ndarray) -> np.ndarray:
        """`cand` 복셀 중 지금 seed 인 것 (host bool). `site_index` 는 **읽기만** 한다."""
        import torch

        t = torch.as_tensor(cand, device=itg._site_index.device, dtype=torch.long)
        m = itg._site_index[t[:, 0], t[:, 1], t[:, 2]] >= 0
        if hasattr(m, "detach"):
            m = m.detach().cpu().numpy()
        return np.asarray(m, bool).reshape(-1)

    def _ball_exclusion(self, itg, ball: "TargetBall", origin: np.ndarray, voxel_size: float,
                        shape, *, owned: Optional[OwnedEvidence] = None,
                        protect: Optional[SurfaceProtectEvidence] = None):
        """target-free 계층의 공 제외 + trace (T44 I0/I1). `(n_removed, sign_idx | None, excl)`.

        `owned is None` → **`ball`**: `_exclude_ball_seeds` 와 같은 `cand` 를 같은 순서로
        `_clear_seeds` 에 넘긴다 (지우는 seed · 부호 교정 집합 · 값 모두 예전과 비트 동일). 지우기
        **전에** cand 의 seed 표시를 읽어 trace (지운 seed 의 ijk) 만 더 만든다.

        `owned` → **`ball_owned`**: 같은 `cand` 를 `_ball_owned_split` 로 가르고 `removed` 만 지운다.
        부호 교정은 `sign` (지운 seed ∪ target 쪽 non-seed) 에만 건다.

        `excl = {"counts": {...}, "trace": {ijk 배열}}` — counts 는 `stats["target_free"]` 로,
        trace 는 `field.exclusion_trace` 로 간다.
        """
        t0 = time.monotonic()
        method = "ball" if owned is None else "ball_owned"
        empty = np.zeros((0, 3), np.int64)
        counts = {"exclusion_method": method, "n_seeds_in_ball": 0, "n_removed": 0,
                  "n_kept_nontarget": 0, "n_ambiguous": 0, "n_unowned_removed": 0,
                  "n_unowned_kept": 0}
        trace = {"removed_ijk": empty, "nontarget_kept_ijk": empty, "ambiguous_ijk": empty,
                 "unowned_kept_ijk": empty}
        cand = self._ball_candidates(ball, origin, voxel_size, shape)
        if cand is None:
            counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return 0, None, {"counts": counts, "trace": trace}
        seed = self._seed_mask(itg, cand)
        counts["n_seeds_in_ball"] = int(seed.sum())
        if owned is None:
            n, idx = self._clear_seeds(itg, cand, shape)
            counts["n_removed"] = int(n)
            trace["removed_ijk"] = cand[seed]
            counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return n, idx, {"counts": counts, "trace": trace}
        centres = origin[None, :] + cand.astype(np.float64) * float(voxel_size)
        if protect is None:
            split = _ball_owned_split(centres, seed, ball, owned, voxel_size)
        else:
            # IFG-CL2 — the same split, then the attention-surface protection (`surface_protect_split`).
            t_p = time.monotonic()
            split = surface_protect_split(centres, seed, ball, owned, protect, voxel_size)
            protect_ms = (time.monotonic() - t_p) * 1e3
        removed = cand[split["removed"]]
        n = 0
        if len(removed):
            n, _ = self._clear_seeds(itg, removed, shape)
        sign_idx = cand[split["sign"]]
        keep_unowned = owned.unowned == "keep"
        counts.update({
            "n_removed": int(n),
            "n_target_owned": int(split["target"].sum()),
            "n_kept_nontarget": int(split["nontarget"].sum()),
            "n_ambiguous": int(split["ambiguous"].sum()),
            "n_unowned_removed": 0 if keep_unowned else int(split["unowned"].sum()),
            "n_unowned_kept": int(split["unowned"].sum()) if keep_unowned else 0,
            "n_sign_candidates": int(len(sign_idx)),
            "n_sign_candidates_nonseed": int((split["sign"] & ~seed).sum()),
            "owned": {"tie_voxels": float(owned.tie_voxels),
                      "max_dist_voxels": float(owned.max_dist_voxels),
                      "unowned": owned.unowned,
                      "n_target_points": int(len(owned.target_points)),
                      "n_nontarget_points": int(len(owned.nontarget_points)),
                      "n_nontarget_near": split["n_nontarget_near"]},
        })
        if owned.ambiguous == "remove":
            # T46 F2 — 지운 ambiguous 수 (`n_ambiguous` 는 분류 수 그대로, `n_removed` 에 이 수가 들어 있다).
            # 기본 keep 이면 키를 만들지 않는다 (stats 비트 동일).
            counts["n_ambiguous_removed"] = int(split["ambiguous"].sum())
            counts["owned"]["ambiguous"] = "remove"
        trace["removed_ijk"] = removed
        trace["nontarget_kept_ijk"] = cand[split["nontarget"]]
        trace["ambiguous_ijk"] = cand[split["ambiguous"]]
        if keep_unowned:
            trace["unowned_kept_ijk"] = cand[split["unowned"]]
        if protect is not None:
            # IFG-CL2 — `n_removed` (above) is after the protection; the `ball_owned` classes keep their meaning
            # (`est`). Kept-as-non-target = cloud non-target ∪ protected, so the trace's `nontarget_kept_ijk` holds
            # every seed kept for a non-target reason (the GT kept-seed scripts read it unchanged) and
            # `protected_ijk` is the protected subset.
            prot = split["protected"]
            srf = split["srf"]
            unowned_c = split["unowned"] & ~prot
            counts["n_kept_nontarget"] = int((split["nontarget"] | prot).sum())
            if not keep_unowned:
                counts["n_unowned_removed"] = int(unowned_c.sum())
            counts["surface_protect"] = {
                "n_protected": int(prot.sum()),
                "n_removed_estimated": int(split["removed"].sum() + prot.sum()),
                "n_protected_from_target": int((prot & split["target"]).sum()),
                "n_protected_from_unowned": int((prot & split["unowned"]).sum()),
                "n_protected_from_ambiguous": int((prot & split["ambiguous"]).sum()),
                "srf": {"n_target": int(srf["target"].sum()), "n_nontarget": int(srf["nontarget"].sum()),
                        "n_ambiguous": int(srf["ambiguous"].sum()), "n_unowned": int(srf["unowned"].sum()),
                        "n_other_near": int(srf["n_nontarget_near"])},
                "n_target_points": int(len(protect.target_points)),
                "n_other_points": int(len(protect.other_points)),
                "split_ms": round(float(protect_ms), 3),
                **{k: v for k, v in (protect.info or {}).items() if k not in ("n_target_points", "n_other_points")},
            }
            trace["nontarget_kept_ijk"] = cand[split["nontarget"] | prot]
            trace["protected_ijk"] = cand[prot]
        counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
        return n, (sign_idx if len(sign_idx) else None), {"counts": counts, "trace": trace}

    def _oracle_exclusion(self, itg, oracle: OracleVisibleEvidence, origin: np.ndarray,
                          voxel_size: float, shape):
        """T46e A13 — 미세 target-free 계층의 **GT** 제외. `(n_removed, sign_idx | None, excl)` — `_ball_exclusion` 과 같은 계약.

        후보 = target 표본에서 `max_dist` 안의 voxel (`oracle_candidates`), 분할 = `oracle_visible_split`. 지운 seed 와
        target 쪽 non-seed 가 부호 교정 후보다. 기록 키는 `ball_owned` 와 같은 이름을 쓴다 (`n_seeds_in_ball` = 후보 안 seed,
        `n_kept_nontarget` = 다른 표본에 더 가까워 남긴 seed; ambiguous · unowned 는 0) — 기존 집계 스크립트가 그대로 읽는다.
        """
        t0 = time.monotonic()
        vs = float(voxel_size)
        empty = np.zeros((0, 3), np.int64)
        counts = {"exclusion_method": "oracle_visible", "n_seeds_in_ball": 0, "n_removed": 0,
                  "n_kept_nontarget": 0, "n_ambiguous": 0, "n_unowned_removed": 0,
                  "n_unowned_kept": 0}
        trace = {"removed_ijk": empty, "nontarget_kept_ijk": empty, "ambiguous_ijk": empty,
                 "unowned_kept_ijk": empty}
        max_d = float(oracle.max_dist_voxels) * vs
        info = {"max_dist_voxels": float(oracle.max_dist_voxels),
                "n_target_points": int(len(oracle.target_points)),
                "n_other_points": int(len(oracle.other_points)), "n_candidates": 0,
                "n_band_cells": int(len(oracle.band_keys)),
                "z_floor_m": None if oracle.z_floor is None else float(oracle.z_floor)}
        cand = oracle_candidates(oracle.target_points, origin, vs, shape, max_d)
        if cand is None:
            counts["oracle"] = info
            counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return 0, None, {"counts": counts, "trace": trace}
        seed = self._seed_mask(itg, cand)
        centres = origin[None, :] + cand.astype(np.float64) * vs
        split = oracle_visible_split(centres, seed, oracle, vs)
        removed = cand[split["removed"]]
        n = 0
        if len(removed):
            n, _ = self._clear_seeds(itg, removed, shape)
        sign_idx = cand[split["sign"]]
        info.update({"n_candidates": int(len(cand)), "n_other_near": split["n_other_near"],
                     "n_removed_surface": int(split["surface"].sum()),
                     "n_removed_band": int(split["band"].sum())})
        counts.update({
            "n_seeds_in_ball": int(seed.sum()),
            "n_removed": int(n),
            "n_target_owned": int(split["removed"].sum()),
            "n_kept_nontarget": int(split["kept_other"].sum()),
            "n_sign_candidates": int(len(sign_idx)),
            "n_sign_candidates_nonseed": int((split["sign"] & ~seed).sum()),
            "oracle": info,
        })
        trace["removed_ijk"] = removed
        trace["nontarget_kept_ijk"] = cand[split["kept_other"]]
        counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
        return n, (sign_idx if len(sign_idx) else None), {"counts": counts, "trace": trace}

    def _coarse_ball_exclusion(self, itg, ball: "TargetBall", origin: np.ndarray,
                               voxel_size: float, shape, spec: CoarseFreeSpec):
        """T44c A5 — `coarse_no_target` 의 공 제외. `(n_removed, sign_idx | None, excl)`.

        미세 계층 (`_ball_exclusion`) 과 **같은 공 (중심 · 반지름 m) · 같은 `target_exclusion`** 을 쓴다.
        다른 것은 해상도 때문에 생기는 셋뿐이고, 셋 다 m 로 정의해 이 voxel 에 적용한다
        (`coarse_exclusion_params`):

        1. **지원면 guard** `z_min` — 지원면이 이 계층 voxel 을 심을 수 있는 높이 위만.
        2. **공유 voxel 보존** — seed 중심에서 `band_m` (= 0.9 · vs, 이 voxel 을 심을 수 있는 표면 거리)
           안에 non-target 관측점이 있으면 그 seed 를 **지우지 않는다** (사용자 판정 1-4: target 과
           obstacle 이 거친 voxel 을 공유하면 obstacle 증거 보존). 거친 voxel 하나에는 target 과 이웃
           과일이 함께 들기 쉽다 (20 mm). 같은 영역의 non-seed voxel 은 부호 교정에서도 뺀다 — 그
           voxel 의 음수가 이웃 물체의 것일 수 있다.
        3. **`ball_owned` 의 거리** — tie 는 m 그대로, unowned 거리는 띠까지 (`coarse_exclusion_params`).

        부호 교정 집합 = 지운 seed ∪ target 쪽 non-seed (`ball`: 공 안 전부, `ball_owned`: `split["sign"]`)
        **− 공유 영역**. 기록: `n_shared_kept` (공유라 안 지운 seed — `ball` 이면 지웠을 것, `ball_owned`
        면 분류가 지우라고 한 것) · `n_seeds_shared` (공유 영역의 seed 전부) · `n_shared_cell` (non-target
        점이 **그 voxel 정육면체 안**에 있는 seed — 문자 그대로의 "같은 voxel").
        """
        t0 = time.monotonic()
        vs = float(voxel_size)
        params = coarse_exclusion_params(ball, vs, spec)
        method = "ball" if spec.owned is None else "ball_owned"
        empty = np.zeros((0, 3), np.int64)
        counts = {"exclusion_method": method, "n_seeds_in_ball": 0, "n_removed": 0,
                  "n_shared_kept": 0, "n_seeds_shared": 0, "n_shared_cell": 0,
                  "n_kept_nontarget": 0, "n_ambiguous": 0, "n_unowned_removed": 0,
                  "n_unowned_kept": 0, "n_sign_candidates": 0,
                  "n_nontarget_points": int(len(spec.nontarget_points)),
                  "params": {k: (round(v, 6) if isinstance(v, float) else v)
                             for k, v in params.items()}}
        trace = {"removed_ijk": empty, "shared_kept_ijk": empty}
        ball_c = TargetBall(centre=ball.centre, radius=ball.radius, z_min=params["z_min_m"],
                            provenance=dict(ball.provenance))
        support_idx = None
        protect_idx = None
        if spec.support in COARSE_SUPPORT_REMOVING:
            # T44e A8: 지원면 seed 를 **먼저** 지운다 — 아래 공 제외 · 띠 부호 교정이 읽는 seed 표시는 그 뒤의 것이다.
            # 지운 지원면 seed 도 부호 교정 집합에 든다 (순수 판정) — 안 그러면 seed 를 지워도 테이블 윗면 바로 아래
            # voxel 의 음수 (−먼 seed 거리) 와 위 voxel 의 양수 사이 보간이 평면 근처에 영교차를 다시 만든다.
            # T44e A8b (`exclude_target`): target 발자국 안 · non-target 발자국 밖만 지운다. 두 발자국이 겹쳐 남긴
            # 지원면 seed 는 `protect` — 아래 공 규칙 (guard 가 공의 z_min 으로 내려갔다) 도 지우지 않는다.
            sup_out: dict = {}
            support_idx = self._coarse_support_exclusion(itg, origin, vs, shape, spec, params, counts,
                                                         ball=ball, out=sup_out)
            trace["support_removed_ijk"] = support_idx
            support_idx = support_idx if len(support_idx) else None
            if spec.support == "exclude_target":
                protect_idx = sup_out.get("protect_ijk", empty)
                trace["support_overlap_kept_ijk"] = protect_idx
                protect_idx = protect_idx if len(protect_idx) else None
        # T44c A6 (`sign_floor="plane"`): guard 아래 · 공의 z_min 위 띠의 부호 교정. seed 를 지우기 **전에**
        # 읽는다 (띠의 seed 는 지우지 않지만, 읽는 순서를 한 곳에 고정한다). "guard" 면 None — A5 그대로.
        plane_idx = self._coarse_plane_band_sign(itg, ball, ball_c, origin, vs, shape, spec,
                                                 params, counts)
        cand = self._ball_candidates(ball_c, origin, vs, shape)
        if cand is None:
            counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return 0, self._join_sign(plane_idx, support_idx), {"counts": counts, "trace": trace,
                                                                 "params": params}
        seed = self._seed_mask(itg, cand)
        counts["n_seeds_in_ball"] = int(seed.sum())
        centres = origin[None, :] + cand.astype(np.float64) * vs
        band = float(params["band_m"])
        nt = spec.nontarget_points
        if len(nt):
            # 공 중심에서 r + band 밖의 점은 어느 cand 의 band 안에도 들 수 없다 — 결과는 전체 점과 같다.
            nt = nt[np.linalg.norm(nt - ball.centre[None, :], axis=1)
                    <= float(ball.radius) + band + vs]
        shared_region = np.zeros(len(cand), bool)
        shared_cell = np.zeros(len(cand), bool)
        if len(nt):
            from scipy.spatial import cKDTree

            tree = cKDTree(nt)
            d_o, _ = tree.query(centres, k=1)
            shared_region = np.asarray(d_o, np.float64) <= band
            d_inf, _ = tree.query(centres, k=1, p=np.inf)
            shared_cell = np.asarray(d_inf, np.float64) <= 0.5 * vs
        if spec.owned is None:
            want_removed = seed
            want_sign = np.ones(len(cand), bool)
        else:
            owned_c = OwnedEvidence(
                target_points=spec.owned.target_points, nontarget_points=spec.owned.nontarget_points,
                tie_voxels=float(params["tie_voxels"]),
                max_dist_voxels=float(params["max_dist_voxels"]),
                unowned=spec.owned.unowned)
            split = _ball_owned_split(centres, seed, ball_c, owned_c, vs)
            want_removed = split["removed"]
            want_sign = split["sign"]
            keep_unowned = owned_c.unowned == "keep"
            # 공유 판정 **뒤**의 분류 수 (공유로 보존된 seed 는 아래 n_shared_kept 로 따로 센다).
            free = ~shared_region
            counts.update({
                "n_target_owned": int((split["target"] & free).sum()),
                "n_kept_nontarget": int((split["nontarget"] & free).sum()),
                "n_ambiguous": int((split["ambiguous"] & free).sum()),
                "n_unowned_removed": 0 if keep_unowned else int((split["unowned"] & free).sum()),
                "n_unowned_kept": int((split["unowned"] & free).sum()) if keep_unowned else 0,
            })
        keep_region = shared_region
        if protect_idx is not None:
            # T44e A8b — 겹침으로 남긴 지원면 seed 는 공유 영역처럼 다룬다 (지우지 않고 부호도 고치지 않는다).
            # 기존 카운트 (`n_shared_kept` 등) 는 공유 판정만 센다 — 이것은 따로 센다.
            dims = np.asarray(shape, np.int64)
            lin = lambda a: (a[:, 0] * dims[1] + a[:, 1]) * dims[2] + a[:, 2]  # noqa: E731
            prot = np.isin(lin(cand), lin(protect_idx))
            counts["n_support_protected_in_ball"] = int((want_removed & prot & ~shared_region).sum())
            keep_region = shared_region | prot
        removed_mask = want_removed & ~keep_region
        sign_mask = want_sign & ~keep_region
        removed = cand[removed_mask]
        n = 0
        if len(removed):
            n, _ = self._clear_seeds(itg, removed, shape)
        sign_idx = cand[sign_mask]
        counts.update({
            "n_removed": int(n),
            "n_shared_kept": int((want_removed & shared_region).sum()),
            "n_seeds_shared": int((seed & shared_region).sum()),
            "n_shared_cell": int((seed & shared_cell).sum()),
            "n_sign_candidates": int(len(sign_idx)),
        })
        if plane_idx is not None:
            # 띠는 guard 위 cand 와 겹치지 않는다 (z 로 가른다) — 그냥 잇는다.
            sign_idx = np.vstack([sign_idx, plane_idx])
        if support_idx is not None:
            # T44e A8: 공 cand 와 겹칠 수 있다 — 중복 없이 합친다 (keep 에서는 이 줄을 지나지 않는다).
            sign_idx = self._join_sign(sign_idx if len(sign_idx) else None, support_idx)
            sign_idx = np.zeros((0, 3), np.int64) if sign_idx is None else sign_idx
        trace["removed_ijk"] = removed
        trace["shared_kept_ijk"] = cand[want_removed & shared_region]
        counts["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
        return n, (sign_idx if len(sign_idx) else None), {"counts": counts, "trace": trace,
                                                          "params": params}

    @staticmethod
    def _join_sign(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """T44e A8 — 두 부호 교정 ijk 집합의 합집합 (`None` = 빈 집합). 하나가 비면 다른 하나를 그대로."""
        if a is None or not len(a):
            return b
        if b is None or not len(b):
            return a
        return np.unique(np.vstack([a, b]), axis=0)

    def _coarse_support_exclusion(self, itg, origin: np.ndarray, vs: float, shape,
                                  spec: CoarseFreeSpec, params: dict, counts: dict,
                                  ball: Optional["TargetBall"] = None,
                                  out: Optional[dict] = None) -> np.ndarray:
        """T44e A8 — `coarse_no_target` 에서 **지원면 seed** 를 지운다. 지운 voxel ijk `(K, 3)` 를 돌려준다.

        지원면 seed = 다음을 모두 만족하는 seed voxel (중심 `c`):

        | # | 조건 | 왜 |
        |---|---|---|
        | 1 | `|n·c − offset| ≤ reach` (`reach = 0.9·vs + 0.5·vs`) | 그 면이 이 voxel 을 seed 로 심을 수 있는 높이. TSDF 영교차가 맞춘 평면보다 몇 mm 위라 띠에 반 칸을 더한다 |
        | 2 | 그 면의 inlier 점이 `reach` 안 | **관측 근거**. 면의 무한 연장 (테이블 밖 · 다른 물체 밑) 을 지원면으로 읽지 않는다 |
        | 3 | non-target 관측점이 `band` (`0.9·vs`) 안에 **없다** | A5 의 공유 규칙 그대로 — distractor · crate 와 voxel 을 나누면 장애물 증거를 보존한다 |
        | 4 | (A8b, `"exclude_target"` 만) target 발자국 안 · non-target 발자국 밖 (`_coarse_support_footprints`) | 이웃 과일 · 기타 물체 아래의 지원면 seed 는 남긴다 (사용자 판정 2026-10-11). 겹치면 남긴다 |

        면은 공 아래 그 면 하나다 (`AG3S._support_plane_z` 와 같은 규칙으로 pipeline 이 고른다) — 그래서 crate 바닥 등
        다른 평면은 건드리지 않는다. 지운 seed 는 호출자가 부호 교정 집합에 넣는다 (순수 판정, `_coarse_ball_exclusion`) —
        seed 띠 밖 (테이블 속 더 깊은 층) 의 음수는 그대로라 이 계층의 영교차는 평면 아래로 내려간다. 테이블은 미세 `f`
        와 본 합성 `d` 가 답한다 (`max(c, d) ≥ d` — 새 값은 언제나 `min(old, d) ≤ new ≤ old`).

        `"exclude_target"` 이면 `out["protect_ijk"]` 에 **두 발자국이 겹치는** 띠 안 seed (조건 1 + 4 의 겹침, 조건 2
        는 보지 않는다 — 보존에는 근거가 필요 없다) 를 싣는다. 호출자의 공 규칙이 이것을 지우지 않는다.
        """
        t0 = time.monotonic()
        band = float(params["band_m"])
        reach = float(params["support_reach_m"])
        P = spec.support_points
        plane = spec.support_plane
        target_mode = spec.support == "exclude_target"
        rec = {"mode": str(spec.support), "applied": True, "n_support_points": int(len(P)),
               "band_m": round(band, 6), "reach_m": round(reach, 6),
               "plane": [round(float(v), 6) for v in plane],
               "n_seeds": 0, "n_shared_kept": 0, "n_removed": 0}
        if target_mode:
            rec["footprint"] = {"dilation_m": round(float(params["footprint_dilation_m"]), 6),
                                "min_height_m": round(float(params["footprint_min_height_m"]), 6),
                                "n_band_seeds": 0, "n_in_target": 0, "n_overlap_kept": 0,
                                "n_outside_target_kept": 0, "n_target_points": 0,
                                "n_nontarget_points": 0}
        counts["support"] = rec
        empty = np.zeros((0, 3), np.int64)
        if out is not None and target_mode:
            out["protect_ijk"] = empty
        if not len(P):
            rec["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return empty
        dims = np.asarray(shape, np.int64)
        # 후보 = 근거 점의 AABB (± reach) 안 voxel 전부 — 거의 수평인 면이라 z 로 몇 층뿐이다. 그 안에서 평면 띠 →
        # seed → 근거 점 거리 순으로 줄인다 (팽창 · unique 없이; 지원면 하나에 수 ms).
        lo = np.maximum(np.floor((P.min(axis=0) - reach - origin) / vs + 0.5).astype(np.int64), 0)
        hi = np.minimum(np.floor((P.max(axis=0) + reach - origin) / vs + 0.5).astype(np.int64) + 1, dims)
        if np.any(hi <= lo):
            rec["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return empty
        ax = [np.arange(int(lo[i]), int(hi[i]), dtype=np.int64) for i in range(3)]
        cand = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1).reshape(-1, 3)
        centres = origin[None, :] + cand.astype(np.float64) * vs
        n, off = plane[:3], float(plane[3])
        near_plane = np.abs(centres @ n - off) <= reach
        cand, centres = cand[near_plane], centres[near_plane]
        if len(cand):
            seed = self._seed_mask(itg, cand)
            cand, centres = cand[seed], centres[seed]
        in_t = in_n = None
        if target_mode and len(cand):
            # A8b 조건 4 — 띠 안 seed 전부에 대해 (근거 조건 2 앞): 겹침 보존은 근거 없이도 한다.
            in_t, in_n, fp = self._coarse_support_footprints(centres, spec, params, ball)
            rec["footprint"].update(fp)
            rec["footprint"]["n_band_seeds"] = int(len(cand))
            if out is not None:
                out["protect_ijk"] = cand[in_t & in_n]
        if len(cand):
            near_pts = _nearest(P, centres) <= reach
            cand, centres = cand[near_pts], centres[near_pts]
            if in_t is not None:
                in_t, in_n = in_t[near_pts], in_n[near_pts]
        rec["n_seeds"] = int(len(cand))
        if not len(cand):
            rec["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
            return empty
        shared = np.zeros(len(cand), bool)
        nt = spec.nontarget_points
        if len(nt):
            # 후보 상자 (± band) 밖의 점은 어느 후보의 band 안에도 들 수 없다 — 결과는 전체 점과 같다.
            clo, chi = centres.min(axis=0) - band, centres.max(axis=0) + band
            nt = nt[np.all((nt >= clo[None, :]) & (nt <= chi[None, :]), axis=1)]
        if len(nt):
            shared = _nearest(nt, centres) <= band
        rec["n_shared_kept"] = int(shared.sum())
        drop = ~shared
        if in_t is not None:
            # 조건 4: target 발자국 안 · non-target 발자국 밖만. 기록은 근거 (조건 1–3) 를 통과한 seed 기준.
            fp = rec["footprint"]
            fp["n_in_target"] = int((in_t & ~shared).sum())
            fp["n_overlap_kept"] = int((in_t & in_n & ~shared).sum())
            fp["n_outside_target_kept"] = int((~in_t & ~shared).sum())
            drop &= in_t & ~in_n
        remove = cand[drop]
        if len(remove):
            nr, _ = self._clear_seeds(itg, remove, shape)
            rec["n_removed"] = int(nr)
        rec["classify_ms"] = round((time.monotonic() - t0) * 1e3, 3)
        return remove if len(remove) else empty

    @staticmethod
    def _coarse_support_footprints(centres: np.ndarray, spec: CoarseFreeSpec, params: dict,
                                   ball: Optional["TargetBall"]):
        """T44e A8b — 지원면 seed 중심마다 (target 발자국 안, non-target 발자국 안) bool 두 개와 기록.

        **발자국** = 지원면에 수직으로 내린 그림자 (평면 위 거리, m). 둘 다 같은 팽창 `D = √2 · vs` (거친 voxel 대각,
        `COARSE_SUPPORT_FOOTPRINT_DILATION_VOXELS` 의 근거) 를 쓴다.

        | 발자국 | 증거 | seed 중심 `c` 가 안이다 |
        |---|---|---|
        | target | 공 (중심 · 반지름 — 미세 계층이 지운 바로 그 공, pad 포함) + target 점 (평면 위 `h_min` 초과) | 평면 거리 (`c`, 공 중심) ≤ `r + D` 또는 (`c`, 가장 가까운 target 점) ≤ `D` |
        | non-target | non-target 관측점 (fused cloud − target − 지원면 inlier, 평면 위 `h_min` 초과) | 평면 거리 (`c`, 가장 가까운 점) ≤ `D` |

        `h_min` (`support_min_height_m`, 기본 8 mm = RANSAC inlier 문턱) 아래 점은 같은 규칙으로 테이블과 구별되지 않아
        발자국 증거가 아니다 — 안 그러면 inlier 에 못 든 테이블 점이 target 둘레를 모두 non-target 발자국으로 덮는다.
        공이 없으면 (`ball=None`) target 발자국은 target 점만이다.
        """
        plane = spec.support_plane
        n, off = plane[:3], float(plane[3])
        dil = float(params["footprint_dilation_m"])
        hmin = float(params["footprint_min_height_m"])

        def onto(p):
            p = np.asarray(p, np.float64).reshape(-1, 3)
            return p - (p @ n - off)[:, None] * n[None, :]

        def above(p):
            if p is None or not len(p):
                return np.zeros((0, 3))
            return p[(p @ n - off) > hmin]

        cq = onto(centres)
        in_t = np.zeros(len(cq), bool)
        if ball is not None:
            bc = onto(np.asarray(ball.centre, np.float64))[0]
            in_t |= np.linalg.norm(cq - bc[None, :], axis=1) <= float(ball.radius) + dil
        tp = above(spec.target_points)
        if len(tp):
            in_t |= _nearest(onto(tp), cq) <= dil
        in_n = np.zeros(len(cq), bool)
        nt = above(spec.nontarget_points)
        n_nt = 0
        if len(nt) and in_t.any():
            # target 발자국 안 seed 만 본다 (밖은 어차피 남는다). 그 상자 ± D 밖의 점은 볼 필요가 없다.
            q = cq[in_t]
            ntp = onto(nt)
            clo, chi = q.min(axis=0) - dil, q.max(axis=0) + dil
            ntp = ntp[np.all((ntp >= clo[None, :]) & (ntp <= chi[None, :]), axis=1)]
            n_nt = int(len(ntp))
            if n_nt:
                in_n[np.flatnonzero(in_t)] = _nearest(ntp, q) <= dil
        return in_t, in_n, {"n_target_points": int(len(tp)), "n_nontarget_points": n_nt}

    def _coarse_plane_band_sign(self, itg, ball: "TargetBall", ball_c: "TargetBall",
                                origin: np.ndarray, vs: float, shape, spec: CoarseFreeSpec,
                                params: dict, counts: dict) -> Optional[np.ndarray]:
        """T44c A6 — `sign_floor="plane"` 의 **부호 교정 띠**. `(K, 3)` ijk 또는 None (`"guard"` · 빈 띠).

        띠 = 공 안이고 `sign_z_min_m ≤ z < z_min_m` 인 거친 voxel (공의 z_min 은 미세 계층과 같은 m 값,
        guard 는 거친 voxel 때문에 높인 값). 여기서는 **seed 를 하나도 지우지 않는다** — 테이블 seed 가
        이 높이에 있다 (guard 의 이유). 부호만 고친다:

        * **non-seed 만** — seed 는 값이 0 이라 `_force_sign_on_pure_voxels` 가 어차피 건드리지 않지만,
          집합에서도 뺀다 (기록이 "고칠 수 있었던 것" 만 세게).
        * **공유 영역은 뺀다** — seed 중심 `band_m` 안에 non-target 관측점이 있는 voxel (A5 와 같은 규칙).
        * **target 쪽만** — `ball_owned` 는 그 분류의 non-seed 규칙 (`d_t < d_o − tie`), `ball` 은
          `d_t ≤ d_o` (non-target 관측이 더 가깝지 않다). `ball` 에 target 점이 없으면 공유 판정만.

        지원면 아래 (`z < sign_z_min_m`) 는 후보가 아니다 — 테이블 내부의 음수는 그대로다.
        """
        z_lo = params.get("sign_z_min_m")
        if spec.sign_floor != "plane" or z_lo is None or not float(z_lo) < float(params["z_min_m"]):
            return None
        ball_p = TargetBall(centre=ball.centre, radius=ball.radius, z_min=float(z_lo),
                            provenance=dict(ball.provenance))
        cand = self._ball_candidates(ball_p, origin, vs, shape)
        rec = {"n_plane_band": 0, "n_plane_band_seeds": 0, "n_plane_band_shared": 0,
               "n_plane_band_nontarget_side": 0, "n_plane_band_sign": 0}
        counts["plane_band"] = rec
        if cand is None:
            return None
        centres = origin[None, :] + cand.astype(np.float64) * vs
        cand = cand[centres[:, 2] < float(ball_c.z_min)]
        centres = centres[centres[:, 2] < float(ball_c.z_min)]
        if not len(cand):
            return None
        seed = self._seed_mask(itg, cand)
        band = float(params["band_m"])
        nt = spec.nontarget_points
        if len(nt):
            nt = nt[np.linalg.norm(nt - ball.centre[None, :], axis=1)
                    <= float(ball.radius) + band + vs]
        d_o = _nearest(nt, centres)
        shared = d_o <= band
        if spec.owned is not None:
            owned_c = OwnedEvidence(
                target_points=spec.owned.target_points, nontarget_points=spec.owned.nontarget_points,
                tie_voxels=float(params["tie_voxels"]),
                max_dist_voxels=float(params["max_dist_voxels"]),
                unowned=spec.owned.unowned)
            target_side = _ball_owned_split(centres, seed, ball_c, owned_c, vs)["sign"]
        elif spec.target_points is not None and len(spec.target_points):
            target_side = _nearest(spec.target_points, centres) <= d_o
        else:
            target_side = np.ones(len(cand), bool)
        sign = ~seed & ~shared & target_side
        rec.update({"n_plane_band": int(len(cand)), "n_plane_band_seeds": int(seed.sum()),
                    "n_plane_band_shared": int((~seed & shared).sum()),
                    "n_plane_band_nontarget_side": int((~seed & ~shared & ~target_side).sum()),
                    "n_plane_band_sign": int(sign.sum())})
        return cand[sign] if sign.any() else None

    def _clear_seeds(self, itg, cand: np.ndarray, shape):
        """`site_index` 의 seed 를 지우고 `(지운 수, 복셀 인덱스)` 를 돌려준다.

        두 경로(점·공)가 이 한 곳을 쓴다. 부호 교정이 **같은 복셀 집합**에 걸려야 하므로
        인덱스를 함께 돌려주는 것이 계약이다.
        """
        import torch

        nx, ny, nz = (int(v) for v in shape)
        inside = np.all((cand >= 0) & (cand < np.array([nx, ny, nz])), axis=1)
        cand = cand[inside]
        if not len(cand):
            return 0, None
        t = torch.as_tensor(cand, device=itg._site_index.device, dtype=torch.long)
        before = int((itg._site_index[t[:, 0], t[:, 1], t[:, 2]] >= 0).sum().item())
        itg._site_index[t[:, 0], t[:, 1], t[:, 2]] = -1
        return before, cand

    @staticmethod
    def _labels_from_sites(tier: _Tier, grid: VoxelGrid, seeds: dict,
                           label_names: tuple, *, seed_index: Optional["_SeedIndex"] = None):
        """`site_index` 조회로 라벨 격자를 만든다.

        절차는 셋이다.
        1. 실제로 site 로 쓰인 복셀만 모은다 (표면 복셀이고, 보통 격자의 수 % 다).
        2. 그 복셀 중심에서 **가장 가까운 씨앗 점**을 찾아 스냅 반경 안이면 그 라벨을 준다.
           `label_snap_voxels` 가 그 반경이다 — 씨앗 점이 떨어진 복셀과 TSDF 영교차가 정한
           표면 복셀이 같지 않기 때문에 스냅이 필요하다 (라벨 스냅).
        3. 복셀마다 `site_label[site_index[복셀]]` 을 읽는다.

        **T38 B3 — `site_linear` 가 GPU tensor 면 1 · 3 을 GPU 에서 한다** (결과도 GPU int32 tensor).
        1 의 `np.unique` (청크당 정렬 2 백만 × 계층 수 — Phase A 에서 이 함수 134 ms 의 대부분) 는
        "쓰인 site 표시 → `nonzero`" 로 바꾼다: 같은 집합 · 같은 오름차순이다. 2 의 KD-tree 질의는
        CPU 그대로 (scipy, 같은 트리 · 같은 질의점 · 같은 순서) 이고, 내려오는 것은 쓰인 site 의
        인덱스(표면 복셀 수)뿐이다. 트리는 `seed_index` 로 update 당 한 번 짓는다.
        numpy `_Tier` 면 예전 경로 그대로 (numpy 배열을 돌려준다).
        """
        if not seeds:
            return None
        if seed_index is None:
            seed_index = _SeedIndex.build(seeds, label_names)
        site_lin = tier.site_linear
        if not isinstance(site_lin, np.ndarray) and hasattr(site_lin, "device"):
            return CuroboFieldBuilder._labels_from_sites_device(tier, seed_index)

        shape = tuple(int(v) for v in tier.values.shape)
        n_vox = int(np.prod(shape))
        site_lin = site_lin.reshape(-1)
        used = np.unique(site_lin[site_lin >= 0])
        if not used.size:
            return None

        centres = CuroboFieldBuilder._site_centres(used, shape, tier.origin, tier.voxel_size)
        hit, owner = seed_index.snap(centres, 2.0 * tier.voxel_size)
        site_label = np.full(n_vox, -1, np.int32)
        site_label[used[hit]] = owner

        out = np.full(n_vox, -1, np.int32)
        valid = site_lin >= 0
        out[valid] = site_label[site_lin[valid]]
        return out.reshape(shape)

    @staticmethod
    def _site_centres(used: np.ndarray, shape, origin, voxel_size: float) -> np.ndarray:
        """쓰인 site 의 선형 인덱스 → 복셀 **중심** (host, float64). 두 경로가 같은 식을 쓴다."""
        nx, ny, nz = (int(v) for v in shape)
        sx = used // (ny * nz)
        sy = (used // nz) % ny
        sz = used % nz
        return (origin + np.stack([sx, sy, sz], axis=1).astype(float) * voxel_size)

    @staticmethod
    def _labels_from_sites_device(tier: _Tier, seed_index: "_SeedIndex"):
        """GPU 판 — `_labels_from_sites` 의 docstring 참고. `(nx, ny, nz)` int32 tensor 또는 `None`."""
        import torch

        site_lin = tier.site_linear.reshape(-1)
        shape = tuple(int(v) for v in tier.values.shape)
        n_vox = int(np.prod(shape))
        dev = site_lin.device
        valid = site_lin >= 0
        marked = torch.zeros(n_vox, dtype=torch.bool, device=dev)
        marked[site_lin[valid]] = True
        used_d = torch.nonzero(marked).reshape(-1)          # = np.unique(...) — 정렬 · 중복 없음
        if not used_d.numel():
            return None
        used = used_d.cpu().numpy()
        centres = CuroboFieldBuilder._site_centres(used, shape, tier.origin, tier.voxel_size)
        hit, owner = seed_index.snap(centres, 2.0 * tier.voxel_size)
        site_label = torch.full((n_vox,), -1, dtype=torch.int32, device=dev)
        if hit.any():
            site_label[used_d[torch.as_tensor(hit, device=dev)]] = torch.as_tensor(
                owner, dtype=torch.int32, device=dev)
        out = torch.where(valid, site_label[site_lin.clamp(min=0)],
                          torch.full((), -1, dtype=torch.int32, device=dev))
        return out.reshape(shape)

    def _observation_coverage(self, coarse: _Tier) -> dict:
        """"이 부피가 관측됐는가" 에 대한 답. **0.0 으로 보고하지 않는다.**

        legacy 의 `unknown_fraction` 은 dense TSDF 에서 **가중치가 0 인 복셀의 비율**이다.
        광선이 지나간 자유 공간은 가중치를 받으므로 `known` 이고, 아무도 안 본 곳만 `unknown`
        이다. 그 구분이 G2(미관측 100 % 필드가 `valid` · `certified True` 로 보고됐다)의 축이다.

        **cuRobo 의 block-sparse TSDF 는 그 구분을 할 수 없다.** 표면 근처 블록만 할당하므로
        "보고 지나간 자유 공간" 과 "한 번도 안 본 곳" 이 **둘 다 미할당**이다. 따라서
        할당 블록 부피로 비율을 만들면 언제나 "거의 다 미관측" 이 나오고, 그것은 legacy 의
        숫자와 **같은 것을 재지 않는다** — 정의를 안 적고 두 숫자를 나란히 놓는 것이
        C2("거친 33 mm 대 미세 2 mm 는 측정법 산물")를 만든 방식이다.

        그래서 **지어내지 않는다.** `unknown_fraction` 을 `None` 으로 두고 그 사실을 싣는다.
        G2 가 실제로 필요한 것(아무것도 안 봤는가)은 `n_active_blocks == 0` 이 그대로 답한다.
        이 자리는 F16(블록-스파스가 만드는 새 안전 질문 — 할당 안 한 곳을 물으면 뭐라
        답하는가)이 실제로 도착한 지점이다.
        """
        stats = {}
        try:
            st = self._mapper.get_stats()
            active = int(st.get("active_blocks", 0))
        except Exception:  # noqa: BLE001 — 통계가 없어도 필드는 유효하다
            active = -1
        tsdf_vs = float(getattr(self.config, "tsdf_voxel_size", 0.0)
                        or self.config.voxel_size)
        stats["n_active_blocks"] = active
        stats["observed_block_volume_m3"] = max(active, 0) * (8.0 * tsdf_vs) ** 3
        stats["grid_volume_m3"] = float(np.prod(
            np.asarray(coarse.values.shape, float) * coarse.voxel_size))
        # 아무것도 안 봤으면 그것은 확실히 100 % 미관측이다 — 그 경우만 수치로 답한다.
        stats["unknown_fraction"] = 1.0 if active == 0 else None
        stats["unknown_fraction_available"] = active == 0
        stats["unknown_fraction_definition"] = (
            "block-sparse TSDF 는 '보고 지나간 자유 공간' 과 '한 번도 안 본 곳' 을 구분하지 "
            "못한다 (둘 다 미할당). 그래서 legacy 의 per-voxel UNKNOWN 비율에 대응하는 값이 "
            "없다. active_blocks == 0 일 때만 1.0 으로 확정한다.")
        return stats

    def _stats(self, tiers, label_names, per_camera, *, n_attached: int,
               n_static: int) -> dict:
        coarse = tiers[0]
        occupied = int((coarse.values < 0).sum())
        return {
            "backend": "curobo",
            "voxel_size": coarse.voxel_size,
            "grid_shape": tuple(int(v) for v in coarse.values.shape),
            "n_voxels": int(np.prod(coarse.values.shape)),
            "frames": self._frames,
            "n_occupied": occupied,
            **self._observation_coverage(coarse),
            "n_labels": len(label_names),
            "label_names": list(label_names),
            "n_attached_points": n_attached,
            "n_attached_seeds_excluded": sum(
                t.extra.get("n_attached_seeds_excluded", 0) for t in tiers),
            "attached_sign_correction": [
                {"tier": t.name, **{k: v for k, v in t.extra.items()
                                    if k != "n_attached_seeds_excluded"}} for t in tiers],
            "n_static_shapes": n_static,
            # T32 H4-fix: which TSDF truncation each tier was built from.
            "tsdf_truncation": {"coarse_m": self.coarse_truncation,
                                "fine_m": self.fine_truncation,
                                "fine_requested_m": getattr(self.config, "fine_truncation_m", None),
                                "separate_fine_tsdf": self.separate_fine_tsdf},
            "decay": {"decayed": bool(
                float(getattr(self.config, "time_decay", 1.0)) < 1.0
                or float(getattr(self.config, "frustum_decay", 1.0)) < 1.0)},
            "tiers": [{"name": t.name, "voxel_size": t.voxel_size,
                       "shape": [int(v) for v in t.values.shape],
                       "origin": [float(v) for v in t.origin]} for t in tiers],
            "per_camera": per_camera,
        }

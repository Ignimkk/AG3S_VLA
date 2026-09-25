---
name: curobo-esdf
description: cuRobo 로 ESDF/TSDF 를 만들 때의 함정 5개와 eikonal 자가진단. curobo_field, compute_esdf, VoxelGrid, esdf_origin 을 만지거나 .venv-curobo 에서 거리장을 생산할 때 쓴다.
---

# cuRobo ESDF — 함정 다섯

유래는 archive `benchmark/ag3s/docs/AG3S_REVIEW_LOG.md` 다.

1. **`esdf_origin` 은 격자 코너가 아니라 중심이다.**
2. **인덱싱을 손으로 쓰지 않는다** — `create_xyzr_tensor(transform_to_origin=True)` 를 쓴다.
3. **첫 `compute_esdf` 호출 시간(약 1500 ms)은 JIT 다 — 인용하지 않는다.** 두 번째부터 재라.
4. **`compute_esdf()` 결과는 다음 호출 전에 복사한다** — `feature_tensor.detach().clone()`.
   복사하지 않으면 다음 호출이 같은 버퍼를 덮어써 앞 프레임의 값이 조용히 바뀐다.
5. **`VoxelGrid.get_occupied_voxels()` 는 반대 부호 규약이다 — 쓰지 않는다.**

## 자가진단 — 1·3·4 는 이 하나로 잡힌다

자유공간에서 거리장의 기울기 크기가 1 인가 (**eikonal** 조건 `|∇d| ≈ 1`).
거리장은 "가장 가까운 장애물까지의 거리" 이므로, 장애물이 없는 곳에서 한 칸 움직이면
거리도 정확히 한 칸 변해야 한다. 어긋나면 셋 중 하나다: origin 이 반 칸 밀렸거나(1),
버퍼가 덮어써졌거나(4), 아직 JIT 중인 값을 읽었다(3).

## venv 경계

셋이다. **`.venv-openpi-live`**(py3.11, numpy 1.26.4)에 **mujoco 3.11 · casadi · osqp · torch 2.7.1 · curobo 0.8.0 · warp 가 전부** 있다 (2026-09-25 실측) — mujoco 와 curobo 가 **한 프로세스에서 같이 필요한 것은 여기서 돌린다.**

나머지 둘은 분리 경로다: **생산(`.venv-curobo`) → npz → 소비(`.venv-ag3s`)**. 둘은 numpy 버전 때문에
합칠 수 없다 — `.venv-curobo` 는 py3.10 / numpy 1.26.4 (torch·curobo·warp),
`.venv-ag3s` 는 py3.11 / numpy 2.4.6 (mujoco·casadi·osqp·scipy).
`.venv-curobo` 에서 `benchmark` 임포트는 되지만 mujoco/casadi/osqp 가 없다.
**거리장 어댑터는 순수 numpy 라 양쪽에서 돈다.**

GPU 는 H200 한 장이다. 병렬 세션에서 `.venv-curobo` 작업이 겹치지 않게 verifier 가 순서를 쥔다.

## 용어

- **ESDF** (Euclidean Signed Distance Field) — 각 복셀에 "가장 가까운 표면까지의 부호 있는
  거리" 를 담은 격자. 부호는 안(−)/밖(+).
- **TSDF** (Truncated SDF) — 표면 근처에서만 거리를 유지하고 멀면 잘라내는 것. 재구성용.
- **복셀(voxel)** — 3D 격자의 한 칸.
- **eikonal** — `|∇d| = 1`. 참된 거리장이 만족하는 성질.

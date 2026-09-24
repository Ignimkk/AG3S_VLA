# 커밋 메시지 (직접 실행용)

현재 `main` 브랜치에 미커밋 변경 18건. 그중 일부는 이번 검토 세션에서 만든 것이 아니라
**이전부터 있던 것**이라, 저자와 범위가 섞이지 않게 **3개로 나눈다.**

먼저 확인:

```bash
cd /mnt/dev/work/benchmark
git status --short
```

---

## 커밋 1 — 이전부터 있던 서빙 쪽 작업 (내가 만들지 않음)

```bash
git add trajopt/safe_policy.py trajopt/serve_safe.py trajopt/attention_policy.py \
        trajopt/asset/safe_replay.json
git commit -m "trajopt: safe serving 경로 작업 진행분

이번 AG3S 검토 세션 이전부터 작업 트리에 있던 변경. 컨테이너 이관 전에
유실을 막으려고 함께 커밋한다."
```

---

## 커밋 2 — AG3S/trajopt 검토 Step 0~4 + 수정

```bash
git add ag3s/esdf.py ag3s/pipeline.py ag3s/to_adapter.py ag3s/types.py ag3s/config.py \
        trajopt/linearize.py ag3s/docs/AG3S_REVIEW_LOG.md ag3s/docs/AG3S_REVIEW_PLAN.md
git commit -m "ag3s/trajopt: ESDF 경로 검토 Step 0~4 — 침묵하던 결함 8건 수정

발견 17건 중 8건 수정, 1건 기각, 8건 미판정. 전부 실측으로 확정했고 근거는
docs/AG3S_REVIEW_LOG.md 에 명령·출력째로 남겼다.

심각 — 안전 판단이 실제로 바뀌는 것:
- E1 GRASP 의 target carving 이 전신에 대해 target 을 지웠다. 필드에서 파내는 대신
  ClearancePolicy 의 링크별 마진을 ESDF 경로로 옮겼다(target_link_margin).
  MoveIt ACM / cuRobo attached-object 와 같은 '지우지 말고 질의 쪽에서 봐준다' 패턴.
  pipeline.py + to_adapter.py + types.py + trajopt/linearize.py 를 함께 고쳐야 했다.
- G1 pointcloud.depth_scale 이 TSDF 경로에 적용되지 않아, uint16 mm depth 에서
  충돌 필드가 통째로 비는데도 status:ok / certified:True 로 보고됐다.
- G2 미관측 100% 필드가 valid 로 보고됐다. ConstraintValidity 자신의 docstring 이
  '로봇이 안 보이는 벽에 박는 방식' 이라 이름 붙인 실패다. 이제 INCOMPLETE 로 내려간다.

중간:
- E4 격자 밖 조회가 unknown_policy 를 따르게. E6 TSDF 적분을 카메라 절두체로 컬링.
  (둘 다 부분적 — cuRobo 백엔드 교체 시 구조적으로 소멸한다)
- E5 카메라 3대가 per_camera 통계에서 'camera' 한 키로 뭉개졌다. camera_id 로 수정.
- G3 기여 0 인 카메라, G4 격자 밖 로봇 구를 검증 플래그에 반영(_esdf_coverage).
  둘 다 넣기 전에 정상 경로에서 오탐 0 임을 44프레임으로 먼저 측정했다.

기각:
- E7 '로봇 마스크가 클라우드를 재구성한다(중복)' 는 틀린 진단이었다. 두 경로는 해상도가
  달라서, 2단계 결과를 재사용하면 로봇 픽셀의 89.3% 를 놓친다. 하면 안 되는 수정이라
  수치와 함께 기록했다.

미수정(이월): E2 테이블 무한 half-space(Step 8), E3 attached object 가 optimizer 에
도달하지 않음(확정, Step 10).

회귀: run_0004 15청크 esdf_rollout 이 수정 전후 동일
(-28.7→-0.6 / -68.0→+1.7 / -140.3→+3.7 / -73.9→+3.2 mm, feasible 8 / violated 7).
고친 것은 침묵하던 실패이지 정상 경로가 아니다."
```

---

## 커밋 3 — cuRobo 도입 결정 · 검증 · 컨테이너 이관 문서

```bash
git add ag3s/experiments/curobo/ requirements-ag3s.txt requirements-curobo.txt \
        ag3s/docs/CONTAINER_SETUP.md ag3s/docs/VLA_ACTION_CONDITIONED_LOCAL_TSDF_ESDF_PLAN.md \
        ag3s/asset/review/ ag3s/docs/review_board.html
git commit -m "ag3s: ESDF 인프라를 cuRoboV2 로 교체하기로 결정 + API 검증 + 이관 문서

우리가 직접 만들던 ESDF 기계장치가 이미 오픈소스로 존재하고(NVlabs/curobo),
우리 발견의 대부분은 그쪽에 대응물이 없는 attention 영역에서 나왔다. 그래서
인프라는 가져다 쓰고 기여를 attention 쪽에 두기로 했다.

2계층 ESDF — attention 이 해상도 예산을 배분한다:
전체 workspace 는 거친 복셀로 유지하고(먼 장애물도 계속 회피), 관심 영역만 미세
복셀을 한 겹 얹는다. 범위를 줄이는 게 아니라 정밀도를 몰아주는 것이라 잃는 것이 없다.

H200 에서 stock API 로 검증 완료 (포크 불필요):
- Mapper.compute_esdf(esdf_origin, esdf_voxel_size) 가 호출마다 오버라이드를 받는다
  (docstring 이 'sliding window' 용도라고 적고 있다). 적분 1회, 추출 N회.
- SceneCfg.voxel 이 VoxelGrid 리스트를 받아 두 계층이 공존한다.
- nvcc 불필요 — USE_PYBIND 기본값 0, cuda-core JIT 백엔드로 동작.
- 실측(카메라 3대): 적분 1.72ms + ESDF 0.52/0.74ms = 2.98ms. 예산 66.7ms 의 4%.
  우리 numpy 구현 ~2600ms 대비 자릿수 세 개.
- 정확성: 같은 테이블 상판(참값 z=0.823)을 거친 20mm 는 0.790(-33mm),
  미세 5mm 는 0.821(-2mm). esdf_margin 50mm 예산의 66% -> 4%.

검증 중 두 번 틀렸고 그 과정을 남겼다 — esdf_origin 은 코너가 아니라 격자 중심이고,
인덱싱은 손으로 쓰지 말고 create_xyzr_tensor 를 써야 한다. 둘 다 '좌표계 불일치' 처럼
보였지만 필드는 옳았다. extract_occupied_voxels 로 우회 진단해 확정했다.

포함:
- experiments/curobo/ : 재현 스크립트 4개 + README(환경 구성, 함정 3가지)
- requirements-ag3s.txt / requirements-curobo.txt : 환경을 분리해야 하는 이유를 파일에 명시
- docs/CONTAINER_SETUP.md : 새 컨테이너 인계 문서. 빈 venv 로 전체 절차를 재현 확인했다
- docs/review_board.html + asset/review/ : 검토 현황 시각화(실측 렌더 뷰 근거 6종)"
```

---

## 그리고 push

```bash
git push
```

> `main` 에 직접 커밋한다. 브랜치를 쓰고 싶으면 커밋 전에
> `git switch -c ag3s-esdf-review` 를 먼저 하고, push 할 때
> `git push -u origin ag3s-esdf-review`.

---

## 커밋 전 확인할 것

```bash
# 1. 새로 들어가는 파일에 키/토큰이 없는지 (확인했지만 한 번 더)
git diff --cached --stat

# 2. asset/review/ 와 review_board.html 은 합쳐서 3.1 MB 다.
#    생성물이라 저장소에 넣기 싫으면 커밋 3 에서 빼고 .gitignore 에 추가:
#    git restore --staged ag3s/asset/review ag3s/docs/review_board.html
```

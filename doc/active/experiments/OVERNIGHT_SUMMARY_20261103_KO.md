# 야간 수행 요약 (2026-11-02 밤 ~ 11-03 08:00 KST)

판정 목록(임의 판정 전부): [OVERNIGHT_AUTONOMOUS_DECISIONS_20261103_KO.md](OVERNIGHT_AUTONOMOUS_DECISIONS_20261103_KO.md) — 특히 ★ 표시(B1·B2·B5·B8·B14·B15)를 먼저 보세요.

## 끝난 것
| 항목 | 결과 위치 | 한 줄 요약 |
|---|---|---|
| SafeLIBERO에서 외부 segmenter baseline 10개 arm vs 실제 ATS (분할 + 거리장 지움/남김) | `outputs/2026-11-03_SafeLIBERO_baseline분할_ATS비교/`, 문서 [RESULTS](BASELINE_SAFELIBERO_TARGET_OBSTACLE_SEGMENTATION_RESULTS_20261103_KO.md) | 3D IoU: SAM2 track 0.333 · Grounded-SAM-2 frame 0.330 · SAM v1 0.322 · YOLOE 0.03–0.16 · 실제 ATS 0.034–0.038. ATS는 obstacle을 거의 안 지우지만(0.2–0.5%) target도 거의 못 지운다(남김 0.90–0.95). 충돌 직전에도 모든 arm에서 장애물 voxel의 95% 이상이 남음 |
| Object(LIBERO)에서 ATS를 그리퍼 물리 신호로 재생 | `outputs/2026-11-02_실제ATS_vs_외부segmenter_offline_ObjectInit56/` (끝 절) | IoU 0.101 → 0.091, 개선 없음. 규칙은 사후 조정 안 함(B18) |
| SafeLIBERO closed-loop 단순 CBF-QP | `outputs/2026-11-03_CBF-QP_충돌회피_SafeLIBERO/` | 아래 표 |

## CBF-QP closed loop (20 episode, VLA 단독 기준 충돌 18/20 · 안전 성공 2/20)
표 전체는 `outputs/2026-11-03_CBF-QP_충돌회피_SafeLIBERO/cbf_arms.md`. 핵심: **같은 필터라도 장애물을 GT로 주면 충돌이 18 → 7(Multi-link) · 11(KNOWS)로 줄지만, Grounding DINO detector로 주면 거의 줄지 않거나(Multi-link: 15/20) 오히려 성공이 줄어든다(KNOWS: 충돌 19/20 · 공식 성공 4/20, VLA 단독 11/20).** 필터보다 장애물 인식이 결과를 정한다는 것이 이 야간 실험의 가장 큰 관찰이다(원인을 가른 실험은 아님).

## 못 한 것 / 아침에 정할 것
1. **AEGIS-style 필터 2 arm**(GT · detector)과 **LIBERO-Safety 새 rollout**(3번째 benchmark)은 GPU를 development의 exclusive 작업이 계속 잡아 시간 안에 돌지 못했다(마감 08:00 KST에 내 대기 요청을 직접 종료했다: B19). 3번째 benchmark는 기존 action-transfer stress 결과만 있다(B7).
2. **TrajOpt(TO) 충돌 회피**는 새로 재지 않았다. 기존 SL6 판(VLA 단독 성공 11 episode 기준, AG3S 공식 성공 1–2/11 · 충돌 3–7)을 CBF 20 episode와 직접 비교하면 안 된다.
3. **외부 mask를 AG3S 서버의 target 입력에 주입한 closed loop**(정책 b)는 하지 않았다(B5). 거리장 "지움"은 mask voxel 그대로의 기하 지표이며 접촉·성공 결과가 아니다.
4. ATS의 낮은 IoU 원인(cluster 선택 vs extent)은 GT seed 대조로 가르지 않았다. SafeLIBERO ATS 요청은 서버 원본과 bit 비교를 못 했다(B8).
5. **런처 shared lock 미지원**(B15): `launch_method.py`와 `launch_libero_safety_paired.py`는 exclusive flock만 인정한다. shared로 바꿀지는 런처 수정 판단이 필요하다.

## 정리 상태
내 대기·실행 프로세스는 모두 종료했다(다른 세션 것은 건드리지 않았다). 다시 돌리려면 `experiments/segmentation/run_cbf_arms_safelibero.sh cbf20261103 aegis:gt aegis`. 코드/문서는 모두 untracked 상태이고 commit하지 않았다.

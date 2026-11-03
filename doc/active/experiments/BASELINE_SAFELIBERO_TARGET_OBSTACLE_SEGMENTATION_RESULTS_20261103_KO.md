# SafeLIBERO에서 외부 segmenter baseline의 target·obstacle 분할과 실제 ATS 비교 (baseline 10개 arm 전부, 2026-11-03 새벽)

규약: [BASELINE_SAFELIBERO_TARGET_OBSTACLE_SEGMENTATION_PROTOCOL_20261102_KO.md](BASELINE_SAFELIBERO_TARGET_OBSTACLE_SEGMENTATION_PROTOCOL_20261102_KO.md) (결과 전 작성). 야간에 내린 임의 판정은 [OVERNIGHT_AUTONOMOUS_DECISIONS_20261103_KO.md](OVERNIGHT_AUTONOMOUS_DECISIONS_20261103_KO.md)에 있다(특히 B5 · B8 · B9).
결과 폴더: `outputs/2026-11-03_SafeLIBERO_baseline분할_ATS비교/` (표 · 그림 5개 · README).

**상태: baseline 10개 arm 전부 완료**(04:16 KST). Grounded-SAM-2 frame 2개와 SAM2 track phrase는 야간 GPU 대기 끝에 마지막으로 돌렸다.

## 무엇을 쟀나
SafeLIBERO spatial Level I dev, VLA 단독 기록 20 episode의 policy call **796 frame × 2 camera**. GT는 저장된 상태로 재렌더했고 두 camera의 RGB가 저장 관측과 모든 frame에서 bit 단위로 일치했다. class는 target · destination(plate) · **obstacle**(active 장애물) · other_object · fixture · support · robot이다. 모델 입력은 RGB와 지시문에서 뽑은 문구뿐이다. 실제 ATS는 Panda 최종 구성(`noun_target:L8H6`)이며 단계 신호는 HSR probe 없이 그리퍼 물리 신호(직전 chunk의 마지막 실행 gripper 명령이 닫힘이면 identity 고정)이다. `none`은 그 신호가 없는 대조다.

## 결과 (전체 frame, 4 task 동일 가중, `table_all_frames.md`)

| arm | 3D IoU | 대상 선택(최대 class) | target 지움 | obstacle 잘못 지움 |
|---|---:|---:|---:|---:|
| SAM2 track (instruction) | 0.333 | 0.482 | 0.416 | 0.001 |
| Grounded-SAM v1 (phrase) | 0.322 | 0.424 | 0.512 | 0.075 |
| Grounded-SAM v1 (instruction) | 0.311 | 0.414 | 0.473 | 0.065 |
| YOLOE 0.05 (phrase) | 0.163 | 0.236 | 0.260 | 0.050 |
| YOLOE 0.05 (instruction) | 0.121 | 0.139 | 0.271 | 0.055 |
| Grounded-SAM-2 frame (phrase) | 0.330 | 0.444 | 0.462 | 0.046 |
| Grounded-SAM-2 frame (instruction) | 0.311 | 0.433 | 0.426 | 0.033 |
| SAM2 track (phrase) | 0.302 | 0.436 | 0.399 | 0.025 |
| 실제 ATS, target cluster (물리 신호) | 0.038 | 0.404 | 0.046 | 0.002 |
| 실제 ATS, 서버가 지우는 voxel (물리 신호) | 0.034 | 0.475 | 0.104 | 0.003 |
| YOLOE 0.25 (phrase / instruction) | 0.030 / 0.017 | 0.048 / 0.026 | 0.050 / 0.036 | 0.034 / 0.028 |

"지움"은 그 arm의 voxel 집합이 덮는 GT class 비율(= 거리장에서 지워지는 몫)이다. 외부 mask는 depth로 올린 5 mm voxel 그대로이며 AG3S의 ball-owned 확장은 적용하지 않았다.

## 해석 (관찰과 추정을 구분)

관찰:
- 어느 arm도 대상 bowl을 안정적으로 고르지 못한다. 대상 선택은 최고 SAM2 track이 0.48이다. 장면에 같은 검은 bowl이 둘 있고 지시문은 위치 관계("between the plate and the ramekin")로 구분하기 때문이다.
- 외부 segmenter 가운데 SAM2 track · Grounded-SAM v1 · Grounded-SAM-2 frame이 3D IoU 0.30–0.33으로 서로 비슷하게 가장 높고 (Grounded-SAM-2 frame은 무출력 8%), YOLOE는 낮다(0.25 기본 confidence는 mask를 거의 내지 않는다: 무출력 0.60–0.63).
- 실제 ATS의 3D IoU는 0.034–0.038로 외부 segmenter보다 낮다. 그러나 예측 voxel의 최대 class가 target인 비율은 0.40–0.48로 SAM v1(0.41–0.42)·SAM2 track(0.48)과 비슷하다. 즉 ATS의 낮은 IoU는 identity 선택보다 **덮는 범위가 작은 것**(target cluster는 target의 4.6%, 서버가 지우는 voxel은 10.4%)에서 온다.
- 거리장의 위험 방향이 arm마다 다르다. ATS는 obstacle을 거의 지우지 않지만(0.2–0.5%) target도 거의 못 지운다(남김 0.90–0.95): 접근 경로를 target이 막을 방향의 위험이다. Grounded-SAM v1은 target을 절반 지우는 대신 obstacle도 6.5–7.5% 지운다(예측에 obstacle이 포함된 frame 24–28%): 충돌 방향의 위험이다. SAM2 track은 둘 다 낮다(obstacle 0.1%).
- attention 자체의 진단(`attention_on_gt.json`): 같은 L8H6의 wrist top-1 patch는 target 54%, head는 33%이고 head에서 obstacle에도 24% 올라간다. V11AB가 쓰던 action cell L8H2는 head에서 obstacle 38%·target 2%다. 이 장면에서는 noun cell이 action cell보다 target에 가깝다.
- 쥠 신호(`physical` 대 `none`)의 효과는 작다. 두 조건의 3D IoU는 0.034–0.038로 거의 같고 obstacle 포함 frame 비율만 줄었다(0.051 → 0.033, target cluster). GT target이 20 mm 이상 움직인 frame에서 `holding` 신호는 239/247(97%)가 움직인 frame과 겹쳤다.

추정(확인 안 됨):
- ATS의 작은 extent는 cluster 선택과 ball-owned 분할 단계의 손실일 수 있다. 원인 층(ATS 선택 / extent)을 가르려면 attention을 정답 위치에 주는 대조(GT seed)나 extent 변형 실험이 필요하다.

## 충돌 직전 구간의 거리장 (VLA 단독의 실제 궤적 기준)

공식 규칙(활성 장애물이 대기 후 위치에서 합계 1 mm 넘게 움직임)을 저장 상태에서 다시 적용하니 **20 episode 중 18개에서 충돌이 시작**됐고(기존 기록의 공식 충돌 18/20과 일치), 시작 시점은 대부분 관측 36–45번(대기 직후 0.8–1.3초)이다(`obstacle_motion_safelibero.py`). 충돌 시작 직전 20 step(policy call 4개) 구간 **72 frame**에서 각 arm이 장애물 voxel을 얼마나 남겼는지 본다.

| arm | 구간 3D IoU | target 지움 | obstacle 잘못 지움 | obstacle 포함 frame 비율 |
|---|---:|---:|---:|---:|
| SAM2 track (instruction) | 0.360 | 0.554 | 0.001 | 0.138 |
| Grounded-SAM-2 frame (phrase) | 0.421 | 0.655 | 0.033 | 0.258 |
| SAM2 track (phrase) | 0.323 | 0.544 | 0.056 | 0.263 |
| Grounded-SAM v1 (phrase) | 0.430 | 0.693 | 0.043 | 0.283 |
| YOLOE 0.05 (phrase) | 0.181 | 0.264 | 0.042 | 0.392 |
| 실제 ATS, 서버가 지우는 voxel (물리 신호) | 0.078 | 0.174 | 0.000 | 0.037 |
| 실제 ATS, target cluster (물리 신호) | 0.062 | 0.065 | 0.000 | 0.025 |

관찰: 충돌 직전에도 모든 arm에서 장애물 voxel의 95% 이상이 거리장에 남는다(잘못 지움 0–4.9%). 곧 이 기록에서 VLA 단독이 장애물을 치는 것은 **장애물이 거리장에서 사라졌기 때문이 아니라 회피하는 제어가 없기 때문**이고, 인식이 만드는 위험은 주로 target을 장애물로 남겨(ATS 남김 0.83–0.94) 접근을 막는 방향이다. 이 구간에서는 SAM v1이 target을 69%까지 지우는 대신 obstacle도 4–5% 지운다. 18개 충돌 episode의 72 frame이며 episode 상관이 크다.

## 한계
- 4 task × 5 init, VLA 단독 1개 정책의 기록이다. frame은 독립 표본이 아니고 순위·일반화 주장이 아니다.
- SafeLIBERO run에는 policy_records가 없어 ATS의 attention 요청을 저장 관측으로 **재구성**했다. 서버가 받은 원본과 bit 비교는 하지 못했다(판정 B8). 같은 코드가 Object에서는 원본과 비교한 입력으로 동작했다.
- 외부 mask를 거리장에 넣는 규칙은 "mask voxel 그대로"이며 AG3S 서버의 target 입력에 주입한 closed loop가 아니다. 이 문서의 "지움"은 voxel 집합과 GT class의 겹침이지 접촉·성공이 아니다.
- 쥠 신호는 서버의 증거 기반 latch(물체 reach·attach)를 재현한 것이 아니다.

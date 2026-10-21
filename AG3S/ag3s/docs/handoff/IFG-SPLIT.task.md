# IFG-SPLIT — 이웃 과일과 붙은 target 의 분리 (AG3S 핵심 수정) · closed-loop 검증

작성: 2026-10-20 KST, lead. 결과를 보기 전에 고정한다. 근거: `IFG-APPLE-FAIL.verify.json`, WORKLOG 2026-10-20 09:40 절.
사용자 판정 (2026-10-20): 궤적 원인은 TO, 관측 · 기하 원인은 AG3S 핵심을 고친다. 권장안(이웃 과일 병합 · attach 중 오인 수정)으로 진행.

## 1. 수정 (코드, 기본 꺼짐)

| flag (serve_safe) | config | 내용 |
|---|---|---|
| `--attention-split-partition` | `clustering.attention_split_partition` | T31b 재군집이 한 덩어리(`one_component`)로 끝난 trigger cluster 를 점별 attention(log, Otsu, 9-NN 다수결)으로 가른다. 가장 큰 질량의 고-attention 연결 부분이 target. 같은 ratio gate(2.0), 최대 길이가 `attention_partition_min_drop`(30 mm) 이상 줄 때만 |
| `--split-supersedes-anchor` | `clustering.split_supersedes_anchor` | 쥐기 전, split 의 target 조각이 merge 된 채 잡힌 anchor 의 subset 이면 (손 가림이 아니라 이웃이 떨어져 나간 것) anchor 를 대체. 조각이 anchor 보다 30 mm 이상 짧을 때만 (ep1313: 손 그늘로 잘린 사과 반쪽 거름) |

- 기존 C1aN 서버에 이미 있는 `--closing-keep-geometry` 가 닫는 동안 이 사과 형상을 유지한다 → attach 중 바나나가 "쥔 물체"로 관측되지 않는다.
- 꺼짐이면 출력 불변: 관련 기존 test 203 + 새 test 15 통과.

## 2. offline 근거 (기록 replay, 결과 보기 전)

- replay(T43 ground_replay 방식, C1aN 서버 flag 전부)가 기록을 24 장면 중 23 장면에서 75 chunk 전부 정확히 재현(덩어리 길이 · 중심 차 0). ep979 일부 frame 만 다름
- 수정 켬: merge 4 장면의 첫 닫기 전 이웃 과일 점 비율 0.34–0.41 → 0.05–0.14, anchor 길이 135–150 → 58–81 mm. 나머지 20 장면은 19 장면 동일, ep979 은 늦은 병합 anchor 105 → 67 mm
- ep250 · 572 attach 중: 사과 형상 유지, 관측 없음 (기록에서는 바나나가 관측되어 attach 해제)

## 3. closed-loop (고정)

- 장면 · seed: IFG-APPLE 의 24 사과 장면 (`seeds.json` sha256 448acbf0…, 같은 obstacle config), seed 1
- arm SPL = C1aN flag 전부 + `--attention-split-partition --split-supersedes-anchor`. snapshot_split (tree ab054291)
- 비교: 이미 있는 E0 (VLA 단독) · C1aN (같은 seed, 결정적 — S2 에서 VLA 첫 chunk 동일 확인). C1aN 은 다시 돌리지 않는다
- run: `verify/IFG-APPLE-SPLIT` (S2 kit 복사, smoke gate: SPL 첫 chunk = E0, split 기록 존재)
- 주 지표: 접촉 없는 작업 성공 (잠근 정의 그대로). 함께: 이웃 과일 접촉 포함 strict 성공, 실패 모드, HOLD, attach 해제, 이웃 과일 ≤ 8.2 cm 7 장면 / 그 밖 17 장면
- paired 차이 + 95 % CI (episode bootstrap 10,000). 미리 정한 판단: 7 장면의 결과는 방향만 본다(장면 수 작음). 24 장면 전체에서 SPL − C1aN 의 CI 와 discordant 장면별 원인을 적는다
- 하지 않을 것: seed 를 결과 보고 늘리기, 결과 보고 flag 바꾸기

## 4. 독립 장면 확인 (IFG-CL3 40 장면, 2026-10-20 사용자 "수행하세요", 결과 전 고정)

- 장면: IFG-CL3 선별 없는 40 장면 (clear 14 · adjacent 13 · place 13, 여러 과일 target), seeds.json byte 동일, seed 1
- arm: SPL (위와 같은 snapshot_split · flag). 비교 기준: IFG-CL3 의 C1aN (같은 snapshot tree f0d8046b, 같은 client 인자 — 경로만 다름, 같은 서버 flag)
- run: `20261019T132719+0900_f15e587290f6_11408fd6d27f/verify/IFG-CL3-SPLIT`. smoke = item 0, SPL 첫 VLA 출력 = IFG-CL3 C1aN 첫 chunk
- 보고: 주 지표(잠근 정의) paired 2×2 · 차 · 95 % CI (episode bootstrap 10,000, seed 20261019), stratum 별, 이웃 과일 접촉 포함 strict,
  실행이 C1aN 과 bit 동일한 장면 수, 결과가 뒤집힌 장면별 서버 판단(덩어리 길이 · attach 해제 · 사과 근처 HOLD)
- 미리 정한 해석: 실패 → 성공 ≥ 성공 → 실패 이고 성공 → 실패 장면이 SPL 의 판단으로 설명되지 않으면 SPL 을 기본 구성 후보로 둔다.
  성공 → 실패 장면이 SPL 판단(분할 · anchor 대체)에서 시작되면 그 장면을 진단하고 채택하지 않는다. 40 장면으로 효과 크기는 판정하지 않는다

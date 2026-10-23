# BM-ISSUES — benchmark에서 겪은 문제와 해결 방법 정리 문서

> writer: lead · 읽는 쪽: scribe · 작성 2026-10-23 KST
> 사용자 지시 2026-10-23: "지금까지 발생했던 문제, 그리고 이를 해결한 방법을 따로 문서로 정리합시다."
> 이 지시로 새 문서 하나를 만든다: `AG3S/ag3s/docs/AG3S_BENCHMARK_ISSUES.md`. `AG3S_BENCHMARK_LOG.md`는 시간순 기록 정본으로 그대로 두고, 이 문서는 **문제별로 다시 모은 요약 · 색인**이다.

## 읽는 사람

- 이 project의 사용자(연구자)
- 나중에 같은 문제를 만날 팀원
- development branch로 옮길 때 참고할 사람

결과를 처음 보는 사람도 따라갈 수 있게 쓴다.

## 이름 규칙 (사용자 2026-10-22)

version과 단계는 **바꾼 내용**으로 부른다. 예: "grasp reach 측정 시점 수정판", "최소 개입 gate", "arm link 제약 추가판", "쥔 물체 hand 면제판", "grasp zone 허용판", "파지 인식 수정판", "파지 인식 + closing 면제판", 역변환 A(지연 역보정) · B(명령 offset) · C(지연 역보정 + 추종). 내부 key(pbN, SLxx)는 괄호로만 붙인다.

## 담을 것

문제마다 한 항목으로 쓴다. 항목 형식은 다음과 같다.

| 칸 | 내용 |
|---|---|
| 문제 | 무엇이 보였나 (증상, 처음 본 단계와 날짜) |
| 원인 | 측정으로 확인된 원인. 확인되지 않았으면 "가설"로 표시한다 |
| 해결 | 무엇을 바꿨나 (file / flag / commit) |
| 검증 | 같은 조건 전후 숫자. **verify.json `numbers`에서만** 가져오고 출처 링크를 단다 |
| 상태 | 해결 / 부분 해결 / 미해결 / 되돌림 |
| 남은 한계 · 회귀 | |
| development branch 이식 메모 | 의존성, 주의점 |

분류(대략, scribe가 조정해도 된다):

1. **환경 · 인프라:** 독립 venv / checkpoint, NFS stall(D state), 공용 GPU 시간 나누기와 lock 규칙, 병렬 GPU run(bit 재현 확인), run 중 code / 파일 변경으로 run이 dirty가 되거나 거부된 일, 결정성과 pairing 근거(VLA-only 재사용), 메모리 working set 정의.
2. **Panda adapter:** camera 좌우 반전(un-mirror), K principal point, depth와 frame 규약, gripper `empty_below` 보정, 실행 피드백 계약, 역변환 A / B / C, LIBERO fixture reset 이력.
3. **평가 · 판정기:** SafeLIBERO 충돌 판정 검증, official success와 safe success 구분, dev / test 분리.
4. **알고리즘 개선판:** 각 version이 겨냥한 원인, 그 효과와 한계. 다음 요소를 포함한다:
   - grasp reach 측정 시점
   - 최소 개입 gate
   - arm link
   - held 면제
   - latch contact phase
   - grasp zone
   - pb4 / pb5 초기 구현 결함과 수정
   - 파지 인식(sticky reach, 열린 개도 기준)
   - closing 면제
   - over-lift와 action 포화의 원인 사슬(SL5h)
5. **아직 풀리지 않은 문제:**
   - 쥔 뒤에도 손을 위로 민다(진단 준비만 됨: 제약 기록 run, SL5j-G는 GPU 정지로 보류)
   - 잘못된 cluster를 잠금(t2e0)
   - Panda 손가락이 테두리를 끝까지 닫아 `blocked` check가 실패
   - VLA-only가 충돌하며 성공한 episode에서 생기는 회귀
   - ATS grounding 탈락(SL4c, 판정 기록 없음)

## 형식

- 맨 앞에 한 장짜리 요약표(문제 · 상태 · 한 줄 해결)를 둔다. 그 뒤에 항목별 상세를 쓴다.
- 기존 figure를 링크한다(새 figure는 만들지 않는다).
- 기술 용어는 English로 쓴다(`.claude/rules/logging.md` 규칙 I). 새 용어는 그 자리에서 풀어 쓴다.
- 판정 문장은 사용자가 이미 판정한 것만 그대로 옮긴다. 판정 대기인 것은 "판정 대기"로 쓴다.
- 숫자 출처가 verify.json에 없으면 쓰지 않는다. 필요하면 "숫자 없음(측정 안 함)"으로 둔다.

## 출처

- `AG3S/ag3s/docs/AG3S_BENCHMARK_LOG.md` 전체
- `AG3S/ag3s/docs/handoff/*.task.md`, `*.impl.md`, `*.verify.json`
- `doc/migration/*.md`(환경 · storage)
- git log

## 넘겨줄 것

- 문서 하나(`AG3S_BENCHMARK_ISSUES.md`)를 만든다. log 끝에 "문제 · 해결 정리 문서 신설" 한 줄과 link를 추가한다.
- commit하지 않는다.
- lead에게 목차와 항목 수, 숫자 출처가 없어 빈칸으로 둔 곳을 보고한다.

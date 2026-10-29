# AG3S system architecture — 발표 대본

- 그림: [`ag3s-system-architecture.png`](ag3s-system-architecture.png)
- 분량: 약 6분, 구간 10개 (구간마다 `[가리킬 곳]`을 먼저 적었다)
- 숫자 출처: 도식 숫자는 `ag3s-system-architecture.json` 의 `numbers`, 예시 장면 숫자는 같은 파일의 `thumbnails`
- 그림 크기: 16:9 슬라이드 한 장(13.33 × 7.5 in)에 100 %로 넣는다. 제목은 슬라이드 제목란에 쓴다
  (예: "AG3S system architecture").
- 썸네일 출처 (슬라이드 노트용): held-out ep1807의 첫 정책 호출(Target = apple, 파지 전).
  RGB는 `outputs/live_test/20260925_ep1807` step 0, 나머지는 `outputs/live_test/20260928_t17` server chunk 0(shadow run).
  attention overlay는 카메라마다 min–max 정규화.

---

## 0. 한 문장 요약 (20초)

`[그림 전체, 맨 위 범례]`

> 오늘 설명드릴 것은 AG3S 전체 시스템입니다.
> 한 문장으로 말씀드리면, **VLA의 visual attention으로 작업 대상인 Target과 장애물인 Obstacle을 구분하고,
> Target을 충돌 대상에서 뺀 거리장을 trajectory optimization의 distance constraint로 넘기는 구조**입니다.
>
> 그림에는 굵은 화살표가 두 줄 있습니다. 보라색은 attention이 거리 제약이 되기까지의 경로이고,
> 파란색은 VLA가 만든 action chunk가 최적화의 초기 궤적이 되는 경로입니다.
> 이 두 줄을 따라가시면 됩니다.

## 1. 전체 배치 — 두 대의 machine (30초)

`[왼쪽 LOCAL PC 점선 상자 → 오른쪽 GPU SERVER 점선 상자]`

> 시스템은 두 machine으로 나뉩니다.
> 왼쪽 **local PC**에서는 MuJoCo simulation이 돌면서 observation을 만들고 받은 동작을 실행합니다.
> 오른쪽 **GPU server**에서는 π0.5, AG3S, trajopt가 **한 process** 안에서 돕니다.
>
> 한 process로 묶은 이유는 attention 때문입니다. Attention은 model 내부 값이라 server 밖으로
> 꺼내려면 매 frame 전송해야 합니다. 같은 process 안이면 그냥 memory에 있는 변수입니다.
> 두 machine은 SSH tunnel 위의 websocket으로 연결됩니다.

## 2. INPUT (30초)

`[LOCAL PC 안의 INPUT 네 상자]`

> 입력은 네 가지입니다.
> **Instruction** — 이 예시에서는 "Pick up the apple and place it in the basket."
> **RGB 세 장** — head camera 하나와 양손 wrist camera 둘. 이것은 VLA로 갑니다.
> **Robot state** — 16차원입니다. 팔 7 joint와 gripper 1개가 양팔에 하나씩 있습니다.
> **RGB-D 세 장** — 같은 세 카메라의 depth, intrinsics K, extrinsics T입니다. 이것은 AG3S로 갑니다.
>
> 한 가지 짚으면, 각 카메라의 depth는 **그 카메라가 찍힌 순간의 robot pose**와 같이 들어갑니다.
> 세 카메라에 현재 pose 하나를 일괄 적용하면, 움직이는 wrist camera에서 로봇 몸을 잘못 지우게 됩니다.

## 3. VLA — π0.5 (60초)

`[VLA 상자: VLM prefix → Action expert의 k=0…k=9 사슬 → Action chunk]`

> VLA는 π0.5이고, 저희 task에 맞춰 16D로 fine-tune한 model입니다.
>
> 왼쪽 **VLM prefix**는 PaliGemma입니다. 이미지 세 장이 카메라당 16×16 patch,
> 즉 image token 256개씩으로 바뀌고, instruction과 state도 token이 되어 prefix를 이룹니다.
>
> 가운데 **action expert**는 flow matching으로 동작을 만듭니다.
> Gaussian noise A₁에서 시작해 **Euler step 10번**을 밟습니다. t가 1.0에서 0.1까지 0.1씩 줄어들고,
> 매 step마다 velocity field v_θ가 action을 조금씩 noise에서 동작 쪽으로 옮깁니다.
> 마지막 k=9 step의 출력이 오른쪽의 **action chunk**, 50 step × 16차원입니다.
>
> `[보라색으로 강조된 k=0 상자와 그 아래 Attention readout]`
>
> 여기서 AG3S가 쓰는 신호를 꺼냅니다. **첫 Euler step인 k=0에서 layer 8, head 2의 attention**을 읽습니다.
> Query는 action token, key는 image token입니다. 즉 "동작을 만들려는 token이 이미지의 어디를 보고 있는가"입니다.
> 카메라마다 16×16 map이 나오고, 오른쪽 세 장이 실제 map을 이미지 위에 겹친 것입니다.
>
> 이 attention은 **별도로 load한 같은 model**에서 읽습니다.
> 그래서 attention을 꺼내는 과정이 실제로 내보내는 action에는 영향을 주지 않습니다.

## 4. AG3S ① Robot mask (30초)

`[AG3S 상자 왼쪽 위 ①]`

> 이제 AG3S입니다. 첫 단계는 **robot mask**, self-filter라고도 부릅니다.
> 카메라가 찍힌 순간의 joint 값으로 forward kinematics를 풀고, URDF의 로봇 구 모델을 이미지에
> 투영해서 **로봇 몸이 찍힌 pixel을 지웁니다.**
> 이 단계가 없으면 로봇이 자기 팔을 장애물로 보고 피하려고 합니다.
> 여기서 두 갈래가 나갑니다. 로봇을 지운 point cloud는 ②로, 로봇을 지운 depth는 아래 cuRobo로 갑니다.

## 5. AG3S ② Attention lifting, ③ Target grounding (60초)

`[② → ③, 보라색 화살표]`

> **② Attention lifting**은 2D attention을 3D로 올리는 단계입니다.
> 각 3D point가 자기가 찍힌 pixel 위치의 attention 값을 읽어 옵니다. 세 카메라 결과를 합칩니다.
> 아래 top view에서 진하게 칠해진 곳이 attention이 높은 point인데, 사과 위치에 모여 있습니다.
>
> **③ Target grounding**은 그 point들을 3D에서 덩어리(cluster)로 묶고, attention과 기하 점수로
> 순위를 매겨 1등 덩어리를 **Target**으로 고릅니다. 나머지는 전부 **Obstacle**입니다.
> Target이 frame마다 흔들리지 않도록, 다른 후보가 **세 frame 연속** 1등이 되어야 바꿉니다.
> 이 장면에서는 빨간 점 111개, 사과가 Target으로 잡혔습니다.
>
> 이 결과가 보라색 화살표를 따라 **cuRobo로** 내려갑니다. "어디가 Target이니 충돌 대상에서 빼라"는 정보입니다.

## 6. ④ TSDF → ESDF — cuRobo (60초)

`[초록 점선 상자 "external · cuRobo"]`

> 초록 점선 상자는 **외부 module인 cuRobo**입니다. 저희가 만든 것이 아니라 가져다 쓰는 부분이라서
> AG3S 안에 상자를 하나 더 그려 구분했습니다.
>
> cuRobo는 로봇이 지워진 depth 세 장을 쌓아 **TSDF**를 만듭니다. TSDF는 표면 근처 얇은 띠에만
> "가장 가까운 표면까지의 부호 있는 거리"를 적은 3D 격자입니다.
> 이것을 펼쳐 **모든 칸**에 거리를 채운 것이 **ESDF**이고, 최적화가 실제로 묻는 거리장이 이것입니다.
>
> ESDF는 해상도를 달리해 두 번 뽑습니다. 작업공간 전체를 덮는 **coarse 20 mm**와,
> Target 주변만 덮는 **fine 5 mm**입니다. 먼 곳은 coarse가, 가까운 곳은 fine이 맡습니다.
>
> `[아래 두 단면: fine ESDF → target-free layer]`
>
> 핵심은 오른쪽 **target-free layer**입니다. Target 부분을 지우고 만든 fine 계층 한 겹입니다.
> 두 단면은 사과 높이에서 수평으로 자른 것이고, 검은색이 물체 안쪽입니다.
> 왼쪽 fine ESDF에는 빨간 원 안에 사과가 있고, 오른쪽 target-free layer에서는 **사과만 사라졌습니다.**
> 옆의 다른 물체들은 그대로 남아 있습니다. **Target만 충돌 대상에서 빠진다**는 것을 눈으로 확인할 수 있습니다.

## 7. ⑤ Distance field adapter (40초)

`[가운데 아래 ⑤ Distance adapter]`

> **⑤ Distance field adapter**는 저희가 만든 얇은 연결층입니다.
> 로봇을 구 여러 개로 근사해 두고, 각 구의 중심 위치를 물으면 거리 d와 그 기울기 ∇d를 돌려줍니다.
> Coarse와 fine 중 **작은 값**을 씁니다.
>
> 여기서 물체마다 요구하는 여유가 달라집니다. 일반 장애물은 50 mm, 물건을 놓을 목적지는 20 mm입니다.
> 그리고 **Target을 잡아야 하는 손가락은 target-free layer를 묻습니다.** 그래서 손가락은 사과에 닿을 수 있고,
> 팔꿈치 같은 나머지 부위는 여전히 사과를 피해야 합니다.
>
> 아래 그림은 이 장면의 로봇 구를 여유거리(clearance)로 칠한 것입니다.
> 여유거리는 거리에서 구 반지름과 margin을 뺀 값이고, 0 이상이면 안전입니다. 이 장면의 최소값은 64.1 mm입니다.

## 8. ⑥ Trajectory optimization — trajopt (50초)

`[노란 점선 상자 "external · trajopt", 위에서 내려오는 파란 화살표]`

> 노란 점선 상자가 두 번째 외부 module, **trajopt**입니다.
>
> 두 줄이 여기서 만납니다. 위에서 내려오는 **파란 화살표**는 VLA의 action chunk이고,
> 이것이 최적화의 **초기 궤적 Q⁰**이자 "되도록 이것에서 멀어지지 말라"는 기준이 됩니다.
> 왼쪽에서 들어오는 **보라 화살표**는 ⑤가 준 거리와 기울기이고, 이것이
> **"모든 구, 모든 step에서 d − r − margin ≥ 0"** 이라는 distance constraint가 됩니다.
>
> SQP로 이 곡선 제약을 직선으로 근사하고, 그 QP를 OSQP로 풉니다. 그러면서 ⑤에 새 구 위치를 다시 묻는데,
> 그것이 점선 화살표 query입니다.
>
> 아래 graph는 이 장면에서 최적화가 원래 chunk를 얼마나 바꿨는지입니다. 최대 2.8 mrad 정도로 거의 바꾸지 않았습니다.
> 이 순간에는 팔 근처에 장애물이 없어서, **바꿀 이유가 없으면 VLA의 동작을 그대로 둡니다.**
> 한 번의 SQP 반복으로 feasible, 즉 모든 제약을 만족했습니다.

## 9. Closed loop (30초)

`[OUTPUT → 아래 검은 화살표 → 왼쪽 MuJoCo simulation → next observation]`

> 최적화된 chunk Q*와 안전 판정이 **local PC로 돌아갑니다.**
> Local PC는 50 step 중 **앞의 8 step만** 15 Hz로 실행합니다. 약 533 ms입니다.
> 그리고 새 observation을 찍어 다시 server로 보냅니다.
> 50 step을 다 실행하지 않는 이유는 그동안 장면이 바뀌기 때문입니다.
> 8 step마다 attention부터 거리장, 최적화까지 전부 새로 계산합니다.

## 10. 마무리 (20초)

`[다시 전체]`

> 정리하겠습니다.
> VLA는 **무엇을 할지**를 정하고, 그 attention은 **무엇이 Target인지**를 알려 줍니다.
> AG3S는 그 정보로 Target과 Obstacle을 나누고, cuRobo는 **어디에 무엇이 있는지**를 거리장으로 만듭니다.
> trajopt는 VLA의 동작을 출발점으로 삼아 **Target은 잡게 두고 Obstacle만 피하도록** 고칩니다.
> 감사합니다.

---

## 예상 질문 (필요할 때만)

| 질문 | 답 |
|---|---|
| 왜 Euler step 0의 attention인가? | 앞선 attention 검증 단계에서 layer × head × denoise step × 집계 방식 조합을 비교해 가장 점수가 높았던 조합(k=0 · layer 8 · head 2 · 마지막 action token)을 그대로 씁니다. step 0만 읽으면 denoising 10 step을 다 돌 필요 없이 forward 한 번으로 끝난다는 장점도 있습니다. |
| cuRobo를 왜 쓰나? | 같은 TSDF에서 해상도가 다른 ESDF를 여러 번 뽑는 것이 기본 API로 되고, 격자 밖을 무조건 빈 공간으로 답하던 저희 자체 구현의 결함이 구조적으로 사라지기 때문입니다. 속도는 선택 이유가 아닙니다. |
| attention이 아무것도 못 찾으면? | Target이 없으면 Target을 지우지 않으므로 모든 물체를 장애물로 봅니다. 즉 더 보수적인 쪽으로 실패합니다. |
| 실시간인가? | 아직 아닙니다. 8 step 예산이 533 ms인데, 현재 한 번에 P50(중앙값) 약 2.5초가 걸립니다(T6: 실제 closed loop 시험). 이번 단계의 목표는 판단의 정확성이고, 속도는 다음 과제입니다. |

## 이 대본에 나오는 용어

| 용어 | 뜻 |
|---|---|
| Euler step | noise에서 동작으로 가는 flow를 작은 보폭으로 나눠 밟는 한 걸음. π0.5는 10걸음 |
| action chunk | 정책이 한 번에 내놓는 동작 묶음. 여기서는 50 step × 16차원 |
| TSDF / ESDF | 표면 근처만 거리를 적은 격자 / 모든 칸에 가장 가까운 표면까지의 거리를 적은 격자 |
| coarse / fine | 20 mm 칸으로 작업공간 전체를 덮는 ESDF / 5 mm 칸으로 Target 주변만 덮는 ESDF |
| target-free layer | Target을 지우고 만든 fine ESDF. Target을 잡아도 되는 링크(손가락)만 이것을 묻는다 |
| clearance (여유거리) | 거리 − 구 반지름 − margin. 0 이상이면 안전 |
| SQP / OSQP | 곡선 제약을 직선으로 근사해 반복해서 푸는 방법 / 그 한 번의 QP를 푸는 solver |
| feasible | 최적화 후 모든 제약을 만족함 (최악 여유거리 ≥ 0) |

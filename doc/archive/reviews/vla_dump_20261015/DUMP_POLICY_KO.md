**VLA 내부 덤프 저장 명세 v1 — 다중 프레임 수집용**

작성: 2026-10-15 KST. 이 문서는 다음 덤프를 위한 **선택·저장 설계**다. 기존 full collector는 수정하지 않았고 CORE/SURVEY flag도 아직 구현되어 있지 않다. 원본 full dump를 삭제하라는 지시가 아니다. 실험 정본은 [EXPERIMENT_PROTOCOL_KO.md](../../../active/reviews/vla_dump_20261015/EXPERIMENT_PROTOCOL_KO.md).

**1. “필요 없는 값”의 정확한 구분**

모든 미래 연구에 무용하다고 할 수 있는 내부 값은 거의 없다. 그러나 이번 target segmentation·phase 연구에서 매 frame 저장할 이유가 없는 값은 많다. 다음 세 가지를 구분한다.

- **확실한 중복:** 다른 key에 같은 배열이 이미 존재. 한 authoritative copy만 보존하면 됨.
- **현재 목적에 불필요:** 실험 입력으로 사용하지 않음. 향후 기계적 해석/언어 head 연구용 audit에서만 저장.
- **다시 계산 가능한 중간값:** immutable checkpoint·입력·noise·code가 있으면 재추출 가능. 같은 비트 패턴을 보장한다는 뜻은 아님. 정밀도·backend 차이를 기록해야 함.

**2. 현재 실제 저장량과 큰 절감 항목**

현재 `inventory_frame022.json`의 611개 배열 payload 합계는 **4,109.41 MB/frame**다. 아래 MB는 10^6 bytes이며 disk의 GiB·압축 후 크기와 다르다. 실제 수치는 [dump_budget.json](dump_budget.json), 계산기는 [dump_budget.py](dump_budget.py)에 있다.

| 항목 | 현재 총량/frame | 권고 | 근거 |
|---|---:|---|---|
| Prefix `mlp_gate`, `mlp_up`, `mlp_act` | **1,712.85 MB** | CORE/SURVEY에서 제외 | segmentation/phase baseline에 필요 없음; MLP neuron/causal 연구 audit에 한정 |
| Suffix `L_mlp_gate/up/act` | **221.18 MB** | CORE/SURVEY에서 제외 | 동일 |
| Prefix `attn_logits` | **269.86 MB** | 제외 | 저장한 probs/sliced rows로 계획된 분석 가능; exact-logit 진단에만 필요 |
| Suffix `L_attn_logits` | **146.59 MB** | 제외 | 동일; QK 재계산은 mask/RoPE/scaling/precision 필요 |
| Prefix full `attn_probs` | **269.86 MB** | named rows만 보존 | target/destination/verb query는 작게 저장 가능; image–image routing 연구는 audit |
| SigLIP 27×16 full attention | **169.87 MB** | 제외 | 첫 실험은 encoded/selected features 사용; 현 attention은 재계산 parity 문제도 있음 |
| SigLIP 모든 block LN/sa/+sa/mlp/+mlp | 각 종류 약 **47.78 MB**, 6종 합계 약 **286.65 MB** | encoded만 CORE, 중간 layer는 audit | backbone depth/경계 형성 분석 필요 시 selected outputs만 추가 |
| Prefix/suffix 전 layer residual·norm·branch 출력 | 수백 MB | selected prefix 출력·small summaries로 축소 | 전층 full tensor는 mechanism audit용 |
| LM-head 전체/Top32 logits·pieces·next-token log-prob | 약 **1.20 MB** | 기본 제외 | 이 경로는 양방향 prefix; next-token 값을 일반 causal perplexity/작업 confidence로 사용할 수 없음 |

LM-head를 빼는 것은 의미상 정리이지 주요 용량 절감은 아니다. 큰 절감은 MLP 중간 activation·full logits·불필요한 full attention에서 나온다.

**확실하게 확인한 중복**

- `siglip_<camera>.projector_out` 3개를 연결한 결과와 `embed.image_tokens`는 이번 frame에서 bitwise 동일했다. `embed.image_tokens`만 보존한다. 약 3.146 MB 절감.
- 모든 18 layer의 `prefix_Lxx.k_rope[:,0]`와 `prefix_final.k_cache[layer]`, `prefix_Lxx.v[:,0]`와 `prefix_final.v_cache[layer]`는 이번 frame에서 각각 bitwise 동일했다. `prefix_final`의 canonical K/V cache만 보존한다. 중복 사본 약 17.842 MB 절감.
- derived serving/noun map은 본래 attention slice에서 얻을 수 있다. 용량이 작으므로 기존 consumer 호환을 위해 유지해도 좋다. authoritative source와 index를 함께 적고 여러 사본이 어긋나지 않게 검사한다.
- 연속 layer의 `resid_out`/다음 `resid_in`, norm/branch 합 관계도 중복 후보지만 모든 구현에서 exact라고 가정하지 않는다. residual과 branch가 각각 반정밀도로 저장되면 재합산 결과가 원래 값과 다를 수 있다.
- RoPE 전/후 Q/K를 모두 저장할 필요는 없다. 해당 분석이 필요한 AUDIT에서 position·scaling·원본 dtype와 canonical 한 representation을 선택한다. 저정밀도 변환을 역산하여 bitwise 복원 가능하다고 쓰지 않는다.

**3. 권장 세 가지 프로파일**

| 프로파일 | 목적 | 내부 tensor payload/frame | 100 frame |
|---|---|---:|---:|
| **CORE** | 전체 episode의 segmentation·phase 및 기본 head 탐색 | **62.42 MB** | **6.24 GB** |
| **SURVEY** | 모든 denoise step에서 head/query를 비교 | **193.44 MB** | **19.34 GB** |
| **FULL_AUDIT** | 소수 frame의 extractor 검증·전체 mechanism 분석 | 현재 **4,109.41 MB** | **410.94 GB** |

이 값은 현재 3-camera/968 prefix slot/18 layer/8 head/50 action/10 denoise/FP16 bulk의 **비압축 배열 설계치**다. metadata, zip header, RGB-D 외부 기록, GT는 포함하지 않는다. query span 수와 dtype가 달라지면 다시 계산한다. profiler로 측정한 추출 시간·disk size가 아니다. 실제 profile collector를 구현한 뒤 sanity frame으로 다시 측정한다.

기본 수집 권고: 모든 policy observation은 CORE. dev의 phase transition/가림/실패를 포함한 대표 구간에는 SURVEY. 서로 다른 상태의 5–10개 observation은 FULL_AUDIT. 초기 데이터가 아주 적으면 전체를 SURVEY로 모으는 것도 가능하나, 연속된 몇 frame보다 episode/물체/배치 다양성을 우선한다. test의 대표 audit 선정으로 test label을 미리 보며 feature를 선택하지 않는다.

**4. CORE의 필수 내부 tensor 계약**

수치는 현재 shape 기준. 실제 token/step 수를 metadata로 기록하고 hardcode하지 않는다.

| 저장 항목 | Shape / 선택 | 약 MB |
|---|---|---:|
| `inputs` | 기존 preprocessed RGB, state, token IDs/pieces/masks, prompt, image order | 1.813 |
| `embed` | image_tokens(768,2048), text_embeddings(200,2048), prefix_mask | 3.966 |
| SigLIP `encoded` | (3,256,1152) | 1.769 |
| Prefix selected `resid_out` | L4/L8/L12의 전체 slot (3,968,2048) | 11.895 |
| Prefix `final_hidden` | (968,2048) | 3.965 |
| Canonical prefix K/V cache | 각각 (18,968,256) | 17.842 |
| Prefix attention named rows | target/destination/verb, 모든 layer/head, 전체 key: 예 (18,8,3,968) | 0.836 |
| Suffix attention `s=0` | **모든** layer/head/query/key (18,8,50,1018) | 14.659 |
| Suffix attention `s=1..9` | L8H2의 **모든 50 query와 전체 key** (9,50,1018) | 0.916 |
| Suffix V | 모든 step/layer/action token (10,18,50,256) | 4.608 |
| Flow path | 각 step x_t/v_t FP32, time FP32 | 0.128 |
| outputs/derived | x0, 실제 16D action, recorded action, HSR feature/prob, 기존 map | 0.022 |

왜 이 정도를 남기는가:

- 현재 L8H2가 특정 query에서 background에 집중했다. **L8H2만 남기면 다른 head를 찾을 수 없다.** 따라서 step0은 모든 head/query를 보존한다.
- `first/last/실제 실행 K개 평균`은 나중에 계산할 수 있어야 한다. query를 먼저 평균해서 저장하지 않는다.
- 1018 key 전체를 저장하면 image 외에 task text/state/suffix로 흐르는 질량을 분석할 수 있다. image 768만 보존하면 lost mass를 별도 저장해야 한다. 현재 비용 차이는 약 3.6 MB라 전체 key를 권장한다.
- named query row는 target noun만 아니라 **destination noun과 verb**도 필요하다. span이 여러 token이면 개별 row와 token indices를 저장한다. 예산의 3개 row는 단일-token 예시다. prompt에 역할이 없으면 missing flag, 임의 다른 token으로 대체 금지.
- prefix K/V와 final hidden은 segmentation에 전부 필수는 아니지만 용량이 작아 layer별 phase/value 연구를 보존하기 위해 권장한다. 미래 phase feature를 kv_L4 512개로 미리 제한하지 않는다.
- 원본 V cache와 action probabilities가 있으면 A×V, contribution direction을 다시 계산할 수 있다. suffix V는 suffix-key contribution까지 포함하려고 보존한다. output projection·gate를 거친 최종 action sensitivity에는 checkpoint 재추론 또는 AUDIT가 여전히 필요하다.
- x_t/v_t는 전체32를 FP32로 보존한다. 양이 작고 실제 초기 noise 및 Euler 경로 검증에 필요하다. 평가에서는 active16과 padding16을 분리한다.

대형 배열은 source inference dtype와 저장 dtype를 둘 다 기록한다. source BF16을 FP16으로 cast할 때 overflow/underflow 범위를 검사한다. FP16에 안 들어가는 텐서는 원본 bit-preserving BF16 storage 또는 FP32로 저장하며 budget을 다시 계산한다. FP16이 항상 무손실이라는 뜻이 아니다. INT8/PCA/평균 축소는 raw CORE를 대체하지 말고 후속 파생 산출물로 둔다.

**5. SURVEY와 FULL_AUDIT**

SURVEY는 CORE의 suffix attention을 **모든 10 step × 18 layer × 8 head × 50 query × 1018 key**로 확장한다. 총 attention 146.592 MB이며 CORE의 두 attention block을 대체한다. 나머지는 CORE와 같다. 약 193.44 MB/frame로 late-step head 탐색을 보존한다. Full prefix image–image attention, MLP neurons는 여전히 없다.

FULL_AUDIT는 기존 full tensor를 소수 frame에서 유지한다. 감사 목적에 따라 SigLIP 중간 block, all prefix attention, residual/gate/QK/MLP 등을 선택해도 된다. 다음 오류를 검증할 최소 original-vs-replay reference를 남긴다: mask, RoPE, prefix cache, selected probability slices, action_out, denoise trajectory. full GPU tensor를 host로 옮기기 위한 별도 forward가 original behavior를 바꾸는지 확인한다.

CORE만으로 불가능한 분석도 명시한다: late-step의 미보존 head 검색, full prefix image-to-image attention rollout, SigLIP attention 경로, MLP neuron-level 분석, 실제 intervention/gradient. 필요하면 SURVEY/FULL_AUDIT 또는 원본 입력의 checkpoint 재추론으로 수행한다. 저장된 activation만으로 counterfactual forward가 자동으로 가능해지는 것은 아니다.

**6. 줄이면 안 되는 외부 관측·재현 메타데이터**

오히려 앞으로 반드시 추가/연결해야 한다.

- **RGB-D + calibration:** depth 원본 단위/encoding/invalid mask, RGB-depth registration, K, distortion 정보, T_world_cam 또는 T_base_cam+T_world_base, capture timestamps, camera ID/order, source→224 resize/crop/pad mapping.
- **시간 기록:** observation 시각, policy request/response 시각, actual executed action indices/times, controller dt, HOLD/TO 수정/중단, episode reset. 예정 K와 실제 실행 K를 구분.
- **원본 state:** q/gripper, 가능하면 qdot와 해당 timestamp, FK에 필요한 joint definition. 관측 gripper feedback와 predicted action gripper를 구분.
- **noise와 모델:** 실제 x_initial 또는 x_t step0의 canonical copy, seed/RNG algorithm, checkpoint/config/hash, norm stats, LoRA/base weight identity, tokenizer, code commit, dtype/backend, solver/time grid.
- **독립 평가:** simulator object/geom mask 또는 별도 annotation, contact/grasp/release/placement label 및 label definition. GT 파일은 feature input과 분리하고 paired fingerprint를 둔다.
- **추출 품질:** original/replay 구분, finite/shape checks, maxabs/RMSE, missing camera/token/span, extraction time, output checksum.

metric depth float32 (3,480,640)는 **3.686 MB**, raw RGB uint8은 **2.765 MB**다. 3D point cloud/normal은 depth+K+T에서 재생성할 수 있으므로 매번 중복 저장하지 않아도 된다. 기존 원본 기록에 이미 있다면 immutable URI+hash/frame index로 참조한다. 원본을 별도로 보존하지 않은 채 feature만 남기면 이후 재정합·재추론이 막힌다.

Depth가 없다면 simulator의 정확한 XML/assets/state와 render version으로 재생성할 수 있는지 확인한다. 실물에서는 이런 복구가 불가능하므로 직접 저장한다. uint16 mm가 원본 sensor encoding이면 그대로 보존하고 scale을 기록한다. 새로 float depth를 mm로 양자화할지는 접촉 경계 오차 예산에 맞춰 결정한다. byte 절감을 위해 새로운 깊이 오차를 조용히 도입하지 않는다.

**7. collector 구현 시 필요한 파일 구조와 검사**

권장 구조(설계, 아직 producer 없음):

```text
run_manifest.json
split_manifest.jsonl
observations/<episode>/...       # 또는 immutable source references
features/<episode>/<frame>/
    metadata.json
    inputs.npz
    embed.npz
    vision_encoded.npz
    prefix_selected.npz
    prefix_cache.npz
    prefix_named_attention.npz
    action_attention.npz
    action_values.npz
    flow.npz
    outputs.npz
    quality.json
labels/<episode>/...             # evaluator만 읽음
```

- metadata: `schema_version, profile, episode/frame/t_step/timestamp, observation_hash, checkpoint_hash, camera_order, layer_indices, head_indices, denoise_step_indices, action_query_indices, prefix_key_offsets, named_query_token_indices, source_dtype, stored_dtype`.
- axis 이름과 저장된 index array를 명시한다. selected L4를 compact axis0에 넣어도 layer0으로 오해하지 않게 한다.
- profile이 빠뜨린 key와 실제 extraction 실패를 구분한다. required key가 없으면 loader가 오류를 내야 하며 zero tensor로 조용히 대체하지 않는다.
- atomic temp-write→rename으로 완료 표시하고 checksum을 남긴다. 덤프 중단 후 partial npz를 정상 sample로 보지 않는다.
- all-frame CORE의 입력/state/outputs는 실제 serving trace와 pairing되어야 한다. 별도 noise 재추론 결과는 recorded output으로 덮어쓰지 않는다.
- collector가 큰 배열을 먼저 host copy한 뒤 버리면 저장량만 줄고 GPU/전송 비용은 남는다. 가능하면 forward hook/capture 단계에서 slice/reduce한다. actual original forward parity를 유지하는지 확인한다.
- 이미지/feature 압축은 lossless container부터 시작한다. 압축비와 CPU time은 측정값으로 보고한다. 저장량 계산을 throughput 보장으로 바꾸지 않는다.

**8. 덤프 시작 전 최소 체크리스트**

1. 이전 full frame022와 소수 다양한 FULL_AUDIT를 보존했는가?
2. RGB-D·K·T·전처리 좌표가 해당 VLA frame과 정확히 대응하는가?
3. action/noun 중 한 source만 남기지 않았는가? 대상·목적지 span이 구분되는가?
4. 모든 frame의 실제 time·state·grasp feedback·실행 chunk trace가 보존되는가?
5. CORE slice를 같은 forward의 FULL slice와 대조했는가?
6. profile budget은 byte 계산이고 실제 collector 구현/실행 완료와 구분했는가?
7. 연속 frame뿐 아니라 독립 episode/물체/배치/가림/실패의 다양성을 확보했는가?

위 조건을 충족한다면 반복 full dump보다 **긴 CORE timeline + 대표 SURVEY + 소수 FULL_AUDIT**가 현재 연구 질문에 더 직접적인 데이터를 제공한다.


**T43–T61 감사 보완 (2026-10-15)**

[중복·차이 감사](HISTORICAL_PROTOCOL_AUDIT_KO.md)에 따라 기존 producer를 우선 재사용한다. CORE의 d0 전체 action probs는 T61 `action:mean:L10H7`을, prefix canonical KV는 T55 noun `v5`를 포함한다. 새 head 선택만을 위해 full dump를 늘릴 필요는 없다. 후속 denoise step에서 T61 map을 비교하려면 현재 L8H2 selected-map 외에 mean:L10H7을 명시적으로 추가하고 budget을 다시 계산해야 한다. 기존 CORE 용량 수치는 그 추가를 포함하지 않는다. Feature-aware extent의 새 검증에는 encoded/image_tokens 및 정합된 RGB-D·K/T·시간·손 pose를 유지한다.

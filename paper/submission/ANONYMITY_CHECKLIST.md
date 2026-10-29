# Anonymity Checklist

Updated: 2026-10-05 23:52 KST

2027 세부 규정: `UNKNOWN`; 작업 기준: `PROVISIONAL_2026`와 `PROJECT_POLICY`.
[정책](../guidelines/CVPR_ANONYMITY_POLICY.md),
[2026 double-blind 안내](https://cvpr.thecvf.com/Conferences/2026/AuthorGuidelines).

## 현재 scaffold에서 확인한 범위

- [x] Working metadata는 Anonymous Authors이며 review PDF는 anonymous submission으로 출력한다.
- [x] Main/supplementary의 author metadata와 제한된 text·link scan을 확인했다.
- [x] 현재 PDF에 내부 `/mnt/` 경로·email/기관 예시·source sidecar를 삽입하지 않았다.
- [x] Supplementary는 별도의 anonymous draft PDF다.

위 항목은 [WORKSPACE_VALIDATION](WORKSPACE_VALIDATION.json)의 정확한 PDF hash와 검사 범위에 한한다.
자동 scan의 no-finding을 모든 identity leakage·hidden instruction 부재의 증명으로 쓰지 않는다.

## 최종 제출물에 필요한 검사

- [ ] Main/supplementary 전체 내용의 저자·기관·grant·acknowledgment·식별 가능 문구를 수동 검토한다.
- [ ] Self-citation을 정상적으로 유지하며 제3자 문장으로 설명한다.
- [ ] Concurrent submission 인용이 있으면 익명 copy·차이 설명 등 해당 규정을 따른다.
- [ ] 모든 URL의 identity·tracking·review 우회 여부와 허용되는 citation/dataset 예외를 판단한다.
- [ ] PDF metadata·annotation·attachment·filename과 모든 공개 package entry를 확인한다.
- [ ] Video/audio/screenshot의 자막·화면·목소리·소속·metadata를 검토한다.
- [ ] 공개 code/config/log 안의 username·host·absolute path·secret·식별 comment를 검토한다.
- [ ] Hidden reviewer instruction·prompt injection을 내용/렌더링/원본에서 수동 확인한다.
- [ ] 실제 uploaded/downloaded package를 마지막으로 대조한다.

영상·code package가 없으므로 관련 항목은 `NOT_CHECKED`다. 내부 provenance용 registry·sidecar를
자동으로 제출 package에 포함하지 않는다. Camera-ready identity 복원은 승인된 이후 별도 작업이다.

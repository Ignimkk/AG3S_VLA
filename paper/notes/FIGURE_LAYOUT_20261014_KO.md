# Figure 1–3 배치 수정

2026-10-14 사용자 요청에 따라 Figure 1을 원본 전체가 보이도록 복원했다. 이전의 crop 관련 기록은 과거 개정 상태이며, 현재 배치는 이 문서를 따른다.

- Figure 1: PNG의 하단 방법 strip과 오른쪽 performance 영역까지 전부 표시. 제목 아래, Abstract 이전의 첫 페이지 title block에 고정했다. 전체 그림과 Abstract가 첫 페이지에 들어가도록 종횡비를 유지하여 본문 폭의 75%로 배치했다.
- Figure 2: 이전 Figure 1이 있던 2쪽 상단에 배치했다.
- Figure 3: 이전 Figure 2가 있던 4쪽 상단, Method 영역에 배치했다.
- 2단 figure가 다음 페이지로 밀리는 LaTeX 특성 때문에 Figure 2/3의 float 선언은 main.tex에서 미리 배치했다. 그림 번호와 본문의 참조는 유지된다.
- PNG 원본과 공식 cvpr.sty는 변경하지 않았다. Figure 1의 performance 수치는 원본의 예시 값임을 caption에 명시했으며, probe 학습과 frozen VLA의 관계도 설명했다.
- Tectonic으로 PDF와 SyncTeX를 재생성하고 첫 페이지·그림 페이지의 실제 배치를 확인했다. 미정의 참조 및 box overflow 여부도 점검했다.

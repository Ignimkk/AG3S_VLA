# 역할별 설정과 실행

새 checkout에서 아래 실행기를 사용한다. 기존 세션을 resume하지 않는다.

```bash
bash .claude/roles/launch.sh implementer
bash .claude/roles/launch.sh verifier
bash .claude/roles/launch.sh scribe
```

`*.settings.json`은 project root 기준의 논리 경로를 가진 **템플릿**이다.
`scripts/setup/materialize_role_settings.py`가 자기 checkout의 실제 절대 경로
`//mnt/dev/AG3S_VLA-.../...`로 변환한 ignored `*.local.settings.json`을 만들고,
실행기가 그 파일을 `--settings`에 전달한다. 템플릿을 CLI에 직접 전달하지 않는다.
프로젝트 루트의 `.claude/settings.json`은 공통 hook과 이전 workspace 편집 금지를 적용한다.

`--settings` 파일 안의 단일 `/path`는 그 설정 파일의 디렉터리를 기준으로 해석된다.
`//path`가 filesystem 절대 경로다. 이 차이 때문에 별도 materialization을 사용한다.
공식 문서: https://code.claude.com/docs/en/permissions#read-and-edit

권한 설정은 역할 소유권의 보조 장치다. Python 등으로 직접 파일을 쓰는 모든 작업을
sandbox처럼 막는다고 가정하지 않는다. 자신의 branch와 소유 파일을 먼저 확인한다.

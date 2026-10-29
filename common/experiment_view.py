"""A human-readable view of a closed-loop experiment inside the checkout (`outputs/`, git-ignored).

The raw run (server records, replays — gigabytes) stays in `results/development/<run>/`. This makes
`outputs/<YYYY-MM-DD>_<experiment>/` with one folder per scene holding each configuration's third-person video,
named `<n>_<configuration>_<result>.mp4`, and a README with what the experiment is, the result table and the raw
paths (user instruction 2026-10-23). Videos are hard links (no extra space); a copy where linking fails.

  python -m common.experiment_view --name 2026-10-23_사과10장면_고정형상따라감_비교 \
      --about "무슨 실험인가 …" \
      --config "VLA단독=<verify>/IFG-APPLE:E0" --config "새비교기준구성=<verify>/IFG-APPLE-NBC-TARGETSURFACE:NBCTS" \
      --config "새비교기준+고정형상따라감=<verify>/IFG-APPLE-TARGETSURFACE-FOLLOW:NBCTF" [--scenes 56_s563 211_s2113 …]

Each `--config LABEL=KIT_DIR:CONFIGURATION` reads `KIT_DIR/analysis/<CONFIGURATION>_ep<scene>.json` (success,
failure_mode) and `KIT_DIR/runs/<CONFIGURATION>/<CONFIGURATION>_ep<scene>/third_person_front.mp4`. Scenes default to
the ones every configuration has.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[1]
#: failure_mode (analyze_cl.py) -> file-name word
RESULT_WORD = {"ok": "성공", "no_grasp": "쥐지못함", "slip_at_lift": "들때미끄러짐",
               "slip_in_transport": "운반중미끄러짐", "regrasp": "다시잡음"}


def _result(a: dict) -> str:
    mode = a.get("failure_mode")
    word = RESULT_WORD.get(mode, str(mode))
    if a.get("success"):
        return "성공" if mode in (None, "ok") else f"성공({word} 표시)"
    return f"실패-{word}"


def _seconds(run: Path) -> str:
    """Recorded length (control rows / 15 Hz) and whether the run ended at the place stop."""
    try:
        import numpy as np
        with np.load(run / "traj.npz") as z:
            n = int(z["executed_actions"].shape[0])
            placed = "place_stop_t" in z.files and int(z["place_stop_t"]) >= 0
        return f"{n / 15.0:.1f} s" + (", place 뒤 종료" if placed else "")
    except Exception:
        return "길이 미상"


def _scenes(kit: Path, conf: str) -> set[str]:
    out = set()
    for f in (kit / "analysis").glob(f"{conf}_ep*.json"):
        m = re.match(rf"{re.escape(conf)}_ep(\d+_s\d+)\.json$", f.name)
        if m:
            out.add(m.group(1))
    return out


def build(name: str, about: list[str], configs: list[tuple[str, Path, str]], scenes: list[str] | None = None,
          root: Path | None = None) -> Path:
    out = (root or CHECKOUT / "outputs") / name
    if out.exists():
        raise SystemExit(f"{out} already exists — pick a new name (a view is never overwritten)")
    if scenes is None:
        common = None
        for _, kit, conf in configs:
            s = _scenes(kit, conf)
            common = s if common is None else common & s
        scenes = sorted(common or [], key=lambda s: int(s.split("_")[0]))
    rows, how = [], set()
    for sc in scenes:
        d = out / f"장면{sc.split('_')[0]}"
        d.mkdir(parents=True)
        row = [sc.split("_")[0]]
        for i, (label, kit, conf) in enumerate(configs, 1):
            res = _result(json.loads((kit / "analysis" / f"{conf}_ep{sc}.json").read_text()))
            src = kit / "runs" / conf / f"{conf}_ep{sc}" / "third_person_front.mp4"
            dst = d / f"{i}_{label}_{res}.mp4"
            try:
                os.link(src, dst)
                how.add("hard link")
            except OSError:
                shutil.copy2(src, dst)
                how.add("복사본")
            row.append(f"{res} ({_seconds(src.parent)})")
        rows.append(row)
    head = " | ".join(f"{i}. {label}" for i, (label, _, _) in enumerate(configs, 1))
    lines = [f"# {name}", "", *[f"- {a}" for a in about],
             "- 영상: 3인칭(녹화 카메라) 시점. 파일 이름 = 순번_구성_결과. 표의 괄호 = 기록 길이",
             "- 기록 길이 규칙(2026-10-21 이후 run): place 판정 2.5 초 뒤 종료. place 에 이르지 못한 run 과 그 규칙 전 기록은 최대 step(40 s)까지",
             f"- 영상 파일은 원본의 {' · '.join(sorted(how))}이다 (원시 기록은 storage 에 그대로 있다)",
             "- 원시 기록:", *[f"  - {label}: `{kit}`" for label, kit, _ in configs], "",
             f"| 장면 | {head} |", "|---|" + "---|" * len(configs)]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    lines += ["", "성공 수: " + ", ".join(
        f"{label} {sum(r[i].startswith('성공') for r in rows)} / {len(rows)}" for i, (label, _, _) in enumerate(configs, 1))]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", required=True, help="outputs/<name>: YYYY-MM-DD_<실험 설명>")
    ap.add_argument("--about", action="append", default=[], help="README 첫 줄들 (무슨 실험이고 무엇을 썼는지)")
    ap.add_argument("--config", action="append", required=True, metavar="LABEL=KIT_DIR:CONFIGURATION")
    ap.add_argument("--scenes", nargs="*", default=None, help="<episode>_s<seed> list; default = common to all")
    a = ap.parse_args()
    configs = []
    for spec in a.config:
        label, rest = spec.split("=", 1)
        kit, conf = rest.rsplit(":", 1)
        configs.append((label, Path(kit), conf))
    print(build(a.name, a.about, configs, a.scenes))


if __name__ == "__main__":
    main()

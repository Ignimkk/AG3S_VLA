"""Build presentation figures from existing videos and verified measurements; no rollout."""
from pathlib import Path
import hashlib
import json
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OUT = HERE / "assets"
OUT.mkdir(exist_ok=True)
RUN = Path("/mnt/dev/AG3S_VLA-storage/results/development/20261019T132719+0900_f15e587290f6_11408fd6d27f/verify")
FIG = ROOT / "AG3S/ag3s/docs/figures/ifg-split"
VERIFY = ROOT / "AG3S/ag3s/docs/handoff"
font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
font_manager.fontManager.addfont(str(font))
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=str(font)).get_name(),
                     "axes.unicode_minus": False, "font.size": 11, "savefig.facecolor": "white"})

def read(path):
    return json.loads(path.read_text())

def provenance(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def save(fig, name, data):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    (OUT / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    plt.close(fig)

# Exact decoded video frames: no synthetic scene, annotation, or trajectory overlay added.
sources = []
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for row, (kit, arm, steps, label) in enumerate([
    ("IFG-APPLE", "E0", [96, 144, 180], "VLA 단독"),
    ("IFG-APPLE-SPLIT", "SPL", [96, 146, 180], "AG3S 적용 · target 분리")]):
    video = RUN / kit / "runs" / arm / f"{arm}_ep954_s9543" / "third_person_front.mp4"
    sources.append({**provenance(video), "frames_zero_based": steps, "fps": 15})
    for col, step in enumerate(steps):
        frame = OUT / f"{arm}_frame_{step:03d}.png"
        if not frame.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select=eq(n\\,{step})",
                            "-frames:v", "1", "-threads", "1", str(frame)], check=True)
        axes[row, col].imshow(plt.imread(frame))
        axes[row, col].set_axis_off()
        stage = ["접근", "첫 닫힘 명령 부근", "닫힘 이후"][col]
        axes[row, col].set_title(f"{label}\n{stage} · 영상 {step/15:.2f} s (frame {step})", fontsize=12, pad=9)
fig.suptitle("같은 사과·바나나 장면에서 나타난 파지 실패  |  episode 954 · seed 9543", fontsize=17, y=1.015)
fig.text(.5, .012, "위: 사과를 들어 올림   /   아래: 사과가 탁자에 남음     ·     원본 영상의 TO 화살표는 시각화용 8배 배율", ha="center", fontsize=12)
fig.tight_layout(rect=(0, .04, 1, 1), h_pad=3.5)
save(fig, "01_video_comparison", {"sources": sources, "note": "Original decoded frames; times are video timestamps. Closing indices from ifg-goal-drift-ep954.json.", "closing_source": provenance(FIG / "ifg-goal-drift-ep954.json")})

base = read(FIG / "ifg-goal-drift-ep954.json")["arms"]
anchor = read(VERIFY / "IFG-APPLE-TOA2.verify.json")["numbers"]["ep954"]
nbc = read(VERIFY / "IFG-APPLE-NBC.verify.json")["numbers"]["first_close"]["954"]
rows = [
    ("VLA 단독", base["E0"]["tcp_at_first_close_mm"], 144, "성공"),
    ("AG3S 기준", base["SPL"]["tcp_at_first_close_mm"], 146, "실패"),
    ("목표 위치 고정", anchor["tcp_minus_apple_t0_at_first_close_mm"]["SPLA"], 113, "실패"),
    ("손목 가중치", nbc["SPLW"]["tcp_rel_apple_t0_mm"], 146, "실패"),
    ("손목 가중치 + 접촉 허용", nbc["SPLT"]["tcp_rel_apple_t0_mm"], 146, "성공"),
]
colors = ["#159b8b", "#e96a40", "#4776b4", "#9973b0", "#687d37"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)
for j, ax in enumerate(axes):
    vals = [r[1][j] for r in rows]
    ax.barh(np.arange(len(rows)), vals, color=colors, height=.6)
    ax.axvline(0, color="#9ba0a5", linewidth=.7)
    ax.set_xlim(-12, 45)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_title(["앞 방향 x", "옆 방향 y", "높이 z"][j])
    for i, v in enumerate(vals):
        ax.text(v + (1 if v >= 0 else -1), i, f"{v:.1f}", ha="left" if v >= 0 else "right", va="center")
    ax.set_xlabel("초기 사과 위치 기준 TCP 좌표 (mm)")
    ax.spines[["top", "right"]].set_visible(False)
axes[0].invert_yaxis()
fig.suptitle("목표 위치의 어긋남을 줄여도 파지 성공이 보장되지는 않았다  |  episode 954", fontsize=16)
fig.text(.5, .015, "목표 위치 고정: 옆 방향은 15 mm로 복원됐으나 31 step 일찍, VLA 단독보다 약 13 mm 높게 닫음\n각 구성의 첫 닫힘 시점 비교 · TCP 좌표이며 목표 중심과의 거리나 손바닥 anchor 오차가 아님", ha="center", fontsize=10)
fig.tight_layout(rect=(0, .12, 1, .94))
save(fig, "02_close_position", {"rows": [{"method": r[0], "tcp_rel_apple_t0_mm": r[1], "first_close_step": r[2], "outcome": r[3]} for r in rows],
    "sources": [provenance(FIG / "ifg-goal-drift-ep954.json"), provenance(VERIFY / "IFG-APPLE-TOA2.verify.json"), provenance(VERIFY / "IFG-APPLE-NBC.verify.json")],
    "note": "Anchor values are rounded in the source. Values are recorded first-close TCP coordinates, not a newly simulated measurement."})
print(f"Figures written to {OUT}")

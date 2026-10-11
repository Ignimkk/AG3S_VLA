"""X3 검증 영상 — 정책이 낸 청크와 TO 가 고친 궤적을 나란히 돌린다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/X3/make_x3_video.py

왼쪽은 정책이 그대로 낸 청크(reference), 오른쪽은 SQP 가 고쳐 내놓은 궤적이다. 같은 프레임,
같은 카메라. 오른쪽 패널의 숫자는 `outputs/verify/X3/probe_traj.json` 에서 온다 —
영상 안에서 만들어 내는 숫자는 하나도 없다.

`traj_headfix.npz` 는 probe 가 `TrajectoryOptimizer._finish` 를 감싸 그 자리에서 받은
`result.trajectory` 와 `reference` 다. 다시 계산하지 않는다.
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import argparse
import functools
import json
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, _ag3s_workspace_path(''))

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import mujoco

from AG3S.ag3s.experiments.common import figstyle
from AG3S.ag3s.experiments.sources.policy_record import load_run, pose_scene, replay_scene

FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


@functools.lru_cache(maxsize=16)
def font(size: int):
    """캐시한다. 20 MB CJK ttc 를 프레임마다 여섯 번 열면 장당 2 s 가 거기서 간다."""
    try:
        return ImageFont.truetype(FONT_PATH, size, index=2)  # index 2 = CJK KR
    except Exception:
        return ImageFont.load_default()


KO = figstyle.use_korean()
C = figstyle.CATEGORICAL
OK, BAD = C[2], C[7]
RECORD = _ag3s_workspace_path('outputs/live_test/20260925_headfix16d/records/run_0000/run_0000')
V = pathlib.Path(_ag3s_workspace_path('outputs/verify/X3'))
OUT = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/x3'))


def T(ko: str, en: str) -> str:
    return ko if KO else en


def free_camera(model, data):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 테이블을 본다. 자리는 모델에서 읽는다 — 눈으로 맞추지 않는다.
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    if tid < 0:
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "apple")
    base = data.xpos[tid] if tid >= 0 else [0.65, 0.0, 0.0]
    # 테이블 body 원점은 바닥(z=0)이라 그대로 보면 다리만 잡힌다. 상판 높이로 올린다.
    cam.lookat[:] = [base[0], base[1], 0.95]
    cam.distance = 2.6
    cam.azimuth = 135.0
    cam.elevation = -30.0
    return cam


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--stride", type=int, default=1, help="지평 스텝을 몇 개마다 그릴까")
    ap.add_argument("--out", default=str(OUT / "x3-verify.mp4"))
    args = ap.parse_args()

    npz = np.load(V / "traj_headfix.npz", allow_pickle=True)
    jnames = [str(x) for x in npz["joint_names"]]
    probe = json.loads((V / "probe_traj.json").read_text())
    rows = {m["i"]: m for m in probe["frames"]}

    run = load_run(RECORD, limit=15)
    scene = replay_scene(run, height=480, width=640)
    renderer = scene._renderer_for()
    renderer.disable_depth_rendering()
    renderer.disable_segmentation_rendering()

    qadr = [scene._qadr[j] for j in jnames]
    probe_shape = npz["traj_00"].shape
    if probe_shape[0] != len(jnames):
        raise SystemExit(
            f"궤적이 {probe_shape[0]} 행인데 관절 이름은 {len(jnames)} 개다. "
            "probe 가 layout.full_q 로 펼쳐 저장하는지 확인해라 — 그냥 붙이면 팔 각도가 "
            "토르소로 들어가 로봇이 뒤틀린 영상이 나온다 (2026-09-25 에 실제로 냈다)."
        )
    n_chunks = sum(1 for k in npz.files if k.startswith("traj_"))
    clearances = [rows[i]["clearance_after_mm_rollout"] for i in range(n_chunks)]
    befores = [None] * n_chunks

    frames_dir = pathlib.Path("/tmp/claude-x3-video")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    cam = None
    k = 0
    for i in range(n_chunks):
        traj = npz[f"traj_{i:02d}"]
        ref = npz[f"ref_{i:02d}"]
        row = rows[i]
        horizon = min(traj.shape[1], ref.shape[1])
        pose_scene(scene, run.steps[i])
        if cam is None:
            cam = free_camera(scene.model, scene.data)

        for t in range(0, horizon, args.stride):
            imgs = []
            for arr in (ref, traj):
                pose_scene(scene, run.steps[i])
                for a, q in zip(qadr, arr[:, t]):
                    scene.data.qpos[a] = q
                mujoco.mj_forward(scene.model, scene.data)
                renderer.update_scene(scene.data, camera=cam)
                imgs.append(np.asarray(renderer.render()))

            H, W = imgs[0].shape[:2]
            PAD, BAR, PLOT_W = 8, 118, 300
            canvas = Image.new("RGB", (W * 2 + PAD * 3 + PLOT_W, H + BAR + PAD * 2),
                               (252, 252, 251))
            canvas.paste(Image.fromarray(imgs[0]), (PAD, BAR))
            canvas.paste(Image.fromarray(imgs[1]), (PAD * 2 + W, BAR))
            d = ImageDraw.Draw(canvas)
            col = (27, 175, 122) if row["status"] == "feasible" else (227, 73, 72)

            d.text((PAD, 8), T(f"청크 {i} / {n_chunks - 1}    지평 스텝 {t} / {horizon - 1}"
                               f"    단계 {row['phase']}",
                               f"chunk {i} / {n_chunks - 1}    horizon step {t} / {horizon - 1}"
                               f"    phase {row['phase']}"), font=font(19), fill=(11, 11, 11))
            d.text((PAD, 36), T(f"status {row['status']}    validity {row['validity']}"
                                f"    clearance_after {row['clearance_after_mm_rollout']:+.2f} mm",
                                f"status {row['status']}    validity {row['validity']}"
                                f"    clearance_after {row['clearance_after_mm_rollout']:+.2f} mm"),
                   font=font(19), fill=col)
            d.text((PAD, 66), T("왼쪽: 정책이 낸 청크 (reference)   |   오른쪽: SQP 가 고친 궤적"
                                "   |   오른쪽 끝: 프레임별 clearance",
                                "left: policy chunk   |   right: SQP-corrected   |   far right: "
                                "clearance per frame"),
                   font=font(16), fill=(82, 81, 78))
            d.text((PAD, 92), T(f"기록 {pathlib.Path(RECORD).parent.parent.parent.name}"
                                f"   ·   숫자 출처 probe_traj.json",
                                f"record {pathlib.Path(RECORD).parent.parent.parent.name}"
                                f"   ·   numbers from probe_traj.json"),
                   font=font(14), fill=(82, 81, 78))

            # 오른쪽 끝: clearance 타임라인 (직접 그린다 — matplotlib 없이)
            x0, y0 = W * 2 + PAD * 3, BAR + 20
            pw, ph = PLOT_W - 40, H - 70
            lo, hi = min(clearances + [0.0]), max(clearances + [0.0])
            span = (hi - lo) or 1.0
            ypix = lambda v: y0 + ph - (v - lo) / span * ph
            d.line([(x0, ypix(0.0)), (x0 + pw, ypix(0.0))], fill=(11, 11, 11), width=2)
            d.text((x0 + pw + 4, ypix(0.0) - 8), "0", font=font(13), fill=(11, 11, 11))
            for j, cv in enumerate(clearances):
                cx = x0 + (j / max(n_chunks - 1, 1)) * pw
                cy = ypix(cv)
                cc = (27, 175, 122) if rows[j]["status"] == "feasible" else (227, 73, 72)
                r_ = 7 if j == i else 4
                d.ellipse([cx - r_, cy - r_, cx + r_, cy + r_], fill=cc)
            d.text((x0, y0 - 18), T("clearance_after (mm)", "clearance_after (mm)"),
                   font=font(14), fill=(11, 11, 11))
            d.text((x0, y0 + ph + 6), T(f"{lo:+.1f} … {hi:+.1f} mm", f"{lo:+.1f} ... {hi:+.1f} mm"),
                   font=font(13), fill=(82, 81, 78))

            canvas.save(frames_dir / f"f{k:05d}.png")
            k += 1
        print(f"  청크 {i:2d}  {row['status']:9s}  clearance {row['clearance_after_mm_rollout']:+7.2f} mm"
              f"   누적 {k} 장", flush=True)

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(args.fps),
                    "-i", str(frames_dir / "f%05d.png"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(out)],
                   check=True)
    print(f"\nwrote {out}  ({k} 장 · {args.fps} fps · {k / args.fps:.1f} s)")
    (out.with_suffix(".json")).write_text(json.dumps({
        "video": out.name, "n_frames": k, "fps": args.fps,
        "source_trajectories": "outputs/verify/X3/traj_headfix.npz",
        "source_numbers": "outputs/verify/X3/probe_traj.json",
        "record": RECORD,
        "panels": ["정책이 낸 청크 (reference)", "SQP 가 고친 궤적",
                   "프레임별 clearance_after 와 status"],
        "numbers": {"clearance_after_mm": clearances,
                    "status": [rows[i]["status"] for i in range(n_chunks)],
                    "validity": [rows[i]["validity"] for i in range(n_chunks)]},
    }, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

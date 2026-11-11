"""CVPR 2027 Fig. 1 v6: a richer teaser (user request 2026-11-10, after two figurelabs mocks).

  top     person icon + instruction -> frozen VLA (robot icon); colour key
  middle  one SafeLIBERO episode (spatial Level I, task 2, episode 9, init 9, seed 7, wine bottle), same moment (obs 110):
          prior hero (AEGIS, detector boxes; bottle knocked over) | three paired axes | AG3S hero (bottle upright)
          axes  Identity   prior: the bottle gets no box             AG3S: noun-attention seeds on the bowl
                Extent     prior: one box spans the cabinet + bowl   AG3S: the bowl's depth-bounded surface
                Permission prior: one end-effector ellipsoid (KNOWS) AG3S: contact links + attached bowl
  bottom  four more episodes (other tasks / obstacles), same initial state and seed: policy alone after the collision
          vs. the AG3S episode's last frame (bowl on the plate, obstacle in place)

Every raster is a recorded frame or a recorded-geometry overlay (v4 assets: fig1_v4_assets.py; gallery: episode npz
frames read here). The mocks themselves (AI-drawn) are not used. No number is printed in the figure.

    .venv-ag3s/bin/python paper/tools/plotting/cvpr_v6_fig1_compose.py
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np
from matplotlib.patches import Ellipse, Polygon
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import cvpr_v3_fig1_compose as v3  # noqa: E402
import cvpr_v5_fig1_compose as v5  # noqa: E402
from cvpr_v3_fig1_compose import C, FS, PDF_DPI, PNG_DPI, REPO, STRICT_LAYOUT, TEXTWIDTH_IN, Fig, sha  # noqa: E402
from cvpr_v5_fig1_compose import INK, MUTE, rbox, rpanel, shadow  # noqa: E402

V6 = REPO / "paper/figures/fig01_teaser/v6"
BENCH = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
INSTR = "pick up the black bowl on the stove and place it on the plate"
PRIOR_BG, OURS_BG = "#FDF3F3", "#F2F8F1"
# gallery: (task, episode, obstacle label); selected because the policy alone collides and AG3S succeeds without
# collision on the same initial state and seed (episodes.jsonl of both runs)
GALLERY = [(0, 9, "wine bottle"), (2, 6, "moka pot"), (2, 8, "milk carton"), (3, 5, "book")]
GAL_AFTER = 40                     # policy-alone frame shown = collide_t + GAL_AFTER (obstacle visibly displaced)


# ------------------------------------------------------------------------------------------- gallery frames
def _episode(run, ep):
    for line in open(run / "episodes.jsonl"):
        e = json.loads(line)
        if e["episode_idx"] == ep:
            return e
    raise KeyError(ep)


def gallery_frames():
    out = []
    for t, ep, lab in GALLERY:
        rv = pathlib.Path(glob.glob(str(BENCH / f"*qe-vla-e1a-spatial-LI-t{t}-*"))[0])
        ra = pathlib.Path(glob.glob(str(BENCH / f"*qe-pe123458-e1a-spatial-LI-t{t}-LI-*pe123458-C"))[0])
        ev, ea = _episode(rv, ep), _episode(ra, ep)
        assert (ev["init_state_index"], ev["seed"]) == (ea["init_state_index"], ea["seed"])
        sv, sa = ev["safelibero"], ea["safelibero"]
        assert sv["collided"] and not sa["collided"] and ea["official_success"], (t, ep)
        nv, na = rv / "episodes" / pathlib.Path(ev["npz"]).name, ra / "episodes" / pathlib.Path(ea["npz"]).name
        fv = np.load(nv)["obs_agentview_image"]; fa = np.load(na)["obs_agentview_image"]
        iv = min(int(sv["collide_t"]) + GAL_AFTER, len(fv) - 1); ia = len(fa) - 1
        big = lambda a: np.asarray(Image.fromarray(np.ascontiguousarray(a[::-1])).resize((768, 768), Image.LANCZOS))  # noqa: E731
        out.append(dict(task=t, episode=ep, label=lab, task_description=ev["task_description"], obstacle=sv["obstacle"],
                        init=ev["init_state_index"], seed=ev["seed"], vla=big(fv[iv]), ag3s=big(fa[ia]),
                        meta=dict(vla_run=str(rv), vla_npz=str(nv), vla_npz_sha256=sha(nv), vla_frame=iv,
                                  vla_collide_t=sv["collide_t"], vla_success=ev["official_success"],
                                  ag3s_run=str(ra), ag3s_npz=str(na), ag3s_npz_sha256=sha(na), ag3s_frame=ia,
                                  ag3s_collided=sa["collided"], ag3s_success=ea["official_success"])))
    return out


def crop(a, centre, width, aspect):
    H, W = a.shape[:2]
    w = int(round(min(width, W, H * aspect))); h = int(round(w / aspect))
    x0 = int(np.clip(centre[0] - w / 2, 0, W - w)); y0 = int(np.clip(centre[1] - h / 2, 0, H - h))
    return a[y0:y0 + h, x0:x0 + w], [x0, y0, w, h]


# ------------------------------------------------------------------------------------------- icons
def robot_icon(f, x, y, s=1.0):
    """Minimal robot head (rounded square, two eyes, antenna), centre (x, y)."""
    col = "#3A4A6B"
    rbox(f, x - 0.11 * s, y - 0.085 * s, 0.22 * s, 0.17 * s, 0.045 * s, fc=col, ec="none", zorder=5)
    for dx in (-0.05, 0.05):
        f.fig.add_artist(Ellipse(((x + dx * s) / f.W, (y + 0.005) / f.H), 0.045 * s / f.W, 0.045 * s / f.H,
                                 transform=f.fig.transFigure, fc="white", ec="none", zorder=6))
    f.line([(x, y + 0.085 * s), (x, y + 0.13 * s)], col, lw=1.0, z=5)
    f.fig.add_artist(Ellipse((x / f.W, (y + 0.14 * s) / f.H), 0.035 * s / f.W, 0.035 * s / f.H, transform=f.fig.transFigure,
                             fc=col, ec="none", zorder=5))


def bubble(f, x, y, s, size=7.5, tail="left", **kw):
    t = f.text(x, y, s, size=size, color=INK, zorder=6,
               bbox=dict(boxstyle="round,pad=0.42,rounding_size=0.75", fc="white", ec="#9AA3B2", lw=0.8), **kw)
    return t


# ------------------------------------------------------------------------------------------- figure
def fig01_v6():
    W = TEXTWIDTH_IN
    M = 0.03
    TOPB = 0.33                                  # instruction band
    HDR = 0.20                                   # lane headers
    HH = 1.32                                    # hero height
    HW = 1.70                                    # hero width
    TW, GAPT = 1.10, 0.06                        # thumb width, hero-thumb gap
    TH = (HH - 2 * 0.06) / 3
    GH, GT = 0.60, 0.20                          # gallery image height, gallery title band
    H = round(M + GH + 0.19 + GT + 0.08 + HH + 0.06 + HDR + TOPB, 3)
    f = Fig(W, H)
    used = []
    A4 = v5.A4

    def asset(name, slot, centre, width, aspect):
        p = A4 / f"{name}.png"
        a = np.asarray(Image.open(p).convert("RGB"))
        c, box = crop(a, centre, width, aspect)
        used.append(dict(slot=slot, file=str(p.relative_to(REPO)), sha256=sha(p), crop_px=box, provenance=v5.META.get(f"{name}.png", {})))
        return c

    # ---- top band: person -> instruction -> frozen VLA (robot)
    yb = H - TOPB / 2 - 0.02
    v5.person(f, M + 0.10, yb - 0.12, s=1.0)
    bubble(f, 0.30, yb, f"“{INSTR}”", style="italic")
    f.fig.add_artist(Polygon([(0.30 / W, (yb + 0.03) / H), (0.30 / W, (yb - 0.05) / H), (0.215 / W, (yb - 0.07) / H)],
                             closed=True, transform=f.fig.transFigure, fc="white", ec="none", zorder=7))
    f.line([(0.215, yb - 0.07), (0.292, yb - 0.05)], "#9AA3B2", lw=0.8, z=7)
    xr = 3.76
    f.arrow((3.63, yb), (xr - 0.02, yb), color="#7A8496", lw=0.9, ms=6)
    robot_icon(f, xr + 0.13, yb - 0.02)
    f.text(xr + 0.29, yb + 0.045, "frozen VLA", size=FS, fontweight="bold", color=INK)
    f.text(xr + 0.29, yb - 0.075, r"$\pi_{0.5}$" + " ❄", size=FS, color=MUTE)
    key = [(C["target"], "target surface"), (C["neigh"], "obstacle"), (C["gold"], "seeds / contact links"),
           (C["attached"], "carried object")]
    kx, ky = W - M - 2.16, yb + 0.07
    for i, (col, lab) in enumerate(key):
        cx, cy = kx + (i // 2) * 1.00, ky - (i % 2) * 0.15
        f.fig.add_artist(Ellipse(((cx + 0.035) / W, cy / H), 0.07 / W, 0.07 / H, transform=f.fig.transFigure, fc=col, ec="none", zorder=6))
        f.text(cx + 0.10, cy, lab, size=FS, color=MUTE)

    # ---- middle: hero | thumbs | axes | thumbs | hero
    yH = H - TOPB - HDR - HH
    xP, xO = M, W - M - HW
    xtP, xtO = xP + HW + GAPT, xO - GAPT - TW
    xc0, xc1 = xtP + TW + 0.04, xtO - 0.04
    rbox(f, M - 0.0, yH - 0.05, xc0 - 0.02 - M + 0.0, HH + HDR + 0.03, 0.08, fc=PRIOR_BG, ec="#F0D2D2", lw=0.6, zorder=0)
    rbox(f, xc1 + 0.02, yH - 0.05, W - M - xc1 - 0.02, HH + HDR + 0.03, 0.08, fc=OURS_BG, ec="#CFE3CC", lw=0.6, zorder=0)
    yh = yH + HH + 0.10
    f.text(xP + 0.06, yh, "Prior: add-on detector + safety filter", size=7.5, fontweight="bold", color=C["red"])
    f.text(W - M - 0.06, yh, "AG3S (ours): the policy’s own attention", size=7.5, fontweight="bold",
           color=C["green"], ha="right")
    asp = HW / HH
    rpanel(f, (xP, yH, HW, HH), asset("l_main", "prior hero (AEGIS, obs 110)", (400, 400), 700, asp), C["red"],
           badge="✗ bottle knocked over", bcol=C["red"])
    rpanel(f, (xO, yH, HW, HH), asset("r_main", "AG3S hero (obs 110)", (420, 400), 700, asp), C["green"],
           badge="✓ bottle untouched", bcol=C["green"])
    tasp = TW / TH
    rows = [
        ("Identity", "which object?",
         ("l_boxes", (430, 380), 520, "✗ bottle: no box"), ("r_identity", (300, 345), 420, "✓ seeds on bowl")),
        ("Extent", "how much space?",
         ("l_boxes", (190, 335), 500, "✗ coarse box"), ("r_extent", (300, 370), 460, "✓ bowl surface")),
        ("Permission", "who may touch it?",
         ("l_ee", (560, 380), 460, "✗ hand only"), ("r_permission", (330, 390), 520, "✓ fingers + bowl")),
    ]
    for i, (axis, q, (pn, pc, pw, pb), (on, oc, ow, ob)) in enumerate(rows):
        y = yH + HH - (i + 1) * TH - i * 0.06
        rpanel(f, (xtP, y, TW, TH), asset(pn, f"prior {axis.lower()}", pc, pw, tasp), C["red"], badge=pb, bcol=C["red"], r=0.05)
        rpanel(f, (xtO, y, TW, TH), asset(on, f"AG3S {axis.lower()}", oc, ow, tasp), C["green"], badge=ob, bcol=C["green"], r=0.05)
        cx = (xc0 + xc1) / 2
        f.text(cx, y + TH / 2 + 0.07, axis, size=7.5, fontweight="bold", color=INK, ha="center")
        f.text(cx, y + TH / 2 - 0.07, q, size=FS, color=MUTE, ha="center")
    f.text((xc0 + xc1) / 2, yh - 0.02, "same scene\nsame seed", size=FS, color=MUTE, ha="center", fontweight="bold", linespacing=1.1,
           bbox=dict(boxstyle="round,pad=0.3,rounding_size=0.6", fc="white", ec="#9AA3B2", lw=0.7), zorder=6)

    # ---- bottom: gallery
    yG = M + 0.19
    yt = yG + GH + 0.13
    f.line([(M, yt + GT / 2 + 0.02), (W - M, yt + GT / 2 + 0.02)], "#E1E4EA", lw=0.6, z=0)
    f.text(M + 0.02, yt, "More scenes, same initial state and seed:", size=FS, fontweight="bold", color=INK)
    f.text(M + 2.42, yt, "policy alone", size=FS, fontweight="bold", color=C["red"])
    f.text(M + 3.10, yt, "vs.", size=FS, color=MUTE)
    f.text(M + 3.32, yt, "with AG3S", size=FS, fontweight="bold", color=C["green"])
    gal = gallery_frames()
    PG = 0.10
    pw = (W - 2 * M - 3 * PG) / 4
    iw = (pw - 0.04) / 2
    gside = []
    for k, g in enumerate(gal):
        x = M + k * (pw + PG)
        gasp = iw / GH
        a, bv = crop(g["vla"], (400, 430), 560, gasp)
        b, ba = crop(g["ag3s"], (400, 430), 560, gasp)
        rpanel(f, (x, yG, iw, GH), a, C["red"], badge="✗", bcol=C["red"], r=0.05)
        rpanel(f, (x + iw + 0.04, yG, iw, GH), b, C["green"], badge="✓", bcol=C["green"], r=0.05)
        f.text(x + pw / 2, yG - 0.105, g["label"], size=FS, color=MUTE, ha="center")
        gside.append(dict({k2: v for k2, v in g.items() if k2 not in ("vla", "ag3s")}, crop_vla_px=bv, crop_ag3s_px=ba))
    save(f, "fig01_v6", dict(gallery=gside, image_panels=used))


def save(f, stem, extra):
    mn, mx, cnt = f.audit()
    assert mn >= FS - 1e-6, f"{stem}: font below {FS} pt: {mn}"
    viol = f.layout_audit()
    n_viol = sum(len(v) for v in viol.values())
    print(f"[{stem}] {f.W:.3f} x {f.H:.3f} in, layout audit: " + ", ".join(f"{k}={len(v)}" for k, v in viol.items()))
    for k, v in viol.items():
        for item in v[:40]:
            print("   ", k, item)
    texts = [t.get_text() for t in f.fig.findobj(v3.mtext.Text) if t.get_text().strip() and t.get_visible()]
    if STRICT_LAYOUT:
        assert n_viol == 0, f"{stem}: {n_viol} layout violations"
    V6.mkdir(parents=True, exist_ok=True)
    pdf = V6 / f"{stem}.pdf"
    f.fig.savefig(pdf, dpi=PDF_DPI)
    f.fig.savefig(pdf.with_suffix(".png"), dpi=PNG_DPI)
    v3.plt.close(f.fig)
    side = dict(
        artifact=stem, pdf=str(pdf.relative_to(REPO)), png=str(pdf.with_suffix(".png").relative_to(REPO)),
        size_in=[f.W, f.H], textwidth_in=TEXTWIDTH_IN, raster_dpi_in_pdf=PDF_DPI, png_dpi=PNG_DPI,
        fonts=dict(family="DejaVu Sans", pdf_fonttype=42, min_pt=mn, max_pt=mx, n_text_items=cnt, floor_pt=FS),
        printed_text=texts, printed_numbers=[],
        layout_audit=dict(tolerance_pt=1.0, min_text_to_panel_border_in=0.05, violations={k: len(v) for k, v in viol.items()},
                          n_panels=len(f.panels), n_texts=cnt),
        episode=v5.META.get("_episode"),
        honesty=["middle row: SafeLIBERO spatial Level I task 2 episode 9 (init 9, seed 7); prior hero and identity / extent "
                 "thumbs = AEGIS run (detector boxes recorded in method_calls), permission thumb = KNOWS run (end-effector "
                 "ellipsoid, recorded pose); AG3S = final run pe123458; both heroes at obs 110",
                 "bottom row: episodes selected because the policy alone collides and AG3S succeeds without collision on the same "
                 "initial state and seed; they are examples, not a rate. Left = policy-alone frame collide_t + 40 (or last), "
                 "right = the AG3S episode's last frame",
                 "person / robot icons, chips and arrows are schematic; the robot icon does not imply a dialogue system",
                 "no number printed in the figure"],
        script="paper/tools/plotting/cvpr_v6_fig1_compose.py", generated=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip(),
        is_measured_result=False, scope="qualitative examples; no numbers")
    side.update(extra)
    pdf.with_suffix(".json").write_text(json.dumps(side, indent=1, ensure_ascii=False, default=str) + "\n")
    print(f"wrote {pdf.relative_to(REPO)}  min font {mn} pt")


if __name__ == "__main__":
    fig01_v6()

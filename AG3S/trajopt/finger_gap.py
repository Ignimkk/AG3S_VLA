"""T91b — "is the target between the fingers?" evidence from the grasping hand's wrist camera.

A held object sits in a fixed region of the wrist-camera image, between the fingertips, at a fixed short distance from
the palm; an empty closed hand sees the table (or the far scene) there. Two scalars per request, both in 0..1 and both
computed from inputs any benchmark with a wrist camera provides:

  surface fraction (main)      share of the ROI pixels whose wrist **depth** is in ``[d_min, d_max]`` (metres along the
                               optical axis, i.e. "a surface within finger reach"). Needs no target recognition.
  target projection fraction   share of the ROI pixels covered by the ATS target points projected into the wrist camera
                               (last known geometry when the target is hidden; each point splats a small disc). Depends
                               on the ATS target and on the hand pose of the request.

Everything robot-specific is a `FingerGapConfig` (ROI rectangle as image fractions, depth window). Pure numpy: no MuJoCo,
no robot name, no AG3S pipeline. The state machine (`phase_machine.py`) only receives the two numbers.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np


@dataclasses.dataclass(frozen=True)
class FingerGapConfig:
    """ROI = image rectangle between the fingertips, as fractions of (width, height); depth window in metres."""

    roi_u: tuple = (0.35, 0.65)          # (left, right) fraction of the image width
    roi_v: tuple = (0.65, 0.97)          # (top, bottom) fraction of the image height (row 0 = top)
    d_min: float = 0.03
    d_max: float = 0.11
    splat_frac: float = 0.012            # projected point -> disc radius as a fraction of the image width

    def __post_init__(self) -> None:
        for lo, hi in (self.roi_u, self.roi_v):
            if not (0.0 <= lo < hi <= 1.0):
                raise ValueError("ROI fractions must satisfy 0 <= lo < hi <= 1")
        if not 0.0 <= self.d_min < self.d_max:
            raise ValueError("need 0 <= d_min < d_max")

    def box(self, h: int, w: int) -> tuple[int, int, int, int]:
        """(row0, row1, col0, col1) of the ROI in an image of h rows and w columns (at least one pixel each way)."""
        r0, r1 = int(np.floor(self.roi_v[0] * h)), int(np.ceil(self.roi_v[1] * h))
        c0, c1 = int(np.floor(self.roi_u[0] * w)), int(np.ceil(self.roi_u[1] * w))
        return r0, max(r1, r0 + 1), c0, max(c1, c0 + 1)


def surface_fraction(depth_m: np.ndarray, cfg: FingerGapConfig) -> Optional[float]:
    """Share of ROI pixels with ``d_min <= depth <= d_max`` (invalid = non-finite or <= 0 counts as not a surface)."""
    d = np.asarray(depth_m, float)
    if d.ndim != 2 or d.size == 0:
        return None
    r0, r1, c0, c1 = cfg.box(*d.shape)
    roi = d[r0:r1, c0:c1]
    ok = np.isfinite(roi) & (roi >= cfg.d_min) & (roi <= cfg.d_max)
    return float(ok.mean())


def project_points(points: np.ndarray, K: np.ndarray, T_base_cam: np.ndarray) -> np.ndarray:
    """(n, 3) ``(u, v, z)`` of points given in the frame ``T_base_cam`` maps *camera -> that frame*; z <= 0 -> NaN row.
    Pinhole, OpenCV camera axes (x right, y down, z forward) as ``t79_events.project``."""
    P = np.asarray(points, float).reshape(-1, 3)
    T = np.asarray(T_base_cam, float)
    pc = (P - T[:3, 3]) @ T[:3, :3]                      # inverse rigid transform of row vectors
    z = pc[:, 2]
    out = np.full((len(P), 3), np.nan)
    ok = z > 1e-6
    out[ok, 0] = K[0, 0] * pc[ok, 0] / z[ok] + K[0, 2]
    out[ok, 1] = K[1, 1] * pc[ok, 1] / z[ok] + K[1, 2]
    out[ok, 2] = z[ok]
    return out


def projection_fraction(uvz: np.ndarray, image_hw: tuple, cfg: FingerGapConfig) -> Optional[float]:
    """Share of ROI pixels covered by splatted projected target points. ``uvz`` rows are pixels of an image whose size
    is ``image_hw`` (rows, cols). ``None`` when no point is given (nothing to project)."""
    a = np.asarray(uvz, float).reshape(-1, 3)
    a = a[np.isfinite(a).all(axis=1)]
    if len(a) == 0:
        return None
    h, w = image_hw
    r0, r1, c0, c1 = cfg.box(h, w)
    rad = max(1.0, cfg.splat_frac * w)
    ri = int(np.ceil(rad))
    mask = np.zeros((r1 - r0, c1 - c0), bool)
    off = np.array([[dr, dc] for dr in range(-ri, ri + 1) for dc in range(-ri, ri + 1) if dr * dr + dc * dc <= rad * rad])
    rr = np.rint(a[:, 1]).astype(int)[:, None] + off[None, :, 0] - r0
    cc = np.rint(a[:, 0]).astype(int)[:, None] + off[None, :, 1] - c0
    ok = (rr >= 0) & (rr < mask.shape[0]) & (cc >= 0) & (cc < mask.shape[1])
    mask[rr[ok], cc[ok]] = True
    return float(mask.mean())


# --------------------------------------------------------------------------- robot settings

def rby1_wrist() -> FingerGapConfig:
    """RB-Y1 wrist cameras (640 x 480 depth, fx = fy = 240): the apple sits at u 240-400, v 330-480 between the fingers.
    ROI / depth window are the values `t91_phase_eval gap-select` chose on the RBY1 train scenes (`handoff/T91b.impl.md`)."""
    return FingerGapConfig(roi_u=(0.35, 0.65), roi_v=(0.72, 0.98), d_min=0.03, d_max=0.09)


def panda_eye_in_hand() -> FingerGapConfig:
    """Panda ``robot0_eye_in_hand`` (256 x 256). ROI = column band centred on the grip-site column (the site projects to pixel
    (127.5, 213.5) at 0.097 m) and the rows directly above the finger-pad rows 195..212, so the pads are not "a surface between the
    fingers". The depth window is the RBY1-train one (the grip site is at the same 0.1 m along the optical axis on both robots);
    nothing here was selected on SafeLIBERO phase labels (`handoff/T91b.impl.md`)."""
    return FingerGapConfig(roi_u=(0.38, 0.62), roi_v=(0.62, 0.76), d_min=0.03, d_max=0.09)

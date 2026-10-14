"""pi05_libero EE-delta chunk  <->  V11AB joint chunk for the LIBERO Panda (SL5a).

pi05_libero emits ``A[H, 7]``: robosuite **OSC_POSE** delta actions (``a[:3]`` position, ``a[3:6]`` axis-angle,
``a[6]`` gripper; each in ``[-1, 1]``). V11AB's TrajOpt refines **absolute joint** chunks. This module is the
adapter between the two; nothing here is a V11AB method change.

robosuite 1.4.1 OSC_POSE (``controllers/config/osc_pose.json``, ``controllers/osc.py``, ``base_controller.py``),
checked against the installed source by ``tests/benchmark/test_panda_sl5a.py``:

| quantity | value |
|---|---|
| input clip | each component to ``[-1, 1]`` |
| position | ``Δp = 0.05 · a[:3]`` (m), **world** frame, goal = current EE + Δp |
| orientation | ``Δθ = 0.5 · a[3:6]`` (rad axis-angle), goal = ``R(Δθ) · R_ee`` (**world** frame, left multiply) |
| gripper | ``-1`` open, ``+1`` close (not optimized: V11AB passes grippers through) |

**Controller response model** (``robots/panda/description/osc_response_fit.json``). OSC does not reach a row's
goal in one 50 ms step: on the dev rollouts (init 0-4) the realised world-frame EE step ``y`` follows the
commanded step ``x`` (``0.05 a`` / ``0.5 a``) as a first-order lag ``y_t = alpha y_{t-1} + beta x_t``
(position alpha 0.717 beta 0.071, R² 0.992; rotation alpha 0.875 beta 0.031, R² 0.962). This is a model of the
robot's controller (adapter), not of V11AB.

**Forward (policy -> V11AB, ``delta_chunk_to_joint_chunk``).** Predict the EE path the chunk will produce
from the current pose and the last realised step ``y_{-1}`` with that model, express it in the robot base frame,
and solve IK row by row from ``q_now`` (damped least squares, previous row as seed). The joint chunk is
``[q1..q7, g]`` with ``g = (1 - clip(a[6], -1, 1)) / 2`` (normalised opening, 1 = open — the V11AB gripper
convention).

**Reverse (V11AB -> robot, ``joint_chunk_to_delta_chunk``).** With ``Dy_k`` = (refined chained EE step) −
(reference chained EE step) from FK (``Q*_{-1} = Qref_{-1} = q_now``), the commanded step that realises the
refined path under the same model differs by ``Dx_k = (Dy_k - alpha Dy_{k-1}) / beta``; row ``k`` becomes
``a'_k = a_k + Dx_k / scale`` (world frame; position and rotation separately). A row whose refined joints equal the reference bit for bit (float32, as
sent on the wire) **and** whose previous row did too is returned as the **original action, bit for bit**, so a
chunk V11AB leaves alone is executed exactly as VLA-only would. Clipping to ``[-1, 1]`` after the correction
is recorded per component (``saturated``). The gripper column is always the original.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Optional, Sequence

import numpy as np

#: robosuite 1.4.1 ``controllers/config/osc_pose.json``: output_max [0.05]*3 + [0.5]*3, input_max 1.
OSC_POS_SCALE = 0.05
OSC_ORI_SCALE = 0.5
OSC_INPUT_CLIP = 1.0
#: The EE frame OSC drives: robosuite ``gripper0_grip_site`` == adapter-URDF link ``panda_grip_site`` (SL2).
EE_LINK = "panda_grip_site"
#: Joint chunk layout: 7 arm joints then the normalised gripper opening.
ARM_DOF = 7
GRIPPER_COLUMN = 7
JOINT_CHUNK_WIDTH = 8

IK_MAX_ITERS = 60

import json as _json
import pathlib as _pathlib

OSC_RESPONSE_FIT = _pathlib.Path(__file__).resolve().parents[3] / "robots/panda/description/osc_response_fit.json"


def load_response_model(path=OSC_RESPONSE_FIT) -> dict:
    fit = _json.loads(_pathlib.Path(path).read_text())
    return {"pos": (float(fit["pos"]["alpha"]), float(fit["pos"]["beta"])),
            "rot": (float(fit["rot"]["alpha"]), float(fit["rot"]["beta"])), "path": str(path)}


RESPONSE = load_response_model()
IK_DAMPING = 1e-3
IK_POS_TOL_M = 1e-5
IK_ROT_TOL_RAD = 1e-4
IK_STEP_CLIP_RAD = 0.2


# --- rotations --------------------------------------------------------------------------------------
def rotvec_to_matrix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float64).reshape(3)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def matrix_to_rotvec(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, np.float64)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return 0.5 * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if np.pi - th < 1e-6:  # near pi: axis from the symmetric part
        M = (R + np.eye(3)) / 2.0
        i = int(np.argmax(np.diag(M)))
        axis = M[:, i] / np.sqrt(max(M[i, i], 1e-12))
        return axis / np.linalg.norm(axis) * th
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(th))
    return w * th


# --- kinematics ---------------------------------------------------------------------------------------
class PandaKinematics:
    """FK / IK of the adapter URDF's ``panda_grip_site`` in the robot base frame (``panda_link0``)."""

    def __init__(self, model: Any, ee_link: str = EE_LINK):
        self.model = model
        self.ee_link = ee_link
        lo, hi = model.joint_limits()
        self.lower = np.asarray(lo, np.float64)
        self.upper = np.asarray(hi, np.float64)
        if self.lower.shape != (ARM_DOF,):
            raise ValueError(f"expected a {ARM_DOF}-joint model, got joint limits of shape {self.lower.shape}")

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        T = np.asarray(self.model.link_pose(np.asarray(q, np.float64), self.ee_link), np.float64)
        return T[:3, 3].copy(), T[:3, :3].copy()

    def jacobian(self, q: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        """Geometric Jacobian ``(6, 7)`` (linear; angular as rotvec rate) by central differences."""
        q = np.asarray(q, np.float64)
        J = np.zeros((6, ARM_DOF))
        for j in range(ARM_DOF):
            dq = np.zeros(ARM_DOF)
            dq[j] = eps
            p1, R1 = self.fk(q + dq)
            p0, R0 = self.fk(q - dq)
            J[:3, j] = (p1 - p0) / (2 * eps)
            J[3:, j] = matrix_to_rotvec(R1 @ R0.T) / (2 * eps)
        return J

    def ik(self, p_tgt: np.ndarray, R_tgt: np.ndarray, q_seed: np.ndarray) -> tuple[np.ndarray, dict]:
        """Damped least squares from ``q_seed``. Returns ``(q, record)``; never raises on non-convergence."""
        q = np.clip(np.asarray(q_seed, np.float64).copy(), self.lower, self.upper)
        for it in range(IK_MAX_ITERS):
            p, R = self.fk(q)
            e = np.concatenate([p_tgt - p, matrix_to_rotvec(R_tgt @ R.T)])
            if np.linalg.norm(e[:3]) < IK_POS_TOL_M and np.linalg.norm(e[3:]) < IK_ROT_TOL_RAD:
                break
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + IK_DAMPING * np.eye(6), e)
            n = float(np.max(np.abs(dq)))
            if n > IK_STEP_CLIP_RAD:
                dq *= IK_STEP_CLIP_RAD / n
            q = np.clip(q + dq, self.lower, self.upper)
        p, R = self.fk(q)
        rec = {"iters": it + 1, "pos_err_m": float(np.linalg.norm(p_tgt - p)),
               "rot_err_rad": float(np.linalg.norm(matrix_to_rotvec(R_tgt @ R.T)))}
        return q, rec


# --- forward --------------------------------------------------------------------------------------------
@dataclasses.dataclass
class Conversion:
    """Everything the reverse step and the logs need for one chunk."""

    delta: np.ndarray            # (H, 7) the policy's chunk, as received
    q_now: np.ndarray            # (7,)
    R_world_base: np.ndarray     # (3, 3)
    joint_chunk: np.ndarray      # (H, 8) [q7, g] — Q_ref + normalised gripper
    ik: list                     # per-row IK record
    targets_base: list           # per-row (p, R) in the base frame (predicted with the response model)
    prev_step_world: tuple = (np.zeros(3), np.zeros(3))  # last realised (pos, rotvec) step, world frame

    def record(self) -> dict:
        pe = [r["pos_err_m"] for r in self.ik]
        re = [r["rot_err_rad"] for r in self.ik]
        return {"rows": int(self.delta.shape[0]), "ik_pos_err_m_max": float(max(pe) if pe else 0.0),
                "ik_rot_err_rad_max": float(max(re) if re else 0.0),
                "ik_iters_max": int(max((r["iters"] for r in self.ik), default=0))}


def gripper_norm_from_action(a6: np.ndarray) -> np.ndarray:
    return (1.0 - np.clip(np.asarray(a6, np.float64), -1.0, 1.0)) / 2.0


def delta_chunk_to_joint_chunk(kin: PandaKinematics, delta: np.ndarray, q_now: np.ndarray,
                               T_world_base: np.ndarray, prev_step_world=None, response: Optional[dict] = None
                               ) -> Conversion:
    """``prev_step_world = (dp, drotvec)``: the last realised EE step in the world frame (zeros at the start)."""
    A = np.asarray(delta)
    if A.ndim != 2 or A.shape[1] < 7:
        raise ValueError(f"expected an (H, 7) EE-delta chunk, got {A.shape}")
    resp = response or RESPONSE
    (ap, bp), (ar, br) = resp["pos"], resp["rot"]
    A64 = np.asarray(A[:, :7], np.float64)
    R_wb = np.asarray(T_world_base, np.float64)[:3, :3]
    R_bw = R_wb.T
    yp, yr = (np.zeros(3), np.zeros(3)) if prev_step_world is None else (
        np.asarray(prev_step_world[0], np.float64).reshape(3), np.asarray(prev_step_world[1], np.float64).reshape(3))
    prev = (yp.copy(), yr.copy())
    p, R = kin.fk(q_now)
    q = np.asarray(q_now, np.float64).copy()
    rows, iks, targets = [], [], []
    for k in range(A64.shape[0]):
        a = np.clip(A64[k], -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
        yp = ap * yp + bp * (OSC_POS_SCALE * a[:3])
        yr = ar * yr + br * (OSC_ORI_SCALE * a[3:6])
        p = p + R_bw @ yp
        R = rotvec_to_matrix(R_bw @ yr) @ R
        q, rec = kin.ik(p, R, q)
        rows.append(np.concatenate([q, [gripper_norm_from_action(A64[k, 6])]]))
        iks.append(rec)
        targets.append((p.copy(), R.copy()))
    return Conversion(delta=A, q_now=np.asarray(q_now, np.float64), R_world_base=R_wb,
                      joint_chunk=np.asarray(rows, np.float64), ik=iks, targets_base=targets, prev_step_world=prev)


def applied_command_rows(kin: PandaKinematics, conv: Conversion, applied: np.ndarray) -> np.ndarray:
    """``exec_feedback.applied_arm`` for Panda: the OSC commands **actually applied** (``(n, 7)`` EE deltas, HOLD
    rows included), expressed as joint command rows in the same convention as the chunk the server optimised.

    Same forward map and the same start state (``q_now`` · ``R_world_base`` · ``prev_step_world``) as ``conv`` —
    the conversion of the request whose chunk these rows came from. Rows equal to the policy's own deltas give
    the reference rows bit for bit; this is a command history, not the measured joint path.
    """
    a = np.asarray(applied, np.float64)
    if a.ndim != 2 or a.shape[1] < 7:
        raise ValueError(f"expected applied OSC rows (n, 7), got {a.shape}")
    if a.shape[0] == 0:
        return np.zeros((0, ARM_DOF), np.float64)
    T = np.eye(4)
    T[:3, :3] = conv.R_world_base
    rows = delta_chunk_to_joint_chunk(kin, a, conv.q_now, T, prev_step_world=conv.prev_step_world)
    return rows.joint_chunk[:, :ARM_DOF]


# --- reverse --------------------------------------------------------------------------------------------
#: Reverse-conversion modes (explicit; recorded per run).
#: ``A`` / ``lag_inverse``: realise the refined *path* under the response model, Dx = (Dy - alpha Dy_prev) / beta.
#: ``B`` / ``command_offset``: add the refined path change to the *commanded* step unchanged, Dx = Dy
#: (V11AB's change applied as a change of the commanded target, without inverting the controller lag).
CONVERSION_MODES = {"A": "lag_inverse", "B": "command_offset"}


def joint_chunk_to_delta_chunk(kin: PandaKinematics, conv: Conversion, refined: np.ndarray,
                               reference: np.ndarray, response: Optional[dict] = None,
                               mode: str = "A") -> tuple[np.ndarray, dict]:
    """``(actions (H, 7) in the policy's dtype, record)``. See the module docstring for the rule."""
    if mode not in CONVERSION_MODES:
        raise ValueError(f"conversion mode {mode!r} not in {sorted(CONVERSION_MODES)}")
    resp = response or RESPONSE
    (ap, bp), (ar, br) = resp["pos"], resp["rot"]
    Qs = np.asarray(refined)[:, :ARM_DOF]
    Qr = np.asarray(reference)[:, :ARM_DOF]
    if Qs.shape != Qr.shape:
        raise ValueError(f"refined {Qs.shape} and reference {Qr.shape} differ in shape")
    H = min(Qs.shape[0], conv.delta.shape[0])
    out = np.array(conv.delta, copy=True)
    unchanged = np.all(Qs[:H] == Qr[:H], axis=1)  # bit equality as sent on the wire (float32)
    R_wb = conv.R_world_base
    fk_s = [kin.fk(conv.q_now)] + [kin.fk(np.asarray(Qs[k], np.float64)) for k in range(H)]
    fk_r = [kin.fk(conv.q_now)] + [kin.fk(np.asarray(Qr[k], np.float64)) for k in range(H)]
    dyp_prev = np.zeros(3)
    dyr_prev = np.zeros(3)
    passthrough, corrected, sat_rows = [], [], []
    max_corr = 0.0
    for k in range(H):
        (ps0, Rs0), (ps1, Rs1) = fk_s[k], fk_s[k + 1]
        (pr0, Rr0), (pr1, Rr1) = fk_r[k], fk_r[k + 1]
        dyp = R_wb @ ((ps1 - ps0) - (pr1 - pr0))
        dyr = R_wb @ (matrix_to_rotvec(Rs1 @ Rs0.T) - matrix_to_rotvec(Rr1 @ Rr0.T))
        if unchanged[k] and (k == 0 or bool(unchanged[k - 1])):
            passthrough.append(k)
            dyp, dyr = np.zeros(3), np.zeros(3)  # identical chain: exactly no change
        else:
            if mode == "A":
                dxp = (dyp - ap * dyp_prev) / bp
                dxr = (dyr - ar * dyr_prev) / br
            else:
                dxp, dxr = dyp, dyr
            corr = np.concatenate([dxp / OSC_POS_SCALE, dxr / OSC_ORI_SCALE])
            raw = np.asarray(conv.delta[k, :6], np.float64) + corr
            clipped = np.clip(raw, -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
            if np.any(clipped != raw):
                sat_rows.append(k)
            out[k, :6] = clipped.astype(out.dtype)
            corrected.append(k)
            max_corr = max(max_corr, float(np.max(np.abs(corr))))
        dyp_prev, dyr_prev = dyp, dyr
    rec = {"mode": mode, "mode_name": CONVERSION_MODES[mode],
           "rows": int(H), "passthrough_rows": passthrough, "corrected_rows": corrected,
           "saturated_rows": sat_rows, "max_abs_correction": max_corr,
           "max_abs_joint_change_rad": float(np.max(np.abs(np.asarray(Qs[:H], np.float64)
                                                             - np.asarray(Qr[:H], np.float64)))) if H else 0.0}
    return out, rec


def predict_ee_path(kin: PandaKinematics, conv: Conversion, actions: np.ndarray, response: Optional[dict] = None):
    """EE positions (base frame, (H, 3)) the response model predicts for ``actions`` from the chunk's start state."""
    resp = response or RESPONSE
    (ap, bp), (ar, br) = resp["pos"], resp["rot"]
    R_bw = conv.R_world_base.T
    yp, yr = (np.asarray(conv.prev_step_world[0], np.float64).copy(), np.asarray(conv.prev_step_world[1], np.float64).copy())
    p, R = kin.fk(conv.q_now)
    out = []
    for k in range(np.asarray(actions).shape[0]):
        a = np.clip(np.asarray(actions[k, :6], np.float64), -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
        yp = ap * yp + bp * (OSC_POS_SCALE * a[:3])
        yr = ar * yr + br * (OSC_ORI_SCALE * a[3:6])
        p = p + R_bw @ yp
        R = rotvec_to_matrix(R_bw @ yr) @ R
        out.append(p.copy())
    return np.asarray(out)


def realisation_record(kin: PandaKinematics, conv: Conversion, actions: np.ndarray, refined: np.ndarray,
                       rows: int) -> dict:
    """How much of V11AB's path change the output actions realise (response model), over the first ``rows`` rows.

    ``p_ref`` = FK(reference rows), ``p_star`` = FK(refined rows), ``p_hat`` = path predicted for the output actions.
    """
    H = min(int(rows), np.asarray(actions).shape[0], np.asarray(refined).shape[0])
    p_hat = predict_ee_path(kin, conv, np.asarray(actions)[:H])
    p_ref = np.asarray([t[0] for t in conv.targets_base[:H]])
    p_star = np.asarray([kin.fk(np.asarray(refined[k, :ARM_DOF], np.float64))[0] for k in range(H)])
    want = np.linalg.norm(p_star - p_ref, axis=1)
    got = np.linalg.norm(p_hat - p_ref, axis=1)
    err = np.linalg.norm(p_hat - p_star, axis=1)
    sel = want > 1e-4
    return {"rows": H, "refined_change_mm_max": float(1000 * want.max()) if H else 0.0,
            "predicted_change_mm_max": float(1000 * got.max()) if H else 0.0,
            "predicted_vs_refined_mm_max": float(1000 * err.max()) if H else 0.0,
            "realised_fraction_median": float(np.median(got[sel] / want[sel])) if sel.any() else None,
            "realised_projection_median": float(np.median(np.einsum("ij,ij->i", p_hat[sel] - p_ref[sel], p_star[sel] - p_ref[sel])
                                                          / want[sel] ** 2)) if sel.any() else None,
            "p_hat_base": p_hat.tolist()}


def panda_chunk_layout(joint_names: Sequence[str]):
    """V11AB ``ChunkLayout`` for the joint chunk ``[q1..q7, g]``: 7 optimised joints, gripper passed through."""
    from AG3S.trajopt.types import ChunkLayout

    names = tuple(joint_names)
    if len(names) != ARM_DOF:
        raise ValueError(f"expected {ARM_DOF} arm joints, got {names}")
    return ChunkLayout(action_dim=JOINT_CHUNK_WIDTH, n_valid=JOINT_CHUNK_WIDTH,
                       action_to_q=tuple((i, i) for i in range(ARM_DOF)), fixed_q=(),
                       passthrough=(GRIPPER_COLUMN,), nq_model=ARM_DOF, joint_names=names)


def conventions() -> dict:
    return {"osc_pos_scale_m": OSC_POS_SCALE, "osc_ori_scale_rad": OSC_ORI_SCALE, "input_clip": OSC_INPUT_CLIP,
            "frame": "world (robosuite OSC); converted with T_world_base", "ee_link": EE_LINK,
            "response_model": {"pos": RESPONSE["pos"], "rot": RESPONSE["rot"], "fit": RESPONSE["path"],
                               "form": "y_t = alpha*y_{t-1} + beta*x_t (world)"},
            "forward": "predict the EE path with the response model from FK(q_now) and the last realised step; "
                       "DLS IK per row seeded by the previous row",
            "reverse": {"A (lag_inverse)": "a'_k = a_k + (Dy_k - alpha*Dy_{k-1})/beta/scale",
                        "B (command_offset)": "a'_k = a_k + Dy_k/scale",
                        "Dy": "refined minus reference chained FK step (world)",
                        "passthrough": "rows with refined == reference (float32, this and previous row) pass the original bits",
                        "clip": "[-1, 1] per component after the correction"},
            "gripper": "joint-chunk column 7 = (1 - clip(a6)) / 2 (1 = open); output gripper = original a6",
            "ik": {"max_iters": IK_MAX_ITERS, "damping": IK_DAMPING, "pos_tol_m": IK_POS_TOL_M,
                   "rot_tol_rad": IK_ROT_TOL_RAD, "step_clip_rad": IK_STEP_CLIP_RAD}}

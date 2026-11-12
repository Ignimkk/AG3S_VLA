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
is recorded per component (``saturated``). The gripper column is the original, except on a chunk where the geometric
grasp (G1) deferred the close (SL6-A A8, ``apply_deferred_close``).

Modes ``D`` / ``E`` (SL15, 2026-11-06, off unless ``--conversion D|E``): the steady-state inverse of the same lag,
``Dx_k = Dy_k (1 - alpha) / beta`` (gain 4.01 position / 4.03 rotation instead of C's dead-beat 1/beta = 14 / 32), row by row,
no cumulative tracking and no carry-over; an action that would leave ``[-1, 1]`` is scaled back along the correction
direction instead of clipped per axis. ``E`` adds a deadband (tiny refined changes keep the policy's bits).
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


def load_response_model(path=OSC_RESPONSE_FIT, variant: Optional[str] = None) -> dict:
    """``{"pos": (alpha, beta), "rot": (alpha, beta), "path"}`` in the convention of this module (``x = 0.05 a`` /
    ``0.5 a`` of the action ``a`` the server returns). ``variant=None``: the fit's own ``pos`` / ``rot`` (robosuite 1.4.1).
    ``variant="libero_safety_fork"`` (SL11): the fork fit's ``libero_scale`` block -- its fork-unit ``beta`` re-expressed for
    the LIBERO-scale action the client rescales with ``libero_to_safety`` (``fit_response.server_units``)."""
    fit = _json.loads(_pathlib.Path(path).read_text())
    if variant is None:
        return {"pos": (float(fit["pos"]["alpha"]), float(fit["pos"]["beta"])),
                "rot": (float(fit["rot"]["alpha"]), float(fit["rot"]["beta"])), "path": str(path)}
    if variant == "libero_safety_fork_native":
        # SL-LS1: the fit's own fork-unit blocks, beta re-expressed for the internal action a_int = a_fork / NATIVE_UNIT
        # (this module's model is y = alpha y + beta (OSC_*_SCALE a); the fork plant is y = alpha y + beta_fork a_fork)
        return {"pos": (float(fit["pos"]["alpha"]), float(fit["pos"]["beta"]) * NATIVE_UNIT[0] / OSC_POS_SCALE),
                "rot": (float(fit["rot"]["alpha"]), float(fit["rot"]["beta"]) * NATIVE_UNIT[1] / OSC_ORI_SCALE),
                "path": str(path), "variant": variant, "policy_unit": NATIVE_UNIT}
    if variant != "libero_safety_fork" or "libero_scale" not in fit:
        raise ValueError(f"{path}: no response variant {variant!r}")
    ls = fit["libero_scale"]
    return {"pos": (float(ls["pos"]["alpha"]), float(ls["pos"]["beta"])),
            "rot": (float(ls["rot"]["alpha"]), float(ls["rot"]["beta"])), "path": str(path), "variant": variant}


#: SL11 -- server option ``--response-model``. ``robosuite141`` (default) is the module's ``RESPONSE``.
RESPONSE_MODELS = ("robosuite141", "libero_safety_fork", "libero_safety_fork_native")
RESPONSE_FORK_FIT = _pathlib.Path(__file__).resolve().parents[3] / "robots/panda/description/osc_response_fit_libero_safety_fork.json"
#: The client request key that names the env's action adapter; the fork model needs ``libero_to_safety`` (see server).
ACTION_ADAPTER_KEY = "panda/action_adapter"
FORK_REQUIRED_ADAPTER = "libero_to_safety"
#: SL-LS1 (2026-11-12, user: "LIBERO-Safety 의 controller 변환") -- the official ``pi05_libero_safety`` checkpoint emits
#: fork-unit actions (the fork OSC realises ~1.0 m / rad per unit; the policy's steps are |a| <~ 0.023 m / 0.06 rad -- norm
#: stats q99 0.022 / 0.057). The server works in an internal action ``a_int = a_fork / NATIVE_UNIT``
#: (:func:`policy_to_internal`) and the LIBERO-Safety client multiplies the returned rows back (action adapter
#: ``fork_native_unit``). Powers of two, so a chunk the server does not change reaches the env bit for bit.
#: The unit also fixes what this module's +-1 input clip means: the fork env itself clips at +-1 m per step, i.e. never, so
#: the clip is the only bound on a corrected action. (1/32 m, 1/8 rad) puts it just above the policy's own largest step
#: (|a_int| <~ 0.75 / 0.5), so a saturated correction moves 31 mm / 0.125 rad per step (LIBERO: 12.5 mm / 0.124 rad).
#: The first closed-loop run used (1/16, 1/2): a saturated action-QP row moved the hand 62.5 mm per step per axis
#: (smoke 2026-11-12 23:50, TSA L2 task 0 call 15: predicted vs refined 157 mm) -- too large, changed before any table run.
NATIVE_UNIT = (0.03125, 0.125)
NATIVE_REQUIRED_ADAPTER = "fork_native_unit"
#: response model -> the client action adapter its numbers are expressed for (the server refuses any other client)
REQUIRED_ADAPTER = {"libero_safety_fork": FORK_REQUIRED_ADAPTER, "libero_safety_fork_native": NATIVE_REQUIRED_ADAPTER}


def response_for(name: Optional[str]) -> Optional[dict]:
    """``None`` for the default (callers then use ``RESPONSE``: bit-identical to before SL11), else the named model."""
    if name in (None, "robosuite141"):
        return None
    if name in ("libero_safety_fork", "libero_safety_fork_native"):
        return load_response_model(RESPONSE_FORK_FIT, variant=name)
    raise ValueError(f"response model {name!r} not in {RESPONSE_MODELS}")


def policy_to_internal(delta: np.ndarray, response: Optional[dict]) -> np.ndarray:
    """SL-LS1 -- the policy's chunk in the server's internal action unit. Only a response model with ``policy_unit`` (the
    native fork model) changes anything: arm columns divided by the unit (exact: powers of two); every other model returns
    ``delta`` itself (same object, the earlier behaviour)."""
    unit = None if response is None else response.get("policy_unit")
    if unit is None:
        return delta
    out = np.array(delta, copy=True)
    out[..., :3] = out[..., :3] / out.dtype.type(unit[0])
    out[..., 3:6] = out[..., 3:6] / out.dtype.type(unit[1])
    return out


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
    response: Optional[dict] = None  # SL11: this conversion's response model (None = ``RESPONSE``, robosuite 1.4.1)

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
                      joint_chunk=np.asarray(rows, np.float64), ik=iks, targets_base=targets, prev_step_world=prev,
                      response=response)


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
    rows = delta_chunk_to_joint_chunk(kin, a, conv.q_now, T, prev_step_world=conv.prev_step_world,
                                      response=conv.response)
    return rows.joint_chunk[:, :ARM_DOF]


# --- reverse --------------------------------------------------------------------------------------------
#: Reverse-conversion modes (explicit; recorded per run).
#: ``A`` / ``lag_inverse``: realise the refined *path* under the response model, Dx = (Dy - alpha Dy_prev) / beta.
#: ``B`` / ``command_offset``: add the refined path change to the *commanded* step unchanged, Dx = Dy
#: (V11AB's change applied as a change of the commanded target, without inverting the controller lag).
#: ``C`` / ``lag_inverse_tracking`` (V11AB-pb, 2026-10-14): A's inverse, but against the deviation the response model
#: predicts the commands *actually* produce (clipping included) and aimed at the refined path's cumulative position:
#: ``Dx_k = ((D*_k - D^_{k-1}) - alpha d^_{k-1}) / beta`` with ``d^_k = alpha d^_{k-1} + beta Dx_k(clipped)``,
#: ``D^_k = D^_{k-1} + d^_k``. A clipped row's deficit carries into the next rows, and once the refined rows rejoin the
#: reference the lag tail is stopped (rows stay corrected until ``d^`` and ``D* - D^`` are zero). Without clipping and
#: with every row corrected it equals A; a chunk without any change passes the policy's bits.
#: ``D`` / ``lag_steady`` (SL15): B's per-row change ``Dy_k`` (and B's ``unchanged`` pass-through rule), but the commanded
#: change is the *steady-state* inverse of the lag, ``Dx_k = Dy_k (1 - alpha) / beta`` (position and rotation separately):
#: a constant ``Dx`` then realises ``Dy`` once the lag has settled, within a few steps, with a gain of about 4 instead of
#: C's ~14 / ~32. The corrected action is ``a_pol + s corr`` with the largest ``s in [0, 1]`` that keeps every axis in
#: ``[-1, 1]`` (direction preserving; ``s < 1`` lists the row in ``saturated_rows``). No cumulative state, no carry-over.
#: ``E`` / ``lag_steady_deadband`` (SL15): D, except that a changed row passes the policy's bits when its own change is
#: below ``DEADBAND_POS_M`` and ``DEADBAND_ROT_RAD`` (norms) and the cumulative change up to and including that row is below
#: ``DEADBAND_CUM_POS_M`` and ``DEADBAND_CUM_ROT_RAD``. Those rows are listed in ``passthrough_rows`` and ``deadband_rows``.
CONVERSION_MODES = {"A": "lag_inverse", "B": "command_offset", "C": "lag_inverse_tracking",
                    "D": "lag_steady", "E": "lag_steady_deadband"}
#: E: deadband thresholds (own change per row / cumulative change from the chunk start), world-frame norms.
DEADBAND_POS_M = 0.0005
DEADBAND_ROT_RAD = float(np.radians(0.3))
DEADBAND_CUM_POS_M = 0.001
DEADBAND_CUM_ROT_RAD = float(np.radians(0.5))
#: C: a row is "back on the reference" when the predicted deviation state is below this (m / rad).
TRACKING_EPS = 1e-9


def joint_chunk_to_delta_chunk(kin: PandaKinematics, conv: Conversion, refined: np.ndarray,
                               reference: np.ndarray, response: Optional[dict] = None,
                               mode: str = "A") -> tuple[np.ndarray, dict]:
    """``(actions (H, 7) in the policy's dtype, record)``. See the module docstring for the rule."""
    if mode not in CONVERSION_MODES:
        raise ValueError(f"conversion mode {mode!r} not in {sorted(CONVERSION_MODES)}")
    resp = response or conv.response or RESPONSE
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
    # C: predicted actual deviation state (step d^, cumulative D^) and the refined cumulative target D*
    hat_dp, hat_dr, hat_Dp, hat_Dr = np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)
    star_Dp, star_Dr = np.zeros(3), np.zeros(3)
    passthrough, corrected, sat_rows = [], [], []
    max_corr = 0.0
    # D / E: cumulative change from the chunk start (deadband test of E), smallest direction-preserving scale used
    cum_dp, cum_dr = np.zeros(3), np.zeros(3)
    deadband, min_s = [], 1.0
    for k in range(H):
        (ps0, Rs0), (ps1, Rs1) = fk_s[k], fk_s[k + 1]
        (pr0, Rr0), (pr1, Rr1) = fk_r[k], fk_r[k + 1]
        dyp = R_wb @ ((ps1 - ps0) - (pr1 - pr0))
        dyr = R_wb @ (matrix_to_rotvec(Rs1 @ Rs0.T) - matrix_to_rotvec(Rr1 @ Rr0.T))
        if mode == "C":
            if unchanged[k] and (k == 0 or bool(unchanged[k - 1])):
                dyp, dyr = np.zeros(3), np.zeros(3)  # identical chain: exactly no change in the target
            star_Dp, star_Dr = star_Dp + dyp, star_Dr + dyr
            settled = (max(np.abs(hat_dp).max(), np.abs(hat_dr).max(), np.abs(star_Dp - hat_Dp).max(),
                           np.abs(star_Dr - hat_Dr).max()) <= TRACKING_EPS)
            if unchanged[k] and (k == 0 or bool(unchanged[k - 1])) and settled:
                passthrough.append(k)
                continue
            dxp = ((star_Dp - hat_Dp) - ap * hat_dp) / bp
            dxr = ((star_Dr - hat_Dr) - ar * hat_dr) / br
            corr = np.concatenate([dxp / OSC_POS_SCALE, dxr / OSC_ORI_SCALE])
            raw = np.asarray(conv.delta[k, :6], np.float64) + corr
            clipped = np.clip(raw, -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
            if np.any(clipped != raw):
                sat_rows.append(k)
            out[k, :6] = clipped.astype(out.dtype)
            corrected.append(k)
            max_corr = max(max_corr, float(np.max(np.abs(corr))))
            eff = np.asarray(out[k, :6], np.float64) - np.asarray(np.clip(conv.delta[k, :6], -OSC_INPUT_CLIP,
                                                                           OSC_INPUT_CLIP), np.float64)
            hat_dp = ap * hat_dp + bp * OSC_POS_SCALE * eff[:3]
            hat_dr = ar * hat_dr + br * OSC_ORI_SCALE * eff[3:]
            hat_Dp, hat_Dr = hat_Dp + hat_dp, hat_Dr + hat_dr
            continue
        if mode in ("D", "E"):
            if unchanged[k] and (k == 0 or bool(unchanged[k - 1])):
                passthrough.append(k)
                continue  # identical chain: exactly no change, the policy's bits
            cum_dp, cum_dr = cum_dp + dyp, cum_dr + dyr
            if (mode == "E" and np.linalg.norm(dyp) < DEADBAND_POS_M and np.linalg.norm(dyr) < DEADBAND_ROT_RAD
                    and np.linalg.norm(cum_dp) < DEADBAND_CUM_POS_M and np.linalg.norm(cum_dr) < DEADBAND_CUM_ROT_RAD):
                passthrough.append(k)
                deadband.append(k)
                continue
            dxp = dyp * (1.0 - ap) / bp
            dxr = dyr * (1.0 - ar) / br
            corr = np.concatenate([dxp / OSC_POS_SCALE, dxr / OSC_ORI_SCALE])
            a_pol = np.clip(np.asarray(conv.delta[k, :6], np.float64), -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
            s = 1.0
            for a_i, c_i in zip(a_pol, corr):  # largest s in [0, 1] with |a_i + s c_i| <= clip on every axis
                if c_i > 0.0:
                    s = min(s, (OSC_INPUT_CLIP - a_i) / c_i)
                elif c_i < 0.0:
                    s = min(s, (-OSC_INPUT_CLIP - a_i) / c_i)
            s = max(s, 0.0)
            if s < 1.0:
                sat_rows.append(k)
            min_s = min(min_s, s)
            out[k, :6] = np.clip(a_pol + s * corr, -OSC_INPUT_CLIP, OSC_INPUT_CLIP).astype(out.dtype)  # clip: ulp only
            corrected.append(k)
            max_corr = max(max_corr, float(np.max(np.abs(corr))))
            continue
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
    if mode == "C":
        rec_track = {"tracking_residual_mm": float(1000 * np.linalg.norm(star_Dp - hat_Dp)),
                     "tracking_residual_rad": float(np.linalg.norm(star_Dr - hat_Dr))}
    elif mode in ("D", "E"):
        rec_track = {"gain_pos": float((1.0 - ap) / bp), "gain_rot": float((1.0 - ar) / br), "min_scale_s": float(min_s),
                     **({"deadband_rows": deadband} if mode == "E" else {})}
    else:
        rec_track = {}
    rec = {"mode": mode, "mode_name": CONVERSION_MODES[mode], **rec_track,
           "rows": int(H), "passthrough_rows": passthrough, "corrected_rows": corrected,
           "saturated_rows": sat_rows, "max_abs_correction": max_corr,
           "max_abs_joint_change_rad": float(np.max(np.abs(np.asarray(Qs[:H], np.float64)
                                                             - np.asarray(Qr[:H], np.float64)))) if H else 0.0}
    return out, rec


def action_box_from_conversion(conv: Conversion, *, margin: float, n_rows: int, mode: str = "C",
                               response: Optional[dict] = None):
    """SL10 §2.2 -- the TrajOpt action-limit rows of this chunk (``AG3S.trajopt.action_box.ActionBox``).

    Every constant is this module's own (never copied): the response model ``RESPONSE`` (fit json), the OSC scales and the
    input clip. Reverse modes A and C invert the lag, ``Dx = (Dy - alpha Dy_prev) / beta``; mode B adds ``Dy`` unchanged
    (``alpha = 0``, ``beta = 1``). Modes D and E (SL15) apply the steady gain ``Dx = Dy (1 - alpha) / beta`` row by row, which
    in the box's form ``g (dy_k - alpha_box dy_{k-1})`` is ``alpha_box = 0`` and ``beta_box = beta / (1 - alpha)`` (B's
    structure with D's gain). The box keeps its per-axis ``|a + corr| <= 1 - margin`` rows: that is sufficient for D's
    direction-preserving scale to be 1, so the box needs no scale-back. E's deadband is not modelled: a deadband row passes
    the policy's bits, which is inside the box whenever the policy action is.
    """
    from AG3S.trajopt.action_box import ActionBox

    if mode not in CONVERSION_MODES:
        raise ValueError(f"conversion mode {mode!r} not in {sorted(CONVERSION_MODES)}")
    resp = response or conv.response or RESPONSE
    (ap, bp), (ar, br) = resp["pos"], resp["rot"]
    if mode == "B":
        ap = ar = 0.0
        bp = br = 1.0
    elif mode in ("D", "E"):
        bp, br = bp / (1.0 - ap), br / (1.0 - ar)
        ap = ar = 0.0
    A = np.asarray(conv.delta, np.float64)[:, :6]
    return ActionBox(link=EE_LINK, actions=np.clip(A, -OSC_INPUT_CLIP, OSC_INPUT_CLIP),
                     R_world_base=np.asarray(conv.R_world_base, np.float64), q_now=np.asarray(conv.q_now, np.float64),
                     alpha=np.array([ap] * 3 + [ar] * 3), beta=np.array([bp] * 3 + [br] * 3),
                     scale=np.array([OSC_POS_SCALE] * 3 + [OSC_ORI_SCALE] * 3), margin=float(margin),
                     n_rows=int(n_rows), input_clip=OSC_INPUT_CLIP, mode=mode)


def apply_deferred_close(actions: np.ndarray, refined: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, dict]:
    """SL6-A A8 — the geometric grasp (G1, ``geo_grasp.close_gate``) deferred the policy's close on this chunk: the
    refined joint chunk keeps the gripper column (7) open from the close row, but the reverse conversion sends the
    policy's gripper value (``a[6]``) unchanged -- the robot would close anyway. Here every row whose refined gripper
    value differs from the reference gets ``a[6] = clip(1 - 2 g)`` (the inverse of ``g = (1 - clip(a6)) / 2``); the
    other rows keep their bits. Called only on a deferred chunk (G1 off: never)."""
    out = np.array(actions, copy=True)
    g_ref = np.asarray(reference, np.float64)[:, GRIPPER_COLUMN]
    g_new = np.asarray(refined, np.float64)[:, GRIPPER_COLUMN]
    H = min(out.shape[0], g_ref.shape[0], g_new.shape[0])
    rows = [k for k in range(H) if np.float32(g_new[k]) != np.float32(g_ref[k])]
    for k in rows:
        out[k, 6] = np.clip(1.0 - 2.0 * g_new[k], -OSC_INPUT_CLIP, OSC_INPUT_CLIP).astype(out.dtype)
    return out, {"rows": rows, "a6_policy": [float(np.asarray(actions)[k, 6]) for k in rows],
                 "a6_sent": [float(out[k, 6]) for k in rows]}


def predict_ee_path(kin: PandaKinematics, conv: Conversion, actions: np.ndarray, response: Optional[dict] = None):
    """EE positions (base frame, (H, 3)) the response model predicts for ``actions`` from the chunk's start state."""
    resp = response or conv.response or RESPONSE
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


# --- SL-GTTO9 (F7, lead self-ruling G47, 2026-11-05) — the action-space map for the EE-delta QP ---------------------
#: Damping of the Jacobian pseudo-inverse in :func:`ee_action_map` (same order as the IK's ``IK_DAMPING``).
EE_MAP_DAMPING = 1e-4


def ee_action_map(kin: PandaKinematics, conv: Conversion, horizon: int, response: Optional[dict] = None) -> dict:
    """Linear map from a change of the policy's OSC actions to the change of the joint chunk the forward conversion
    predicts, about the reference chunk ``conv.joint_chunk``.

    With the response model (``y_t = alpha y_{t-1} + beta s x_t`` per position / rotation, world frame) a change
    ``Da_j`` of row ``j`` changes the chained EE pose at row ``k >= j`` by ``beta s (1 - alpha^(k-j+1)) / (1 - alpha)
    Da_j``; in the base frame through ``R_bw`` and to joints through the damped pseudo-inverse of the EE Jacobian at
    the reference row ``k``: ``Dq_k = J_k^+ R_bw DP_k``. Returns ``{"M": (H*7, H*6), "a_ref": (H, 6)}`` with ``Dq``
    stacked step-major (``k*7 + j``) and ``Da`` step-major (``k*6 + [pos 3, rot 3]``); ``a_ref`` is the policy's
    rows clipped to the OSC input range. First order: the safety check of the result is done on the real geometry.
    """
    resp = response or RESPONSE
    (ap, bp), (ar, br) = resp["pos"], resp["rot"]
    H = int(horizon)
    if conv.delta.shape[0] < H or conv.joint_chunk.shape[0] < H:
        raise ValueError(f"the conversion has {conv.delta.shape[0]} rows, the TO horizon is {H}")
    R_bw = conv.R_world_base.T
    Rb = np.zeros((6, 6))
    Rb[:3, :3] = R_bw
    Rb[3:, 3:] = R_bw
    M = np.zeros((H * ARM_DOF, H * 6))
    for k in range(H):
        J = kin.jacobian(np.asarray(conv.joint_chunk[k, :ARM_DOF], np.float64))
        Jp = J.T @ np.linalg.inv(J @ J.T + EE_MAP_DAMPING * np.eye(6))   # (7, 6)
        A = Jp @ Rb
        for j in range(k + 1):
            n = k - j + 1
            cp = bp * OSC_POS_SCALE * (1.0 - ap ** n) / (1.0 - ap)
            cr = br * OSC_ORI_SCALE * (1.0 - ar ** n) / (1.0 - ar)
            M[k * ARM_DOF:(k + 1) * ARM_DOF, j * 6:j * 6 + 3] = A[:, :3] * cp
            M[k * ARM_DOF:(k + 1) * ARM_DOF, j * 6 + 3:j * 6 + 6] = A[:, 3:] * cr
    a_ref = np.clip(np.asarray(conv.delta[:H, :6], np.float64), -OSC_INPUT_CLIP, OSC_INPUT_CLIP)
    # SL-SOTA2: "R_world_base" lets a reader turn a base-frame direction into the world-frame position action
    return {"M": M, "a_ref": a_ref, "R_world_base": np.asarray(conv.R_world_base, np.float64).copy()}

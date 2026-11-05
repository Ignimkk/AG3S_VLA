"""Fit the OSC first-order response ``y_t = alpha y_{t-1} + beta x_t`` of a Panda env from recorded rollouts (SL11).

Same method as ``outputs/verify/SL5a/scripts/fit_osc_response.py`` (robosuite 1.4.1 / LIBERO, ``osc_response_fit.json``):
per group (position, rotation) one ``(alpha, beta)`` shared by the three components, ``x`` the commanded world-frame
step, ``y`` the realised world-frame EE step (position difference; rotation vector of ``R_{t+1} R_t^-1``), wait steps
skipped and ``y_{t-1}`` taken only from the previous non-wait step.

LIBERO-Safety fork (``benchmark/LIBERO-Safety/third_party/robosuite-1.4``: OSC kp 750, output +-2). What the VLA-only
episode npz of ``benchmark/runners/libero_safety_client.py`` hold: ``action`` is the **policy's action (LIBERO scale,
before ``adapt_action``)**; ``env_action`` is not saved. With ``action_adapter = libero_to_safety`` (recorded in the
run manifest) the fork received ``x = g * action`` with ``g_pos = 0.05 * 0.2494 / 1.0`` and ``g_rot = 0.5 * 0.2484 / 1.0``
(``libero_safety_client.REALISED_GAIN``; ``GAINS_LIBERO_TO_SAFETY`` below, checked against the client by the test). The fit is
done in the fork's own action units (``x`` = ``g * action``) so ``steady_state_gain`` is the fork's measured
gain (~1, ``outputs/verify/BM-SETUP-B/controller/fork_osc_gain.json``), and :func:`server_units` re-expresses ``beta`` in the
server's convention (``x_server = 0.05 a`` / ``0.5 a`` of the LIBERO-scale action ``a`` the server returns), which is what
``action.py`` needs: ``beta_server = beta * g / scale``.

Usage (CPU, numpy + scipy)::

    python -m AG3S.adapters.panda.fit_response fit --out robots/panda/description/osc_response_fit_libero_safety_fork.json
    python -m AG3S.adapters.panda.fit_response gate --fork robots/panda/description/osc_response_fit_libero_safety_fork.json
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

import numpy as np

STORAGE = pathlib.Path("/mnt/dev/AG3S_VLA-storage/results/benchmark")
LS_VLA_ONLY_GLOB = "*sl7-liberosafety-vla30-*vla_only"
LIBERO_VLA_ONLY_GLOB = "*sl5f-libero10-init0-vla_only"
SERVER_SCALE = {"pos": 0.05, "rot": 0.5}
#: ``libero_to_safety``: fork action = LIBERO-scale action x these (``REALISED_GAIN[1.4.1] / REALISED_GAIN[fork]``).
GAINS_LIBERO_TO_SAFETY = {"pos": 0.05 * 0.2494372096817311 / 1.0, "rot": 0.5 * 0.2484488995069122 / 1.0}


def episode_files(run_glob: str, root: pathlib.Path = STORAGE) -> list[str]:
    return sorted(f for d in sorted(glob.glob(str(root / run_glob))) for f in glob.glob(d + "/episodes/*.npz")
                  if not f.endswith(("_geom.npz", "_calls.npz")))


def _step(group, p, q, t):
    from scipy.spatial.transform import Rotation as Rot

    if group == "pos":
        return p[t + 1] - p[t]
    return (Rot.from_quat(q[t + 1]) * Rot.from_quat(q[t]).inv()).as_rotvec()


def collect(files, group: str, gain: float):
    """Per-step regression data: ``(y_prev, x, y)`` each ``(N, 3)`` with ``x = gain * clip(a)`` (fork units)."""
    sl = slice(0, 3) if group == "pos" else slice(3, 6)
    P, X, Y = [], [], []
    for f in files:
        e = np.load(f)
        p, q, a, w = e["obs_robot0_eef_pos"], e["obs_robot0_eef_quat"], e["action"], e["is_wait"].astype(bool)
        prev = None
        for t in range(len(a)):
            if w[t]:
                prev = None
                continue
            y = _step(group, p, q, t)
            if prev is not None:
                P.append(prev)
                X.append(gain * np.clip(a[t, sl], -1, 1))
                Y.append(y)
            prev = y
    return np.array(P), np.array(X), np.array(Y)


def _lstsq(P, X, Y, keep):
    A = np.c_[P[keep].T.ravel(), X[keep].T.ravel()]
    b = Y[keep].T.ravel()
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef


def _r2_rmse(P, X, Y, coef, keep):
    b = Y[keep].T.ravel()
    res = b - (coef[0] * P[keep].T.ravel() + coef[1] * X[keep].T.ravel())
    return float(1 - (res ** 2).sum() / ((b - b.mean()) ** 2).sum()), float(np.sqrt((res ** 2).mean()))


TRIM_SIGMA = 4.0
TRIM_ITERS = 6
TRIM_FLOOR = 1e-9   # m / rad: noiseless data must not be trimmed on floating-point residuals


def fit_group(files, group: str, gain: float, trim: bool = True) -> dict:
    """Least squares of ``y_t = alpha y_{t-1} + beta x_t`` over all non-wait steps.

    ``trim=True`` (default): iteratively drop steps whose largest component residual exceeds ``TRIM_SIGMA`` robust sigmas
    (1.4826 x median of the retained steps' residual). On LIBERO-Safety the ~4 % dropped steps (position) are not
    controller response -- contact / release events where the realised step differs from the command by 2x or more
    (mean |y| 10 mm vs 7 mm commanded) -- and they dominate the plain-OLS R^2. Both are recorded: ``r2`` (the retained
    steps, what the offline gate uses), ``r2_all_steps`` (the same coefficients on every step) and ``ols`` (plain
    least squares on every step)."""
    P, X, Y = collect(files, group, gain)
    allk = np.ones(len(Y), bool)
    ols = _lstsq(P, X, Y, allk)
    keep = allk.copy()
    coef = ols
    if trim:
        for _ in range(TRIM_ITERS):
            coef = _lstsq(P, X, Y, keep)
            res = np.abs(Y - (coef[0] * P + coef[1] * X)).max(axis=1)
            keep = res < max(TRIM_SIGMA * 1.4826 * np.median(res[keep]), TRIM_FLOOR)
        coef = _lstsq(P, X, Y, keep)
    r2, rmse = _r2_rmse(P, X, Y, coef, keep)
    r2_all, rmse_all = _r2_rmse(P, X, Y, coef, allk)
    r2_ols, _ = _r2_rmse(P, X, Y, ols, allk)
    return {"alpha": float(coef[0]), "beta": float(coef[1]), "r2": r2, "n": int(keep.sum()) * 3, "rmse": rmse,
            "steady_state_gain": float(coef[1] / (1 - coef[0])),
            "trim": {"on": bool(trim), "sigma": TRIM_SIGMA, "iters": TRIM_ITERS, "steps_total": int(len(Y)),
                     "steps_kept": int(keep.sum()), "dropped_fraction": float(1 - keep.mean()),
                     "r2_all_steps": r2_all, "rmse_all_steps": rmse_all,
                     "ols": {"alpha": float(ols[0]), "beta": float(ols[1]), "r2": r2_ols,
                             "steady_state_gain": float(ols[1] / (1 - ols[0]))}}}


def server_units(group_fit: dict, group: str, gains=GAINS_LIBERO_TO_SAFETY) -> dict:
    """``(alpha, beta)`` for ``x_server = scale * a`` (the convention of ``action.py``) of the LIBERO-scale action ``a``."""
    return {"alpha": group_fit["alpha"], "beta": group_fit["beta"] * gains[group] / SERVER_SCALE[group]}


def fit(files, gains=GAINS_LIBERO_TO_SAFETY) -> dict:
    out = {"model": "y_t = alpha*y_{t-1} + beta*x_t (world frame, per component, shared coefficients); x = fork env action "
                    "= gains * LIBERO-scale policy action (libero_to_safety), y = realised EE step",
           "response_model": "libero_safety_fork",
           "data": {"runs_glob": LS_VLA_ONLY_GLOB, "runs": sorted({str(pathlib.Path(f).parents[1].name) for f in files}),
                    "episodes": len(files),
                    "recorded_action": "policy action (LIBERO scale, before adapt_action); env_action = gains * action "
                                       "(action_adapter libero_to_safety, run manifest)"}}
    for g in ("pos", "rot"):
        out[g] = fit_group(files, g, gains[g])
    out["libero_scale"] = {
        "gains_libero_to_safety": dict(gains),
        "pos": server_units(out["pos"], "pos", gains), "rot": server_units(out["rot"], "rot", gains),
        "meaning": "(alpha, beta) in the convention of action.py: x = 0.05*a (pos) / 0.5*a (rot) of the LIBERO-scale action "
                   "a the server returns; beta = beta_fork * gains / scale. The client multiplies a by gains "
                   "(libero_to_safety) before the fork sees it."}
    out["provenance"] = {
        "script": "AG3S/adapters/panda/fit_response.py (fit)", "date": "2026-11-05",
        "data_split": "LIBERO-Safety L0 VLA-only pi05_libero (3 categories x 10 episodes, sl7-liberosafety-vla30)",
        "controller": "LIBERO-Safety robosuite-1.4 fork OSC_POSE (kp 750, output_max +-2), control_freq 20 Hz",
        "use": "AG3S/adapters/panda/action.py via server --response-model libero_safety_fork (conversion C forward + inverse)"}
    return out


# --- offline gate -------------------------------------------------------------------------------------------------
def chunk_prediction_errors(files, response: dict, gains=None, rows: int = 5):
    """Per chunk start ``t``: |EE(t+rows) predicted by the model - actual| (mm) and the rotation analogue (rad).

    The prediction uses the policy chunk ``action_chunks[i][:rows]`` and the realised step ``y_{-1} = p_t - p_{t-1}``
    exactly like the conversion's forward model (``action.delta_chunk_to_joint_chunk``): ``response`` is in the
    server convention (``x = 0.05 a`` / ``0.5 a``). The model is applied to the LIBERO-scale action; ``gains`` is
    not needed (it lives in ``response["pos"]`` beta)."""
    from scipy.spatial.transform import Rotation as Rot

    (ap, bp), (ar, br) = response["pos"], response["rot"]
    epos, erot = [], []
    for f in files:
        e = np.load(f)
        p, q, C, ct = e["obs_robot0_eef_pos"], e["obs_robot0_eef_quat"], e["action_chunks"], e["chunk_t"]
        for i, t in enumerate(ct):
            t = int(t)
            if t < 1 or t + rows >= len(p):
                continue
            yp = p[t] - p[t - 1]
            yr = (Rot.from_quat(q[t]) * Rot.from_quat(q[t - 1]).inv()).as_rotvec()
            pp, Rr = p[t].copy(), Rot.from_quat(q[t])
            for k in range(rows):
                a = np.clip(C[i, k, :6], -1, 1)
                yp = ap * yp + bp * 0.05 * a[:3]
                yr = ar * yr + br * 0.5 * a[3:6]
                pp = pp + yp
                Rr = Rot.from_rotvec(yr) * Rr
            epos.append(1000 * float(np.linalg.norm(pp - p[t + rows])))
            erot.append(float(np.linalg.norm((Rr * Rot.from_quat(q[t + rows]).inv()).as_rotvec())))
    return np.array(epos), np.array(erot)


def _summ(x):
    return {"n": int(len(x)), "median": float(np.median(x)), "p90": float(np.percentile(x, 90)), "mean": float(np.mean(x))}


def gate(fork_json: pathlib.Path, rows: int = 5) -> dict:
    from AG3S.adapters.panda import action as pa

    fork = json.loads(pathlib.Path(fork_json).read_text())
    ls_files = episode_files(LS_VLA_ONLY_GLOB)
    libero_files = episode_files(LIBERO_VLA_ONLY_GLOB)
    r141 = pa.load_response_model()
    rfork = pa.load_response_model(fork_json, variant="libero_safety_fork")
    out = {"rows": rows, "r2": {"pos": fork["pos"]["r2"], "rot": fork["rot"]["r2"]},
           "ls_episodes": len(ls_files), "libero_episodes": len(libero_files)}
    for name, files, model in (("ls_robosuite141_current", ls_files, r141), ("ls_fork", ls_files, rfork),
                               ("libero_robosuite141", libero_files, r141)):
        ep, er = chunk_prediction_errors(files, model, rows=rows)
        out[name] = {"pos_mm": _summ(ep), "rot_rad": _summ(er)}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--out", required=True)
    f.add_argument("--runs-glob", default=LS_VLA_ONLY_GLOB)
    g = sub.add_parser("gate")
    g.add_argument("--fork", required=True)
    g.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "fit":
        res = fit(episode_files(args.runs_glob))
        pathlib.Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
    else:
        res = gate(args.fork)
        if args.json_out:
            pathlib.Path(args.json_out).write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""SL6-A — the Panda hand geometry the AG3S final configuration needs: palm shell spheres (A5), the grasp-moment site
(A6) and the parallel-gripper spec of the geometric grasp (A8, ``AG3S.ag3s.grasp.gripper.PANDA_LEFT``).

**A5 palm shell (fixed rule, task SL6 §2 A5).** RB-Y1's rule (``gripper_cover.palm_shell_spheres``: cover the hold
link's collision surface except the grasp face between the open fingers, keep out of the space under it, 5 mm
overflow) applied to the Panda hand's collision geometry -- the robosuite ``PandaGripper`` hand mesh
(``gripper0_hand_collision`` on body ``gripper0_right_gripper``, rigid on URDF ``panda_hand``). Nothing of the
rule changes; what differs is only how the surface is read:

* the MuJoCo model is rebuilt from the LIBERO geometry export (``cover.build_mujoco_model``) with
  ``collidable=True`` (the export model marks every geom ``contype = 0``, which the V11AB sampler
  ``gripper_cover._geom_pieces`` skips -- only this build reads them) and ``joints=True`` (so the rigid group of the
  hand is the hand, ``panda_link7`` and the eef frame, not the whole robot: the hidden-face rule then flags exactly the
  hand's top face under the link-7 flange, as RB-Y1 flags the face under its FT sensor);
* the URDF link ``panda_hand`` has no MuJoCo body of its own geometry (``robot0_right_hand`` carries none), so the
  surface is the one of ``gripper0_right_gripper`` expressed in the ``panda_hand`` frame (``static_urdf_ancestor``);
* the grasp channel (``gripper_cover._palm_channel``) comes from the Panda constraint model (franka.yml finger
  spheres, ``panda_finger_joint1/2`` open at 0.04 m).

The shell is fitted once (``python -m AG3S.adapters.panda.hand_geometry --geometry <export.npz> --out <json>``,
~10 s) and committed as :data:`PALM_SHELL_JSON` with the export's digests -- the server does not need the export
(``outputs/`` is not in git). ``tests/benchmark/test_sl6a_panda_final.py`` refits it from the export when the export
is present and checks the file bit for bit.

**A6 site.** ``panda_grip_site`` = ``panda_hand`` origin + (0, 0, 0.097) m (``panda_libero.urdf`` joint
``panda_grip_site_joint``; robosuite ``gripper0_grip_site``), the point between the pads. RB-Y1's site is the hold
link + (0, 0, -0.100) m (``left_ee``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
from typing import Any, Optional

import numpy as np

from AG3S.adapters.panda import cover as pc
from AG3S.adapters.panda import model as pm

#: Committed palm shell (generated; provenance inside).
PALM_SHELL_JSON = pathlib.Path(__file__).resolve().with_name("panda_palm_shell.json")
#: The hold link the shell hangs on (= ``model.PANDA_PALM_LINKS``).
PALM_LINK = "panda_hand"
#: MuJoCo body that carries the hand collision mesh (robosuite ``PandaGripper`` root).
HAND_MJCF_BODY = pc.HAND_BODY
#: A6 — the grasp-moment site in the ``panda_hand`` frame (``panda_grip_site``).
PANDA_HOLD_SITE_OFFSET_M: tuple[float, float, float] = (0.0, 0.0, 0.097)
#: ``--palm-shell`` hands -> Panda palm links (the Panda is the single ``left`` manipulator).
PALM_SHELL_HANDS = {"both": (PALM_LINK,), "left": (PALM_LINK,)}


def collision_model(tables: dict):
    """The export's collision geometry as a MuJoCo model the V11AB surface sampler reads (module docstring)."""
    model, _ = pc.build_mujoco_model(tables, ("collision",), collidable=True, joints=True)
    return model


def hand_surface(model, robot_model, link: str, *, spacing: float):
    """`gripper_cover.LinkSurface` of the hand collision mesh in the ``panda_hand`` frame (no grasp face marked here:
    the palm-shell rule marks its own)."""
    from AG3S.ag3s.experiments.sources.gripper_cover import LinkSurface, body_collision_surface
    from AG3S.ag3s.experiments.sources.mujoco_source import static_urdf_ancestor

    found, T = static_urdf_ancestor(model, HAND_MJCF_BODY, robot_model.model.links)
    if found != link:
        raise ValueError(f"{HAND_MJCF_BODY!r} hangs off URDF link {found!r}, not {link!r}")
    pts, nrm, verts, hulls, hidden = body_collision_surface(model, HAND_MJCF_BODY, T, spacing)
    return LinkSurface(link=link, body=HAND_MJCF_BODY, points=pts, normals=nrm, grasp_face=np.zeros(pts.shape[0], bool),
                       vertices=verts, hidden=hidden, hulls=hulls)


def fit_palm_shell(tables: dict, robot_model=None) -> tuple[list, dict]:
    """``(capsules, report)`` — RB-Y1's palm-shell rule (``gripper_cover`` constants, 5 mm overflow) on the Panda hand.
    ``robot_model`` = the shell-less Panda constraint model (default: palm + fingers)."""
    from AG3S.ag3s.experiments.sources import gripper_cover as gc
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE

    robot_model = robot_model or pm.build_panda_constraint_model(pm.PANDA_GRIPPER_LINKS)
    model = collision_model(tables)
    return gc._body_capsules(
        model, robot_model, [PALM_LINK], role=PALM_SHELL_ROLE, spacing=gc.PALM_COVER_SPACING,
        key=("panda_palm_shell", float(gc.PALM_SHELL_OVERFLOW), float(gc.PALM_COVER_SPACING), float(gc.PALM_COVER_GRID),
             float(gc.PALM_COVER_UNION_EPS), float(gc.PALM_SHELL_FACE_COS)),
        solve=lambda surf, link: gc.palm_shell_spheres(surf, robot_model, link, overflow=gc.PALM_SHELL_OVERFLOW,
                                                       grid=gc.PALM_COVER_GRID, union_eps=gc.PALM_COVER_UNION_EPS,
                                                       face_cos=gc.PALM_SHELL_FACE_COS),
        surface_fn=hand_surface)


def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    return v


def shell_rows(capsules) -> list[dict]:
    return [{"link": c.link, "centre": [float(x) for x in np.asarray(c.origin, np.float64)[:3, 3]],
             "radius": float(c.radius)} for c in capsules]


def shell_digest(capsules) -> str:
    return hashlib.sha256(json.dumps(shell_rows(capsules), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_palm_shell_json(path, capsules, report: dict, tables: dict, geometry_path) -> dict:
    from AG3S.ag3s.experiments.sources import gripper_cover as gc

    source = json.loads(str(tables["provenance_json"])) if "provenance_json" in tables else {}
    document = {
        "_comment": "Generated by AG3S/adapters/panda/hand_geometry.py (SL6-A A5). Palm shell spheres of the LIBERO "
                    "Panda hand, panda_hand frame of robots/panda/description/panda_libero.urdf. Do not edit by hand.",
        "provenance": {
            "rule": "AG3S.ag3s.experiments.sources.gripper_cover.palm_shell_spheres (RB-Y1 SC rule, unchanged)",
            "constants": {"overflow": gc.PALM_SHELL_OVERFLOW, "spacing": gc.PALM_COVER_SPACING,
                          "grid": gc.PALM_COVER_GRID, "union_eps": gc.PALM_COVER_UNION_EPS,
                          "face_cos": gc.PALM_SHELL_FACE_COS},
            "mujoco_body": HAND_MJCF_BODY, "link": PALM_LINK,
            "geometry_file": pathlib.Path(geometry_path).name,
            "geometry_digest": str(tables.get("geometry_digest", "")),
            "kinematic_digest": str(tables.get("kinematic_digest", "")),
            "geometry_export": source,
            "urdf_sha256": pm.file_sha256(pm.PANDA_URDF),
            "franka_yml_sha256": pm.file_sha256(pm.CUROBO_FRANKA_YML),
            "shell_digest": shell_digest(capsules),
        },
        "report": _jsonable(report),
        "spheres": shell_rows(capsules),
    }
    pathlib.Path(path).write_text(json.dumps(document, indent=1) + "\n")
    return document


def load_palm_shell(path=PALM_SHELL_JSON) -> tuple[list, dict]:
    """``(capsules (role palm_shell, file order), document)``; the digest must match the spheres."""
    from AG3S.ag3s.robot_models.urdf_sphere_chain import PALM_SHELL_ROLE, UrdfCapsule

    document = json.loads(pathlib.Path(path).read_text())
    capsules = []
    for row in document["spheres"]:
        origin = np.eye(4)
        origin[:3, 3] = np.asarray(row["centre"], np.float64)
        capsules.append(UrdfCapsule(link=str(row["link"]), origin=origin, radius=float(row["radius"]), length=0.0,
                                    role=PALM_SHELL_ROLE))
    if shell_digest(capsules) != document["provenance"]["shell_digest"]:
        raise ValueError(f"{path}: the spheres do not match the recorded shell digest")
    return capsules, document


def palm_shell_table(capsules) -> dict[str, Any]:
    """Count / radius / centre ranges (``panda_hand`` frame, mm) — the impl.md table and the server config record."""
    if not capsules:
        return {"n_spheres": 0}
    C = np.asarray([np.asarray(c.origin, np.float64)[:3, 3] for c in capsules]) * 1e3
    R = np.asarray([float(c.radius) for c in capsules]) * 1e3
    return {"n_spheres": int(len(capsules)), "link": PALM_LINK,
            "radius_mm": {"min": round(float(R.min()), 2), "median": round(float(np.median(R)), 2),
                          "max": round(float(R.max()), 2)},
            "centre_mm": {ax: [round(float(C[:, i].min()), 1), round(float(C[:, i].max()), 1)]
                          for i, ax in enumerate("xyz")},
            "extent_mm": {ax: [round(float((C[:, i] - R).min()), 1), round(float((C[:, i] + R).max()), 1)]
                          for i, ax in enumerate("xyz")}}


# --- A8 gripper spec ------------------------------------------------------------------------------------------
def measure_gripper_spec(tables: dict) -> dict[str, Any]:
    """The Panda numbers of ``gripper.PANDA_LEFT`` from the export, in the **gripper frame** (``x_G = y_H``,
    ``y_G = x_H``, ``z_G = -z_H``), m. Pads = pad boxes + finger-mesh inner face (z_H >= the face start), measured with
    the fingers at the open stop for ``max_opening`` and closed for the rest."""
    import mujoco
    from scipy.spatial import ConvexHull

    m = collision_model(tables)
    d = mujoco.MjData(m)
    hb = m.body(PALM_LINK).id

    def verts(geom: str, q: tuple[float, float]) -> np.ndarray:
        d.qpos[:] = 0.0
        d.qpos[m.joint("gripper0_finger_joint1").qposadr[0]] = q[0]
        d.qpos[m.joint("gripper0_finger_joint2").qposadr[0]] = q[1]
        mujoco.mj_kinematics(m, d)
        R, p0 = d.xmat[hb].reshape(3, 3), d.xpos[hb]
        g = m.geom(geom).id
        mid = int(m.geom_dataid[g])
        a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        V = (d.geom_xmat[g].reshape(3, 3) @ m.mesh_vert[a:a + n].T).T + d.geom_xpos[g]
        return (R.T @ (V - p0).T).T

    closed, opened = (0.0, 0.0), (0.04, -0.04)
    pad1 = verts("gripper0_finger1_pad_collision", closed)
    fin1 = verts("gripper0_finger1_collision", closed)
    pad1_open = verts("gripper0_finger1_pad_collision", opened)
    pad2_open = verts("gripper0_finger2_pad_collision", opened)
    hand = verts("gripper0_hand_collision", closed)
    inner = float(pad1[:, 1].min())                       # pad face (finger 1 on +y_H at joint 0)
    flat = fin1[fin1[:, 1] <= float(fin1[:, 1].min()) + 1e-3]   # finger-mesh inner face vertices
    z_lo = float(pad1[:, 2].min())
    z_hi = float(max(pad1[:, 2].max(), flat[:, 2].max()))
    eq = ConvexHull(fin1).equations
    best = inner
    for z in np.linspace(z_lo, z_hi, 41):
        for x in np.linspace(pad1[:, 0].min(), pad1[:, 0].max(), 17):
            ys = np.linspace(inner, inner + 0.05, 501)
            P = np.c_[np.full_like(ys, x), ys, np.full_like(ys, z)]
            ins = (P @ eq[:, :3].T + eq[:, 3] <= 1e-9).all(1)
            if ins.any():
                best = max(best, float(ys[ins].max()))
    lo, hi = hand.min(0), hand.max(0)
    return {"max_opening": float(pad1_open[:, 1].min() - pad2_open[:, 1].max()),
            "pad_half_y": float(max(abs(pad1[:, 0].min()), abs(pad1[:, 0].max()))),
            "pad_z": (-z_hi, -z_lo),
            "finger_thickness": float(best - inner),
            "palm": ((float(lo[1]), float(lo[0]), -float(hi[2])), (float(hi[1]), float(hi[0]), -float(lo[2]))),
            "pad_inner_face_closed_y": inner}


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Fit the LIBERO Panda palm shell (SL6-A A5).")
    ap.add_argument("--geometry", required=True, help="robot_geometry.py export npz (--source libero)")
    ap.add_argument("--out", default=str(PALM_SHELL_JSON))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out = pathlib.Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out} (--force)")
    tables = pc.load_geometry(args.geometry)
    capsules, report = fit_palm_shell(tables)
    doc = write_palm_shell_json(out, capsules, report, tables, args.geometry)
    print(json.dumps({"out": str(out), "shell": palm_shell_table(capsules),
                      "shell_digest": doc["provenance"]["shell_digest"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["HAND_MJCF_BODY", "PALM_LINK", "PALM_SHELL_HANDS", "PALM_SHELL_JSON", "PANDA_HOLD_SITE_OFFSET_M",
           "collision_model", "fit_palm_shell", "hand_surface", "load_palm_shell", "measure_gripper_spec",
           "palm_shell_table", "shell_digest", "write_palm_shell_json"]

"""Self-filter covering capsules for the LIBERO Panda, fitted by V11AB's own code (SL2).

RB-Y1 (T29/T30): `mujoco_source.self_filter_covering_capsules(model, urdf_links, bodies, existing=bare)`
samples each MJCF body's mesh surface, keeps the points **outside** the bare model's spheres on the
body's rigidly attached URDF link, and fits capsules (role `COVER_ROLE`) to what is left. The same
function is called here, unchanged, on a MuJoCo model rebuilt from the robosuite Panda geometry that
``benchmark/adapters/libero_panda/robot_geometry.py export`` writes (Python 3.8 side):

* bodies are renamed to the adapter URDF link they coincide with (``model.BODY_TO_LINK``;
  ``robot0_base`` and ``robot0_link0`` -- same pose -- become one body ``panda_link0``), so
  `static_urdf_ancestor` finds the link by name, exactly as it finds ``ee_left`` for RB-Y1;
  bodies with no URDF counterpart (``gripper0_right_gripper``, the finger pads, the ``mount0_*``
  pedestal) keep their names and are walked up to their rigid URDF ancestor (``panda_hand``,
  ``panda_left/rightfinger``, ``panda_link0``);
* robot base placement is dropped (``panda_link0`` at the origin) -- covers are link-frame shapes;
* which geoms are surface: ``surfaces=("collision",)`` = robosuite ``group 0`` (the collision meshes
  and the primitive pads / mount boxes), ``("visual",)`` = ``group 1`` (what the cameras render).
  Primitives (box, cylinder) are triangulated, since the V11AB sampler reads mesh geoms only.

The fitted capsules are written to ``robots/panda/description/libero_panda_self_filter_cover.json``
with the geometry digest of the export they came from; :func:`main` regenerates it::

    .venv-libero/bin/python benchmark/adapters/libero_panda/robot_geometry.py export --source libero --out G.npz
    .venv-ag3s/bin/python -m AG3S.adapters.panda.cover --geometry G.npz --out <cover.json>
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Optional, Sequence

import numpy as np

from AG3S.adapters.panda import model as panda

#: Geom group per surface name (robosuite: group 0 = collision, group 1 = visual).
SURFACE_GROUPS = {"collision": 0, "visual": 1}
#: Default surfaces the cover is fitted to (SL2 task: the robosuite collision meshes). Checked in SL2
#: (impl.md): a cover fitted to the collision geometry also contains 100 % of the visual surface the
#: cameras render (every robot body, 4 mm surface samples), so the visual meshes add nothing.
DEFAULT_SURFACES: tuple[str, ...] = ("collision",)
#: The Panda's static base assembly (robosuite ``RethinkMount``), rigidly on ``panda_link0``. The head
#: camera sees ``mount0_pedestal`` (SL1 segmentation: ~1.4 k px / frame) and a little of
#: ``mount0_torso``; all of it is covered, as RB-Y1 covers its base.
MOUNT_BODIES: tuple[str, ...] = ("mount0_controller_box", "mount0_pedestal_feet", "mount0_torso", "mount0_pedestal")
#: Bodies fitted with V11AB's *confined* rule (``COVER_CONFINED_BODIES``, T30 F5: split until every
#: capsule stays in the body's surface box). **Default: none** -- V11AB's function is used as is. SL2
#: measured the alternative on the pedestal: unconfined 3 capsules (9 spheres; the top one reaches
#: base z +0.173 m and x +0.394 m, i.e. over the table near the robot), confined 239 capsules (1039
#: spheres, still x +0.219 m, 55 s to fit). Pass ``confined=MOUNT_BODIES`` (CLI ``--confine mount``) for that
#: variant. The hand (``gripper0_right_gripper``) confined: 246 capsules / 893 spheres, reach 13 mm below
#: the palm instead of 54 mm.
CONFINED_BODIES: tuple[str, ...] = ()
CYLINDER_SEGMENTS = 48
_GEOM_MESH, _GEOM_BOX, _GEOM_CYLINDER, _GEOM_SPHERE, _GEOM_CAPSULE = 7, 6, 5, 2, 3


def load_geometry(path: str | pathlib.Path) -> dict:
    with np.load(str(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _quat_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, np.float64) / np.linalg.norm(q))
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def primitive_mesh(geom_type: int, size: np.ndarray, pos: np.ndarray, quat: np.ndarray):
    """Closed triangle mesh (body frame) of a MuJoCo box / cylinder / sphere / capsule geom."""
    s = np.asarray(size, np.float64)
    if geom_type == _GEOM_BOX:
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], np.float64) * s
        f = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                      [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], np.int64)
    elif geom_type == _GEOM_CYLINDER:
        n = CYLINDER_SEGMENTS
        a = 2 * np.pi * np.arange(n) / n
        ring = np.stack([s[0] * np.cos(a), s[0] * np.sin(a)], 1)
        v = np.vstack([np.column_stack([ring, np.full(n, -s[1])]), np.column_stack([ring, np.full(n, s[1])]),
                       [[0.0, 0.0, -s[1]], [0.0, 0.0, s[1]]]])
        f = []
        for i in range(n):
            j = (i + 1) % n
            f += [[i, j, n + j], [i, n + j, n + i], [2 * n, j, i], [2 * n + 1, n + i, n + j]]
        f = np.asarray(f, np.int64)
    elif geom_type in (_GEOM_SPHERE, _GEOM_CAPSULE):
        n, m = 24, 12
        r = s[0]
        half = s[1] if geom_type == _GEOM_CAPSULE else 0.0
        verts, faces = [], []
        for k in range(m + 1):
            phi = np.pi * k / m
            z = r * np.cos(phi) + (half if k <= m // 2 else -half)
            for i in range(n):
                th = 2 * np.pi * i / n
                verts.append([r * np.sin(phi) * np.cos(th), r * np.sin(phi) * np.sin(th), z])
        for k in range(m):
            for i in range(n):
                a0, a1 = k * n + i, k * n + (i + 1) % n
                faces += [[a0, a1, a1 + n], [a0, a1 + n, a0 + n]]
        v, f = np.asarray(verts), np.asarray(faces, np.int64)
    else:
        raise ValueError(f"no triangulation for geom type {geom_type}")
    return v @ _quat_matrix(quat).T + np.asarray(pos, np.float64), f


def mjcf_body_name(robosuite_body: str) -> str:
    return panda.BODY_TO_LINK.get(robosuite_body, robosuite_body)


def surface_geoms(tables: dict, surfaces: Sequence[str]) -> list[int]:
    unknown = sorted(set(surfaces) - set(SURFACE_GROUPS))
    if unknown or not surfaces:
        raise ValueError(f"surfaces must be a non-empty subset of {sorted(SURFACE_GROUPS)}, got {list(surfaces)}")
    groups = {SURFACE_GROUPS[s] for s in surfaces}
    return [g for g in range(len(tables["geom_names"])) if int(tables["geom_group"][g]) in groups]


def geom_mesh(tables: dict, g: int) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices (body frame), faces)`` of geom ``g`` of the export."""
    if int(tables["geom_type"][g]) == _GEOM_MESH:
        a, n = int(tables["geom_vertadr"][g]), int(tables["geom_vertnum"][g])
        fa, fn = int(tables["geom_faceadr"][g]), int(tables["geom_facenum"][g])
        return np.asarray(tables["mesh_vert_body"][a:a + n], np.float64), np.asarray(tables["mesh_face"][fa:fa + fn])
    return primitive_mesh(int(tables["geom_type"][g]), tables["geom_size"][g], tables["geom_pos"][g],
                          tables["geom_quat"][g])


def build_mujoco_model(tables: dict, surfaces: Sequence[str] = DEFAULT_SURFACES):
    """MuJoCo model of the robot subtree with only the ``surfaces`` geoms, as meshes in body frames.

    Returns ``(mujoco.MjModel, {mjcf body name: [geom names]})``. No joints: the covers are fitted
    per body in its link frame, and `static_urdf_ancestor` only needs the fixed chain.
    """
    import mujoco

    names = [str(n) for n in tables["body_names"]]
    parents = [str(n) for n in tables["body_parent"]]
    base = names.index("robot0_base")
    link0 = names.index("robot0_link0")
    if parents[link0] != "robot0_base" or not (np.allclose(tables["body_pos"][link0], 0.0, atol=0.0)
                                               and np.allclose(tables["body_quat"][link0], [1, 0, 0, 0], atol=0.0)):
        raise ValueError("robot0_link0 is not at the robot0_base pose; panda_link0 cannot stand for both")
    spec = mujoco.MjSpec()
    bodies = {}
    for k, name in enumerate(names):
        target = mjcf_body_name(name)
        if target in bodies:
            continue
        if k == base:
            bodies[target] = spec.worldbody.add_body(name=target, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
            continue
        parent = bodies[mjcf_body_name(parents[k])]
        bodies[target] = parent.add_body(name=target, pos=[float(x) for x in tables["body_pos"][k]],
                                         quat=[float(x) for x in tables["body_quat"][k]])
    geoms_by_body: dict[str, list[str]] = {}
    for g in surface_geoms(tables, surfaces):
        body = mjcf_body_name(names[int(tables["geom_body"][g])])
        verts, faces = geom_mesh(tables, g)
        if faces.shape[0] == 0:
            continue
        mesh_name = f"m{g}_{tables['geom_names'][g]}"
        mesh = spec.add_mesh(name=mesh_name, uservert=verts.reshape(-1).tolist(),
                             userface=np.asarray(faces, np.int64).reshape(-1).tolist())
        mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        bodies[body].add_geom(name=str(tables["geom_names"][g]), type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh_name,
                              contype=0, conaffinity=0)
        geoms_by_body.setdefault(body, []).append(str(tables["geom_names"][g]))
    return spec.compile(), geoms_by_body


def _confined_cover(mj_model, body: str, bare_model) -> list:
    """`self_filter_covering_capsules` steps 1-4 for one body with ``confined = True``.

    The V11AB function confines exactly the bodies named in its module constant
    ``COVER_CONFINED_BODIES`` (RB-Y1 ``base``), which this adapter does not modify. The same steps are
    spelled out here with the same helpers (`_body_surface_points`, `static_urdf_ancestor`, the
    existing-sphere mask, `_cover_capsules_for(box=...)`) and the same constants.
    """
    import mujoco
    from scipy.spatial import cKDTree

    from AG3S.ag3s.experiments.sources import mujoco_source as ms

    link, T = ms.static_urdf_ancestor(mj_model, body, bare_model.model.links)
    bid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body)
    pts = ms._body_surface_points(mj_model, bid, ms.SURFACE_SAMPLE_SPACING)
    if not pts.shape[0]:
        return []
    pts = pts @ T[:3, :3].T + T[:3, 3]
    box = (pts.min(axis=0), pts.max(axis=0))
    centres, radii = bare_model.local_spheres(link)
    if radii.size:
        inside = np.zeros(pts.shape[0], bool)
        tree = cKDTree(pts)
        for c, r in zip(centres, radii):
            hits = tree.query_ball_point(c, float(r))
            if hits:
                inside[hits] = True
        pts = pts[~inside]
    if not pts.shape[0]:
        return []
    return ms._cover_capsules_for(pts, link, pad=ms.COVER_PAD, eps=ms.COVER_CLUSTER_EPS, box=box)


def fit_self_filter_cover(tables: dict, bare_model, surfaces: Sequence[str] = DEFAULT_SURFACES,
                          bodies: Optional[Sequence[str]] = None,
                          confined: Sequence[str] = CONFINED_BODIES) -> tuple[list, dict]:
    """V11AB `self_filter_covering_capsules` on the Panda: ``(capsules, {body: n_capsules})``.

    One body at a time, in export order; bodies in ``confined`` go through :func:`_confined_cover`.
    """
    from AG3S.ag3s.experiments.sources.mujoco_source import self_filter_covering_capsules

    mj_model, geoms_by_body = build_mujoco_model(tables, surfaces)
    bodies = tuple(bodies) if bodies is not None else tuple(geoms_by_body)
    unknown = sorted(set(bodies) - set(geoms_by_body))
    if unknown:
        raise ValueError(f"no {list(surfaces)} geometry on bodies {unknown}; have {sorted(geoms_by_body)}")
    per_body = {}
    capsules = []
    for body in bodies:
        if body in set(confined):
            caps = _confined_cover(mj_model, body, bare_model)
        else:
            caps = self_filter_covering_capsules(mj_model, bare_model.model.links, bodies=(body,),
                                                 existing=bare_model)
        per_body[body] = len(caps)
        capsules.extend(caps)
    return capsules, per_body


# --- JSON ---------------------------------------------------------------------------------------------
def capsule_to_json(capsule) -> dict:
    return {"link": capsule.link, "origin": np.asarray(capsule.origin, np.float64).tolist(),
            "radius": float(capsule.radius), "length": float(capsule.length), "role": str(capsule.role)}


def write_cover_json(path: str | pathlib.Path, capsules: Sequence, provenance: dict, *, per_body: dict) -> dict:
    document = {
        "_comment": "Generated by AG3S/adapters/panda/cover.py (SL2). Self-filter covering capsules for the LIBERO "
                    "Panda, link frames of robots/panda/description/panda_libero.urdf. Do not edit by hand.",
        "provenance": provenance,
        "per_body": per_body,
        "capsules": [capsule_to_json(c) for c in capsules],
    }
    pathlib.Path(path).write_text(json.dumps(document, indent=1) + "\n")
    return document


def read_cover_json(path: str | pathlib.Path) -> tuple[list, dict]:
    from AG3S.ag3s.robot_models.urdf_sphere_chain import COVER_ROLE, UrdfCapsule

    document = json.loads(pathlib.Path(path).read_text())
    capsules = []
    for row in document["capsules"]:
        if row["role"] != COVER_ROLE:
            raise ValueError(f"{path}: capsule role {row['role']!r} is not {COVER_ROLE!r}")
        capsules.append(UrdfCapsule(link=row["link"], origin=np.asarray(row["origin"], np.float64),
                                    radius=float(row["radius"]), length=float(row["length"]), role=COVER_ROLE))
    return capsules, document


def provenance_of(tables: dict, geometry_path: str | pathlib.Path, surfaces: Sequence[str],
                  confined: Sequence[str] = CONFINED_BODIES) -> dict:
    from AG3S.ag3s.experiments.sources import mujoco_source as ms

    source = json.loads(str(tables["provenance_json"])) if "provenance_json" in tables else {}
    return {
        "geometry_digest": str(tables.get("geometry_digest", "")),
        "kinematic_digest": str(tables.get("kinematic_digest", "")),
        "geometry_export": source,
        "geometry_file": pathlib.Path(geometry_path).name,
        "surfaces": list(surfaces),
        "confined_bodies": list(confined),
        "fitter": "AG3S.ag3s.experiments.sources.mujoco_source.self_filter_covering_capsules",
        "fitter_constants": {"SURFACE_SAMPLE_SPACING": ms.SURFACE_SAMPLE_SPACING, "COVER_PAD": ms.COVER_PAD,
                             "COVER_CLUSTER_EPS": ms.COVER_CLUSTER_EPS,
                             "COVER_CONFINED_BODIES": list(ms.COVER_CONFINED_BODIES)},
        "bare_model": "franka.yml collision spheres on panda_libero.urdf (all links)",
        "urdf_sha256": panda.file_sha256(panda.PANDA_URDF),
        "franka_yml_sha256": panda.file_sha256(panda.CUROBO_FRANKA_YML),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit the LIBERO Panda self-filter cover (SL2).")
    parser.add_argument("--geometry", required=True, help="robot_geometry.py export npz")
    parser.add_argument("--out", required=True)
    parser.add_argument("--surfaces", nargs="+", default=list(DEFAULT_SURFACES), choices=sorted(SURFACE_GROUPS))
    parser.add_argument("--confine", nargs="+", default=list(CONFINED_BODIES), metavar="BODY",
                        help="variant (not the committed cover): fit these MJCF bodies with V11AB's confined rule; "
                             "'mount' = MOUNT_BODIES (e.g. --confine mount gripper0_right_gripper)")
    parser.add_argument("--force", action="store_true", help="overwrite --out")
    args = parser.parse_args(argv)
    out = pathlib.Path(args.out)
    if out.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {out} (--force)")
    tables = load_geometry(args.geometry)
    confined = tuple(dict.fromkeys(b for name in args.confine for b in (MOUNT_BODIES if name == "mount" else (name,))))
    capsules, per_body = fit_self_filter_cover(tables, panda.build_panda_bare_model(), args.surfaces, confined=confined)
    document = write_cover_json(out, capsules, provenance_of(tables, args.geometry, args.surfaces, confined),
                                per_body=per_body)
    print(json.dumps({"out": str(out), "capsules": len(document["capsules"]), "per_body": per_body}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

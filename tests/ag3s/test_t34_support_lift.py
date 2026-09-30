"""T34 J1 — 쥔 질의 구는 받침면을 넘지 않는다 (support lift) · self-filter 복사본은 그대로.

두 묶음:

* **합성** — `held_object.support_plane_under` · `field_support_band` · `lift_off_support` 의 규칙과
  `AG3S.attach` 배선 (질의 구는 올라가고 self-filter 구는 맞춤 그대로, 기록 `held_record["support"]`).
* **T33 기록 재구성** — attach 9 건을 `AG3S.attach` 로 다시 한다. 입력은 live attach 가 본 것과 같다:
  스냅샷 프레임의 `target_points` (manipulated 기하) · 그 프레임 planning 행 q (`geometry_state`) ·
  attach 요청 행 q (`robot_state`) · **attach 직전 청크**의 받침 평면 (support mask 점, z > 0.5 m 의
  TLS) 과 거리장 (띠 측정). 재구성 충실도: 맞춘 중심이 기록의 `fit.centre_m` 과 < 1 mm.
  판정: 모든 구 바닥 ≥ 평면 (스냅샷 자세) · ≥ 평면 + 띠 (attach 자세) · ≥ GT 테이블 상판 (0.82 m);
  E3b ep1800 r2 t=152 의 held 행 (기록 거리장, q_now + reference 0–7 행) 이 음수가 아니고, 그 run 의
  교착 56 청크도 그렇다. 기록이 이 머신에 없으면 건너뛴다.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest

from benchmark.ag3s.robot_models.held_object import (
    HELD_FILTER_SLIP_M,
    field_support_band,
    held_spheres_of,
    lift_off_support,
    support_plane_under,
)
from benchmark.ag3s.types import SupportSurface

MAX_OPENING = 0.07146


def _plane(normal, offset, sid=0):
    return SupportSurface(id=sid, normal=np.asarray(normal, float), offset=float(offset),
                          point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)


# =============================================================================================
# 합성
# =============================================================================================


def test_the_support_under_an_object_is_the_highest_horizontal_plane_below_it():
    floor, table = _plane([0, 0, 1], 0.0, 0), _plane([0, 0, 1], 0.82, 1)
    wall = _plane([1, 0, 0], 0.5, 2)
    upside = _plane([0, 0, -1], -0.82, 3)                   # same table, normal pointing down
    n, off, s = support_plane_under([floor, wall, table], [0.4, 0.3, 0.85])
    assert s is table and np.allclose(n, [0, 0, 1]) and off == pytest.approx(0.82)
    n, off, s = support_plane_under([upside], [0.4, 0.3, 0.85])
    assert np.allclose(n, [0, 0, 1]) and off == pytest.approx(0.82)     # flipped up
    assert support_plane_under([wall], [0.4, 0.3, 0.85]) is None
    assert support_plane_under([table], [0.4, 0.3, 0.80]) is None       # the plane is above it


def test_a_sphere_through_the_table_is_lifted_along_the_normal_and_keeps_its_radius():
    n, off = np.array([0.0, 0.0, 1.0]), 0.82
    primary = (np.array([0.45, 0.33, 0.849]), 0.0386)      # bottom 9.6 mm under the table (T33 ~9.3)
    stem = (np.array([0.46, 0.33, 0.89]), 0.011)            # well above: untouched
    out, rec = lift_off_support([primary, stem], n, off, clearance=0.009)
    (c0, r0), (c1, r1) = out
    assert r0 == primary[1] and r1 == stem[1]               # radii are never capped
    assert np.allclose(c0[:2], primary[0][:2])              # only along the normal
    assert c0[2] - r0 - off == pytest.approx(0.009, abs=1e-9)
    assert np.allclose(c1, stem[0])
    assert rec["spheres"][0]["lift_mm"] == pytest.approx(18.6, abs=0.01)
    assert rec["spheres"][1]["lift_mm"] == 0.0


def test_the_lift_holds_at_both_instants_with_their_own_clearance():
    """Capture pose: the plane itself. Attach request pose (the hand pressed 10 mm down since the
    capture, T33 E3b ep1800 r2): the plane + the field band. The larger need wins."""
    n, off = np.array([0.0, 0.0, 1.0]), 0.82
    s = (np.array([0.45, 0.33, 0.849]), 0.0386)             # capture: bottom −9.6 mm
    press = np.eye(4)
    press[2, 3] = -0.010                                    # attach pose: 10 mm lower
    out, rec = lift_off_support([s], n, off, clearance=0.009, relative_poses=[press],
                                snapshot_clearance=0.0)
    (c, r), = out
    assert c[2] - r - off == pytest.approx(0.019, abs=1e-9)             # capture pose: +19 mm
    assert (c[2] - 0.010) - r - off == pytest.approx(0.009, abs=1e-9)   # attach pose: the band
    assert rec["spheres"][0]["bottom_above_plane_mm_before"] == pytest.approx([-9.6, -19.6], abs=0.01)
    # a pose rotated about the normal moves the lift direction with the palm, not off the normal
    rot = np.eye(4)
    rot[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    out2, _ = lift_off_support([s], n, off, clearance=0.009, relative_poses=[rot])
    c2 = out2[0][0]
    assert (rot[:3, :3] @ c2)[2] - s[1] - off >= 0.009 - 1e-9


class _TableField:
    """Analytic stand-in for the recorded fine ESDF over a table: d = 0 up to `band` above the plane
    (the recorded ±8.8 mm surface band), then `h − band`; an orange 100 mm from the apple."""

    def __init__(self, z0, band, obstacle=None):
        self.z0, self.band, self.obstacle = z0, band, obstacle

    def distance(self, p):
        p = np.asarray(p, float).reshape(-1, 3)
        d = np.maximum(p[:, 2] - self.z0 - self.band, 0.0)
        if self.obstacle is not None:
            c, r = self.obstacle
            d = np.minimum(d, np.linalg.norm(p - c, axis=1) - r)
        return d


def test_the_field_band_is_measured_over_the_table_and_ignores_a_neighbour():
    apple = np.array([0.45, 0.33, 0.85])
    orange = (np.array([0.55, 0.33, 0.855]), 0.035)
    band, rec = field_support_band(_TableField(0.82, 0.0088, orange), [0, 0, 1], 0.82, apple)
    assert band == pytest.approx(0.0088, abs=1e-6)
    assert rec["n_table_only"] < rec["n_samples"]            # the orange's samples were dropped
    none, rec2 = field_support_band(_TableField(0.82, 0.0088, (apple, 0.30)), [0, 0, 1], 0.82, apple)
    assert none is None and "too few" in rec2["reason"]      # nothing but the obstacle nearby


@pytest.fixture(scope="module")
def rby1():
    from benchmark.ag3s.robot_models import RBY1_URDF, UrdfSphereChain, parse_urdf

    return UrdfSphereChain(parse_urdf(RBY1_URDF),
                           link_filter=["ee_left", "ee_right", "ee_finger_l1", "ee_finger_l2",
                                        "ee_finger_r1", "ee_finger_r2"])


def _ag3s(model):
    from benchmark.ag3s.config import AG3SConfig
    from benchmark.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": MAX_OPENING}})
    return AG3S(cfg, robot_model=model, constraint_robot_model=model,
                attached_parent_links=("ee_left", "ee_right"))


def _target(points, tid=3):
    from benchmark.ag3s.stages.geometry import fit_sphere
    from benchmark.ag3s.types import TargetGeometry

    pts = np.asarray(points, np.float64)
    return TargetGeometry(id=tid, points=pts, point_indices=np.arange(len(pts)),
                          centroid=pts.mean(axis=0), bounding_geometry=fit_sphere(pts),
                          attention_score=0.9, confidence=0.9)


def _oblate_apple_cap(centre, n=3000, seed=0):
    """A one-sided view of an oblate apple (33 mm wide, 28 mm half-height) — the sphere fit comes
    out ~33 mm and reaches under the table it stands on (T33: 1.8–10.1 mm with the pad)."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    v = v[(v @ np.array([1.0, 0.0, 0.4]) / np.linalg.norm([1.0, 0.0, 0.4])) > 0.25]
    return centre + v * np.array([0.033, 0.033, 0.028])


def test_attach_lifts_the_query_spheres_and_leaves_the_filter_copy(rby1):
    q = np.zeros(rby1.nq)
    T_l1, T_l2 = rby1.link_pose(q, "ee_finger_l1"), rby1.link_pose(q, "ee_finger_l2")
    c = 0.5 * (T_l1[:3, 3] + T_l2[:3, 3])
    table_z = c[2] - 0.028                                   # the apple stands on the table
    ag3s = _ag3s(rby1)
    ag3s._last_support_surfaces = [_plane([0, 0, 1], table_z - 0.8, 0),        # noqa: SLF001
                                   _plane([0, 0, 1], table_z, 1)]
    ag3s._last_esdf_field = _TableField(table_z, 0.0088)                        # noqa: SLF001
    held = ag3s.attach(_target(_oblate_apple_cap(c)), robot_state=q, parent_link="ee_finger_l1")
    rec = ag3s.held_record
    sup = rec["support"]
    assert sup["applied"] and sup["clearance_source"] == "field_band"
    assert sup["plane"]["surface_id"] == 1 and sup["clearance_mm"] == pytest.approx(8.8, abs=0.01)
    assert sup["snapshot_clearance_mm"] == 0.0
    T = rby1.link_pose(q, "ee_left")
    query = [(T[:3, :3] @ cc + T[:3, 3], r) for cc, r in held_spheres_of(held)]
    fit_bottom = min(p["bottom_above_plane_mm_before"][0] for p in sup["spheres"])
    assert fit_bottom < -3.0                                 # the fit did cross the table
    for cc, r in query:
        assert cc[2] - r - table_z >= 0.0088 - 1e-9          # now clears plane + band
    # the filter copy is the fit as T32 made it (not lifted), radius + slip in the filter
    filt = ag3s._held_filter_local                                             # noqa: SLF001
    assert len(filt) == len(query)
    fc = T[:3, :3] @ filt[0][0] + T[:3, 3]
    assert np.allclose(fc, rec["fit"]["centre_m"], atol=1e-6)
    assert fc[2] < query[0][0][2] - 0.003
    ag3s._prepare_frame_filter()                                               # noqa: SLF001
    hc, hr = ag3s._frame_filter_model.held_spheres(q)                          # noqa: SLF001
    assert np.allclose(hc[0], fc) and hr[0] == pytest.approx(filt[0][1] + HELD_FILTER_SLIP_M)
    assert rec["filter_spheres_parent_frame"][0]["radius_mm"] == pytest.approx(filt[0][1] * 1000, abs=1e-3)
    # the per-frame TSDF free follows the query spheres; the capture-pose free is the fit
    assert np.allclose(ag3s._held_first_free[0][0], fc)                        # noqa: SLF001
    ag3s.detach()
    assert ag3s._held_filter_local is None and ag3s.held_observation() is None  # noqa: SLF001


def test_no_plane_or_switch_off_changes_nothing(rby1):
    q = np.zeros(rby1.nq)
    T_l1, T_l2 = rby1.link_pose(q, "ee_finger_l1"), rby1.link_pose(q, "ee_finger_l2")
    c = 0.5 * (T_l1[:3, 3] + T_l2[:3, 3])
    for setup, reason in ((lambda a: None, "no_support_plane_under_the_object"),
                          (lambda a: setattr(a, "held_support_lift", False), "switched_off")):
        ag3s = _ag3s(rby1)
        setup(ag3s)
        if reason == "switched_off":
            ag3s._last_support_surfaces = [_plane([0, 0, 1], c[2] - 0.028)]    # noqa: SLF001
        held = ag3s.attach(_target(_oblate_apple_cap(c)), robot_state=q, parent_link="ee_finger_l1")
        rec = ag3s.held_record
        assert rec["support"] == {"applied": False, "reason": reason}
        T = rby1.link_pose(q, "ee_left")
        assert np.allclose(T[:3, :3] @ held_spheres_of(held)[0][0] + T[:3, 3], rec["fit"]["centre_m"],
                           atol=1e-6)


def test_without_a_field_the_plane_alone_is_cleared_and_a_given_clearance_wins(rby1):
    q = np.zeros(rby1.nq)
    T_l1, T_l2 = rby1.link_pose(q, "ee_finger_l1"), rby1.link_pose(q, "ee_finger_l2")
    c = 0.5 * (T_l1[:3, 3] + T_l2[:3, 3])
    z = c[2] - 0.028
    ag3s = _ag3s(rby1)
    ag3s._last_support_surfaces = [_plane([0, 0, 1], z)]                        # noqa: SLF001
    ag3s.attach(_target(_oblate_apple_cap(c)), robot_state=q, parent_link="ee_finger_l1")
    assert ag3s.held_record["support"]["clearance_source"] == "no_field"
    assert ag3s.held_record["support"]["clearance_mm"] == 0.0
    ag3s.detach()
    ag3s.held_support_clearance_m = 0.004
    ag3s._last_esdf_field = _TableField(z, 0.0088)                              # noqa: SLF001
    ag3s.attach(_target(_oblate_apple_cap(c)), robot_state=q, parent_link="ee_finger_l1")
    assert ag3s.held_record["support"]["clearance_source"] == "given"
    assert ag3s.held_record["support"]["clearance_mm"] == pytest.approx(4.0)
    ag3s.detach()
    ag3s.held_support_clearance_m = None
    ag3s._last_esdf_field = _TableField(z, 0.0088, (c, 0.5))                    # noqa: SLF001
    ag3s.attach(_target(_oblate_apple_cap(c)), robot_state=q, parent_link="ee_finger_l1")
    assert ag3s.held_record["support"]["clearance_source"] == "fallback"
    assert ag3s.held_record["support"]["clearance_mm"] == pytest.approx(9.0)


# =============================================================================================
# T33 기록 재구성 — attach 9 건
# =============================================================================================

V = pathlib.Path("/mnt/dev/work/outputs/verify/T33")
PORT = {"E3a": 8204, "E3b": 8206}
EVENTS = [("E3a", "E3a_ep1800_r1", 144), ("E3a", "E3a_ep1800_r2", 152),
          ("E3a", "E3a_ep1807_r1", 104), ("E3a", "E3a_ep1807_r2", 120),
          ("E3b", "E3b_ep1800_r2", 152), ("E3b", "E3b_ep1807_r1", 104),
          ("E3b", "E3b_ep1807_r2", 136), ("E3b", "E3b_ep1808_r1", 128),
          ("E3b", "E3b_ep1808_r2", 400)]
_have_records = (V / "p/E3b_analysis.json").exists() and \
    (V / "E3b_rec/server_8206/server_constraints/run_0000/chunk_00244.npz").exists()
records = pytest.mark.skipif(not _have_records, reason="T33 records not present")

_CACHE: dict = {}


def _robot(cond, run):
    if cond not in _CACHE:
        from benchmark.trajopt.experiments import fixed_input_ablation as T

        rec = V / f"{cond}_rec"
        args = T.build_parser().parse_args([
            "--constraints", str(rec / f"server_{PORT[cond]}/server_constraints/run_0000"),
            "--frames", str(rec / run / "frames.jsonl"),
            "--server-log", str(rec / f"serve_safe_{PORT[cond]}.log"), "--out", "/dev/null"])
        settings = T.resolve_settings(args, T.parse_server_log(args.server_log),
                                      json.loads((rec / run / "manifest.json").read_text()))
        robot, qadr, _ = T.build_robot(settings)
        _CACHE[cond] = (robot, qadr, T)
    return _CACHE[cond]


def _chunks(cond, run):
    A = json.loads((V / "p" / f"{cond}_analysis.json").read_text())["runs"][run]
    rows = {int(x["seq"]): x for x in (json.loads(line) for line in
                                        (V / f"{cond}_rec" / run / "frames.jsonl").read_text()
                                        .splitlines() if line.strip())
            if x.get("kind") == "planning"}
    out = []
    for c in A["chunks"]:
        if c.get("server_chunk"):
            f = V / f"{cond}_rec" / c["server_chunk"]
            out.append({"t": int(c["t"]), "file": f, "row": rows[int(c["seq"])]})
    return out


def _npz(c):
    z = np.load(c["file"], allow_pickle=False)
    return z, json.loads(str(z["summary_json"]))


def _table_plane(z):
    pts = np.asarray(z["raw_cloud"], float)[np.asarray(z["support_mask"], bool)]
    tab = pts[pts[:, 2] > 0.5]
    m = tab.mean(axis=0)
    n = np.linalg.svd(tab - m, full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    return _plane(n, float(n @ m)), n, float(n @ m)


def _gt_table_top_base(row):
    import mujoco

    key = "mj"
    if key not in _CACHE:
        m = mujoco.MjModel.from_xml_path(
            "/mnt/dev/work/pi05_TO_hybrid/rby1_description/models/rby1a/mujoco/model_transport.xml")
        _CACHE[key] = (m, mujoco.MjData(m), mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base"))
    m, d, base = _CACHE[key]
    d.qpos[:] = np.asarray(row["qpos"])
    mujoco.mj_kinematics(m, d)
    return float((np.array([0.0, 0.0, 0.82]) - d.xpos[base]) @ d.xmat[base].reshape(3, 3)[:, 2])


def rebuild(cond, run, t_attach):
    """`AG3S.attach` on the recorded attach inputs → (ag3s, held, context)."""
    robot, qadr, T = _robot(cond, run)
    ch = _chunks(cond, run)
    a = next(i for i, c in enumerate(ch) if c["t"] == t_attach)
    z_a, s_a = _npz(ch[a])
    held_rec = s_a["grasp"]["attached"]["held"]
    f = int(re.match(r"frame (\d+)", held_rec["snapshot_state"]).group(1))
    isnap = next(i for i, c in enumerate(ch) if int(np.load(c["file"])["t_step"]) == f + 1)
    z_snap, _ = _npz(ch[isnap])
    z_prev, _ = _npz(ch[a - 1])
    q_snap = T.q_now_from_qpos(ch[isnap]["row"]["qpos"], qadr)
    q_now = T.q_now_from_qpos(ch[a]["row"]["qpos"], qadr)
    surface, n, off = _table_plane(z_prev)
    ag3s = _ag3s(robot)
    ag3s._last_support_surfaces = [surface]                                    # noqa: SLF001
    ag3s._last_esdf_field = T.build_field(z_prev, outside_distance=1.0)        # noqa: SLF001
    held = ag3s.attach(_target(np.asarray(z_snap["target_points"], float)), robot_state=q_now,
                       parent_link="ee_finger_l1", geometry_state=q_snap)
    return ag3s, held, dict(robot=robot, qadr=qadr, T=T, ch=ch, a=a, n=n, off=off, q_now=q_now,
                            q_snap=q_snap, recorded=held_rec, z_a=z_a)


@records
@pytest.mark.parametrize("cond,run,t", EVENTS)
def test_every_t33_attach_sphere_bottom_is_at_or_above_the_table(cond, run, t):
    ag3s, held, k = rebuild(cond, run, t)
    rec = ag3s.held_record
    # faithful: the same fit as the live attach (T33 record)
    assert np.linalg.norm(np.asarray(rec["fit"]["centre_m"])
                          - np.asarray(k["recorded"]["fit"]["centre_m"])) < 1e-3
    sup = rec["support"]
    assert sup["applied"] and sup["clearance_source"] == "field_band"
    assert 8.0 <= sup["clearance_mm"] <= 9.5                   # the recorded band (q90)
    # the fit crossed the table in every T33 attach (the T33 finding) ...
    assert min(p["bottom_above_plane_mm_before"][0] for p in sup["spheres"]) < -1.5
    robot, n, off = k["robot"], k["n"], k["off"]
    gt_top = _gt_table_top_base(k["ch"][k["a"]]["row"])
    for q, need in ((k["q_snap"], 0.0), (k["q_now"], sup["clearance_mm"] / 1000.0)):
        Tq = robot.link_pose(q, held.parent_link)
        for c, r in held_spheres_of(held):
            cb = Tq[:3, :3] @ c + Tq[:3, 3]
            assert float(n @ cb) - off - r >= need - 1e-6     # ... and none does now
            assert cb[2] - r >= gt_top                        # nor the true table top
    ag3s.detach()


def _held_rows(k, link, local, z, q_now):
    from benchmark.trajopt.types import ChunkLayout

    robot, T = k["robot"], k["T"]
    field = T.build_field(z, outside_distance=1.0)
    lay = ChunkLayout.rby1(robot.joint_names)
    traj = lay.chunk_to_trajectory(np.asarray(z["reference_chunk"], float))[:, :8]
    out = []
    for q in [q_now] + list(lay.full_q(traj, q_now).T):
        Tq = robot.link_pose(q, link)
        cs = [(Tq[:3, :3] @ c + Tq[:3, 3], r) for c, r in local]
        d = np.asarray(field.distance(np.array([c for c, _ in cs])), float)
        out.append(min(float(x - r) for x, (_, r) in zip(d, cs)))
    return np.array(out)


@records
def test_the_e3b_ep1800_r2_held_table_row_no_longer_violates():
    """T33: the held sphere read −11.0 mm (TO initial) against the table at t=152 and the run
    deadlocked for 56 chunks (−1.6 … −10.5 mm). With J1 every held row of that chunk (q_now +
    reference rows 0–7, the recorded field, `esdf_margin` 0) is ≥ 0, and so is every held row of
    the 56 chunks that followed (each with its own field and q_now)."""
    ag3s, held, k = rebuild("E3b", "E3b_ep1800_r2", 152)
    link, local = held.parent_link, held_spheres_of(held)
    rows = _held_rows(k, link, local, k["z_a"], k["q_now"])
    assert rows.min() >= 0.0, rows
    # the pre-J1 spheres (the record) violate the same rows
    rec_local = [(np.asarray(x["centre_m"]), x["radius_mm"] / 1000.0)
                 for x in k["recorded"]["spheres_parent_frame"]]
    assert _held_rows(k, link, rec_local, k["z_a"], k["q_now"]).min() < -0.009
    T = k["T"]
    worst = []
    for c in k["ch"][k["a"]:]:
        z = np.load(c["file"], allow_pickle=False)
        worst.append(_held_rows(k, link, local, z,
                                T.q_now_from_qpos(c["row"]["qpos"], k["qadr"])).min())
    assert len(worst) >= 56 and min(worst) >= 0.0

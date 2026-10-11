"""T43 R2 — 기록 재생: 쥔 사과의 몸통 덮개가 막대 접촉을 TO 에 보이게 하는가 (8 run, T43 S (5)).

입력은 live attach 가 본 것 그대로다 (`test_t34_support_lift.rebuild` 와 같은 규칙): attach 청크의
`target_points` (= 스냅샷 기하, 맞춘 중심이 기록과 0 mm), 스냅샷 프레임 planning 행 q (`geometry_state`),
attach 요청 행 q, 기록의 J1 받침 평면, attach 직전 청크의 거리장 (띠 측정). 그 위에서 `AG3S.attach` 를
`held_body_cover` off / on 으로 다시 하고, 청크마다 **TO 의 쥔 물체 행** 을 `CollisionLinearizer.
esdf_clearance` 로 잰다 (기록 거리장 · 기록 refined 계획 · `esdf_margin` 기록값). on 의 프레임 선택
(받침 가까이 → J1 lift) 은 `AG3S._frame_held_attached` 가 기록의 q_now · reference 실행 창으로 한다.
참값 (사과 mesh ↔ 막대) 은 verifier 의 `outputs/verify/T43/S/raw/held_runs_all.json`.

고정하는 것:

1. off 는 기록의 attach 와 같다 — 맞춘 중심 · 질의 구 (중심 · 반지름) 가 기록 그대로, 프레임 선택은
   `_attached` 그 객체, TO 행은 verifier 재구성 (기록 worst row 와 대조된 것) 과 0.05 mm 안.
2. 덮임 — 접촉 전 모든 held 청크에서 on 의 구가 사과 mesh 꼭짓점을 최대 3 mm 바깥까지 덮는다
   (off: 4.6–16.3 mm 바깥, T43 S).
3. TO 가 막대를 본다 — 접촉 청크 (접촉 행이 든 청크) 의 on 쥔 물체 행 최솟값 ≤ 0, 그리고 계획 단계 중
   참값 ≤ 0 인 단계마다 on ≤ 0.
4. 파지 · 들어 올림 — 받침 가까이 (J1 lift) 로 고른 청크에서 on 이 새로 음수가 되지 않는다.
5. 오른손 — T42-1 오른팔 사과 run (`ee_finger_r1` → `ee_right`) 에서 같은 길: off = 기록, on 은 attach
   청크에서 lift (받침 + 띠 위), 들어 올린 뒤 lift 없이 사과 mesh 를 최대 3 mm 바깥까지 덮는다.
"""

from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import dataclasses
import functools
import json
import pathlib
import re

import numpy as np
import pytest

from AG3S.ag3s.robot_models.held_object import held_spheres_of
from AG3S.ag3s.types import SupportSurface

V = pathlib.Path(_ag3s_workspace_path('legacy/outputs/verify'))
S_RAW = V / "T43/S/raw"
VERIFY = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/handoff/T43S.verify.json'))
_have = (S_RAW / "chunk_index_t40t41.json").exists() and (S_RAW / "held_runs_all.json").exists() \
    and VERIFY.exists()
records = pytest.mark.skipif(not _have, reason="T43 S records not present")
RUNS = (json.loads(VERIFY.read_text())["numbers"]["q5_held_rows"]["apple_bar_contact_runs"]
        if _have else [])
H = 8
MAX_OPENING = 0.07146


class _Args:
    links = sphere_spacing = max_spheres_per_capsule = capsule_radius_scale = None
    max_sphere_radius = esdf_margin = target_field_policy = phase = None
    manipulators = model_xml = hand = rows_per_step = None
    allow_uncertified = False


@functools.lru_cache(maxsize=4)
def _setup(server_log: str, manifest: str):
    from AG3S.trajopt.experiments import fixed_input_ablation as FIA
    from AG3S.trajopt.linearize import CollisionLinearizer
    from AG3S.trajopt.types import ChunkLayout

    man = json.loads(pathlib.Path(manifest).read_text())
    settings = FIA.resolve_settings(_Args, FIA.parse_server_log(server_log), man)
    robot, qadr, _ = FIA.build_robot(settings)
    layout = ChunkLayout.rby1(robot.joint_names)
    *_, max_distance, unknown = FIA.contact_policy(settings, robot.sphere_link_names,
                                                   has_target=False, has_free=False,
                                                   attached=False)
    return dict(FIA=FIA, settings=settings, robot=robot, qadr=qadr, layout=layout,
                outside=max_distance if unknown == "free" else -max_distance,
                lin=CollisionLinearizer(robot, layout, H), xml=man["scene"]["model_xml"])


def _ag3s(robot, body_cover):
    from AG3S.ag3s.config import AG3SConfig
    from AG3S.ag3s.runtime.pipeline import AG3S

    cfg = AG3SConfig().with_overrides({"constraint": {"horizon": 2, "max_attached_primitives": 4},
                                       "clustering": {"gripper_max_opening": MAX_OPENING},
                                       "esdf": {"backend": "curobo", "voxel_size": 0.02,
                                                "fine_voxel_size": 0.005,
                                                "tsdf_voxel_size": 0.005}})
    a = AG3S(cfg, robot_model=robot, constraint_robot_model=robot,
             attached_parent_links=("ee_left", "ee_right"))
    a.held_body_cover = body_cover
    return a


def _surface(pl):
    return SupportSurface(id=1, normal=np.asarray(pl["normal"], float), offset=float(pl["offset_m"]),
                          point_indices=np.zeros(0, np.int64), point_count=1000, rms_error=0.001)


class Run:
    """One recorded run: rebuild its first attach (off / on) and evaluate the TO's held rows."""

    def __init__(self, rel: str, index: str = "chunk_index_t40t41.json"):
        idx = json.loads((S_RAW / index).read_text())
        self.key = str(V / rel)
        ix = idx[self.key]
        rd = pathlib.Path(self.key)
        self.G = _setup(str(pathlib.Path(ix["server"]) / "serve_safe.log"), str(rd / "manifest.json"))
        rows = [json.loads(x) for x in (rd / "frames.jsonl").read_text().splitlines() if x.strip()]
        self.plan = {r["seq"]: r for r in rows if r["kind"] == "planning"}
        self.ctrl = {r["t_step"]: r for r in rows if r["kind"] == "control"}
        self.npz = {int(s): v["npz"] for s, v in ix["chunks"].items()}
        self.summ = {}
        for s, f in self.npz.items():
            with np.load(f, allow_pickle=True) as z:
                self.summ[s] = json.loads(str(z["summary_json"]))
        self.attach_seq = next(s for s in sorted(self.summ)
                               if (self.summ[s].get("grasp") or {}).get("attach"))
        self.recorded = self.summ[self.attach_seq]["grasp"]["attached"]["held"]
        self.truth = json.loads((S_RAW / "held_runs_all.json").read_text())["runs"].get(self.key)
        self.off = self.rebuild(False)
        self.on = self.rebuild(True)

    def q_now(self, seq):
        return self.G["FIA"].q_now_from_qpos(np.asarray(self.plan[seq]["qpos"], float), self.G["qadr"])

    def rebuild(self, body_cover):
        from AG3S.ag3s.stages.geometry import fit_sphere
        from AG3S.ag3s.types import TargetGeometry

        a, FIA = self.attach_seq, self.G["FIA"]
        g = self.summ[a]["grasp"]
        f = int(re.match(r"frame (\d+)", self.recorded["snapshot_state"]).group(1))
        with np.load(self.npz[a], allow_pickle=True) as z:
            pts = np.asarray(z["target_points"], float).reshape(-1, 3)
        with np.load(self.npz[a - 1], allow_pickle=True) as zp:
            field_prev = FIA.build_field(zp, self.G["outside"])
        ag = _ag3s(self.G["robot"], body_cover)
        ag._last_support_surfaces = [_surface(self.recorded["support"]["plane"])]  # noqa: SLF001
        ag._last_esdf_field = field_prev                                           # noqa: SLF001
        tg = TargetGeometry(id=0, points=pts, point_indices=np.arange(len(pts)),
                            centroid=pts.mean(0), bounding_geometry=fit_sphere(pts),
                            attention_score=0.9, confidence=0.9)
        att = ag.attach(tg, robot_state=self.q_now(a), parent_link=g["parent_link"],
                        geometry_state=self.q_now(f + 1))
        return ag, att

    def held_seqs(self):
        t_c = self.truth["apple_bar_contact_t"] if self.truth else 10 ** 9
        out = []
        for s in sorted(self.summ):
            if s < self.attach_seq:
                continue
            if (self.summ[s].get("grasp") or {}).get("state") != "held" \
                    or self.plan[s]["t_step"] > t_c:
                break
            out.append(s)
        return out

    def chunk(self, s):
        with np.load(self.npz[s], allow_pickle=True) as z:
            return (self.G["FIA"].build_field(z, self.G["outside"]),
                    np.asarray(z["refined_chunk"], float), np.asarray(z["reference_chunk"], float))

    def frame_attached(self, s, which):
        ag, att = self.off if which == "off" else self.on
        _, _, ref = self.chunk(s)
        lay = self.G["layout"]
        path = lay.full_q(lay.chunk_to_trajectory(ref)[:, :H], self.q_now(s)).T
        out = ag._frame_held_attached(                                          # noqa: SLF001
            self.q_now(s), [_surface(self.recorded["support"]["plane"])], path)
        return out, (None if which == "off" else ag.held_record["body_cover"]["frame"])

    def held_rows(self, s, att):
        """(H,) min over held rows of the TO's ESDF clearance along the recorded refined plan (m)."""
        field, fin, _ = self.chunk(s)
        G, lin = self.G, self.G["lin"]
        traj = G["layout"].chunk_to_trajectory(fin)[:, :H]
        sp = held_spheres_of(att)
        local = np.array([c for c, _ in sp])
        radii = np.array([r for _, r in sp])
        lin.set_attached(local, att.parent_link, radii)
        try:
            scene = G["FIA"].make_scene(field, G["robot"].radii, G["settings"].esdf_margin)
            scene = dataclasses.replace(scene, attached_points=local,
                                        attached_parent_link=att.parent_link, attached_radii=radii)
            cl = lin.esdf_clearance(traj, self.q_now(s), scene)[..., 0]
        finally:
            lin.set_attached(None, None)
        return cl[:, lin.n_spheres:].min(axis=1)


_RUNS: dict = {}


def _run(rel) -> Run:
    if rel not in _RUNS:
        _RUNS[rel] = Run(rel)
    return _RUNS[rel]


@records
@pytest.mark.parametrize("rel", RUNS)
def test_off_rebuilds_the_recorded_attach_and_the_recorded_to_rows(rel):
    R = _run(rel)
    ag, att = R.off
    rec = ag.held_record
    assert np.allclose(rec["fit"]["centre_m"], R.recorded["fit"]["centre_m"], atol=1e-6)
    for got, want in zip(rec["spheres_parent_frame"], R.recorded["spheres_parent_frame"], strict=True):
        assert np.allclose(got["centre_m"], want["centre_m"], atol=2e-6)
        assert got["radius_mm"] == pytest.approx(want["radius_mm"], abs=2e-3)
    assert "body_cover" not in rec
    by_seq = {c["seq"]: c for c in R.truth["held_chunks"]}
    for s in R.held_seqs():
        fa, _ = R.frame_attached(s, "off")
        assert fa is att                                    # off: the same object, every frame
        rows = R.held_rows(s, fa)
        np.testing.assert_allclose(1e3 * rows, by_seq[s]["TO_view_held_clearance_mm_per_step"],
                                   atol=0.05)


def _apple_in_palm(R, s, parent):
    """MuJoCo apple mesh vertices at chunk `s`'s planning row, in the parent (palm) link frame."""
    import mujoco

    key = ("mj", R.G["xml"])
    if key not in _RUNS:
        m = mujoco.MjModel.from_xml_path(R.G["xml"])
        _RUNS[key] = (m, mujoco.MjData(m))
    m, d = _RUNS[key]
    d.qpos[:] = np.asarray(R.plan[s]["qpos"], float)
    mujoco.mj_kinematics(m, d)
    g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "apple_geom")
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
    did = int(m.geom_dataid[g])
    va, nv = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
    vw = d.geom_xpos[g] + m.mesh_vert[va:va + nv].astype(float) @ d.geom_xmat[g].reshape(3, 3).T
    vb = (vw - d.xpos[b]) @ d.xmat[b].reshape(3, 3)
    T = np.linalg.inv(R.G["robot"].link_pose(R.q_now(s), parent))
    return vb @ T[:3, :3].T + T[:3, 3]


def _gap(v, att):
    sp = held_spheres_of(att)
    return np.min(np.stack([np.linalg.norm(v - c, axis=1) - r for c, r in sp]), axis=0)


@records
@pytest.mark.parametrize("rel", RUNS)
def test_on_the_body_spheres_cover_the_apple_to_3_mm_until_the_contact(rel):
    R = _run(rel)
    _, att_off = R.off
    worst_on, worst_off = [], []
    for s in R.held_seqs():
        fa, frame = R.frame_attached(s, "on")
        v = _apple_in_palm(R, s, fa.parent_link)
        if not frame["near_support"]:          # on the table the lifted copy is used (J1)
            g = _gap(v, fa)
            worst_on.append(g.max())
            assert np.mean(g <= 0.0) >= 0.75
        worst_off.append(_gap(v, att_off).max())
    assert worst_on, "no held chunk away from the table"
    assert max(worst_on) <= 0.003, 1e3 * np.array(worst_on)
    assert max(worst_off) > 0.004                         # the T43 S finding (4.6–16.3 mm)


@records
@pytest.mark.parametrize("rel", RUNS)
def test_on_the_to_sees_the_bar_contact_in_the_contact_chunk(rel):
    R = _run(rel)
    t_c = R.truth["apple_bar_contact_t"]
    s_c = max(s for s in R.held_seqs() if R.plan[s]["t_step"] <= t_c)
    by_seq = {c["seq"]: c for c in R.truth["held_chunks"]}
    fa, frame = R.frame_attached(s_c, "on")
    assert frame["near_support"] is False                 # over the bar: no lift
    on = 1e3 * R.held_rows(s_c, fa)
    true = np.asarray(by_seq[s_c]["true_apple_bar_mm_per_step"], float)
    assert on.min() <= 0.0, (on, true)
    assert np.all(on[true <= 0.0] <= 0.0), (on, true)


@records
@pytest.mark.parametrize("rel", RUNS)
def test_on_grasp_and_lift_off_chunks_get_no_new_negative_held_row(rel):
    R = _run(rel)
    _, att_off = R.off
    n_near = 0
    for s in R.held_seqs():
        fa, frame = R.frame_attached(s, "on")
        if not frame["near_support"]:
            continue
        n_near += 1
        off = R.held_rows(s, att_off).min()
        on = R.held_rows(s, fa).min()
        assert on >= min(off, 0.0) - 1e-9, (s, 1e3 * off, 1e3 * on)
    assert n_near >= 1                                    # the attach chunk is on the table


RIGHT = "T42/1/runs/E3b/noobs/E3b_ep1834_s18341"            # right arm, apple, lifted and placed
_have_right = _have and (S_RAW / "chunk_index_t42.json").exists() and (V / RIGHT).exists()


@pytest.mark.skipif(not _have_right, reason="T42-1 right-arm record not present")
def test_the_right_hand_takes_the_same_path():
    key = ("right", RIGHT)
    if key not in _RUNS:
        _RUNS[key] = Run(RIGHT, "chunk_index_t42.json")
    R = _RUNS[key]
    ag_off, att_off = R.off
    ag_on, att_on = R.on
    assert R.summ[R.attach_seq]["grasp"]["parent_link"] == "ee_finger_r1"
    assert att_off.parent_link == att_on.parent_link == "ee_right"
    for got, want in zip(ag_off.held_record["spheres_parent_frame"],
                         R.recorded["spheres_parent_frame"], strict=True):
        assert np.allclose(got["centre_m"], want["centre_m"], atol=2e-6)
    bc = ag_on.held_record["body_cover"]
    assert bc["mode"] == "sphere" and bc["lift"]["max_lift_mm"] > 0.0
    pl = R.recorded["support"]["plane"]
    n, off = np.asarray(pl["normal"], float), float(pl["offset_m"])
    seqs = R.held_seqs()
    # this grasp latched after the lift-off (the attach chunk's body bottom is already 38.6 mm up):
    # every chunk's choice is the rule — near iff the body bottom (now or on the reference path)
    # is within band + hysteresis of the plane — and the J1 copy cleared the plane at the attach
    from AG3S.ag3s.robot_models.held_object import HELD_LIFT_HYSTERESIS_M

    for s in seqs:
        _, frame = R.frame_attached(s, "on")
        limit = bc["band_mm"] + HELD_LIFT_HYSTERESIS_M * 1e3
        assert frame["near_support"] is (frame["min_bottom_gap_mm"] <= limit + 1e-9)
    assert all(min(sp["bottom_above_plane_mm_before"]) < bc["band_mm"]
               for sp in bc["lift"]["spheres"] if sp["lift_mm"] > 0.0)
    # lifted off: the unlifted body covers the apple (MuJoCo mesh) to 3 mm, the recorded J1 spheres
    # do not
    away = [s for s in seqs if not R.frame_attached(s, "on")[1]["near_support"]]
    assert away
    c0 = _apple_in_palm(R, seqs[0], "ee_right").mean(axis=0)
    worst_on, worst_off = [], []
    for s in away:
        fa, _ = R.frame_attached(s, "on")
        v = _apple_in_palm(R, s, "ee_right")
        if np.linalg.norm(v.mean(axis=0) - c0) > 0.010:     # the gripper opened: the apple left
            continue                                        # the hand (seq 27–28 of this run)
        worst_on.append(_gap(v, fa).max())
        worst_off.append(_gap(v, att_off).max())
    assert len(worst_on) >= 8
    assert max(worst_on) <= 0.003, 1e3 * np.array(worst_on)
    assert max(worst_off) > 0.005                           # the recorded J1 spheres: 2.2 → 6.9 mm

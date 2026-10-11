"""V11 snapshot check (V8 check + TA flags: serve_safe CLI -> TrajOptConfig -> to_dict round trip, announce_ta log lines for all three arms): collision_backend after with_overrides + new flags survive AG3SConfig round trip + CLI -> clustering section."""
import sys, json, types
from common.workspace import PROJECT_ROOT
from pathlib import Path
sys.path.insert(0, str(PROJECT_ROOT))
import AG3S.ag3s.config as C
print("config file", C.__file__)
assert Path(C.__file__).resolve().is_relative_to(PROJECT_ROOT)
cls = C.AG3SConfig
base = cls.from_dict({"collision_backend": "esdf"})
print("from_dict esdf:", base.collision_backend)
o = base.with_overrides({"clustering": {"held_slip_detach": True, "held_cover_cap": 0.01, "hold_follow_to": 3, "subtask_gate": True}})
print("after with_overrides(held_slip_detach=True, held_cover_cap=0.01, hold_follow_to=3, subtask_gate=True): collision_backend", o.collision_backend)
d = o.to_dict()
print("to_dict keys has collision_backend:", "collision_backend" in d, d.get("collision_backend"))
cl = d["clustering"]
rt = cls.from_dict(d)
print("round trip: held_slip_detach", rt.clustering.held_slip_detach, "held_cover_cap", rt.clustering.held_cover_cap, "hold_follow_to", rt.clustering.hold_follow_to, "subtask_gate", rt.clustering.subtask_gate)
print("round trip equal to_dict:", rt.to_dict() == d, "| collision_backend after round trip:", rt.collision_backend)
assert rt.clustering.held_slip_detach is True and abs(rt.clustering.held_cover_cap - 0.01) < 1e-12 and rt.clustering.hold_follow_to == 3 and rt.clustering.subtask_gate is True
assert rt.collision_backend == "esdf" and o.collision_backend == "esdf"
# CLI -> clustering section
import AG3S.trajopt.serve_safe as ss
a = types.SimpleNamespace(closing_keep_geometry=True, latched_identity_hold=True, refused_identity_release=None, supported_max_bottom_mm=60.0,
    held_slip_detach=True, held_cover_cap_mm=10.0, hold_follow_to=3)
for k in ("held_free_observed_mm", "gripper_cover", "rest_start"): setattr(a, k, None)
sec = ss.t43t_identity_section(a)
print("t43t_identity_section:", json.dumps(sec, sort_keys=True))
print("subtask_config_section(True, None):", ss.subtask_config_section(True, None))
print("OK")

# ---- T43 TA flags: CLI -> TrajOptConfig -> round trip (V11A / V11AB / V11ABC) + announce_ta log text
import logging, io, dataclasses
from AG3S.trajopt.config import TrajOptConfig
V8 = ("--esdf-backend curobo --voxel 0.020 --fine-voxel 0.005 --tsdf-voxel 0.005 --links gripper --target-field-policy exclude-authorized "
      "--esdf-margin 0 --no-self-collision --capsule-radius-scale 0.05 --sphere-spacing 0.4 --max-spheres-per-capsule 32 "
      "--obstacle-margin 0.010 --held-body-cover --closing-keep-geometry --latched-identity-hold --supported-max-bottom-mm 60 "
      "--held-slip-detach --held-cover-cap-mm 10 --hold-follow-to 3 --finger-cover both").split()
TA_A = "--plan-horizon 16 --lookahead obstacles --lookahead-gate-mm 20 --qp-eps 1e-4".split()
TA_B = "--w-deflection-rate 20".split()
TA_C = "--servo-accel relaxed --servo-accel-tolerance-deg 1 --servo-accel-weight 0.01".split()
ap = ss.build_parser()
base_args = ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", str(PROJECT_ROOT / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")] + V8)
base_cfg = ss.trajopt_config_from_args(base_args)
print("V8 args servo_model:", getattr(base_args, "servo_model", None), "held_obstacle_margin:", getattr(base_args, "held_obstacle_margin", None))
for arm, extra in (("V11A", TA_A), ("V11AB", TA_A + TA_B), ("V11ABC", TA_A + TA_B + TA_C)):
    a = ap.parse_args(["--config", "x", "--checkpoint", "y", "--model-xml", str(PROJECT_ROOT / "robots/rby1/description/models/rby1a/mujoco/model_transport.xml")] + V8 + extra)
    cfg = ss.trajopt_config_from_args(a)
    d = cfg.to_dict()
    rt = TrajOptConfig.from_dict(d)
    ok_rt = (rt == cfg and rt.to_dict() == d and cfg.with_overrides({}) == cfg)
    h, c, q, l = cfg.horizon, cfg.cost, cfg.qp, cfg.limits
    print(f"{arm}: collision.backend={cfg.collision.backend} plan_horizon={h.plan_horizon} planned={h.planned} execution_length={h.execution_length} "
          f"lookahead={h.lookahead} gate={h.lookahead_gate} eps_abs={q.eps_abs} eps_rel={q.eps_rel} w_deflection_rate={c.w_deflection_rate} "
          f"servo_accel={l.servo_accel} tol_rad={l.servo_accel_tolerance} weight={l.servo_accel_weight} servo_model={l.servo_model} "
          f"held_obstacle_margin={cfg.collision.held_obstacle_margin} round_trip_equal={ok_rt}")
    assert ok_rt and cfg.collision.backend == "esdf" and h.planned == 16 and h.lookahead == "obstacles" and abs(h.lookahead_gate - 0.02) < 1e-12
    assert q.eps_abs == q.eps_rel == 1e-4 and not l.servo_model and not cfg.collision.held_obstacle_margin
    assert (c.w_deflection_rate == 20.0) == (arm != "V11A") and (l.servo_accel == "relaxed") == (arm == "V11ABC")
    # only the TA keys differ from the V8 config
    db = base_cfg.to_dict(); diff = {sec: sorted(k for k in d[sec] if d[sec][k] != db[sec][k]) for sec in d if any(d[sec][k] != db[sec][k] for k in d[sec])}
    print(f"{arm}: keys differing from the V8 config: {json.dumps(diff, sort_keys=True)}")
    buf = io.StringIO(); hd = logging.StreamHandler(buf); hd.setLevel(logging.DEBUG)
    root = logging.getLogger(); root.addHandler(hd); lv = root.level; root.setLevel(logging.DEBUG)
    ss.announce_ta(cfg); ss.announce_path_check(cfg)
    root.removeHandler(hd); root.setLevel(lv)
    print(f"{arm}: announce_ta lines:"); [print("   |", ln) for ln in buf.getvalue().splitlines()]
print("TA OK")

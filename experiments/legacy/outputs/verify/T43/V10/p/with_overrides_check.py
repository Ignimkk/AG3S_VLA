"""V10 snapshot check: collision_backend after with_overrides + new flags survive AG3SConfig round trip + CLI -> clustering section."""
import sys, json, types
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t43v10")
import AG3S.ag3s.config as C
print("config file", C.__file__)
assert C.__file__.startswith("${AG3S_PROJECT_ROOT}-t43v10/")
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
# --- T43 HM: TrajOptConfig round trip of collision.held_obstacle_margin
import AG3S.trajopt.config as TC
assert TC.__file__.startswith("${AG3S_PROJECT_ROOT}-t43v10/")
t = TC.TrajOptConfig.from_dict({"collision": {"obstacle_margin": 0.01, "held_obstacle_margin": True}})
td = t.to_dict()
t2 = TC.TrajOptConfig.from_dict(td)
print("TrajOptConfig round trip: held_obstacle_margin", t2.collision.held_obstacle_margin, "obstacle_margin", t2.collision.obstacle_margin, "equal to_dict:", t2.to_dict() == td)
assert t2.collision.held_obstacle_margin is True and t2.to_dict() == td
t0 = TC.TrajOptConfig.from_dict({}); print("default held_obstacle_margin:", t0.collision.held_obstacle_margin)
try:
    TC.TrajOptConfig.from_dict({"collision": {"held_obstacle_margin": True}}); print("ValueError NOT raised (unexpected)")
except ValueError as e:
    print("held_obstacle_margin without obstacle_margin>0 -> ValueError:", str(e)[:80])
print("OK")

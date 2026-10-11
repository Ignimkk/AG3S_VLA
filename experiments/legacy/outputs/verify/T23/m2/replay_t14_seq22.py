"""T23 측정 #2 — T14 seq 22 (chunk 21, no_seed) 를 SafePolicy._verdict (T23) 로 단위 재생.

manipulated 기록: T20 파이프라인(ground_target + TargetConfirm)을 T14 chunk 0..21 에 차례로 돌려
seq 22 시점의 `confirm.manipulated.record()` 를 얻는다 (T20 verify 의 replay_t14.py 와 같은 입력).
switch_min_score 0.0 (기본) · 0.05 두 경우. 대조: state 를 lost 로 바꾼 것, manipulated 없음.
result: 기록된 TO (status violated — 미인증으로 내린 것, clearance +13.55 mm → max_violation 0).
  time_budget_hit: 기록 notes "time budget 50 ms reached" → True (False 도 함께 돌린다).
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, types
from pathlib import Path
import numpy as np
sys.path.insert(0, _ag3s_workspace_path(''))
from AG3S.ag3s.config import AG3SConfig
from AG3S.ag3s.types import PointCloud, AttentionPointCloud, ConstraintValidity, ContactPolicyContext
from AG3S.ag3s.stages.target_grounding import ground_target, TargetConfirm
from AG3S.ag3s.constraints.clearance import ClearancePolicy
from AG3S.trajopt.safe_policy import SafePolicy
from AG3S.trajopt.config import TrajOptConfig
from AG3S.trajopt.types import TrajOptStatus

ROOT = Path(_ag3s_workspace_path(''))
REC = ROOT / "outputs/live_test/20260928_t14/server_constraints/run_0000"
OUT = ROOT / "outputs/verify/T23/m2"
cfg = AG3SConfig()


def manipulated_at_seq22(min_score):
    confirm = TargetConfirm(cfg.clustering, switch_min_score=min_score, lost_frames=None)
    rec = None
    for k in range(22):
        z = np.load(REC / f"chunk_{k:05d}.npz", allow_pickle=True)
        apc = AttentionPointCloud(PointCloud(z["attention_cloud"].astype(np.float64)), z["attention_values"],
                                  raw_attention=z["attention_raw"])
        g = ground_target(apc, cfg.clustering, seed_percentile=cfg.attention.seed_percentile,
                          seed_threshold=cfg.attention.seed_threshold, exclude_mask=z["support_mask"],
                          min_radius=cfg.geometry.min_radius, timestamp=float(z["t_step"]), confirm=confirm)
        if k == 21:
            rec = None if confirm.manipulated is None else confirm.manipulated.record()
            return rec, str(g.status.value)


z = np.load(REC / "chunk_00021.npz", allow_pickle=True)
s = json.loads(str(z["summary_json"]))
to = s["to"]; met = to["metrics"]
config = AG3SConfig.from_dict({"collision_backend": "esdf"})
cpolicy = ClearancePolicy.from_config(config.contact, config.geometry, config.support_surface)
ctx = ContactPolicyContext.make("approach", ["left"])


def verdict(man_rec, has_manip, budget_hit):
    result = types.SimpleNamespace(
        status=TrajOptStatus(to["status"]), max_violation=max(0.0, -float(met["clearance_m"])),
        trajectory=np.zeros((1, 8)), notes=list(to.get("notes", [])),
        metrics={"max_violation_pair": __import__("ast").literal_eval(met["max_violation_pair"]), "time_budget_hit": budget_hit,
                 "max_iterations_hit": False, "sqp_iterations": 1})
    cs = types.SimpleNamespace(
        status=types.SimpleNamespace(value=s["status"]),
        grounding_status=types.SimpleNamespace(value=s["grounding_status"]),
        validity=ConstraintValidity(s["validity"]),
        geometry_certified=ConstraintValidity(s["validity"]).certified,
        manipulated=(object() if has_manip else None), contact_context=ctx, candidates=[],
        metrics=({"manipulated": man_rec} if man_rec is not None else {}))
    sp = SafePolicy.__new__(SafePolicy)
    sp.to_config = TrajOptConfig(); sp.verdict_policy = "reasons"
    sp.allowed_contact_pad_m = 0.01; sp.allow_unresolved_contact_tier = False
    sp.ag3s = types.SimpleNamespace(builder=types.SimpleNamespace(clearance_policy=cpolicy), manipulated=None)
    sp.refiner = types.SimpleNamespace(last_result=result, optimizer=None)
    sp._last_constraint_set = cs; sp._last_snapshot = None; sp._last_q_now = None
    v = SafePolicy._verdict(sp, None)
    return {"safe": bool(v.safe), "kinds": [r["kind"] for r in v.verdict_reasons],
            "reasons": [{k: r.get(k) for k in ("kind", "action", "detail")} for r in v.verdict_reasons],
            "trajopt_status": v.trajopt_status, "geometry_certified": v.geometry_certified,
            "ag3s_status": v.ag3s_status, "max_violation_m": v.max_violation_m,
            "notes_tail": v.notes[-1:]}


out = {"record": {"chunk": 21, "seq": 22, "ag3s_status": s["status"], "validity": s["validity"],
                  "grounding_status": s["grounding_status"], "has_target": s["has_target"],
                  "to_status": to["status"], "clearance_m": met["clearance_m"], "to_notes": to.get("notes")},
       "cases": []}
for ms in (0.0, 0.05):
    man, gstat = manipulated_at_seq22(ms)
    for budget in (True, False):
        out["cases"].append({"case": f"T20 replay switch_min_score={ms}, manipulated as replayed", "time_budget_hit": budget,
                             "replay_grounding_status": gstat, "manipulated": man,
                             **verdict(man, man is not None, budget)})
    lost = dict(man); lost["state"] = "lost"
    out["cases"].append({"case": f"control: same record with state=lost (min_score={ms})", "time_budget_hit": True,
                         "manipulated": lost, **verdict(lost, False, True)})
out["cases"].append({"case": "control: no manipulated record (T23 before T20)", "time_budget_hit": True,
                     "manipulated": None, **verdict(None, False, True)})
(OUT / "t14_seq22_verdict.json").write_text(json.dumps(out, indent=1, default=str, ensure_ascii=False))
for c in out["cases"]:
    print(c["case"], "budget", c["time_budget_hit"], "->", c["kinds"], "safe", c["safe"],
          (c["manipulated"] or {}).get("state"), (c["manipulated"] or {}).get("id"), (c["manipulated"] or {}).get("age_frames"))

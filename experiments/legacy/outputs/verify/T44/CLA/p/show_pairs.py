from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys
d = json.load(open(_ag3s_workspace_path('outputs/verify/T44/CLA/diverge_C1a_C1b.json')))
def L(name): return {r["seq"]: r for r in map(json.loads, open(f"{_ag3s_workspace_path()}/outputs/verify/T44/CLA/chunks/{name}.jsonl"))}
for k, r in d.items():
    A = L("C1a_" + k); B = L("C1b_" + k)
    print("==", k, r["failure_mode"], r["grasp_t"])
    for seq in range(12, 22):
        for tag, X in (("A", A), ("B", B)):
            x = X.get(seq)
            if not x: continue
            f = x.get("fk") or {}; m = x.get("mvp") or {}; cw = x.get("collision_worst") or {}
            rs = ";".join(f"{q['kind']}:{q['detail'][:70]}" for q in x["reasons"] if q["action"] != "execute")
            print(f"  {seq:2d}{tag} h{int(x['hold'])} {x['to_status'][:4]} {x['grasp_state'][:6]:6s} mvp {str(m.get('tier'))[:9]:9s} {str(m.get('link'))[-7:]:7s} {1e3*(m.get('clearance_m') or 0):6.1f} "
                  f"dref7 {f.get('d_mid_target_ref_mm',[0]*8)[7]:6.1f} push7 {f.get('push_mm',[0]*8)[7]:6.1f} disp {max(f.get('disp_hand_mm',[0])[:8]):5.1f} gT {f.get('gap_target_ref_mm',0):6.1f}/{f.get('gap_target_fin_mm',0):6.1f} gO {f.get('gap_other_fin_mm',0):6.1f} "
                  f"tfc_now {(x.get('win_by_tier_now') or {}).get('target_free_coarse')} grip {f.get('grip_ref_rows0_7',[None])[-1]} | {rs}")

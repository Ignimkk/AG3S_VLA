"""T43 T — where is the surface the q_now collision rows see? (record field, gradient step to the surface)"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, pathlib
import numpy as np
sys.argv += []
from AG3S.trajopt.experiments import fixed_input_ablation as FIA
S = pathlib.Path(_ag3s_workspace_path('outputs/verify/T43/S/raw'))
IX = {**json.load(open(S / "chunk_index_t42.json"))}
RUNS = {"T42_R1908": _ag3s_workspace_path('outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1908_s19082'),
        "T42_L1992": _ag3s_workspace_path('outputs/verify/T42/1/runs/E3b/noobs/E3b_ep1992_s19922')}
out = {}
for key, seq in (("T42_R1908", 16), ("T42_R1908", 24), ("T42_L1992", 16), ("T42_L1992", 24)):
    rd = RUNS[key]; f = IX[rd]["chunks"][str(seq)]["npz"]
    q = json.load(open(f"{_ag3s_workspace_path()}/outputs/impl/T43T/qnow_{key[4:]}.json"))[str(seq)]
    plan = [json.loads(l) for l in open(pathlib.Path(rd) / "frames.jsonl") if '"planning"' in l[:40]]
    poses = [p for p in plan if p["seq"] == seq][0]["object_poses"]
    man = json.load(open(pathlib.Path(rd) / "manifest.json"))["scene"]["episode_target_fruit"]
    with np.load(f) as z:
        field = FIA.build_field(z, 0.5)
        fine = FIA.build_field(type("Z", (), {"files": [k for k in z.files if not k.startswith("esdf_coarse")], "__getitem__": lambda s, k: z[k]})(), 0.5)
    rows = [r for r in q["stay"]["rows"] if r["class"] != "allowed" and r["step"] == 0]
    res = []
    for r in rows:
        p = np.asarray(r["p"], float)
        d = float(field.distance(p[None])[0]); g = np.asarray(field.gradient(p[None]), float)[0]
        g = g / (np.linalg.norm(g) + 1e-12); surf = p - d * g
        tier = str(np.asarray(field.answer_tier(p[None], target_free=np.zeros(1, bool))).reshape(-1)[0])
        res.append({"link": r["link"], "why": r["why"], "cl_mm": r["cl_mm"], "tier": tier, "d_field_mm": round(1e3 * d, 2),
                    "d_fine_only_mm": round(1e3 * float(fine.distance(p[None])[0]), 2),
                    "surface_p": [round(v, 4) for v in surf], "surface_above_table_mm": round(1e3 * (surf[2] - q["plane"][1]), 1),
                    "surface_to_fruit_centre_mm": round(1e3 * float(np.linalg.norm(surf - np.asarray(poses[man]["pos"]))), 1),
                    "point_to_fruit_centre_mm": round(1e3 * float(np.linalg.norm(p - np.asarray(poses[man]["pos"]))), 1)})
    out[f"{key}:{seq}"] = {"fruit": man, "fruit_pos": poses[man]["pos"], "held_world": q["held_world"], "held_r_mm": q["held_r_mm"], "rows": res}
    print(key, seq, man, json.dumps(res[:1]), json.dumps(res[-1:]))
pathlib.Path(_ag3s_workspace_path('outputs/impl/T43T/probe.json')).write_text(json.dumps(out, indent=1))

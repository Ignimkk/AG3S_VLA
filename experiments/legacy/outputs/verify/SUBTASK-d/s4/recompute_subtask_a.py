"""SUBTASK-d §4 — recompute the SUBTASK (a) numbers (2026-09-30 text-decode / score probe) from the raw json in
outputs/verify/subtask_probe/ and compare with the figure sidecars (figures/subtask/*.json), auroc.json and the audit text.
Definitions follow outputs/verify/subtask_probe/subtask_figs.py (metrics(), pooled(), auroc(), fig_discover()) and are
re-implemented here, not imported. CPU only, no model. usage: python recompute_subtask_a.py -> s4/subtask_a_recomputed.json"""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, pathlib
import numpy as np
SP = pathlib.Path(_ag3s_workspace_path('outputs/verify/subtask_probe'))
FIG = pathlib.Path(_ag3s_workspace_path('AG3S/ag3s/docs/figures/subtask'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/SUBTASK-d/s4/subtask_a_recomputed.json'))
E = json.load(open(SP / "episode.json")); DISC = json.load(open(SP / "discover.json")); SMOKE = json.load(open(SP / "smoke.json"))
V = ["base", "ft16d", "ft_nolora"]; EXPECT = {"reach": "pick", "held": "place", "released": "home"}
FAST_LO, FAST_HI = 254976, 257023          # audit: FAST region ids 254,976-257,023 (2048)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


res = {"inputs": {k: str(SP / k) for k in ("episode.json", "discover.json", "smoke.json", "auroc.json")}}
# --- frames / phases
res["frames"] = {rk: {"n": len(r["frames"]), "phase_counts": {p: sum(f["gt"]["phase"] == p for f in r["frames"]) for p in EXPECT}}
                 for rk, r in E["runs"].items()}
res["frames_total"] = sum(v["n"] for v in res["frames"].values())
# --- AUROC
au = {}
for mk in V:
    for rk, r in E["runs"].items():
        fr = r["frames"]; ph = [f["gt"]["phase"] for f in fr]; s = [f["score"][mk] for f in fr]
        d = [x["place"] - x["pick"] for x in s]; h = [x["home"] - max(x["pick"], x["place"]) for x in s]
        au.setdefault(mk, {})[rk] = {
            "held_vs_reach": auroc([x for x, p in zip(d, ph) if p == "held"], [x for x, p in zip(d, ph) if p == "reach"]),
            "released_vs_rest": auroc([x for x, p in zip(h, ph) if p == "released"], [x for x, p in zip(h, ph) if p != "released"]),
            "argmax_counts": {c: sum(max(x, key=x.get) == c for x in s) for c in ("pick", "place", "home")}}
res["auroc"] = au
res["argmax_home_all_frames"] = {mk: sum(au[mk][rk]["argmax_counts"]["home"] for rk in E["runs"]) for mk in V}
res["majority_released_frames"] = sum(v["phase_counts"]["released"] for v in res["frames"].values())
# --- generation
gen = {}
for mk in V:
    for tk in E["gen_templates"]:
        n = acc = act = 0; obj_n = obj_ok = 0; held_act = {}; held_n = {}
        for rk, r in E["runs"].items():
            for f in r["frames"]:
                g = f["gen"][mk][tk]; ph = f["gt"]["phase"]; n += 1
                acc += g["cls"] == EXPECT[ph]; act += g["cls"] == "action_tokens"
                if g["cls"] in ("pick", "place"):
                    obj_n += 1; obj_ok += r["fruit"] in g["text"].lower()
                if ph == "held":
                    held_n[rk] = held_n.get(rk, 0) + 1; held_act[rk] = held_act.get(rk, 0) + (g["cls"] == "action_tokens")
        gen.setdefault(mk, {})[tk] = {"n": n, "accuracy": acc / n, "n_correct": acc, "action_token_rate": act / n, "n_action_tokens": act,
                                     "object_name_ok": (obj_ok / obj_n) if obj_n else None, "object_name_n": obj_n, "object_name_ok_n": obj_ok,
                                     "held_action_tokens_by_run": {rk: [held_act[rk], held_n[rk]] for rk in held_n}}
res["generation"] = gen
res["generation_templates"] = E["gen_templates"]
res["latency_ms_per_token"] = E["timing"]
# --- FAST share (smoke.json, ep1807 2 frames): templates whose prompt contains 'State:' (openpi_action, state_subtask)
fast = {}
for r in SMOKE["records"]:
    for mk, g in r["gen"].items():
        for tk, o in g.items():
            ids = o.get("ids", []); k = f"{mk}|{tk}"
            fast.setdefault(k, [0, 0]); fast[k][0] += sum(FAST_LO <= i <= FAST_HI for i in ids); fast[k][1] += len(ids)
state_keys = [k for k in fast if k.split("|")[1] in ("openpi_action", "state_subtask")]
res["fast_tokens"] = {"per_model_template": {k: {"n_fast": v[0], "n_tok": v[1]} for k, v in sorted(fast.items())},
                      "state_templates_pooled": {"templates": {t: SMOKE["templates"][t] for t in ("openpi_action", "state_subtask")},
                                                 "n_fast": sum(fast[k][0] for k in state_keys), "n_tok": sum(fast[k][1] for k in state_keys)},
                      "smoke_frames_t_step": [r["t_step"] for r in SMOKE["records"]], "models": SMOKE["models"], "fast_id_range": [FAST_LO, FAST_HI]}
p = res["fast_tokens"]["state_templates_pooled"]; p["share"] = p["n_fast"] / p["n_tok"]
oa = [k for k in fast if k.endswith("|openpi_action")]
res["fast_tokens"]["openpi_action_pooled"] = {"n_fast": sum(fast[k][0] for k in oa), "n_tok": sum(fast[k][1] for k in oa)}
# --- template discovery
disc = {}
for r in DISC:
    t = r["text"]; eng = t.strip() != "" and sum(ord(c) > 127 for c in t) / max(1, len(t)) < 0.2
    cls = "english" if eng else ("empty" if t.strip() == "" else "non_english(action tokens)")
    key = r["prefix"] + (" (causal)" if r["causal"] else "")
    disc.setdefault(r["model"], {}).setdefault(key, {})[f"t={r['t_step']} {r['phase']}"] = {"class": cls, "text": t, "n_fast": r["n_fast"], "n_tok": r["n_tok"]}
res["template_discovery"] = disc
res["template_discovery_counts"] = {mk: {c: sum(v["class"] == c for pr in disc[mk].values() for v in pr.values())
                                         for c in ("english", "empty", "non_english(action tokens)")} for mk in disc}
res["template_discovery_n_templates"] = len({k for mk in disc for k in disc[mk]})
# --- cross-checks against sidecars
side = json.load(open(FIG / "subtask-accuracy-table.json")); au_old = json.load(open(SP / "auroc.json"))
diffs = []
for mk in V:
    for rk in E["runs"]:
        for k in ("held_vs_reach", "released_vs_rest"):
            a, b = au[mk][rk][k], side["auroc"][mk][rk][k]
            if a != b:
                diffs.append({"where": f"figure auroc {mk} {rk} {k}", "recomputed": a, "sidecar": b})
        a, b = au[mk][rk]["held_vs_reach"], au_old[rk][mk]["auroc_held_vs_reach"]
        if a != b:
            diffs.append({"where": f"auroc.json {rk} {mk} held_vs_reach", "recomputed": a, "auroc_json": b})
        a, b = au[mk][rk]["released_vs_rest"], au_old[rk][mk]["auroc_released"]
        if abs(a - b) > 0:
            diffs.append({"where": f"auroc.json {rk} {mk} released (auroc.json definition may differ)", "recomputed_released_vs_rest": a, "auroc_json": b})
for k, tk, field in (("gen_acc[p_nl]", "p_nl", "accuracy"), ("gen_acc[T.Sub_causal]", "T.Sub_causal", "accuracy"),
                     ("gen_action_tokens[T;nl]", "T;nl", "action_token_rate"), ("gen_object_ok[T;nl]", "T;nl", "object_name_ok")):
    for mk in V:
        a, b = gen[mk][tk][field], side["pooled_generation"][k][mk]
        if b is None or a is None or abs(a - b) > 1e-12:
            diffs.append({"where": f"figure pooled {k} {mk}", "recomputed": a, "sidecar": b})
res["diffs_vs_sidecars"] = diffs
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print("frames", res["frames_total"], "argmax home", res["argmax_home_all_frames"], "released", res["majority_released_frames"])
print("fast", p, res["fast_tokens"]["openpi_action_pooled"])
for mk in V:
    print(mk, {tk: (round(gen[mk][tk]["accuracy"], 4), round(gen[mk][tk]["action_token_rate"], 4), gen[mk][tk]["object_name_ok"] and round(gen[mk][tk]["object_name_ok"], 4)) for tk in gen[mk]})
    print("   auroc", {rk: (round(au[mk][rk]["held_vs_reach"], 3), round(au[mk][rk]["released_vs_rest"], 3)) for rk in E["runs"]})
print("discovery", res["template_discovery_n_templates"], res["template_discovery_counts"])
print("diffs", json.dumps(diffs, indent=0)[:3000])

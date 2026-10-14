"""T60 step 3d — T47 label-free head selection (attention sum + spatial entropy) on the P / H windows.

`t47_head_selection.record_statistics` / `select_from_stats` unchanged, T47's arguments (kneedle τ,
n_lowest 10, kmax 5, L0 · L1 excluded); only the selection records' frames after completion are dropped
(T60 record masks; of `long` · `t2` · `ep1807` only ep1807 reaches P). The full window must reproduce the
published `select/selection.json` top-k of every setting before a P / H result is written.

usage: t60_t47.py MASK_DIR OUT_JSON
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from AG3S.ag3s.experiments.studies import t47_head_selection as hs  # noqa: E402
from AG3S.ag3s.experiments.studies import t60_termination as T  # noqa: E402

T47_SELECT = hs.storage_root() / "results/development/20261012T141412+0900_bd778e5e3453_d7a51b456f36/verify/T47/select/selection.json"


def main(mask_dir: pathlib.Path, out_json: pathlib.Path) -> None:
    pub = json.loads(T47_SELECT.read_text())
    a = pub["args"]
    keys = list(hs.SELECTION_KEYS)
    atts = {}
    for k in keys:
        att, dn, ag, _ = hs.load_attention(k)
        atts[k] = att
    res = {"protocol": T.PROTOCOL_VERSION, "source": str(T47_SELECT), "keys": keys, "windows": {}}
    for w in ("full", "P", "H"):
        stats, n_frames = {}, {}
        for k in keys:
            att = atts[k]
            if w != "full":
                with np.load(mask_dir / f"{k}.npz") as z:
                    m = z[f"keep_frames_{w}"].astype(bool)
                assert len(m) == att.shape[0], (k, len(m), att.shape)
                att = att[m]
            n_frames[k] = int(att.shape[0])
            s_cam, ent, ncomp = hs.record_statistics(att)
            stats[k] = {"S_cam": s_cam, "entropy": ent, "ncomp": ncomp}
        n_layers, n_heads = stats[keys[0]]["S_cam"].shape[3:5]
        cells = hs.cell_list(n_layers, n_heads)
        out = {"n_frames": n_frames, "settings": {}}
        for di, d in enumerate(dn):
            for ai, agg in enumerate(ag):
                name = hs.setting_name(d, agg)
                block = {}
                for src in ("cam", "img"):
                    r = hs.select_from_stats(stats, keys, di, ai, cells, tau_source=src,
                                             tau_method=a["tau_method"], n_lowest=a["n_lowest"], kmax=a["kmax"])
                    block[src] = {"common_topk": r["common"]["topk"],
                                  "common_top10": r["common"]["ranking"][:10],
                                  "per_camera_top3": {c: r["per_camera"][c]["topk"]["3"] for c in hs.POLICY_CAMERAS}}
                    if w == "full":
                        p = pub["settings"][name]["tau_source"][src]
                        assert r["common"]["topk"] == p["common"]["topk"], (name, src)
                        for c in hs.POLICY_CAMERAS:
                            assert r["per_camera"][c]["topk"] == p["per_camera"][c]["topk"], (name, src, c)
                out["settings"][name] = block
        res["windows"][w] = out
        print(w, n_frames, {n: s["cam"]["common_topk"]["3"] for n, s in out["settings"].items() if n.startswith("d0")})
    for w in ("P", "H"):
        res[f"changed_vs_full_{w}"] = {
            f"{n}|{src}": {"common_top3": [res["windows"]["full"]["settings"][n][src]["common_topk"]["3"],
                                           res["windows"][w]["settings"][n][src]["common_topk"]["3"]]}
            for n in res["windows"]["full"]["settings"] for src in ("cam", "img")
            if res["windows"]["full"]["settings"][n][src]["common_topk"]["3"] != res["windows"][w]["settings"][n][src]["common_topk"]["3"]}
    T.write_json(out_json, res)
    print("changed P", res["changed_vs_full_P"]); print("changed H", res["changed_vs_full_H"])


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))

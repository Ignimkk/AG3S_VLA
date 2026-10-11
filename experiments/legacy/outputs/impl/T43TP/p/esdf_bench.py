"""T43 TP — ESDF read cost per call: host EsdfField vs DeviceEsdfField (GPU), and the D2H copy of the
value grids (what a per-chunk host mirror would pay once). One recorded V8 carry chunk."""
import json, time, argparse
import numpy as np, torch
from AG3S.trajopt.experiments import ta_offline as TA
from AG3S.trajopt.experiments import tp_timing as TP
key = "V8/H/V8_ep1807_s18071"
entry = json.load(open("outputs/verify/T43/V8/analysis/analysis.json"))["runs"][key]
run = TA.Run(key, entry, argparse.Namespace(hm=False, no_plane_rows=False))
inp, _ = run.inputs(28)
host_scene, _ = TP.prepare_field(inp["scene"], "cpu")
dev_scene, _ = TP.prepare_field(inp["scene"], "cuda")
hf = host_scene.esdf._field if hasattr(host_scene.esdf, "_field") else host_scene.esdf
df = dev_scene.esdf._field if hasattr(dev_scene.esdf, "_field") else dev_scene.esdf
from AG3S.trajopt.linearize import CollisionLinearizer
lin = CollisionLinearizer(run.robot, run.layout, 16)
ref = run.layout.chunk_to_trajectory(inp["ref_chunk"])[:, :16]
pts16 = lin.sphere_states(ref, inp["q_now"])[0].reshape(-1, 3)
def t(fn, n=40):
    fn(); torch.cuda.synchronize(); ts = []
    for _ in range(n):
        a = time.perf_counter(); fn(); ts.append((time.perf_counter() - a) * 1e3)
    return float(np.median(ts))
out = {}
for N in (100, 1000, len(pts16) // 2, len(pts16)):
    p = pts16[:N]
    out[N] = {"host_distance": t(lambda: hf.distance(p)), "device_distance": t(lambda: df.distance(p)),
              "host_gradient": t(lambda: hf.gradient(p)), "device_gradient": t(lambda: df.gradient(p)),
              "host_label": t(lambda: hf.label(p)), "device_label": t(lambda: df.label(p)),
              "host_distance_and_label": t(lambda: hf.distance_and_label(p, "x")),
              "device_distance_and_label": t(lambda: df.distance_and_label(p, "x"))}
    assert np.array_equal(hf.distance(p), df.distance(p))
layers = list(df.layers) + list(df.target_free_layers)
d2h = t(lambda: [l.host_distance_grid(cache=False) for l in layers], n=10)
sizes = [tuple(l.grid.shape) for l in layers]
res = {"n_points_H16": len(pts16), "per_call_ms": out, "d2h_all_layers_ms": d2h, "layer_shapes": sizes}
print(json.dumps(res, indent=1))
json.dump(res, open("outputs/impl/T43TP/prof/esdf_bench.json", "w"), indent=1)

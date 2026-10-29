from common.workspace import workspace_path as _ag3s_workspace_path
import sys, numpy as np, torch
sys.path.insert(0, sys.argv[1])
import curobo_builder_head as H
from AG3S.ag3s.fields.curobo_builder import CuroboFieldBuilder, FineWindowRequest, TargetBall
sys.path.insert(0, _ag3s_workspace_path(''))
import tests.o4.test_gpu_esdf_field as G
cams = G._scene_cameras()
apple = np.random.default_rng(0).normal([0.51, -0.03, 0.85], 0.012, (300, 3))
other = np.random.default_rng(3).normal([0.51, 0.03, 0.85], 0.012, (300, 3))
kw = dict(target_points=apple, target_free_points=apple,
          target_free_ball=TargetBall(centre=[0.51, -0.03, 0.85], radius=0.045, z_min=0.825),
          labelled_points={"apple": apple},
          fine_window=FineWindowRequest(manipulated_points=apple, placement="target_centroid"))
def vals(f):
    out = [l.values_device.detach().cpu().numpy() for l in f.layers]
    out += [l.values_device.detach().cpu().numpy() for l in f.target_free_layers]
    return out
res = {}
for name, cls, cfg in (("head", H.CuroboFieldBuilder, G._config()), ("new_ball", CuroboFieldBuilder, G._config()),
                       ("new_owned", CuroboFieldBuilder, G._config(target_exclusion="ball_owned"))):
    b = cls(cfg)
    for _ in range(2):
        extra = {"nontarget_points": other} if name == "new_owned" else {}
        f = b.update(cams, **kw, **extra)
    res[name] = (vals(f), f)
h, nb, no = res["head"][0], res["new_ball"][0], res["new_owned"][0]
print("layers", len(h), len(nb))
print("ball vs HEAD bit-identical per layer:", [bool(np.array_equal(a, b)) for a, b in zip(h, nb)])
print("owned main layers == HEAD:", [bool(np.array_equal(a, b)) for a, b in zip(h[:-1], no[:-1])], " free differs:", not np.array_equal(h[-1], no[-1]))
tf = res["new_ball"][1].stats["target_free"]; print({k: tf[k] for k in ("exclusion_method","n_seeds_in_ball","n_removed","n_kept_nontarget","n_ambiguous","n_ball_seeds_excluded")})
tf = res["new_owned"][1].stats["target_free"]; print({k: tf.get(k) for k in ("exclusion_method","n_seeds_in_ball","n_removed","n_kept_nontarget","n_ambiguous","n_unowned_removed","classify_ms")})
tr = res["new_owned"][1].exclusion_trace; print({k: (v.shape if hasattr(v,'shape') else v) for k, v in tr.items()})

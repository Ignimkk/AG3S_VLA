"""T44dA verifier: rebuild the recorded chunk field (CPU numpy) from a CL server_constraints chunk npz.
layers = (main coarse 20 mm `esdf_coarse_*`, main fine 5 mm `esdf_*`), target_free_layers = (`esdf_free_*`),
target_free_coarse_layer = `esdf_free_coarse_*` (only if present / C1b, C2). Grids are the recorded float16 values.
Query: CuroboEsdfField.target_free_distance / answer_tier (snapshot code, read-only) with coarse ON (as served) and with
coarse OFF (same object, target_free_coarse_layer=None -> T44 C1a formula on the same fine layers)."""
import sys
sys.path.insert(0, "${AG3S_PROJECT_ROOT}-t44cl")
import numpy as np
from AG3S.ag3s.fields.curobo_field import CuroboEsdfField, layer_from_arrays

def _layer(z, key):
    return layer_from_arrays(np.asarray(z[key + "_distance"], np.float32), z[key + "_origin"], float(z[key + "_voxel_size"]))

def build(z, coarse=True, outside=0.0):
    has_c = "esdf_free_coarse_distance" in z.files and coarse
    has_f = "esdf_free_distance" in z.files
    return CuroboEsdfField((_layer(z, "esdf_coarse"), _layer(z, "esdf")), outside_distance=outside,
                           target_free_layers=((_layer(z, "esdf_free"),) if has_f else ()),
                           target_free_coarse_layer=(_layer(z, "esdf_free_coarse") if has_c else None))

def query(z, pts, tf_mask):
    """returns dict: d_main, d_tf_on, tier_on, d_tf_off, tier_off  (tf rows only differ)."""
    on = build(z, True); off = build(z, False)
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    m = np.broadcast_to(np.asarray(tf_mask, bool), (len(pts),))
    don, ton, _ = on._answer(pts, m, want_gradient=False)
    doff, toff, _ = off._answer(pts, m, want_gradient=False)
    return {"d_main": on.distance(pts), "d_on": don, "tier_on": ton, "d_off": doff, "tier_off": toff}

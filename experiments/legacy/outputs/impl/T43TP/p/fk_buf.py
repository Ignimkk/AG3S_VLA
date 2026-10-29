import json, time
import numpy as np, casadi as ca, argparse
from AG3S.trajopt.experiments import ta_offline as TA
key = "V8/H/V8_ep1807_s18071"
entry = json.load(open("outputs/verify/T43/V8/analysis/analysis.json"))["runs"][key]
run = TA.Run(key, entry, argparse.Namespace(hm=False, no_plane_rows=False))
from AG3S.trajopt.linearize import CollisionLinearizer
inp, _ = run.inputs(25)
q_now = inp["q_now"]; ref = run.layout.chunk_to_trajectory(inp["ref_chunk"])
H = 16
lin = CollisionLinearizer(run.robot, run.layout, H)
traj = ref[:, :H]; full = run.layout.full_q(traj, q_now)
args = lin._fk_args(full)
fn = lin._fk_map
print(fn.n_in(), [fn.sparsity_in(i).shape for i in range(fn.n_in())], [fn.sparsity_out(i).shape for i in range(fn.n_out())], fn.nnz_out(1))
buf, f_eval = fn.buffer()
ins = [np.asfortranarray(a, dtype=np.float64) for a in args]
c_out = np.zeros(fn.nnz_out(0)); j_out = np.zeros(fn.nnz_out(1))
for i, a in enumerate(ins):
    buf.set_arg(i, memoryview(np.ascontiguousarray(a.T).reshape(-1)) if False else memoryview(a.reshape(-1, order="F")))
buf.set_res(0, memoryview(c_out)); buf.set_res(1, memoryview(j_out))
def run_buf():
    for i, a in enumerate(args):
        flat = np.asarray(a, np.float64).reshape(-1, order="F").copy()
        ins[i] = flat
        buf.set_arg(i, memoryview(flat))
    f_eval()
    return c_out, j_out
run_buf()
cd, jd = fn(*args)
ok_c = np.array_equal(np.asarray(cd).reshape(-1, order="F"), c_out)
ok_j = np.array_equal(np.asarray(jd.nonzeros()), j_out)
def t(f, n=50):
    f(); ts=[]
    for _ in range(n):
        a=time.perf_counter(); f(); ts.append((time.perf_counter()-a)*1e3)
    return np.median(ts)
print("bit-identical centres", ok_c, "jac", ok_j)
print("call+nonzeros list %.2f ms" % t(lambda: np.asarray(fn(*args)[1].nonzeros())))
print("buffer %.2f ms" % t(run_buf))
print("scatter only %.2f ms" % t(lambda: np.zeros(int(np.prod(lin._jac_shape))).__setitem__(lin._jac_dest, j_out)))

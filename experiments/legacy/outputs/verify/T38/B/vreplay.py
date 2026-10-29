"""T38 Phase B verifier replay driver (verifier's own capture; inputs from the implementer's harness).

Inputs only are reproduced with replay_b4.py (`build_server`, `load_requests`, `StubPolicy`): the same
SafePolicy construction as serve_safe with the E3b flags and the re-rendered wire requests in
outputs/verify/T38/frames/<run>/. Nothing of the harness's instrumentation (stage patches, digests, census)
is installed. Everything recorded here is computed by this file.

env:
  VR_CODE   code root (contains benchmark/)       -- also PYTHONPATH
  VR_OUT    output dir
  VR_REC    bg | sync | none   (recorder: background writer / synchronous writer / no recorder in the pass)
  VR_HEAVY  1 = capture field arrays hashes, masks, fusion, manipulated (V1); 0 = actions + timing only (V2/V4)
  VR_CPU    1 = set config.RECON_DEVICE / SUPPORT_RANSAC_DEVICE / SELF_FILTER_DEVICE = "cpu" (V5)
  VR_KEEP   comma list of seqs whose raw per-camera masks / kept uv / depth are saved (figure 4)
usage: vreplay.py <run> <n> [--warm 4]
"""
import argparse, hashlib, importlib.util, json, os, pathlib, subprocess, sys, threading, time

import numpy as np

CODE = os.environ["VR_CODE"]
OUT = pathlib.Path(os.environ["VR_OUT"]); OUT.mkdir(parents=True, exist_ok=True)
REC_MODE = os.environ.get("VR_REC", "sync")
HEAVY = os.environ.get("VR_HEAVY", "1") == "1"
CPU_FALLBACK = os.environ.get("VR_CPU", "0") == "1"
KEEP = {int(x) for x in os.environ.get("VR_KEEP", "").split(",") if x.strip()}
HARNESS = "/tmp/claude-0/-mnt-dev-work/769cea2f-2a69-4fbe-beed-8be40017af85/scratchpad/t38b4/replay_b4.py"

os.environ["T38B_CODE_ROOT"] = CODE
os.environ["T38B_OUT"] = str(OUT / "_harness_unused")
os.environ["T38B_RECORDER_BACKGROUND"] = "1" if REC_MODE == "bg" else "0"
import benchmark  # noqa: E402
assert benchmark.__file__.startswith(CODE + "/"), benchmark.__file__
import torch  # noqa: E402

spec = importlib.util.spec_from_file_location("replay_b4_inputs", HARNESS)
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


def sha(a):
    if a is None:
        return None
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(a.tobytes()).hexdigest() + f"|{a.shape}|{a.dtype}"


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float):
        return {"f": o.hex()}
    if isinstance(o, (np.floating,)):
        return {"f": float(o).hex()}
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return {"sha": sha(o)}
    if isinstance(o, (str, int, bool)) or o is None:
        return o
    if hasattr(o, "value") and isinstance(getattr(o, "value"), (str, int)):
        return str(o.value)
    return repr(o)


# ------------------------------------------------------------------ counters (no behaviour change)
COUNT = {}


def count(mod, name):
    if not hasattr(mod, name):
        return
    fn = getattr(mod, name)

    def w(*a, **k):
        COUNT[name] = COUNT.get(name, 0) + 1
        return fn(*a, **k)
    setattr(mod, name, w)


CAPTURE = {}


def install_capture():
    from AG3S.ag3s.runtime import pipeline as P
    real = P.AG3S._build_esdf

    def build(self, cameras, *a, **k):
        CAPTURE["cameras"] = list(cameras)
        return real(self, cameras, *a, **k)
    P.AG3S._build_esdf = build


class Sampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True); self.path = path; self.stop = False; self.pid = os.getpid()
        self.max_ours = 0; self.max_others = 0; self.max_used = 0; self.seq = 0

    def run(self):
        with open(self.path, "w") as f:
            while not self.stop:
                try:
                    used, util = [int(x) for x in subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                              capture_output=True, text=True, timeout=10).stdout.strip().split(",")]
                    load1 = os.getloadavg()[0]
                    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True, timeout=10).stdout.strip()
                    ours = others = 0
                    for line in apps.splitlines():
                        p, m = [int(x) for x in line.split(",")]
                        if p == self.pid:
                            ours = m
                        else:
                            others += m
                    self.max_ours = max(self.max_ours, ours); self.max_others = max(self.max_others, others)
                    self.max_used = max(self.max_used, used)
                    f.write(json.dumps({"t": time.time(), "seq": self.seq, "used": used, "util": util, "load1": load1, "ours": ours, "others": others}) + "\n"); f.flush()
                except Exception as e:  # noqa: BLE001
                    f.write(json.dumps({"error": repr(e)}) + "\n")
                time.sleep(0.5)


def capture(sp, s, resp, keep):
    cs = sp._last_constraint_set
    ag = sp.ag3s
    rec = {"seq": s["seq"]}
    act = np.ascontiguousarray(np.asarray(resp["actions"]))
    rec["actions_sha"] = sha(act)
    rec["resp"] = jsonable({k: v for k, v in resp.items() if k != "actions"})
    rec["has_target"] = getattr(cs, "target", None) is not None
    rec["ag3s_status"] = None if cs is None else str(getattr(getattr(cs, "status", None), "value", None))
    rec["grasp"] = jsonable(sp._last_grasp_record)
    if HEAVY:
        man = getattr(ag, "manipulated", None)
        geo = getattr(man, "geometry", None) if man is not None else None
        gpts = getattr(geo, "points", geo)
        rec["manipulated"] = None if man is None else {
            "id": jsonable(getattr(man, "id", None)), "state": str(getattr(man, "state", None)),
            "centroid": sha(np.asarray(man.centroid, np.float64)) if getattr(man, "centroid", None) is not None else None,
            "geometry": sha(np.asarray(gpts, np.float64)) if gpts is not None else None}
        tgt = getattr(cs, "target", None)
        rec["target_points"] = None if tgt is None or getattr(tgt, "points", None) is None else sha(np.asarray(tgt.points))
        field = getattr(ag, "_last_esdf_field", None)
        fr = None
        if field is not None:
            fr = {"outside_distance": jsonable(getattr(field, "outside_distance", None)), "tiers": []}
            for grp in ("layers", "target_free_layers"):
                for i, L in enumerate(getattr(field, grp, ()) or ()):
                    g = L.grid
                    fr["tiers"].append({
                        "grp": grp, "i": i, "type": type(L).__name__,
                        "distance": sha(np.asarray(L.distance_grid)),
                        "label": sha(None if L.label_grid is None else np.asarray(L.label_grid)),
                        "origin": sha(np.asarray(g.origin, np.float64)), "voxel": jsonable(float(g.voxel_size)),
                        "shape": list(map(int, g.shape)), "max_distance": jsonable(float(L.max_distance)),
                        "outside": jsonable(L.outside_distance), "label_names": list(L.label_names)})
        rec["field"] = fr
        fus = (sp._last_debug or {}).get("fusion")
        if fus is not None:
            rec["fused"] = {k: sha(getattr(fus.cloud, k, None)) for k in ("points", "obs_offset", "obs_camera", "obs_uv",
                                                                          "obs_time", "obs_attention", "representative")}
            rec["fused"]["attention"] = sha(fus.attention)
            rec["per_camera"] = {r.camera_id.value: {"points": sha(r.cloud.points), "uv": sha(r.cloud.uv),
                                                     "attention": sha(r.attention), "n": int(len(r.cloud))}
                                 for r in fus.per_camera}
            if s["seq"] in keep:
                for r in fus.per_camera:
                    keep_arrays[f"seq{s['seq']:03d}/uv/{r.camera_id.value}"] = np.asarray(r.cloud.uv, np.int32)
                    keep_arrays[f"seq{s['seq']:03d}/pts/{r.camera_id.value}"] = np.asarray(r.cloud.points, np.float64)
        cams = CAPTURE.pop("cameras", None) or []
        rec["masks"] = {}
        for c in cams:
            mk = getattr(c, "robot_mask", None)
            if mk is None:
                rec["masks"][c.name] = None; continue
            mk = np.asarray(mk, bool)
            rec["masks"][c.name] = {"sha": sha(mk), "n": int(mk.sum())}
            masks[f"seq{s['seq']:03d}/{c.name}"] = np.packbits(mk)
            masks[f"seq{s['seq']:03d}/{c.name}/shape"] = np.asarray(mk.shape)
            if s["seq"] in keep:
                keep_arrays[f"seq{s['seq']:03d}/mask/{c.name}"] = mk
                keep_arrays[f"seq{s['seq']:03d}/depth/{c.name}"] = np.asarray(c.depth, np.float32)
    acts[f"seq{s['seq']:03d}"] = act
    return rec


acts, masks, keep_arrays = {}, {}, {}


def run_pass(tag, sp, stub, reqs, smp, f):
    torch.cuda.synchronize()
    for s, req, nxt in reqs:
        smp.seq = s["seq"]
        stub.next = nxt
        COUNT.clear()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        resp = sp.infer(dict(req))
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3
        cs = sp._last_constraint_set
        row = {"pass": tag, "wall_ms": wall,
               "timing_ms": dict((sp._pending or {}).get("timing") or {}),
               "profile_ms": dict(getattr(cs, "profile", {}) or {}) if cs is not None else None,
               "to_timing_ms": (getattr(sp.refiner.last_result, "metrics", None) or {}).get("timing_ms")
               if sp.refiner.last_result is not None else None,
               "torch_max_alloc_mib": torch.cuda.max_memory_allocated() / 2**20,
               "torch_max_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
               "torch_alloc_after_mib": torch.cuda.memory_allocated() / 2**20,
               "recorder_pending": getattr(getattr(sp, "recorder", None), "pending", None),
               "device_calls": dict(COUNT)}
        if tag == "T":
            row.update(capture(sp, s, resp, KEEP))
        else:
            row["seq"] = s["seq"]; row["actions_sha"] = sha(np.asarray(resp["actions"])); CAPTURE.pop("cameras", None)
        f.write(json.dumps(row) + "\n"); f.flush()
        pm = row["profile_ms"] or {}
        print(f"[{tag}] seq {s['seq']:2d} wall {wall:7.1f} total {row['timing_ms'].get('total', float('nan')):7.1f} "
              f"recon {pm.get('scene_reconstruction', 0):6.1f} ground {pm.get('target_grounding', 0):6.1f} "
              f"cgen {pm.get('constraint_generation', 0):6.1f} dev {row['device_calls']}", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("n", type=int)
    ap.add_argument("--warm", type=int, default=4)
    a = ap.parse_args()
    meta = {"run": a.run, "n": a.n, "warm": a.warm, "code": CODE, "rec": REC_MODE, "heavy": HEAVY, "cpu_fallback": CPU_FALLBACK,
            "pid": os.getpid(), "python": sys.executable, "torch": torch.__version__, "benchmark_file": benchmark.__file__,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "harness": HARNESS}
    (OUT / "nvidia_smi_pre.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    (OUT / "uptime_pre.txt").write_text(subprocess.run(["uptime"], capture_output=True, text=True).stdout)
    from AG3S.ag3s import config as CFG
    if CPU_FALLBACK:
        for k in ("RECON_DEVICE", "SUPPORT_RANSAC_DEVICE", "SELF_FILTER_DEVICE"):
            assert hasattr(CFG, k), k
            setattr(CFG, k, "cpu")
    meta["config_devices"] = {k: getattr(CFG, k, None) for k in ("RECON_DEVICE", "SUPPORT_RANSAC_DEVICE", "SELF_FILTER_DEVICE",
                                                                  "KDTREE_WORKERS", "ESDF_QUERY_HOST_ARITH_MAX_POINTS")}
    from AG3S.ag3s.stages import robot_filter as RF, support_surface as SS
    count(RF, "_sphere_mask_device"); count(SS, "_hypothesis_counts_device")
    try:
        from AG3S.ag3s.stages import device_recon as DR
        count(DR, "camera_front_end")
    except ImportError:
        pass
    install_capture()
    smp = Sampler(OUT / "gpu_samples.jsonl"); smp.start()
    t0 = time.perf_counter()
    sp, stub, recorder, args, to_config = H.build_server(OUT / "rec")
    meta["build_s"] = time.perf_counter() - t0
    meta["recorder_dir"] = str(recorder.run_dir)
    meta["recorder_class_background"] = bool(getattr(recorder, "background", False) or getattr(recorder, "_background", False))
    reqs = H.load_requests(a.run, a.n)
    rec_saved = sp.recorder
    with open(OUT / "pass_warm.jsonl", "w") as f:
        run_pass("warm", sp, stub, reqs[:a.warm], smp, f)
    sp.recorder = None if REC_MODE == "none" else rec_saved
    t1 = time.perf_counter()
    with open(OUT / "pass_T.jsonl", "w") as f:
        run_pass("T", sp, stub, reqs, smp, f)
    meta["pass_T_s"] = time.perf_counter() - t1
    if hasattr(recorder, "flush"):
        recorder.flush()
    meta["recorder_n_failed"] = getattr(recorder, "n_failed", None)
    np.savez(OUT / "actions_T.npz", **acts)
    if masks:
        np.savez_compressed(OUT / "masks_T.npz", **masks)
    if keep_arrays:
        np.savez_compressed(OUT / "keep_T.npz", **keep_arrays)
    smp.stop = True; time.sleep(1.2)
    meta.update({"nvsmi_max_ours_mib": smp.max_ours, "nvsmi_max_others_mib": smp.max_others, "nvsmi_max_used_mib": smp.max_used,
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (OUT / "nvidia_smi_post.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    (OUT / "uptime_post.txt").write_text(subprocess.run(["uptime"], capture_output=True, text=True).stdout)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print("DONE", json.dumps(meta, default=str), flush=True)


if __name__ == "__main__":
    main()

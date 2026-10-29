"""T38 Phase A — drive serve_safe's SafePolicy.infer with re-rendered T34 E3b requests and profile it.

Code under test: PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 (worktree, branch o4-gpu-parallel). Nothing in benchmark/ is edited;
all instrumentation is monkeypatching from this script.

The SafePolicy is built exactly like serve_safe.main() (same parser, same E3b flags from
outputs/verify/T34/p/start_server_e3.sh) except the pi0.5 policy: a stub returns the recorded policy chunk
(planning row actions_reference) and the recorded 16x16 attention maps (server record, float16) under the
policy's camera names, so attention_extractor() maps them exactly as live.

passes (one process, each pass replays seq 1..N from reset so the cross-frame state is the server's):
  warm   seq 1..W                        (JIT / first-call; reported separately)
  T      timing, no extra sync, recorder on (same as the server)          -> passT.jsonl
  S      timing with torch.cuda.synchronize() at every stage boundary     -> passS.jsonl
  C      census: KD-tree builds/queries, input hashes, host<->device copies, sizes -> passC.jsonl
  P      cProfile per top-level stage on the selected chunks              -> prof/*.prof
  W      cProfile of the whole infer() on the selected chunks             -> prof/*_infer.prof
  X      torch.profiler (CPU+CUDA) on the selected chunks                 -> torchprof/*.json

usage: MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT}-o4 AG3S_ASSET_ROOT=${AG3S_PROJECT_ROOT}/src \
  AG3S_CLIENT_SCRIPT=${AG3S_PROJECT_ROOT}/integration/rby1/bringup/pi05_infer.py \
  ${AG3S_PROJECT_ROOT}/.venv-openpi-live/bin/python replay_profile.py E3b_ep1807_r2 75 --passes T,S,C,P,W,X
"""
from common.workspace import workspace_path as _ag3s_workspace_path
import argparse, collections, contextlib, cProfile, functools, hashlib, json, os, pathlib, subprocess, sys
import threading, time, traceback

import numpy as np

assert "${AG3S_PROJECT_ROOT}-o4" in os.environ.get("PYTHONPATH", ""), "PYTHONPATH must point at the worktree"
import benchmark  # noqa: E402
assert benchmark.__file__.startswith("${AG3S_PROJECT_ROOT}-o4/"), benchmark.__file__

import torch  # noqa: E402

OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T38'))
FR = OUT / "frames"
CAMS = ("zed_left", "wrist_cam_l", "wrist_cam_r")
POLICY_NAME = {"zed_left": "cam_high", "wrist_cam_l": "cam_left_wrist", "wrist_cam_r": "cam_right_wrist"}
E3B_FLAGS = [
    "--config", "pi05_rby1_randomized_pick_place_16d_lora",
    "--checkpoint", _ag3s_workspace_path('legacy/pi05_TO_hybrid/checkpoints/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'),
    "--model-xml", _ag3s_workspace_path('robots/rby1/description/models/rby1a/mujoco/model_transport.xml'),
    "--port", "8299",  # never bound — no server is started
    "--esdf-backend", "curobo", "--voxel", "0.020", "--fine-voxel", "0.005", "--tsdf-voxel", "0.005",
    "--links", "gripper", "--target-field-policy", "exclude-authorized", "--esdf-margin", "0",
    "--no-self-collision", "--capsule-radius-scale", "0.05", "--sphere-spacing", "0.4",
    "--max-spheres-per-capsule", "32",
]
TOP_STAGES = ("scene_reconstruction", "support_surface", "attention_lifting", "target_grounding",
              "constraint_generation")


# ----------------------------------------------------------------------------- state
class St:
    sync = False          # torch.cuda.synchronize() at stage boundaries / gpu-tagged wrappers
    census = False
    prof_stages = False   # cProfile per top-level stage
    selected = False      # current chunk is a selected one
    seq = 0
    tag = ""
    stage_ms = collections.defaultdict(float)
    stage_n = collections.defaultdict(int)
    fn = collections.defaultdict(lambda: [0, 0.0])
    sizes = collections.defaultdict(list)
    kd = []
    copies = []
    hashes = []
    active_prof = None
    stage_stack = []


def reset_chunk():
    St.stage_ms = collections.defaultdict(float)
    St.stage_n = collections.defaultdict(int)
    St.fn = collections.defaultdict(lambda: [0, 0.0])
    St.sizes = collections.defaultdict(list)
    St.kd, St.copies, St.hashes = [], [], []
    St.stage_stack = []


def callsite(skip_prefixes=(_ag3s_workspace_path('legacy/.venv-openpi-live/lib/python3.11/site-packages/torch'),
                            _ag3s_workspace_path('legacy/.venv-openpi-live/lib/python3.11/site-packages/scipy'),
                            _ag3s_workspace_path('outputs/verify/T38'), "<")):
    f = sys._getframe(2)
    first = None; ours = None
    while f is not None:
        fn = f.f_code.co_filename
        if not fn.startswith(skip_prefixes):
            if first is None:
                first = f"{fn.replace('${AG3S_PROJECT_ROOT}-o4/', '').replace(_ag3s_workspace_path('legacy/.venv-openpi-live/lib/python3.11/site-packages/'), 'site:')}:{f.f_lineno} {f.f_code.co_name}"
            if fn.startswith("${AG3S_PROJECT_ROOT}-o4/benchmark"):
                ours = f"{fn.replace('${AG3S_PROJECT_ROOT}-o4/', '')}:{f.f_lineno} {f.f_code.co_name}"
                break
        f = f.f_back
    return first, ours


def digest(a):
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.blake2b(a.view(np.uint8).reshape(-1).tobytes() if a.size else b"", digest_size=8).hexdigest() + f"/{a.shape}/{a.dtype}"


def current_stage():
    return St.stage_stack[-1] if St.stage_stack else "(outside stages)"


# ----------------------------------------------------------------------------- patches
def install_patches():
    from AG3S.ag3s.runtime import profiler as profmod

    orig_stage = profmod.StageProfiler.stage

    @contextlib.contextmanager
    def stage(self, name):
        St.stage_stack.append(name)
        prof = None
        if St.prof_stages and St.selected and name in TOP_STAGES and St.active_prof is None:
            prof = cProfile.Profile(); St.active_prof = prof
        try:
            if St.sync:
                torch.cuda.synchronize()
            t = time.perf_counter()
            if prof is not None:
                prof.enable()
            with orig_stage(self, name):
                yield
                if St.sync:
                    torch.cuda.synchronize()
            if prof is not None:
                prof.disable()
            St.stage_ms[name] += (time.perf_counter() - t) * 1e3
            St.stage_n[name] += 1
        finally:
            St.stage_stack.pop()
            if prof is not None:
                St.active_prof = None
                d = OUT / "prof"; d.mkdir(exist_ok=True)
                prof.dump_stats(d / f"{St.tag}_seq{St.seq:03d}_{name}.prof")

    profmod.StageProfiler.stage = stage


def timed(name, fn, gpu=False, sizer=None):
    @functools.wraps(fn)
    def w(*a, **k):
        if St.sync and gpu:
            torch.cuda.synchronize()
        t = time.perf_counter()
        try:
            out = fn(*a, **k)
        finally:
            if St.sync and gpu:
                torch.cuda.synchronize()
            r = St.fn[name]; r[0] += 1; r[1] += (time.perf_counter() - t) * 1e3
        if St.census and sizer is not None:
            try:
                St.sizes[name].append(sizer(a, k, out))
            except Exception as e:  # noqa: BLE001
                St.sizes[name].append({"sizer_error": repr(e)})
        return out
    return w


def patch(obj, attr, name=None, gpu=False, sizer=None):
    raw = obj.__dict__.get(attr) if isinstance(obj, type) else None
    fn = getattr(obj, attr)
    label = name or f"{getattr(obj, '__name__', str(obj))}.{attr}"
    wrapped = timed(label, fn, gpu=gpu, sizer=sizer)
    if isinstance(raw, staticmethod):
        wrapped = staticmethod(wrapped)
    elif isinstance(raw, classmethod):
        wrapped = classmethod(timed(label, raw.__func__, gpu=gpu, sizer=sizer))
    setattr(obj, attr, wrapped)


def n_of(x):
    try:
        return int(len(x))
    except Exception:  # noqa: BLE001
        return None


def install_function_timers():
    from AG3S.ag3s.runtime import pipeline as P, multiview as MV
    from AG3S.ag3s.stages import reconstruction as R, robot_filter as RF, support_surface as SS
    from AG3S.ag3s.stages import target_grounding as TG, attention_lifting as AL
    from AG3S.ag3s.stages import admissibility as AD, attention_split as AS
    from AG3S.ag3s.constraints import to_adapter as TA, constraint_builder as CB
    from AG3S.ag3s.fields import curobo_builder as CBF
    from AG3S.trajopt import safe_policy as SP, wire, linearize as LZ, refiner as RFN, sqp as SQ

    def pts_in(a, k, out):  # first positional = points / cloud
        x = a[0] if a else None
        pts = getattr(x, "points", x)
        return {"n_in": n_of(pts), "hash_in": digest(pts) if pts is not None and hasattr(pts, "__len__") else None,
                "stage": current_stage()}

    # --- scene_reconstruction
    patch(P, "fuse_observations", "multiview.fuse_observations")
    patch(MV, "check_freshness", "multiview.check_freshness")
    patch(MV, "process_observation", "multiview.process_observation",
          sizer=lambda a, k, out: {"camera": str(a[0].camera_id.value), "n_cloud_out": len(out.cloud),
                                   **{kk: out.stats.get(kk) for kk in ("n_raw", "n_voxel", "n_final", "final_voxel_size", "capped",
                                                                        "n_self_filtered")}})
    patch(MV, "reconstruct", "reconstruction.reconstruct",
          sizer=lambda a, k, out: {kk: out[1].get(kk) for kk in ("n_raw", "n_voxel", "n_final", "final_voxel_size")})
    patch(R, "backproject", "reconstruction.backproject")
    patch(R, "voxel_downsample", "reconstruction.voxel_downsample", sizer=pts_in)
    patch(MV, "filter_robot_points", "robot_filter.filter_robot_points", sizer=pts_in)
    patch(RF, "robot_sphere_mask", "robot_filter.robot_sphere_mask", sizer=pts_in)
    patch(MV, "lift", "attention_lifting.lift", sizer=pts_in)
    patch(MV, "fuse", "multiview.fuse",
          sizer=lambda a, k, out: {"n_in": sum(len(r.cloud) for r in a[0]), "n_fused": len(out[0])})
    patch(P.AG3S, "_prepare_frame_filter", "AG3S._prepare_frame_filter")
    # --- support
    patch(P, "fit_support_surfaces", "support_surface.fit_support_surfaces", sizer=pts_in)
    patch(SS, "fit_plane_ransac", "support_surface.fit_plane_ransac", sizer=pts_in)
    # --- grounding
    patch(P, "ground_target", "target_grounding.ground_target",
          sizer=lambda a, k, out: {"n_in": len(a[0]), "hash_in": digest(a[0].points), "n_seeds": n_of(out.seed_indices),
                                   "n_clusters": n_of(out.clusters), "status": out.status.value})
    patch(TG, "extract_seeds", "target_grounding.extract_seeds")
    patch(TG, "grow_region", "target_grounding.grow_region",
          sizer=lambda a, k, out: {"n_in": n_of(a[0]), "n_seeds": n_of(a[1]), "n_grown": n_of(out), "hash_in": digest(a[0])})
    patch(TG, "dbscan", "target_grounding.dbscan",
          sizer=lambda a, k, out: {"n_in": n_of(a[0]), "n_labels": int(len(set(np.asarray(out).tolist()) - {-1})), "hash_in": digest(a[0])})
    patch(TG, "connected_components_3d", "target_grounding.connected_components_3d")
    patch(TG, "_split_groups", "target_grounding._split_groups",
          sizer=lambda a, k, out: {"n_groups": n_of(a[0]), "group_sizes": [int(len(g)) for g in a[0]]})
    patch(TG, "_score_cluster", "target_grounding._score_cluster")
    for meth in ("select", "_assess_frame", "observe", "_associate", "split_anchor_on_release", "note_hand",
                 "inject_destination", "freeze"):
        if hasattr(TG.TargetConfirm, meth):
            patch(TG.TargetConfirm, meth, f"TargetConfirm.{meth}")
    for nm in ("principal_extents", "graspable"):
        if hasattr(TG, nm):
            patch(TG, nm, f"admissibility.{nm}")
    # --- constraint generation
    patch(P.AG3S, "_depth_cameras_from", "AG3S._depth_cameras_from", gpu=True)
    patch(P.AG3S, "_robot_mask_for", "AG3S._robot_mask_for")
    patch(P.AG3S, "_build_esdf", "AG3S._build_esdf", gpu=True)
    patch(P.AG3S, "_esdf_coverage", "AG3S._esdf_coverage", gpu=True)
    patch(P.AG3S, "_constraints", "AG3S._constraints")
    patch(P.AG3S, "_exclusion_gate", "AG3S._exclusion_gate")
    patch(P.AG3S, "_exclusion_record", "AG3S._exclusion_record")
    patch(P.AG3S, "_admissibility_frame_record", "AG3S._admissibility_frame_record")
    patch(P.AG3S, "_clearance_metrics", "AG3S._clearance_metrics")
    patch(P.AG3S, "_fine_window_request", "AG3S._fine_window_request")
    patch(P.AG3S, "_target_ball", "AG3S._target_ball", sizer=lambda a, k, out: {"n_in": n_of(a[1])})
    patch(P.AG3S, "_held_seed_points", "AG3S._held_seed_points")
    patch(P.AG3S, "_held_free_points", "AG3S._held_free_points")
    patch(P.AG3S, "_window_query_report", "AG3S._window_query_report", gpu=True)
    patch(P, "build_constraint_set", "to_adapter.build_constraint_set")
    patch(CB.ConstraintBuilder, "build", "ConstraintBuilder.build",
          sizer=lambda a, k, out: {"n_candidates": n_of(a[1]), "n_surfaces": n_of(a[2]),
                                   "n_params": int(np.asarray(out.parameter_values).size)})
    for meth in ("update", "_ensure_mapper", "_make_mapper", "_integrate", "_build_tiers", "_camera_contribution",
                 "_free_tsdf_points", "_exclude_seed_points", "_exclude_ball_seeds", "_clear_seeds",
                 "_labels_from_sites", "_observation_coverage", "_stats", "_force_sign_on_pure_voxels"):
        if hasattr(CBF.CuroboFieldBuilder, meth):
            patch(CBF.CuroboFieldBuilder, meth, f"CuroboFieldBuilder.{meth}", gpu=True)
    if hasattr(CBF, "place_fine_window"):
        patch(CBF, "place_fine_window", "curobo_builder.place_fine_window")
    # --- SafePolicy / TO
    patch(wire, "unpack_camera_observations", "wire.unpack_camera_observations")
    for meth in ("_scene_fn", "_apply_finger_state", "_advance_grasp", "_run_latch", "_finish_grasp_record",
                 "_verdict", "_record", "_preserve_grippers", "_execution_path", "_continuity_input"):
        patch(SP.SafePolicy, meth, f"SafePolicy.{meth}")
    patch(P.AG3S, "process_multi_debug", "AG3S.process_multi_debug", gpu=True)
    patch(SP, "scene_from_constraint_set", "linearize.scene_from_constraint_set",
          sizer=lambda a, k, out: {"spec_present": a[0].constraints is not None,
                                   "candidate_active_any": bool(np.any(out.candidate_active)) if out is not None else None,
                                   "plane_active_any": bool(np.any(out.plane_active)) if out is not None else None,
                                   "n_candidate_slots": int(len(out.candidate_active)) if out is not None else None,
                                   "n_plane_slots": int(len(out.plane_active)) if out is not None else None,
                                   "esdf_present": getattr(out, "esdf", None) is not None})
    patch(SP.TrajOptChunkRefiner, "refine", "refiner.refine", gpu=True)
    for meth in ("linearize", "check"):
        pass


# ----------------------------------------------------------------------------- census-only patches
def install_census():
    import scipy.spatial as sps
    base = sps.cKDTree

    class TrackedKD(base):
        def __init__(self, data, *a, **k):
            t = time.perf_counter()
            super().__init__(data, *a, **k)
            if St.census:
                cs, ours = callsite()
                St.kd.append({"op": "build", "n": int(np.asarray(data).shape[0]), "site": cs, "ours": ours,
                              "stage": current_stage(), "hash": digest(np.asarray(data, np.float64)),
                              "ms": (time.perf_counter() - t) * 1e3})

        def _q(self, op, fn, x, *a, **k):
            t = time.perf_counter()
            out = fn(x, *a, **k)
            if St.census:
                cs, ours = callsite()
                St.kd.append({"op": op, "n_tree": int(self.n), "n_query": n_of(np.atleast_2d(np.asarray(x))) if op != "query_pairs" else None,
                              "site": cs, "ours": ours, "stage": current_stage(), "ms": (time.perf_counter() - t) * 1e3})
            return out

        def query(self, x, *a, **k):
            return self._q("query", super().query, x, *a, **k)

        def query_ball_point(self, x, *a, **k):
            return self._q("query_ball_point", super().query_ball_point, x, *a, **k)

        def query_pairs(self, r, *a, **k):
            t = time.perf_counter()
            out = super().query_pairs(r, *a, **k)
            if St.census:
                cs, ours = callsite()
                St.kd.append({"op": "query_pairs", "n_tree": int(self.n), "site": cs, "ours": ours,
                              "stage": current_stage(), "ms": (time.perf_counter() - t) * 1e3})
            return out

    sps.cKDTree = TrackedKD

    def rec(direction, nbytes, extra=""):
        if St.census:
            cs, ours = callsite()
            St.copies.append({"dir": direction, "bytes": int(nbytes), "site": cs, "ours": ours,
                              "stage": current_stage(), "kind": extra})

    T = torch.Tensor
    o_cpu, o_to, o_cuda, o_item, o_tolist = T.cpu, T.to, T.cuda, T.item, T.tolist

    def cpu(self, *a, **k):
        if self.is_cuda:
            rec("D2H", self.element_size() * self.numel(), "cpu()")
        return o_cpu(self, *a, **k)

    def to(self, *a, **k):
        out = o_to(self, *a, **k)
        if isinstance(out, torch.Tensor) and out.device.type != self.device.type:
            rec("H2D" if out.is_cuda else "D2H", out.element_size() * out.numel(), "to()")
        return out

    def cuda(self, *a, **k):
        if not self.is_cuda:
            rec("H2D", self.element_size() * self.numel(), "cuda()")
        return o_cuda(self, *a, **k)

    def item(self):
        if self.is_cuda:
            rec("D2H", self.element_size(), "item()")
        return o_item(self)

    def tolist(self):
        if self.is_cuda:
            rec("D2H", self.element_size() * self.numel(), "tolist()")
        return o_tolist(self)

    T.cpu, T.to, T.cuda, T.item, T.tolist = cpu, to, cuda, item, tolist
    o_as, o_tensor = torch.as_tensor, torch.tensor

    def _dev_cuda(k, a):
        d = k.get("device")
        return d is not None and "cuda" in str(d)

    def as_tensor(data, *a, **k):
        out = o_as(data, *a, **k)
        if _dev_cuda(k, a) and not (isinstance(data, torch.Tensor) and data.is_cuda):
            rec("H2D", out.element_size() * out.numel(), "as_tensor")
        return out

    def tensor(data, *a, **k):
        out = o_tensor(data, *a, **k)
        if _dev_cuda(k, a) and not (isinstance(data, torch.Tensor) and data.is_cuda):
            rec("H2D", out.element_size() * out.numel(), "tensor")
        return out

    torch.as_tensor, torch.tensor = as_tensor, tensor


# ----------------------------------------------------------------------------- build the server object
class StubPolicy:
    """Returns the recorded chunk + attention. No reset (SafePolicy.reset tolerates its absence)."""
    metadata = {}

    def __init__(self):
        self.next = None

    def infer(self, obs, **kw):
        return {"actions": self.next["actions"], "attention": self.next["attention"]}


def build_server(record_dir):
    from AG3S.trajopt import serve_safe as S
    from AG3S.trajopt.safe_policy import SafePolicy
    from AG3S.ag3s.experiments.sources.constraint_record import ConstraintRecordWriter

    ap = S.build_parser()
    args = ap.parse_args(E3B_FLAGS + ["--record-constraints", str(record_dir)])
    S.reject_bad_flag_combinations(ap, args)
    to_config = S.trajopt_config_from_args(args)
    recorder = ConstraintRecordWriter(args.record_constraints, esdf_mode=args.record_constraints_esdf,
                                      meta={"t38": "replay", "flags": E3B_FLAGS})
    stub = StubPolicy()
    sp = SafePolicy(
        stub,
        ag3s=S.build_ag3s(args.model_xml, voxel=args.voxel, range_max=args.range_max, links=args.links,
                          backend=args.esdf_backend, fine_voxel=args.fine_voxel, tsdf_voxel=args.tsdf_voxel,
                          attached_sign_threshold=args.attached_sign_threshold,
                          max_field_age_sec=args.max_field_age_sec, exclude_links=args.exclude_links,
                          constraint_sphere_options=S.sphere_options(args),
                          self_collision=not args.no_self_collision,
                          target_field_policy=args.target_field_policy,
                          plan_horizon_steps=to_config.horizon.planned,
                          rows_per_step=to_config.reduction.rows_per_step,
                          self_filter=S.self_filter_options(args)),
        static_geometry=S.load_static_geometry(args.static_geometry, args.model_xml, links=args.links),
        to_config=to_config,
        attention_fn=S.attention_extractor(),
        recorder=recorder,
        shadow=args.shadow,
    )
    return sp, stub, recorder, args, to_config


def load_requests(run, n):
    from AG3S.trajopt import wire
    idx = json.loads((FR / f"{run}.json").read_text())
    reqs = []
    for s in idx["seqs"][:n]:
        z = np.load(FR / run / f"seq{s['seq']:03d}.npz")
        obs = {"state": z["state"], "prompt": "Pick up the apple and place it in the basket."}
        fb = s["exec_feedback"]
        req = wire.pack_request(
            obs, cameras=CAMS,
            depth={c: z[f"depth_mm/{c}"].astype(np.float64) / 1000.0 for c in CAMS},
            intrinsics={c: z[f"K/{c}"] for c in CAMS}, extrinsics={c: z[f"T/{c}"] for c in CAMS},
            robot_state={c: z[f"robot_state/{c}"] for c in CAMS},
            stamps={c: float(s["stamps"][c]) for c in CAMS}, phase="approach", active_manipulators=("left",),
            reset=(s["seq"] == 1), seq=int(s["seq"]),
            exec_feedback=fb if fb is not None else wire.no_exec_feedback("first chunk of the episode (reset)"),
            render_stamps={c: float(s["render_stamps"][c]) for c in CAMS} if s.get("render_stamps") else None,
            stamp_mode=s.get("stamp_mode") or None)
        stub_next = {"actions": z["actions_reference"].astype(np.float32),
                     "attention": {POLICY_NAME[c]: z[f"attention/{c}"] for c in CAMS}}
        reqs.append((s, req, stub_next))
    return reqs


# ----------------------------------------------------------------------------- GPU memory sampler
class GpuSampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True)
        self.path = path; self.stop = False; self.pid = os.getpid(); self.max_ours = 0; self.max_total = 0
        self.max_others = 0

    def run(self):
        with open(self.path, "a") as f:
            while not self.stop:
                try:
                    tot = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                         capture_output=True, text=True, timeout=10).stdout.strip()
                    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True, timeout=10).stdout.strip()
                    used, util = [int(x) for x in tot.split(",")]
                    ours = 0; others = 0
                    for line in apps.splitlines():
                        p, m = [int(x) for x in line.split(",")]
                        if p == self.pid:
                            ours = m
                        else:
                            others += m
                    self.max_ours = max(self.max_ours, ours); self.max_total = max(self.max_total, used)
                    self.max_others = max(self.max_others, others)
                    f.write(json.dumps({"t": time.time(), "tag": St.tag, "seq": St.seq, "used_mib": used, "util": util,
                                        "ours_mib": ours, "others_mib": others,
                                        "torch_alloc_mib": torch.cuda.memory_allocated() / 2**20,
                                        "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20}) + "\n")
                    f.flush()
                    if used > 0.8 * 140000:
                        f.write(json.dumps({"warning": "GPU memory > 80 % of 140 GB", "used_mib": used}) + "\n")
                except Exception as e:  # noqa: BLE001
                    f.write(json.dumps({"error": repr(e)}) + "\n")
                time.sleep(1.0)


# ----------------------------------------------------------------------------- one pass
def chunk_record(sp, s, resp, wall_ms):
    cs = sp._last_constraint_set
    dbg = sp._last_debug or {}
    m = dict(getattr(cs, "metrics", {}) or {}) if cs is not None else {}
    esdf = m.get("esdf") or {}
    grounding = dbg.get("grounding")
    to = resp.get("to") if isinstance(resp, dict) else None
    res = sp.refiner.last_result
    rm = dict(getattr(res, "metrics", None) or {}) if res is not None else {}
    fus = dbg.get("fusion")
    rec = {
        "seq": s["seq"], "t_step": s["t_step"], "phase_label": s["phase_label"], "wall_ms": wall_ms,
        "timing_ms": dict((sp._pending or {}).get("timing") or {}),
        "profile_ms": dict(getattr(cs, "profile", {}) or {}) if cs is not None else None,
        "stage_ms_harness": dict(St.stage_ms), "stage_n": dict(St.stage_n),
        "fn": {k: {"n": v[0], "ms": v[1]} for k, v in St.fn.items()},
        "to_timing_ms": rm.get("timing_ms"), "to": to,
        "status": getattr(getattr(cs, "status", None), "value", None),
        "has_target": getattr(cs, "target", None) is not None,
        "grasp_state": (sp._last_grasp_record or {}).get("state"),
        "recorded_grasp_state": s.get("recorded_grasp_state"),
        "n_clusters": None if grounding is None else len(grounding.clusters),
        "recorded_n_clusters": s.get("recorded_n_clusters"),
        "recorded_profile_ms": s.get("recorded_profile_ms"),
        "recorded_to_timing": s.get("recorded_to_timing"),
        "sizes": {
            "n_points_per_camera": m.get("n_points_per_camera"),
            "n_points_before_fusion": m.get("n_points_before_fusion"),
            "n_points_fused": m.get("n_points_fused"),
            "n_observations": m.get("n_observations"),
            "n_self_filtered": m.get("n_self_filtered"),
            "n_points_raw": m.get("n_points_raw"), "n_points_voxel": m.get("n_points_voxel"),
            "n_points_final": m.get("n_points_final"),
            "per_camera_recon": None if fus is None else {
                r.camera_id.value: {kk: r.stats.get(kk) for kk in ("n_raw", "n_voxel", "n_final", "final_voxel_size", "capped", "n_self_filtered")}
                for r in fus.per_camera},
            "n_support_points": m.get("n_support_points"),
            "n_seeds": None if grounding is None or grounding.seed_indices is None else int(len(grounding.seed_indices)),
            "target_n_points": None if getattr(cs, "target", None) is None else n_of(getattr(cs.target, "points", None)),
            "esdf_frames": esdf.get("frames"), "esdf_n_active_blocks": esdf.get("n_active_blocks"),
            "esdf_n_occupied": esdf.get("n_occupied"), "esdf_tiers": esdf.get("tiers"),
            "esdf_n_attached_points": esdf.get("n_attached_points"), "esdf_held_free": esdf.get("held_free"),
            "spec_present": getattr(cs, "constraints", None) is not None,
            "n_support_surfaces": None if cs is None else len(cs.support_surfaces),
        },
        "fn_sizes": {k: v for k, v in St.sizes.items()},
    }
    if St.census:
        rec["kd"] = St.kd; rec["copies"] = St.copies
    return rec


def run_pass(tag, sp, stub, reqs, *, selected=(), sync=False, census=False, prof_stages=False, prof_infer=False,
             torch_prof=False, out_path=None, limit=None):
    St.tag = tag; St.sync = sync; St.census = census; St.prof_stages = prof_stages
    torch.cuda.reset_peak_memory_stats()
    f = open(out_path, "w") if out_path else None
    t_pass = time.perf_counter()
    for i, (s, req, nxt) in enumerate(reqs if limit is None else reqs[:limit]):
        reset_chunk()
        St.seq = s["seq"]; St.selected = s["seq"] in selected
        stub.next = nxt
        req_copy = dict(req)
        prof = None; tp = None
        if prof_infer and St.selected:
            prof = cProfile.Profile()
        if torch_prof and St.selected:
            from torch.profiler import profile, ProfilerActivity
            tp = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False)
            tp.__enter__()
        torch.cuda.synchronize()
        t = time.perf_counter()
        if prof is not None:
            prof.enable()
        resp = sp.infer(req_copy)
        if prof is not None:
            prof.disable()
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) * 1e3
        if tp is not None:
            tp.__exit__(None, None, None)
            d = OUT / "torchprof"; d.mkdir(exist_ok=True)
            evs = tp.key_averages()
            rows = []
            for e in evs:
                rows.append({"key": e.key, "count": e.count,
                             "cpu_time_total_us": e.cpu_time_total, "self_cpu_time_total_us": e.self_cpu_time_total,
                             "device_time_total_us": getattr(e, "device_time_total", getattr(e, "cuda_time_total", 0)),
                             "self_device_time_total_us": getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))})
            (d / f"{tag}_seq{s['seq']:03d}_keyavg.json").write_text(json.dumps(rows))
            try:
                tp.export_chrome_trace(str(d / f"{tag}_seq{s['seq']:03d}_trace.json"))
            except Exception as e:  # noqa: BLE001
                print("chrome trace failed", e)
        if prof is not None:
            d = OUT / "prof"; d.mkdir(exist_ok=True)
            prof.dump_stats(d / f"{tag}_seq{s['seq']:03d}_infer.prof")
        rec = chunk_record(sp, s, resp, wall)
        rec["torch_max_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
        rec["torch_max_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        if f:
            f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n"); f.flush()
        pm = rec["profile_ms"] or {}
        print(f"[{tag}] seq {s['seq']:2d} {s['phase_label']:9s} grasp {rec['grasp_state']}/{s.get('recorded_grasp_state')} "
              f"wall {wall:7.1f}  recon {pm.get('scene_reconstruction', 0):6.1f} ground {pm.get('target_grounding', 0):7.1f} "
              f"cgen {pm.get('constraint_generation', 0):6.1f} esdf {pm.get('esdf', 0):6.1f} "
              f"TO {json.dumps(rec['to_timing_ms'])}", flush=True)
    if f:
        f.close()
    return (time.perf_counter() - t_pass)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("n", type=int)
    ap.add_argument("--passes", default="T,S,C,P,W,X")
    ap.add_argument("--warm", type=int, default=4)
    ap.add_argument("--selected", default="2,10,15,16,22,28,40")
    ap.add_argument("--label", default="", help="suffix for output file names (e.g. _pyspy)")
    a = ap.parse_args()
    selected = tuple(int(x) for x in a.selected.split(","))
    meta = {"run": a.run, "n": a.n, "passes": a.passes, "selected": selected, "pid": os.getpid(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "argv": sys.argv,
            "python": sys.executable, "torch": torch.__version__, "benchmark_file": benchmark.__file__}
    (OUT / "logs").mkdir(exist_ok=True)
    smp = GpuSampler(OUT / "logs" / f"gpu_samples_{a.run}{a.label}.jsonl"); smp.start()
    nv_pre = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
    (OUT / "logs" / f"nvidia_smi_pre_{a.run}{a.label}.txt").write_text(nv_pre)
    install_patches(); install_function_timers()
    t0 = time.perf_counter()
    sp, stub, recorder, args, to_config = build_server(OUT / "rec")
    meta["build_server_s"] = time.perf_counter() - t0
    meta["recorder_dir"] = str(recorder.run_dir)
    reqs = load_requests(a.run, a.n)
    # warm-up: first-call costs (cuRobo/warp JIT, CasADi, scipy imports) — measured, reported separately
    run_pass("warm", sp, stub, reqs, out_path=OUT / f"pass_warm_{a.run}{a.label}.jsonl", limit=a.warm)
    rec_saved = sp.recorder
    for p in a.passes.split(","):
        if p == "C" and not getattr(St, "census_installed", False):
            install_census(); St.census_installed = True   # lazily: its wrappers would cost time in T/S
        sp.recorder = rec_saved if p == "T" else None
        kw = dict(T={}, S={"sync": True}, C={"census": True}, P={"prof_stages": True},
                  W={"prof_infer": True}, X={"torch_prof": True})[p]
        dt = run_pass(p, sp, stub, reqs, selected=selected, out_path=OUT / f"pass{p}_{a.run}{a.label}.jsonl", **kw)
        meta[f"pass_{p}_s"] = dt
        meta[f"pass_{p}_torch_max_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
        (OUT / f"meta_{a.run}{a.label}.json").write_text(json.dumps(meta, indent=1, default=str))
    smp.stop = True; time.sleep(1.5)
    meta.update({"gpu_max_ours_mib": smp.max_ours, "gpu_max_total_mib": smp.max_total, "gpu_max_others_mib": smp.max_others,
                 "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    (OUT / "logs" / f"nvidia_smi_post_{a.run}{a.label}.txt").write_text(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
    (OUT / f"meta_{a.run}{a.label}.json").write_text(json.dumps(meta, indent=1, default=str))
    print("DONE", json.dumps(meta, default=str))


if __name__ == "__main__":
    main()

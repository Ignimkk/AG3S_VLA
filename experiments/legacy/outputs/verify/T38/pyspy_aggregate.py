"""T38: aggregate a py-spy raw (collapsed-stack) file by pipeline stage and leaf function.
usage: python pyspy_aggregate.py pyspy/replay_T.raw.txt
Stage = first match of these markers in the stack (outermost first). Leaf = last frame of the stack.
"""
import collections, json, pathlib, sys

MARK = [(("_record", "safe_policy.py"), "_record (recorder, outside timing)"),
        (("fuse_observations", "multiview.py"), "scene_reconstruction"),
        (("fit_support_surfaces", "support_surface.py"), "support_surface"),
        (("ground_target", "target_grounding.py"), "target_grounding"),
        (("_depth_cameras_from", "pipeline.py"), "constraint_generation: depth robot mask"),
        (("_build_esdf", "pipeline.py"), "esdf"),
        (("_esdf_coverage", "pipeline.py"), "constraint_generation: coverage"),
        (("_constraints", "pipeline.py"), "constraint_generation: build_constraint_set"),
        (("process_multi_debug", "pipeline.py"), "ag3s other"),
        (("_scene_fn", "safe_policy.py"), "safe_policy scene_fn other"),
        (("refine", "refiner.py"), "TO (refiner/sqp)"),
        (("infer", "safe_policy.py"), "safe_policy.infer other")]


def stage_of(frames):
    for (fn, file), name in MARK:
        for f in frames:
            if f.startswith(fn + " (") and file in f:
                return name
    return "outside infer"


src = pathlib.Path(sys.argv[1])
by_stage = collections.Counter(); leaf = collections.defaultdict(collections.Counter); total = 0
for line in src.read_text(errors="replace").splitlines():
    if not line.strip():
        continue
    stack, n = line.rsplit(" ", 1); n = int(n)
    frames = stack.split(";")
    st = stage_of(frames)
    by_stage[st] += n; total += n
    leaf[st][frames[-1]] += n
out = {"file": str(src), "total_samples": total,
       "by_stage_samples": dict(by_stage.most_common()),
       "top_leaf_per_stage": {k: leaf[k].most_common(15) for k, _ in by_stage.most_common()}}
dst = src.with_suffix(".agg.json"); dst.write_text(json.dumps(out, indent=1))
print(json.dumps({"total": total, "by_stage": dict(by_stage.most_common())}, indent=1))

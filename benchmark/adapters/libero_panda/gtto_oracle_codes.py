"""SL-GTTO -- LIBERO GT segmentation -> AG3S oracle code images (``wire.ORACLE_SEG``). Python 3.8 compatible, numpy only.

The LIBERO clients (``libero_safe_client.py`` -- LIBERO / SafeLIBERO, ``libero_safety_client.py`` -- LIBERO-Safety) run
in their own interpreters and cannot import AG3S. This module restates the code table of
``AG3S/ag3s/runtime/oracle_segmentation.py`` (RB-Y1 T46e) for the LIBERO scenes; ``tests/benchmark/
test_sl_gtto_gt_perception.py`` checks that the constants are the same.

=====  ======================================================================================================
code   pixels
=====  ======================================================================================================
0      no geom hit, or the robot (``robot0_*`` / ``gripper0_*`` / ``mount0_*`` bodies, ``eef_marker``) -- never
       evidence (the server's self-filter / depth mask removes the robot)
1      the episode's GT target: the BDDL goal's object (``task_target.goal_roles``, the rule SL3 / ATS /
       ``Baseline/paper_baselines/providers/gt_oracle.py`` use), every body of it (``task_target.instance_body_ids``)
2      static: the table, fixtures (cabinet, stove, ...), walls, floor -- every body that is neither robot nor a
       movable object
3, 4,  every other object, one code each: the BDDL objects other than the target (in BDDL order), then every other
...    root body with a free joint (SafeLIBERO obstacles not listed in the BDDL, by body id)
=====  ======================================================================================================

Input = the client's ``segmentation_geom/<cam>`` image (``camera.capture(..., segmentation=True)``): int16 MuJoCo
**geom** ids, -1 = no geom, **OpenCV rows** (``raw[::-1]``) -- the same pixel grid and orientation as the
``depth_m`` the request carries, so a code pixel and the depth pixel at the same ``(row, column)`` are the same ray.
The segmentation is rendered at ``env.step`` return, one physics substep after the depth image instant (camera.py
conventions ``segmentation``); the sub-step offset is reported, not corrected (as ``gt_oracle``).
"""
from __future__ import annotations

import numpy as np

CODE_NONE = 0
CODE_TARGET = 1
CODE_STATIC = 2
CODE_OBJECT_BASE = 3
#: wire format version of the code image (``ag3s/oracle_seg_meta["version"]``) = ``oracle_segmentation.CODE_VERSION``
CODE_VERSION = 1
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_", "mount0_")
ROBOT_BODY_NAMES = ("eef_marker",)
#: MuJoCo ``mjtJoint.mjJNT_FREE``
MJ_JOINT_FREE = 0
#: wire keys (after ``ag3s/``), = ``AG3S.trajopt.wire.ORACLE_SEG`` / ``ORACLE_SEG_META`` / ``ORACLE_SEG_REQUIRED``
ORACLE_SEG = "oracle_seg"
ORACLE_SEG_META = "oracle_seg_meta"
ORACLE_SEG_REQUIRED = "oracle_segmentation_required"


def is_robot_body(name):
    name = str(name or "")
    return name.startswith(ROBOT_BODY_PREFIXES) or name in ROBOT_BODY_NAMES


def _prefix_ids(name, body_names):
    """``task_target.instance_body_ids`` without the fixture aliases (objects only): ``<name>_*`` bodies."""
    return [i for i, b in enumerate(body_names) if str(b).startswith(str(name) + "_")]


def body_code_table(body_names, body_rootid, free_bodies, target, objects):
    """``(codes (nbody,) uint8, legend {object name: code}, info)`` -- the code of every MuJoCo body.

    ``body_names`` / ``body_rootid``: the model's tables; ``free_bodies``: ids of the bodies that carry a free joint;
    ``target``: the BDDL target object name; ``objects``: the BDDL object names (fixtures are static, not listed).
    A body takes its root body's object code (``X_main`` and its parts). Raises if the target resolves to no body
    (a typo or a region would otherwise label nothing as the target).
    """
    names = [str(b) for b in body_names]
    root = np.asarray(body_rootid, np.int64).reshape(-1)
    n = len(names)
    codes = np.full(n, CODE_STATIC, np.uint8)
    robot = np.array([is_robot_body(names[i]) or is_robot_body(names[int(root[i])]) for i in range(n)], bool)
    target_ids = _prefix_ids(target, names)
    if not target_ids:
        raise ValueError("oracle target {!r} resolves to no MuJoCo body".format(target))
    legend = {str(target): CODE_TARGET}
    owner = {}                                   # body id -> code
    for i in target_ids:
        owner[i] = CODE_TARGET
    unresolved = []
    nxt = CODE_OBJECT_BASE
    for obj in objects:
        obj = str(obj)
        if obj == str(target) or obj in legend:
            continue
        ids = [i for i in _prefix_ids(obj, names) if i not in owner]
        if not ids:
            unresolved.append(obj)
            continue
        if nxt > 254:
            raise ValueError("more than 252 objects: the uint8 code image cannot hold them")
        legend[obj] = nxt
        for i in ids:
            owner[i] = nxt
        nxt += 1
    extra = []
    for b in sorted(int(x) for x in free_bodies):
        r = int(root[b])
        if robot[b] or b in owner or r in owner:
            continue
        if nxt > 254:
            raise ValueError("more than 252 objects: the uint8 code image cannot hold them")
        legend[names[r]] = nxt
        owner[r] = nxt
        extra.append(names[r])
        nxt += 1
    for i in range(n):
        if robot[i]:
            codes[i] = CODE_NONE
        elif i in owner:
            codes[i] = owner[i]
        elif int(root[i]) in owner:
            codes[i] = owner[int(root[i])]
    info = {"n_bodies": n, "n_robot_bodies": int(robot.sum()), "target_bodies": [names[i] for i in target_ids],
            "unresolved_objects": unresolved, "free_joint_objects_not_in_bddl": extra}
    return codes, legend, info


def geom_lut(body_codes, geom_bodyid):
    """``lut[geom_id + 1]`` = the code of a pixel showing that geom; ``lut[0]`` (geom id -1, no hit) = 0."""
    gb = np.asarray(geom_bodyid, np.int64).reshape(-1)
    lut = np.full(gb.size + 1, CODE_NONE, np.uint8)
    lut[1:] = np.asarray(body_codes, np.uint8)[gb]
    return lut


def code_image(segmentation_geom, lut):
    """(H, W) uint8 code image from a geom-id image (-1 = no geom; ids outside the model -> 0). Same shape, same
    pixel grid (no flip, no resize)."""
    g = np.asarray(segmentation_geom, np.int64)
    idx = g + 1
    bad = (idx < 0) | (idx >= len(lut))
    out = np.asarray(lut, np.uint8)[np.clip(idx, 0, len(lut) - 1)]
    out[bad] = CODE_NONE
    return out.astype(np.uint8, copy=False)


def model_code_table(model, target, objects):
    """:func:`body_code_table` + :func:`geom_lut` from a live MuJoCo model (robosuite ``env.sim.model``)."""
    names = [model.body_id2name(i) or "" for i in range(model.nbody)]
    free = sorted({int(model.jnt_bodyid[j]) for j in range(model.njnt) if int(model.jnt_type[j]) == MJ_JOINT_FREE})
    codes, legend, info = body_code_table(names, np.asarray(model.body_rootid), free, target, objects)
    return geom_lut(codes, np.asarray(model.geom_bodyid)), legend, info


class OracleCoder(object):
    """Per-episode code images for the client's requests. Built lazily from the env's model on the first frame."""

    def __init__(self, target, objects, source=None):
        self.target = str(target)
        self.objects = [str(o) for o in objects]
        self.source = dict(source or {})
        self.lut = None
        self.legend = None
        self.info = None
        self.record = {}     # the client's per-episode record (BDDL target rule, connect check)

    def prepare(self, model):
        if self.lut is None:
            self.lut, self.legend, self.info = model_code_table(model, self.target, self.objects)
        return self

    def meta(self):
        return {"version": CODE_VERSION, "target": self.target, "codes": dict(self.legend or {}),
                "source": "benchmark/adapters/libero_panda/gtto_oracle_codes.py", **self.source}

    def request_keys(self, frames, prefix="ag3s/"):
        """``{ag3s/oracle_seg/<cam>: (H, W) uint8, ag3s/oracle_seg_meta: {...}}`` for one request. Every camera needs
        ``segmentation_geom`` of the same shape as its ``depth_m`` (else the request is refused before it is sent,
        as ``wire.pack_request``)."""
        if self.lut is None:
            raise RuntimeError("OracleCoder.prepare(model) was not called")
        out = {}
        for cam, frame in frames.items():
            if "segmentation_geom" not in frame:
                raise KeyError("--oracle-seg needs --capture-segmentation (no segmentation_geom for {})".format(cam))
            seg = np.asarray(frame["segmentation_geom"])
            depth = np.asarray(frame["depth_m"]) if "depth_m" in frame else np.asarray(frame["depth_mm"])
            if seg.shape != depth.shape:
                raise ValueError("oracle_seg/{} {} != depth {}".format(cam, seg.shape, depth.shape))
            out["{}{}/{}".format(prefix, ORACLE_SEG, cam)] = code_image(seg, self.lut)
        out[prefix + ORACLE_SEG_META] = self.meta()
        return out


#: Target rule when ``task_target.goal_roles`` does not apply: LIBERO-Safety affordance goals are
#: ``(CheckGripperContactPart <object> (<parts>))`` -- a 3-argument predicate whose first argument is the object the
#: gripper must hold (e.g. ``knife_1`` of ``grab_the_knife_and_cut_the_lemon_on_the_cutting_board``).
FALLBACK_RULE = ("first argument of the first goal predicate acting on an obj_of_interest, any arity "
                 "(LIBERO-Safety affordance CheckGripperContactPart; task_target.goal_roles takes On/In/unary only)")


def bddl_oracle_target(bddl_file, task_target_module):
    """``(target, objects, record)`` of a BDDL problem file: the ``task_target.goal_roles`` rule (goal's first object,
    the SL3 / gt_oracle rule), else :data:`FALLBACK_RULE`. Raises when neither names an object of interest."""
    problem = task_target_module.parse_bddl(bddl_file)
    try:
        target, destination = task_target_module.goal_roles(problem)
        rule = "task_target.goal_roles (first goal predicate acting on an obj_of_interest)"
    except ValueError:
        interest = list(problem["obj_of_interest"])
        hits = [p for p in problem["goal"] if len(p) >= 2 and isinstance(p[1], str) and p[1] in interest]
        if not hits:
            raise
        target, destination, rule = hits[0][1], None, FALLBACK_RULE
    return target, list(problem["object_names"]), {"bddl_file": str(bddl_file), "target": target,
                                                    "destination": destination, "rule": rule}

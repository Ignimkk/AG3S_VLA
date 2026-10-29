"""T63 step 2 — composed voxel worlds for cuRobo v2 (B0 world of user decision 2, B4 per-link world).

cuRobo's collision kernel **sums** the cost over every enabled obstacle, so two layers loaded into
one `SceneCollision` cannot express "inside the box this layer, outside that one". The composition
is done here instead, **without editing third_party/curobo**: `ComposedSceneCollision` subclasses
`SceneCollision` and overrides its query methods. Each layer lives in its own channel (a plain
`SceneCollision` with one voxel grid) and is queried with cuRobo's own kernel and arguments
(weight, activation distance, sweep, speed metric — exactly what the calling cost passed). The
per-sphere, per-step results are then composed:

| world | rule (per sphere, per step; `in_X` = centre inside layer X's lattice, first..last voxel centre, base frame) |
|---|---|
| `b0` (user decision 2, 2026-10-21) | `in_box ? box : coarse`. `box` = `esdf_free` (target excluded). A chunk recorded without one: `in_fine ? min(coarse, fine) : coarse` = our main field (`CuroboEsdfField._evaluate`) — lead default of user decision 3 (step 3; step 2 used the fine layer alone there, which differed from `min(coarse, fine)` by > 2.5 mm on 2.1 % of the in-box rows) |
| `b0m` (T63b user decision 2) | the `b0` world, each sphere and step queried at radius `r + margin`: margin = `obstacle_margin` when our TO's `_obstacle_rows` would call the row `obstacle`, else 0 (`b0m.py` rule table). The class is decided on the `b0` distance at the step centre (no gradient; cuRobo kernels with weight 1 and `eta` = `CLASS_ETA_M`, activation inverted), the target distance from a `target` channel (5 mm distance grid of the manipulated points / held spheres) and the support planes analytically. The B0m world distance is therefore `d_b0 − margin` |
| `lte` (B4) | target-excluded rows (`target_field_exclude`): the `b0` rule. Other rows: our main field — `in_fine ? min(coarse, fine) : coarse` (`CuroboEsdfField._evaluate`). Plus, for rows with a target margin (`manipulated_link_margin`; the caller passes NaN on the target-excluded rows, which our TO exempts — T30 F1), the target point cloud as an obstacle at radius `r + margin` |

Why this is exact for a discrete query: cuRobo's cost is `w · act(r + η − sdf)`, a non-increasing
function of `sdf` — so `min(sdf_a, sdf_b)` is `max(cost_a, cost_b)` and a region choice is a
`where` on the costs. Gradients: every channel is called with `return_loss=True`, so cuRobo's
autograd function scales its stored gradient by the incoming `grad_output` and `torch.where` /
`torch.maximum` route it to the channel that answered (with `return_loss=False` cuRobo ignores
`grad_output` and both channels would push). For the **swept** query (the optimizer's) each channel
sweeps on its own and the choice is made at the step's centre — a sphere that crosses the box
boundary between two steps uses one layer for the whole sub-sample sweep (stated, not measured).

The composed cost is returned; the caller's own `collision_buffer` is not written (its
`gradient` is read only by `RobotSceneCollision`, not by the trajectory optimizer).

Per-chunk data (grids, frame, box bounds, masks, margins) is **copied into** preallocated tensors,
so CUDA graphs captured on the first solve stay valid.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

import numpy as np
import torch

from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo.scene import VoxelGrid
from curobo.types import DeviceCfg

from AG3S.trajopt.curobo_v2 import b0m as _b0m
from AG3S.trajopt.curobo_v2 import world as W

#: channels per world mode (each one voxel grid)
MODE_CHANNELS = {"b0": ("box", "coarse"), "b0m": ("box", "coarse", "target"),
                 "lte": ("free", "fine", "coarse", "target")}
#: activation distance of the B0m classification queries (m): larger than any in-grid distance
#: + radius, so every in-grid sphere has a non-zero cost and the distance can be read back
#: (`world.sdf_from_activation_cost`; the parity helpers use the same 20 m)
CLASS_ETA_M = 20.0
#: padding of the target-distance grid around the target points (m): >= max margin 0.05 +
#: max sphere radius 0.0085 + activation 0.0025 + two voxels
TARGET_GRID_PAD_M = 0.10
TARGET_GRID_VOXEL_M = 0.005
#: B0m's target grid (T63b): the class test `|d − d_obj| ≤ 5 mm` needs `d_obj` more exactly than
#: B4's margin channel — the trilinear error of an unsigned point distance peaks at the points
#: (≈ √3/2 voxel: 4.3 mm at 5 mm, 2.2 mm at 2.5 mm; parity 2026-10-21). The pad only has to cover
#: rows whose margin can change the cost: `d < r + obstacle_margin + activation` = 8.5 + 10 + 10 mm
#: (`optimizer_collision_activation_distance` 0.01); farther rows read `+inf` = obstacle, whose
#: margin is then inactive. Clouds too large for the channel at 2.5 mm fall back to the B4 grid.
B0M_TARGET_GRID_VOXEL_M = 0.0025
B0M_TARGET_GRID_PAD_M = 0.06


def _channel(dev: DeviceCfg, capacity_shape, voxel_size: float, max_esdf_distance: float
             ) -> SceneCollision:
    cache = {"voxel": {"layers": 1, "dims": [float(n) * voxel_size for n in capacity_shape],
                       "voxel_size": float(voxel_size)}}
    sc = SceneCollision.from_config(SceneCollisionCfg(device_cfg=dev, scene_model=None,
                                                      cache=cache))
    sc.data.voxels.max_esdf_distance = float(max_esdf_distance)
    return sc


def voxel_grid(name: str, distance, origin: np.ndarray, voxel_size: float,
               T_base_frame: np.ndarray, dev: DeviceCfg) -> VoxelGrid:
    """`distance` is a record's numpy grid (float16, loaded bit for bit) or — T63b, serving — the
    live field's GPU grid (`DeviceEsdfField.values_device`, float32). The GPU grid is rounded to
    float16 on the device: the same IEEE round-to-nearest-even the recorder applies on the host
    (`np.asarray(values, np.float16)`), so a served chunk and its record load the same bytes."""
    if isinstance(distance, torch.Tensor):
        shape = tuple(int(v) for v in distance.shape)
        feat = (distance.detach().to(device=dev.device, dtype=torch.float16)
                .reshape(-1, 1).contiguous())
    else:
        d = np.asarray(distance)
        shape = d.shape
        feat = torch.as_tensor(np.ascontiguousarray(d.astype(np.float16)).reshape(-1, 1),
                               device=dev.device)
    centre = W.grid_centre(origin, shape, voxel_size)
    pose = W.grid_pose_in_frame(T_base_frame, centre)
    return VoxelGrid(name=name, pose=pose, dims=W.grid_dims(shape, voxel_size).tolist(),
                     voxel_size=float(voxel_size), feature_tensor=feat,
                     feature_dtype=torch.float16)


def target_distance_grid(points: Optional[np.ndarray], *, voxel_size: float = TARGET_GRID_VOXEL_M,
                         pad: float = TARGET_GRID_PAD_M, device: Any = "cuda:0",
                         max_voxels: int = 128 ** 3,
                         spheres: Optional[tuple[np.ndarray, np.ndarray]] = None
                         ) -> dict[str, Any]:
    """Unsigned distance to the nearest target point (our `_manipulated_distance` for an
    un-held target: `cKDTree(points).query`) sampled on a grid around the points (base frame).
    `spheres = (centres (K, 3), radii (K,))` instead (T63b B0m, a held object): the signed distance
    to their union, `min_k |g − c_k| − r_k` (the held branch of `_manipulated_distance`)."""
    if spheres is not None:
        c = np.asarray(spheres[0], np.float64).reshape(-1, 3)
        rad = np.asarray(spheres[1], np.float64).reshape(-1)
        lo = (c - rad[:, None]).min(0) - pad
        hi = (c + rad[:, None]).max(0) + pad
        shape = np.ceil((hi - lo) / voxel_size).astype(int) + 1
        if int(np.prod(shape)) > max_voxels:
            raise ValueError(f"target grid {shape} exceeds the channel capacity")
        axes = [lo[k] + np.arange(shape[k]) * voxel_size for k in range(3)]
        g = torch.stack(torch.meshgrid(*[torch.as_tensor(a, device=device) for a in axes],
                                       indexing="ij"), -1).reshape(-1, 3)
        out = (torch.cdist(g, torch.as_tensor(c, device=device))
               - torch.as_tensor(rad, device=device)[None]).min(dim=1).values
        return {"distance": out.reshape(*shape).cpu().numpy(), "origin": lo,
                "voxel_size": voxel_size}
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    lo = pts.min(0) - pad
    hi = pts.max(0) + pad
    shape = np.ceil((hi - lo) / voxel_size).astype(int) + 1
    if int(np.prod(shape)) > max_voxels:
        raise ValueError(f"target grid {shape} exceeds the channel capacity")
    axes = [lo[k] + np.arange(shape[k]) * voxel_size for k in range(3)]
    g = torch.stack(torch.meshgrid(*[torch.as_tensor(a, device=device) for a in axes],
                                   indexing="ij"), -1).reshape(-1, 3)
    p = torch.as_tensor(pts, device=device)
    out = torch.empty(g.shape[0], device=device, dtype=torch.float64)
    step = 65536
    for s in range(0, g.shape[0], step):
        out[s:s + step] = torch.cdist(g[s:s + step], p).min(dim=1).values
    return {"distance": out.reshape(*shape).cpu().numpy(), "origin": lo, "voxel_size": voxel_size}


class ComposedSceneCollision(SceneCollision):
    """`SceneCollision` whose queries compose several single-layer channels (see module doc)."""

    @classmethod
    def build(cls, mode: str, n_spheres: int, dev: DeviceCfg, *,
              capacity_shape=(128, 128, 128), voxel_size: float = 0.005,
              max_esdf_distance: float = 10000.0) -> "ComposedSceneCollision":
        if mode not in MODE_CHANNELS:
            raise ValueError(f"mode must be one of {sorted(MODE_CHANNELS)}")
        channels = {name: _channel(dev, capacity_shape, voxel_size, max_esdf_distance)
                    for name in MODE_CHANNELS[mode]}
        obj = cls(data=channels["coarse"].data, checker=channels["coarse"].checker,
                  device_cfg=dev)
        obj.mode = mode
        obj.channels = channels
        obj._capacity = int(np.prod(capacity_shape))
        obj.n_spheres = int(n_spheres)
        z3 = lambda: torch.zeros(3, device=dev.device, dtype=dev.dtype)  # noqa: E731
        obj.R_base_frame = torch.eye(3, device=dev.device, dtype=dev.dtype)
        obj.t_base_frame = z3()
        obj.boxes = {name: (z3(), z3()) for name in ("box", "free", "fine")}
        zn = lambda: torch.zeros(int(n_spheres), device=dev.device, dtype=dev.dtype)  # noqa: E731
        #: b0, chunk without `esdf_free`: 1 = inside the box take `min(coarse, box)` (our main field)
        obj.box_min_coarse = torch.zeros(1, device=dev.device, dtype=dev.dtype)
        obj.excl = zn()  # 1 = target-excluded row (lte)
        obj.target_on = zn()  # 1 = this row gets the target margin channel (lte)
        obj.target_margin = zn()
        # T63b B0m — the obstacle-margin rule of our TO (`b0m.py`), cuRobo sphere order
        z1 = lambda v=0.0: torch.full((1,), float(v), device=dev.device, dtype=dev.dtype)  # noqa: E731
        obj.m_contact = zn()          # 1 = the row's link may touch the target (`target` class)
        obj.m_target_on = z1()        # 1 = something is manipulated (target channel loaded)
        obj.m_planes = torch.zeros(_b0m.MAX_PLANES, 4, device=dev.device, dtype=dev.dtype)
        obj.m_plane_on = torch.zeros(_b0m.MAX_PLANES, device=dev.device, dtype=dev.dtype)
        obj.m_obstacle = z1()         # obstacle_margin (m)
        obj.m_tol_target = z1()       # |d - d_obj| tolerance (finest field voxel)
        obj.m_tol_support = z1()      # |d - s_k| tolerance (2 finest voxels)
        obj.cls_w = z1(1.0)           # weight / activation of the classification queries
        obj.cls_eta = z1(CLASS_ETA_M)
        obj._buffers: dict[tuple, CollisionBuffer] = {}
        obj.loaded: dict[str, Any] = {}
        return obj

    # ---------------------------------------------------------------------------------------------
    # per-chunk data
    # ---------------------------------------------------------------------------------------------
    def _set_frame(self, T_base_frame: np.ndarray) -> None:
        T = np.asarray(T_base_frame, np.float64)
        self.R_base_frame.copy_(torch.as_tensor(T[:3, :3], dtype=self.device_cfg.dtype))
        self.t_base_frame.copy_(torch.as_tensor(T[:3, 3], dtype=self.device_cfg.dtype))

    def _set_box(self, name: str, origin, shape, voxel_size) -> None:
        o = np.asarray(origin, np.float64).reshape(3)
        last = o + (np.asarray(shape, np.float64) - 1.0) * float(voxel_size)
        lo, hi = self.boxes[name]
        lo.copy_(torch.as_tensor(o, dtype=self.device_cfg.dtype))
        hi.copy_(torch.as_tensor(last, dtype=self.device_cfg.dtype))

    def _load(self, channel: str, layer: Mapping[str, Any], T_base_frame) -> None:
        g = voxel_grid(channel, layer["distance"], layer["origin"], layer["voxel_size"],
                       T_base_frame, self.device_cfg)
        self.channels[channel].data.voxels.load_batch([g], env_idx=0)
        self.loaded[channel] = {"origin": np.asarray(layer["origin"], np.float64).tolist(),
                                "shape": [int(v) for v in layer["distance"].shape],
                                "voxel_size": float(layer["voxel_size"])}

    def load_chunk(self, z: Mapping[str, np.ndarray], T_base_frame: np.ndarray, *,
                   excl_curobo: Optional[np.ndarray] = None,
                   target_margin_curobo: Optional[np.ndarray] = None,
                   target_points: Optional[np.ndarray] = None,
                   b0m_rules: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """Load one record's layers. `excl_curobo` / `target_margin_curobo` are `(n_spheres,)` in
        **cuRobo's** sphere order (`spec.sphere_permutation`); `lte` only. `target_margin_curobo`
        with NaN = no target margin on that row. `b0m_rules`: `b0m` only, `b0m.to_curobo(
        b0m.rules_from_scene(...))` (cuRobo order)."""
        layers = {"coarse": W.layer_arrays(z, "coarse"), "main": W.layer_arrays(z, "main")}
        if "esdf_free_distance" in z:
            layers["free"] = W.layer_arrays(z, "free")
        return self.load_layers(layers, T_base_frame, excl_curobo=excl_curobo,
                                target_margin_curobo=target_margin_curobo,
                                target_points=target_points, b0m_rules=b0m_rules)

    def load_layers(self, layers: Mapping[str, Mapping[str, Any]], T_base_frame: np.ndarray, *,
                    excl_curobo: Optional[np.ndarray] = None,
                    target_margin_curobo: Optional[np.ndarray] = None,
                    target_points: Optional[np.ndarray] = None,
                    b0m_rules: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
        """`load_chunk` on layers given directly: `{"coarse", "main", ["free"]}` ->
        `{"distance" (numpy or torch), "origin", "voxel_size"}` (T63b: the live field's layers,
        `serving.live_layers`). No `"free"` key = a chunk without a target-excluded layer."""
        self._set_frame(T_base_frame)
        coarse = layers["coarse"]
        fine = layers["main"]
        has_free = "free" in layers
        free = layers["free"] if has_free else None
        self._load("coarse", coarse, T_base_frame)
        info: dict[str, Any] = {"has_free": has_free}
        if self.mode in ("b0", "b0m"):
            box = free if has_free else fine
            self._load("box", box, T_base_frame)
            self._set_box("box", box["origin"], box["distance"].shape, box["voxel_size"])
            # T63 step 3 (user decision 3, lead default): without a target-excluded layer the box
            # is our main field — min(coarse, fine) inside the fine lattice, as our TO sees it
            self.box_min_coarse.fill_(0.0 if has_free else 1.0)
            info["box_layer"] = "free" if has_free else "main_fine"
            info["box_rule"] = "esdf_free" if has_free else "min(coarse, fine)"
            if self.mode == "b0m":
                info["b0m"] = self._set_b0m(b0m_rules, T_base_frame)
            elif b0m_rules is not None:
                raise ValueError("b0m_rules given to a b0 world")
            return info
        # lte
        self._load("fine", fine, T_base_frame)
        self._set_box("fine", fine["origin"], fine["distance"].shape, fine["voxel_size"])
        f = free if has_free else fine
        self._load("free", f, T_base_frame)
        self._set_box("free", f["origin"], f["distance"].shape, f["voxel_size"])
        ex = np.zeros(self.n_spheres) if excl_curobo is None else np.asarray(excl_curobo, float)
        if not has_free and ex.any():
            raise ValueError("target-excluded rows but no esdf_free layer in this record")
        self.excl.copy_(torch.as_tensor(ex, dtype=self.device_cfg.dtype))
        m = (np.full(self.n_spheres, np.nan) if target_margin_curobo is None
             else np.asarray(target_margin_curobo, np.float64))
        on = np.isfinite(m) & (target_points is not None)
        self.target_on.copy_(torch.as_tensor(on.astype(float), dtype=self.device_cfg.dtype))
        self.target_margin.copy_(torch.as_tensor(np.where(on, m, 0.0),
                                                 dtype=self.device_cfg.dtype))
        if on.any():
            tg = target_distance_grid(target_points, device=self.device_cfg.device)
            self._load("target", tg, T_base_frame)
        else:
            # keep a grid loaded (the graph has a fixed set of kernels); no row queries it
            self._load("target", {"distance": np.full((2, 2, 2), 1.0, np.float32),
                                  "origin": np.zeros(3), "voxel_size": 0.005}, T_base_frame)
        info.update(n_excl=int(ex.sum()), n_target_rows=int(on.sum()))
        return info

    def _set_b0m(self, rules: Optional[Mapping[str, Any]], T_base_frame) -> dict[str, Any]:
        """Copy one chunk's B0m rule (`b0m.rules_from_scene`, cuRobo order) into the fixed tensors
        and load the target channel. Refuses a chunk without a rule (B0m must not silently be B0)."""
        if rules is None:
            raise ValueError("b0m world: this chunk has no margin rule (b0m.rules_from_scene)")
        dt = self.device_cfg.dtype
        contact = np.asarray(rules["contact"], bool).reshape(-1)
        if contact.shape[0] != self.n_spheres:
            raise ValueError(f"b0m contact rows {contact.shape[0]} != {self.n_spheres} spheres")
        self.m_contact.copy_(torch.as_tensor(contact.astype(float), dtype=dt))
        planes = np.asarray(rules["planes"], np.float64).reshape(-1, 4)
        if planes.shape[0] > _b0m.MAX_PLANES:
            raise ValueError(f"b0m: {planes.shape[0]} support planes > {_b0m.MAX_PLANES}")
        P = np.zeros((_b0m.MAX_PLANES, 4))
        P[:planes.shape[0]] = planes
        on = np.zeros(_b0m.MAX_PLANES)
        on[:planes.shape[0]] = 1.0
        self.m_planes.copy_(torch.as_tensor(P, dtype=dt))
        self.m_plane_on.copy_(torch.as_tensor(on, dtype=dt))
        self.m_obstacle.fill_(float(rules["obstacle_margin"]))
        self.m_tol_target.fill_(float(rules["target_tol"]))
        self.m_tol_support.fill_(float(rules["support_tol"]))
        tgt = rules.get("target") or {"kind": None}
        kind = tgt.get("kind")
        if kind in ("points", "spheres"):
            src = ((tgt["points"],), {}) if kind == "points" else (
                (None,), {"spheres": (tgt["centres"], tgt["radii"])})
            try:
                g = target_distance_grid(*src[0], **src[1], voxel_size=B0M_TARGET_GRID_VOXEL_M,
                                         pad=B0M_TARGET_GRID_PAD_M, device=self.device_cfg.device,
                                         max_voxels=self._capacity)
            except ValueError:     # cloud too large for the channel at 2.5 mm
                g = target_distance_grid(*src[0], **src[1], device=self.device_cfg.device,
                                         max_voxels=self._capacity)
        else:
            # keep a grid loaded (fixed kernel set); `m_target_on = 0` makes it unread
            g = {"distance": np.full((2, 2, 2), 1.0, np.float32), "origin": np.zeros(3),
                 "voxel_size": TARGET_GRID_VOXEL_M}
        self._load("target", g, T_base_frame)
        self.m_target_on.fill_(0.0 if kind is None else 1.0)
        return {"obstacle_margin_m": float(rules["obstacle_margin"]),
                "target_grid_voxel_m": None if kind is None else float(g["voxel_size"]),
                "target_tol_m": float(rules["target_tol"]),
                "support_tol_m": float(rules["support_tol"]),
                "n_contact_rows": int(contact.sum()), "n_support_planes": int(planes.shape[0]),
                "target": kind}

    # ---------------------------------------------------------------------------------------------
    # composition
    # ---------------------------------------------------------------------------------------------
    def _distance(self, channel: str, spheres):
        """Signed distance of every sphere centre in one channel (no gradient): cuRobo's kernel at
        weight 1 and activation `CLASS_ETA_M`, inverted (`world.sdf_from_activation_cost`).
        `+inf` = no surface within `r + eta` (e.g. outside the channel's lattice)."""
        buf = self._buffer(self, "cls_" + channel, spheres.shape)
        ch = self.channels[channel]
        c = ch.checker.get_sphere_distance(
            scene=ch.data, query_sphere=spheres, collision_buffer=buf, weight=self.cls_w,
            activation_distance=self.cls_eta, env_query_idx=None, return_loss=True)
        eta = self.cls_eta
        lin = c > 0.5 * eta
        p = torch.where(lin, c + 0.5 * eta, torch.sqrt(torch.clamp(2.0 * eta * c, min=0.0)))
        return torch.where(c > 0.0, spheres[..., 3] + eta - p, torch.full_like(c, float("inf")))

    def _b0m_classes(self, spheres, base):
        """`(d_b0, d_obj, class)` of every sphere and step (`b0m.classify` on the GPU)."""
        d_c = self._distance("coarse", spheres)
        d_b = self._distance("box", spheres)
        d_b = torch.where(self.box_min_coarse > 0.5, torch.minimum(d_b, d_c), d_b)
        d = torch.where(self._in_box(base, "box"), d_b, d_c)
        d_t = self._distance("target", spheres)
        hit = ((d - d_t).abs() <= self.m_tol_target) & (self.m_contact > 0.5) \
            & (self.m_target_on > 0.5)
        # support planes: s_k = n_k . p - o_k in the robot base frame, elementwise (no TF32 matmul)
        s = (base.unsqueeze(-2) * self.m_planes[:, :3]).sum(-1) - self.m_planes[:, 3]
        near = ((s - d.unsqueeze(-1)).abs() <= self.m_tol_support) & (self.m_plane_on > 0.5)
        sup = near.any(-1) & ~hit
        cls = torch.where(hit, torch.full_like(d, float(_b0m.TARGET)),
                          torch.where(sup, torch.full_like(d, float(_b0m.SUPPORT)),
                                      torch.full_like(d, float(_b0m.OBSTACLE))))
        return d, d_t, cls

    def _b0m_spheres(self, spheres, base):
        """`spheres` with radius `r + margin` (margin = `obstacle_margin` on obstacle rows)."""
        with torch.no_grad():
            _, _, cls = self._b0m_classes(spheres, base)
            m = torch.where(cls == float(_b0m.OBSTACLE), self.m_obstacle,
                            torch.zeros_like(self.m_obstacle))
        r = spheres[..., 3:4]
        return torch.cat([spheres[..., :3], torch.where(r > 0.0, r + m.unsqueeze(-1), r)], dim=-1)
    def _buffer(self, key_obj, channel: str, shape) -> CollisionBuffer:
        key = (id(key_obj), channel, tuple(shape))
        buf = self._buffers.get(key)
        if buf is None:
            buf = CollisionBuffer.from_shape(tuple(shape), self.device_cfg)
            self._buffers[key] = buf
        return buf

    def _in_box(self, centres_base, name: str):
        lo, hi = self.boxes[name]
        return ((centres_base >= lo) & (centres_base <= hi)).all(-1)

    def _compose(self, spheres, call, key_obj, *, parts: tuple[str, ...] = ("world", "target")):
        xyz = spheres[..., :3]
        # elementwise, not `xyz @ R.T`: cuRobo enables TF32 matmul globally
        # (`seed_ik_solver.py`: `torch.backends.cuda.matmul.allow_tf32 = True`), whose 10-bit
        # mantissa moves a 1 m coordinate by up to ~0.5 mm — enough to flip the in-box test
        base = (xyz.unsqueeze(-2) * self.R_base_frame).sum(-1) + self.t_base_frame
        shape = spheres.shape

        def q(channel, sph):
            return call(self.channels[channel], sph, self._buffer(key_obj, channel, shape))

        if self.mode == "b0m":
            spheres = self._b0m_spheres(spheres, base)
        c_coarse = q("coarse", spheres)
        if self.mode in ("b0", "b0m"):
            c_box = q("box", spheres)
            # max of the costs = min of the distances (module doc); a 0/1 tensor, not a Python
            # branch, so one captured graph serves chunks with and without `esdf_free`
            c_box = torch.where(self.box_min_coarse > 0.5, torch.maximum(c_box, c_coarse), c_box)
            return torch.where(self._in_box(base, "box"), c_box, c_coarse)
        out = None
        if "world" in parts:
            c_fine = q("fine", spheres)
            c_free = q("free", spheres)
            c_main = torch.where(self._in_box(base, "fine"), torch.maximum(c_fine, c_coarse),
                                 c_coarse)
            c_fc = torch.where(self._in_box(base, "free"), c_free, c_coarse)
            out = torch.where(self.excl > 0.5, c_fc, c_main)
        if "target" in parts:
            r = spheres[..., 3:4]
            on = (self.target_on > 0.5).unsqueeze(-1) & (r > 0.0)
            r_t = torch.where(on, r + self.target_margin.unsqueeze(-1), torch.full_like(r, -1.0))
            c_t = q("target", torch.cat([xyz, r_t], dim=-1))
            out = c_t if out is None else torch.maximum(out, c_t)
        return out

    # cuRobo's query API (state-based and raw) ---------------------------------------------------
    def get_sphere_distance(self, state, collision_buffer, weight, activation_distance,
                            env_query_idx=None, return_loss=False):
        return self.get_sphere_distance_raw(state.robot_spheres, collision_buffer, weight,
                                            activation_distance, env_query_idx, return_loss)

    def get_sphere_collision(self, state, collision_buffer, weight, activation_distance,
                             env_query_idx=None, return_loss=False):
        return self.get_sphere_distance(state, collision_buffer, weight, activation_distance,
                                        env_query_idx, return_loss)

    def get_swept_sphere_distance(self, state, collision_buffer, weight, activation_distance,
                                  trajectory_dt, enable_speed_metric=False, env_query_idx=None,
                                  return_loss=False):
        return self.get_swept_sphere_distance_raw(
            state.robot_spheres, collision_buffer, weight, activation_distance, trajectory_dt,
            enable_speed_metric, env_query_idx, return_loss)

    def get_swept_sphere_collision(self, state, collision_buffer, weight, activation_distance,
                                   trajectory_dt, enable_speed_metric=False, env_query_idx=None,
                                   return_loss=False):
        return self.get_swept_sphere_distance(state, collision_buffer, weight,
                                              activation_distance, trajectory_dt,
                                              enable_speed_metric, env_query_idx, return_loss)

    def get_sphere_distance_raw(self, query_spheres, collision_buffer, weight,
                                activation_distance, env_query_idx=None, return_loss=False,
                                parts: tuple[str, ...] = ("world", "target")):
        def call(ch, sph, buf):
            return ch.checker.get_sphere_distance(
                scene=ch.data, query_sphere=sph, collision_buffer=buf, weight=weight,
                activation_distance=activation_distance, env_query_idx=env_query_idx,
                return_loss=True)
        return self._compose(query_spheres, call, collision_buffer, parts=parts)

    def get_swept_sphere_distance_raw(self, query_spheres, collision_buffer, weight,
                                      activation_distance, trajectory_dt,
                                      enable_speed_metric=False, env_query_idx=None,
                                      return_loss=False):
        def call(ch, sph, buf):
            return ch.checker.get_swept_sphere_distance(
                scene=ch.data, query_sphere=sph, collision_buffer=buf, weight=weight,
                activation_distance=activation_distance, trajectory_dt=trajectory_dt,
                enable_speed_metric=enable_speed_metric, env_query_idx=env_query_idx,
                return_loss=True)
        return self._compose(query_spheres, call, collision_buffer)

    # parity helper ------------------------------------------------------------------------------
    def signed_distance(self, spheres_frame: np.ndarray, *, part: str = "world",
                        eta: float = 20.0) -> np.ndarray:
        """`(N, 4)` spheres (solver frame, **cuRobo order irrelevant — per-row masks are in cuRobo
        order, so pass spheres in cuRobo order for `lte`**) -> composed signed distance of the
        centre (activation inverted, `+inf` = no obstacle within `r + eta`). For `part="target"`
        the radius used is `r + margin` (the returned value is the distance to the target cloud)."""
        sph = np.asarray(spheres_frame, np.float64).reshape(-1, 4)
        n = self.n_spheres
        if sph.shape[0] % n:
            raise ValueError("pass whole sphere sets (multiples of n_spheres)")
        qt = torch.as_tensor(sph, dtype=self.device_cfg.dtype,
                             device=self.device_cfg.device).view(1, -1, n, 4)
        buf = CollisionBuffer.from_shape(qt.shape, self.device_cfg)
        w = torch.ones(1, dtype=self.device_cfg.dtype, device=self.device_cfg.device)
        act = torch.full((1,), float(eta), dtype=self.device_cfg.dtype,
                         device=self.device_cfg.device)
        parts = ("world",) if self.mode in ("b0", "b0m") or part == "world" else ("target",)
        with torch.no_grad():
            cost = self.get_sphere_distance_raw(qt, buf, w, act, parts=parts)
        r = sph[:, 3].copy()
        if part == "target":
            r = r + np.tile(self.target_margin.detach().cpu().numpy(), sph.shape[0] // n)
        return W.sdf_from_activation_cost(cost.detach().cpu().numpy().reshape(-1), r, eta)

    def b0m_classes(self, spheres_frame: np.ndarray) -> dict[str, np.ndarray]:
        """`b0m` parity helper: `(N, 4)` spheres (solver frame, cuRobo order, whole sets) ->
        `{"d_b0", "d_obj", "cls", "margin"}` per sphere, as the optimizer's queries decide them."""
        if self.mode != "b0m":
            raise ValueError("b0m_classes needs a b0m world")
        sph = np.asarray(spheres_frame, np.float64).reshape(-1, 4)
        n = self.n_spheres
        qt = torch.as_tensor(sph, dtype=self.device_cfg.dtype,
                             device=self.device_cfg.device).view(1, -1, n, 4)
        with torch.no_grad():
            xyz = qt[..., :3]
            base = (xyz.unsqueeze(-2) * self.R_base_frame).sum(-1) + self.t_base_frame
            d, d_t, cls = self._b0m_classes(qt, base)
            m = torch.where(cls == float(_b0m.OBSTACLE), self.m_obstacle,
                            torch.zeros_like(self.m_obstacle))
        f = lambda t: t.detach().cpu().numpy().reshape(-1).astype(np.float64)  # noqa: E731
        return {"d_b0": f(d), "d_obj": f(d_t), "cls": f(cls).astype(np.int8), "margin": f(m)}


__all__ = ["ComposedSceneCollision", "MODE_CHANNELS", "target_distance_grid", "voxel_grid"]

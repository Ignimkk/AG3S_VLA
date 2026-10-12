"""pi05_libero attention -> Panda cameras: V11AB's attention read-out, bound to LIBERO (SL3).

V11AB reads one attention cell of a second, attention-enabled copy of the policy
(`AG3S/trajopt/attention_policy.py`): Euler step 0 of the denoising loop, the ``last`` action query,
layer 8, head 2, over the 768 image tokens of the prefix (three 16x16 SigLIP grids,
`policy_record.CAMERA_BINDINGS`). This module applies **the same sampler and the same cell** to
pi05_libero and says where each number lands in the Panda adapter's OpenCV images. Nothing in AG3S
core is modified: `AttentionSampler`, `AGGREGATIONS` and the cell constants are imported as they are.

Token blocks (:data:`PI05_LIBERO_TOKEN_BLOCKS`; :func:`read_token_layout` reads the real order, block
lengths and image masks from the model input instead of assuming them)::

    prefix tokens   pi05_libero image key    Panda wire camera   robosuite camera      used
    0   .. 256      base_0_rgb               head                agentview             yes
    256 .. 512      left_wrist_0_rgb         left_wrist          robot0_eye_in_hand    yes
    512 .. 768      right_wrist_0_rgb        -                   -                     no (zeros, image mask False)

Pixel mapping (:func:`unmirror_grid`, :func:`patch_centre_uv`, :func:`attention_pixel_map`). The client
builds the policy image as ``resize_with_pad(raw[::-1, ::-1], 224)`` and the adapter's OpenCV image is
``raw[::-1]`` (SL1: ``policy256[i, j] = opencv[i, W-1-j]``). With continuous coordinates in which pixel
``i`` spans ``[i, i+1)``:

* ``resize_with_pad`` 256 -> 224 of a square image is a pure scale by 224/256 with zero padding
  (:func:`resize_with_pad_geometry` restates openpi_client's arithmetic and checks it), and PIL's
  resize is centre-aligned, so policy ``x_p`` <-> mirrored ``x_m = x_p * 256/224``;
* undoing the mirror: ``x_o = 256 - x_m``. Rows are unchanged.

Patch ``(r, c)`` of the policy grid (policy pixels ``[14c, 14c+14)``) is therefore exactly OpenCV block
rows ``[16r, 16r+16)``, columns ``[16(15-c), 16(16-c))``: un-mirroring is ``grid[:, ::-1]`` and the
224 -> 256 scale is absorbed by the grid's own resampling. :func:`attention_pixel_map` feeds the
un-mirrored grid to V11AB's ``GridAttentionAdapter.to_pixel_map`` -- the pixel-centre bilinear
resampling that ``attention_lifting.lift`` uses on depth images -- so the Panda result is what V11AB
would lift. ``unmirror=False`` gives the same maps without undoing the mirror (comparison only).

Coordinates: :func:`patch_centre_uv` returns the patch centre in the **index** convention of the
adapter's K (``K_CONVENTION = "pixel_centre"``: pixel centres at integer indices,
``cx = (W-1)/2``) by default; ``convention="edge"`` returns the value of V11AB's
``policy_record.patch_to_uv`` (pixel ``i`` spans ``[i, i+1)``), which is 0.5 px larger.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

#: SigLIP grid side (V11AB `policy_record.ATTENTION_GRID`).
GRID = 16
TOKENS_PER_IMAGE = GRID * GRID
#: Client policy image size (``resize_size``) and adapter / LIBERO render size.
POLICY_SIZE = 224
OPENCV_SIZE = 256


@dataclasses.dataclass(frozen=True)
class TokenBlock:
    """One image token block of the pi05_libero prefix."""

    model_key: str
    tokens: tuple[int, int]
    wire_camera: Optional[str]
    robosuite_camera: Optional[str]
    image_mask: bool
    v11ab_policy_camera: str  # the RB-Y1 `CAMERA_BINDINGS` key at the same token positions

    @property
    def used(self) -> bool:
        return self.wire_camera is not None


#: pi05_libero (``LiberoInputs`` + ``Pi0.embed_prefix``) image token blocks, in prefix order.
PI05_LIBERO_TOKEN_BLOCKS: tuple[TokenBlock, ...] = (
    TokenBlock("base_0_rgb", (0, 256), "head", "agentview", True, "cam_high"),
    TokenBlock("left_wrist_0_rgb", (256, 512), "left_wrist", "robot0_eye_in_hand", True, "cam_left_wrist"),
    TokenBlock("right_wrist_0_rgb", (512, 768), None, None, False, "cam_right_wrist"),
)
#: Wire cameras with attention (adapter ``camera.CAMERA_TABLE`` order).
WIRE_CAMERAS: tuple[str, ...] = tuple(b.wire_camera for b in PI05_LIBERO_TOKEN_BLOCKS if b.used)


def block_for(wire_camera: str) -> TokenBlock:
    for block in PI05_LIBERO_TOKEN_BLOCKS:
        if block.wire_camera == wire_camera:
            return block
    raise ValueError(f"no pi05_libero token block for camera {wire_camera!r}; have {WIRE_CAMERAS}")


def v11ab_cell() -> dict:
    """V11AB's attention cell, read from `AG3S.trajopt.attention_policy` (never restated here)."""
    from AG3S.trajopt import attention_policy as ap

    return {"denoise_step": int(ap._DENOISE_STEP), "aggregation": str(ap._AGG),  # noqa: SLF001
            "layer": int(ap._LAYER), "head": int(ap._HEAD),  # noqa: SLF001
            "source": "AG3S/trajopt/attention_policy.py _DENOISE_STEP/_AGG/_LAYER/_HEAD"}


# --- geometry ------------------------------------------------------------------------------------------
def resize_with_pad_geometry(in_hw: Sequence[int], out_hw: Sequence[int]) -> dict:
    """openpi_client ``_resize_with_pad_pil`` arithmetic: resized size, padding, scale (output / input).

    Restated (the function itself needs PIL images); `tests/benchmark/test_panda_sl3.py` checks it
    against the real function on a marker image.
    """
    in_h, in_w = (int(v) for v in in_hw)
    out_h, out_w = (int(v) for v in out_hw)
    if (in_h, in_w) == (out_h, out_w):
        return {"resized_hw": (in_h, in_w), "pad_top": 0, "pad_left": 0, "scale_y": 1.0, "scale_x": 1.0}
    ratio = max(in_w / out_w, in_h / out_h)
    rh, rw = int(in_h / ratio), int(in_w / ratio)
    return {"resized_hw": (rh, rw), "pad_top": max(0, int((out_h - rh) / 2)), "pad_left": max(0, int((out_w - rw) / 2)),
            "scale_y": rh / in_h, "scale_x": rw / in_w}


def check_square_no_padding(in_size: int = OPENCV_SIZE, out_size: int = POLICY_SIZE) -> dict:
    """The client resize is a pure scale (no padding) for the square LIBERO render; raises otherwise."""
    geom = resize_with_pad_geometry((in_size, in_size), (out_size, out_size))
    if geom["pad_top"] or geom["pad_left"] or geom["resized_hw"] != (out_size, out_size):
        raise ValueError(f"resize_with_pad {in_size}->{out_size} pads: {geom}")
    return geom


def unmirror_grid(grid: np.ndarray, *, unmirror: bool = True) -> np.ndarray:
    """Policy-orientation ``(..., G, G)`` grid -> OpenCV orientation (columns reversed). ``unmirror=False``
    returns the grid unchanged (the "mirror not undone" comparison mapping)."""
    grid = np.asarray(grid)
    return grid[..., ::-1] if unmirror else grid


def opencv_col_of_policy_col(col, *, unmirror: bool = True, grid: int = GRID):
    """Policy grid column -> OpenCV grid column."""
    return (grid - 1 - np.asarray(col)) if unmirror else np.asarray(col)


def patch_centre_uv(row, col, *, size: int = OPENCV_SIZE, unmirror: bool = True, convention: str = "index",
                    grid: int = GRID):
    """Centre of **policy** grid patch ``(row, col)`` in the ``size``x``size`` OpenCV image, ``(u, v)``.

    ``convention="index"``: pixel centres at integer indices (the adapter K's ``pixel_centre``).
    ``convention="edge"``: pixel ``i`` spans ``[i, i+1)`` -- equals V11AB ``patch_to_uv`` of the
    OpenCV-orientation patch.
    """
    if convention not in ("index", "edge"):
        raise ValueError(f"convention must be 'index' or 'edge', got {convention!r}")
    c = opencv_col_of_policy_col(col, unmirror=unmirror, grid=grid)
    shift = 0.5 if convention == "index" else 0.0
    u = (np.asarray(c, np.float64) + 0.5) * size / grid - shift
    v = (np.asarray(row, np.float64) + 0.5) * size / grid - shift
    return u, v


def policy_patch_of_uv(u, v, *, size: int = OPENCV_SIZE, unmirror: bool = True, grid: int = GRID):
    """OpenCV pixel (index convention) -> the **policy** grid patch ``(row, col)`` whose footprint holds it."""
    cu = np.clip(np.floor((np.asarray(u, np.float64) + 0.5) * grid / size), 0, grid - 1).astype(np.int64)
    rv = np.clip(np.floor((np.asarray(v, np.float64) + 0.5) * grid / size), 0, grid - 1).astype(np.int64)
    return rv, opencv_col_of_policy_col(cu, unmirror=unmirror, grid=grid)


def policy_pixel_to_opencv_uv(x_p, y_p, *, policy_size: int = POLICY_SIZE, size: int = OPENCV_SIZE,
                              unmirror: bool = True):
    """Policy-image pixel (index convention) -> OpenCV pixel (index convention). Continuous: centre-aligned
    scale ``size/policy_size``, then ``x -> size - x`` (edge convention) to undo the mirror."""
    s = size / policy_size
    xm = (np.asarray(x_p, np.float64) + 0.5) * s  # edge convention, mirrored image
    ym = (np.asarray(y_p, np.float64) + 0.5) * s
    xo = (size - xm) if unmirror else xm
    return xo - 0.5, ym - 0.5


def patch_footprint(row: int, col: int, *, size: int = OPENCV_SIZE, unmirror: bool = True, grid: int = GRID):
    """``(row slice, col slice)`` of OpenCV pixels that policy patch ``(row, col)`` covers (``size % grid == 0``)."""
    if size % grid:
        raise ValueError(f"size {size} is not a multiple of the grid {grid}")
    p = size // grid
    c = int(opencv_col_of_policy_col(col, unmirror=unmirror, grid=grid))
    return slice(int(row) * p, (int(row) + 1) * p), slice(c * p, (c + 1) * p)


def attention_pixel_map(grid_policy: np.ndarray, out_hw: Sequence[int] = (OPENCV_SIZE, OPENCV_SIZE), *,
                        unmirror: bool = True, interpolation: str = "bilinear") -> np.ndarray:
    """``(G, G)`` policy-orientation grid -> ``(H, W)`` OpenCV attention map by V11AB's own resampling
    (``GridAttentionAdapter(interpolation).to_pixel_map``, pixel-centre alignment), after un-mirroring."""
    from AG3S.ag3s.stages.attention_lifting import GridAttentionAdapter

    grid = np.ascontiguousarray(unmirror_grid(np.asarray(grid_policy, np.float32), unmirror=unmirror))
    return GridAttentionAdapter(interpolation=interpolation).to_pixel_map(grid, (int(out_hw[0]), int(out_hw[1])))


def opencv_mask_to_policy_patches(mask: np.ndarray, *, unmirror: bool = True, grid: int = GRID) -> np.ndarray:
    """Fraction of each **policy** patch covered by an OpenCV-orientation boolean mask ``(H, W)``
    (``H``, ``W`` multiples of ``grid``). The inverse direction of :func:`attention_pixel_map`, for scoring
    a token-order attention grid against a segmentation mask without resampling the attention."""
    mask = np.asarray(mask, np.float64)
    H, W = mask.shape
    if H % grid or W % grid:
        raise ValueError(f"mask {mask.shape} is not a multiple of the grid {grid}")
    cov = mask.reshape(grid, H // grid, grid, W // grid).mean(axis=(1, 3))  # OpenCV orientation
    return unmirror_grid(cov, unmirror=unmirror)  # the column reversal is its own inverse


def mapping_conventions() -> dict:
    """JSON-able statement of the mapping (manifest / sidecars)."""
    geom = check_square_no_padding()
    return {
        "policy_image": "client: resize_with_pad(raw[::-1, ::-1], 224) (openpi_client, PIL bilinear); "
                        "adapter OpenCV image = raw[::-1]; policy256[i, j] = opencv[i, W-1-j] (SL1)",
        "resize_with_pad_256_to_224": geom,
        "grid": GRID, "policy_size": POLICY_SIZE, "opencv_size": OPENCV_SIZE,
        "unmirror": "opencv_grid = policy_grid[:, ::-1]; policy patch (r, c) = OpenCV rows [16r, 16r+16), "
                    "cols [16(15-c), 16(16-c))",
        "pixel_map": "AG3S GridAttentionAdapter(interpolation='bilinear').to_pixel_map(opencv_grid, (256, 256)) "
                     "(attention_lifting pixel-centre resampling)",
        "patch_centre_uv_default": "index convention (pixel centres at integers, = adapter K pixel_centre); "
                                   "V11AB patch_to_uv = edge convention = index + 0.5",
        "comparison_mapping": "unmirror=False: the policy grid used as if it were OpenCV-oriented",
    }


# --- token layout (read from the model input) --------------------------------------------------------------
def read_token_layout(model, observation, *, sampler=None) -> dict:
    """Image token blocks of the prefix **as the model builds them**, for one (batched) observation.

    * order: ``Pi0.embed_prefix`` iterates ``obs.images`` of the preprocessed observation;
    * tokens per image: ``PaliGemma.img`` output length (``jax.eval_shape``, no compute);
    * image masks: the observation's ``image_masks``; the prefix input mask actually used is taken from
      V11AB's ``AttentionSampler`` prefix pass (``sampler``; one pass) and compared block by block.
    """
    import flax.nnx as nnx
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    obs = _model.preprocess_observation(None, observation, train=False)
    graphdef, state = nnx.split(model)

    def image_tokens(state, image):
        m = nnx.merge(graphdef, state)
        return m.PaliGemma.img(image, train=False)[0]

    blocks, start = [], 0
    for key in obs.images:  # embed_prefix order
        shape = jax.eval_shape(image_tokens, state, obs.images[key]).shape
        n = int(shape[1])
        blocks.append({"model_key": str(key), "start": start, "stop": start + n, "tokens": n,
                       "image_mask": bool(np.asarray(obs.image_masks[key]).reshape(-1)[0])})
        start += n
    text = int(obs.tokenized_prompt.shape[1]) if obs.tokenized_prompt is not None else 0
    layout = {"blocks": blocks, "n_image_tokens": start, "text_tokens": text, "prefix_len": start + text}
    if sampler is not None:
        _, mask = sampler._jit_prefix(sampler._state, obs)  # noqa: SLF001 -- V11AB's own prefix pass
        mask = np.asarray(mask)[0]
        layout["prefix_mask_len"] = int(mask.shape[0])
        for b in blocks:
            seg = mask[b["start"]:b["stop"]]
            b["prefix_mask_all"] = bool(seg.all())
            b["prefix_mask_any"] = bool(seg.any())
        layout["text_mask_valid"] = int(mask[start:].sum())
    return layout


def check_token_layout(layout: Mapping[str, Any]) -> list[str]:
    """Differences between a read layout and :data:`PI05_LIBERO_TOKEN_BLOCKS` (empty list = agrees).

    Also checks that V11AB's ``CAMERA_BINDINGS`` (what `AttentionSampler` slices by) puts its three
    slices at the same token positions.
    """
    from AG3S.ag3s.experiments.sources.policy_record import CAMERA_BINDINGS

    problems = []
    blocks = list(layout["blocks"])
    if [b["model_key"] for b in blocks] != [t.model_key for t in PI05_LIBERO_TOKEN_BLOCKS]:
        problems.append(f"order {[b['model_key'] for b in blocks]} != {[t.model_key for t in PI05_LIBERO_TOKEN_BLOCKS]}")
    for b, t in zip(blocks, PI05_LIBERO_TOKEN_BLOCKS):
        if (b["start"], b["stop"]) != t.tokens:
            problems.append(f"{t.model_key}: tokens {(b['start'], b['stop'])} != {t.tokens}")
        if b["image_mask"] != t.image_mask:
            problems.append(f"{t.model_key}: image mask {b['image_mask']} != {t.image_mask}")
        for key in ("prefix_mask_all", "prefix_mask_any"):
            if key in b and b[key] != t.image_mask:
                problems.append(f"{t.model_key}: {key} {b[key]} != image mask {t.image_mask}")
        if tuple(CAMERA_BINDINGS[t.v11ab_policy_camera][2]) != t.tokens:
            problems.append(f"V11AB CAMERA_BINDINGS[{t.v11ab_policy_camera}] {CAMERA_BINDINGS[t.v11ab_policy_camera][2]} "
                            f"!= {t.tokens}")
    if layout.get("n_image_tokens") != TOKENS_PER_IMAGE * len(PI05_LIBERO_TOKEN_BLOCKS):
        problems.append(f"n_image_tokens {layout.get('n_image_tokens')} != {TOKENS_PER_IMAGE * len(PI05_LIBERO_TOKEN_BLOCKS)}")
    return problems


# --- the V11AB read-out ----------------------------------------------------------------------------------
class PandaAttentionReader:
    """V11AB's attention read-out on pi05_libero inputs.

    ``policy``: an openpi ``Policy`` whose model has ``return_attn_probs=True``
    (`pi05_attention.load_policy`; V11AB's `load_attention_model` returns its ``_model``). Its
    ``_input_transform`` turns a recorded request (``observation/image`` ...) into the model input, as
    `AttentionPolicy._attention_and_feature` does with the serving policy (the transforms are the same).

    :meth:`read` = V11AB `AttentionPolicy._attention_and_feature` with the subtask feature on (the
    served default): one `AttentionSampler.attention_and_subtask_feature` call at denoise step 0,
    ``AGGREGATIONS[agg]``, the ``[layer, head]`` cell -- plus, for diagnosis, the whole ``[L, H, 768]``
    pooled array of the same pass.
    """

    def __init__(self, policy, *, num_denoise_steps: int = 10, noise_seed: int = 0):
        from AG3S.ag3s.experiments.sources.pi05_attention import AttentionSampler

        self.policy = policy
        self.cell = v11ab_cell()
        self.noise_seed = int(noise_seed)
        self.num_denoise_steps = int(num_denoise_steps)
        self.sampler = AttentionSampler(policy._model, num_steps=num_denoise_steps)  # noqa: SLF001

    def observation(self, request: Mapping[str, Any]):
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model

        inputs = self.policy._input_transform(dict(request))  # noqa: SLF001
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None, ...], inputs)
        return _model.Observation.from_dict(inputs)

    def token_layout(self, request: Mapping[str, Any]) -> dict:
        return read_token_layout(self.policy._model, self.observation(request), sampler=self.sampler)  # noqa: SLF001

    def read(self, request: Mapping[str, Any], *, full: bool = True, with_feature: bool = True) -> dict:
        """One recorded request -> ``{"cell_tokens": (768,) f32, "grids": {wire: (16,16) f32 policy
        orientation}, "pooled": (L, H, 768) f32 or None, "feature": (512,) f32 or None, "seconds": float}``."""
        from AG3S.ag3s.experiments.sources.pi05_attention import AGGREGATIONS

        model_obs = self.observation(request)
        step = self.cell["denoise_step"]
        t0 = time.perf_counter()
        if with_feature:
            blocks, feature = self.sampler.attention_and_subtask_feature(model_obs, [step], noise_seed=self.noise_seed)
        else:
            blocks, feature = self.sampler.attention(model_obs, [step], noise_seed=self.noise_seed), None
        seconds = time.perf_counter() - t0  # np.asarray inside the sampler synchronizes the device
        pooled = np.asarray(AGGREGATIONS[self.cell["aggregation"]](blocks[step]), np.float32)  # [L, H, 768]
        cell = pooled[self.cell["layer"], self.cell["head"]]
        grids = {b.wire_camera: cell[b.tokens[0]:b.tokens[1]].reshape(GRID, GRID)
                 for b in PI05_LIBERO_TOKEN_BLOCKS if b.used}
        return {"cell_tokens": cell, "grids": grids, "pooled": pooled if full else None,
                "feature": feature, "seconds": seconds}


def v11ab_attention_policy_grids(policy, request: Mapping[str, Any], *, noise_seed: int = 0,
                                 num_denoise_steps: int = 10) -> dict:
    """The cell V11AB's own `AttentionPolicy` returns for ``request`` (no subtask probe), keyed by the
    RB-Y1 policy camera names -- the reference :class:`PandaAttentionReader` is checked against."""
    from AG3S.trajopt.attention_policy import AttentionPolicy

    wrapper = AttentionPolicy(policy, policy._model, num_denoise_steps=num_denoise_steps,  # noqa: SLF001
                              noise_seed=noise_seed, subtask_probe=None)
    return {k: np.asarray(v, np.float32) for k, v in wrapper._attention(dict(request)).items()}  # noqa: SLF001


__all__ = [
    "GRID", "OPENCV_SIZE", "PI05_LIBERO_TOKEN_BLOCKS", "POLICY_SIZE", "PandaAttentionReader", "TokenBlock",
    "WIRE_CAMERAS", "attention_pixel_map", "block_for", "check_square_no_padding", "check_token_layout",
    "mapping_conventions", "opencv_col_of_policy_col", "opencv_mask_to_policy_patches", "patch_centre_uv",
    "patch_footprint", "policy_patch_of_uv", "policy_pixel_to_opencv_uv", "read_token_layout",
    "resize_with_pad_geometry", "unmirror_grid", "v11ab_attention_policy_grids", "v11ab_cell",
]

"""T49 — noun-token query: the pure pieces (no jax, no MuJoCo, no GPU).

pi0.5's prefix is fully bidirectional (`third_party/openpi/src/openpi/models/pi0.py:108-139`:
image and language tokens all get ``ar_mask=False``), so a **text** row of the prefix attention
looks at the 768 image columns. T49 reads the rows of two prompt nouns — the target fruit and
``basket`` — and turns each into a three-camera 16x16 map, the way Kang et al. (arXiv 2503.06287,
appendix D.4) build one map per noun.

Prefix layout (`policy_record.CAMERA_BINDINGS`, `LANGUAGE_TOKEN_START`)::

    [cam_high 0:256 | cam_left_wrist 256:512 | cam_right_wrist 512:768 | text 768:768+max_token_len]

so the query row of text token ``i`` is ``768 + i`` and text token 0 is ``<bos>``
(`models/tokenizer.py:28-29`, ``add_bos=True``). The prompt string is
``"Task: {prompt}, State: {ints};\\nAction: "`` (`tokenizer.py:22-29`): the task comes first, so the
noun positions never depend on the state string.

Everything here is deterministic numpy (or numpy-API compatible, see `slice_probs`) so
`tests/ag3s/test_t49_noun_query.py` can pin it on synthetic data.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from AG3S.ag3s.experiments.sources.policy_record import (
    ATTENTION_GRID, CAMERA_BINDINGS, LANGUAGE_TOKEN_START, POLICY_CAMERA_NAMES)
from AG3S.ag3s.experiments.studies.t47_head_metrics import (
    PATCH_HIT_COV, combine_maps, map_scores)

#: Image tokens before the text in the prefix (3 cameras x 16 x 16).
N_IMAGE_TOKENS = LANGUAGE_TOKEN_START
#: Query rows read per prefix pass, in this order. ``target`` = the prompt's fruit noun (for a swap
#: variant: the substituted fruit), ``basket`` = the destination noun, ``last_text`` = the last valid
#: text token (the ``"▁"`` after ``"Action:"``; exploratory, recorded so a later query change needs
#: no second GPU pass — T49.task.md "되돌아올 지점").
QUERY_NAMES = ("target", "basket", "last_text")
#: The destination noun every 16D prompt uses; its GT body is ``crate``
#: (`attention_report.py:64-70` alias).
DESTINATION_NOUN = "basket"
DESTINATION_BODY = "crate"
#: Fruits of the transport scene (`t47_replay.FRUITS`).
FRUITS = ("apple", "banana", "orange", "pear")
#: SentencePiece word-start marker.
SP_SPACE = "▁"
#: GT class index of the destination in `t47_head_metrics.CLASS_NAMES`.
DESTINATION_CLASS = 1
TARGET_CLASS = 0


# ------------------------------------------------------------------------- noun positions


def noun_piece(noun: str) -> str:
    """The SentencePiece piece of a word that follows a space: ``"▁" + noun``."""
    return SP_SPACE + str(noun)


def piece_ids(sp, nouns: Sequence[str]) -> dict[str, int]:
    """``{noun: piece id}`` for single-piece nouns; raises when a noun is not one known piece.

    `sp` is a `sentencepiece.SentencePieceProcessor` (or anything with ``piece_to_id``,
    ``id_to_piece`` and ``unk_id``). The round trip ``id_to_piece(piece_to_id(p)) == p`` and
    ``id != unk`` together say that ``p`` is one vocabulary entry (T49.task.md: ``▁apple · ▁banana ·
    ▁pear · ▁orange · ▁basket`` are single pieces).
    """
    out = {}
    unk = int(sp.unk_id())
    for noun in nouns:
        piece = noun_piece(noun)
        pid = int(sp.piece_to_id(piece))
        if pid == unk or str(sp.id_to_piece(pid)) != piece:
            raise ValueError(f"{piece!r} is not a single SentencePiece piece (id {pid}, unk {unk})")
        out[str(noun)] = pid
    return out


def find_positions(token_ids: np.ndarray, mask: np.ndarray, piece_id: int) -> np.ndarray:
    """Indices of `piece_id` among the **valid** (mask True) text tokens."""
    ids = np.asarray(token_ids).reshape(-1)
    m = np.asarray(mask, bool).reshape(-1)
    if ids.shape != m.shape:
        raise ValueError(f"token ids {ids.shape} and mask {m.shape} differ")
    return np.flatnonzero(m & (ids == int(piece_id)))


def query_positions(token_ids: np.ndarray, mask: np.ndarray, noun_ids: Mapping[str, int], *,
                    bos_id: Optional[int] = None,
                    queries: Sequence[str] = QUERY_NAMES) -> tuple[dict[str, int], list[str]]:
    """Text indices of the query tokens of one tokenized prompt, and the errors found.

    `noun_ids` maps a query name (``target``, ``basket``) to the piece id it must match. A noun
    query is an error unless its piece occurs **exactly once** among the valid tokens (0 or >= 2
    matches -> the frame is recorded as an error, never guessed). ``last_text`` is the last valid
    index. With `bos_id`, text token 0 must be ``<bos>`` (the prefix row of index 0 is then BOS).
    """
    ids = np.asarray(token_ids).reshape(-1)
    m = np.asarray(mask, bool).reshape(-1)
    errors: list[str] = []
    pos: dict[str, int] = {}
    valid = np.flatnonzero(m)
    if valid.size == 0:
        return pos, ["no valid text token"]
    if not np.array_equal(valid, np.arange(valid.size)):
        errors.append("valid text tokens are not a prefix of the sequence")
    if bos_id is not None and int(ids[0]) != int(bos_id):
        errors.append(f"text token 0 is {int(ids[0])}, not <bos> {int(bos_id)}")
    for q in queries:
        if q == "last_text":
            pos[q] = int(valid[-1])
            continue
        if q not in noun_ids:
            errors.append(f"{q}: no piece id given")
            continue
        hits = find_positions(ids, m, noun_ids[q])
        if hits.size != 1:
            errors.append(f"{q}: {hits.size} matches of piece id {int(noun_ids[q])} "
                          f"(need exactly 1)")
            continue
        pos[q] = int(hits[0])
    return pos, errors


def prefix_rows(positions: Mapping[str, int], queries: Sequence[str] = QUERY_NAMES) -> np.ndarray:
    """Prefix-attention rows (``768 + text index``) of the queries, int32, in `queries` order."""
    return np.asarray([N_IMAGE_TOKENS + int(positions[q]) for q in queries], np.int32)


# ----------------------------------------------------------------------- probs -> maps


def slice_probs(probs, rows, *, n_image: int = N_IMAGE_TOKENS, xp=np) -> dict:
    """Rows of a prefix attention tensor and their mass summary — runs inside `jax.jit` (``xp=jnp``).

    `probs` is ``[L, B, K, G, T, S]`` (`gemma.py:233,254` stacked over layers by `nn.scan`;
    gemma_2b is multi-query so ``K == 1`` and ``G`` is the head axis), batch 0 and kv-head 0 are
    read. `rows` are absolute prefix rows (int). Returns float32:

    ``image`` ``[L, G, R, n_image]`` — the image columns (the only big array leaving the device);
    ``image_mass`` / ``text_mass`` ``[L, G, R]`` — sums over the image columns / over every text
    column (``n_image:``, BOS included; padded columns are masked to ~0 by the softmax);
    ``bos`` ``[L, G, R]`` — column ``n_image`` (text token 0 = ``<bos>``);
    ``self`` ``[L, G, R]`` — the row's own column (``probs[..., r, r]``).
    """
    p = probs[:, 0, 0]  # [L, G, T, S]
    sel = xp.take(p, rows, axis=2).astype(xp.float32)  # [L, G, R, S]
    image = sel[..., :n_image]
    text = sel[..., n_image:]
    idx = xp.broadcast_to(xp.reshape(rows, (1, 1, -1, 1)), sel.shape[:3] + (1,))
    self_col = xp.take_along_axis(sel, idx, axis=-1)[..., 0]
    return {"image": image, "image_mass": image.sum(axis=-1), "text_mass": text.sum(axis=-1),
            "bos": sel[..., n_image], "self": self_col}


def rows_to_maps(image: np.ndarray) -> np.ndarray:
    """``[L, G, R, 768]`` image columns -> ``(R, L, G, C, 16, 16)`` in `POLICY_CAMERA_NAMES` order.

    The same reshape as `pi05_attention.main` (`pooled[..., lo:hi].reshape(L, H, 16, 16)` per
    camera, slices from `CAMERA_BINDINGS`), so a noun map and an action map share one convention.
    """
    a = np.asarray(image)
    if a.ndim != 4 or a.shape[-1] != N_IMAGE_TOKENS:
        raise ValueError(f"expected [L, G, R, {N_IMAGE_TOKENS}], got {a.shape}")
    a = np.moveaxis(a, 2, 0)  # [R, L, G, 768]
    g = ATTENTION_GRID
    cams = []
    for cam in POLICY_CAMERA_NAMES:
        lo, hi = CAMERA_BINDINGS[cam][2]
        cams.append(a[..., lo:hi].reshape(a.shape[:3] + (g, g)))
    return np.stack(cams, axis=3)


# --------------------------------------------------------------------------- prompts


def swap_prompt(prompt: str, old: str, new: str) -> str:
    """`prompt` with the word `old` replaced by `new`; `old` must occur exactly once as a word."""
    pat = re.compile(rf"\b{re.escape(old)}\b")
    hits = pat.findall(prompt)
    if len(hits) != 1:
        raise ValueError(f"{old!r} occurs {len(hits)} times as a word in {prompt!r} (need 1)")
    return pat.sub(new, prompt)


def swap_fruits(target: str, fruits: Sequence[str] = FRUITS) -> list[str]:
    """The scene's other fruits (the prompt-swap nouns), in `FRUITS` order."""
    if target not in fruits:
        raise ValueError(f"target {target!r} is not one of {tuple(fruits)}")
    return [f for f in fruits if f != target]


def variant_name(fruit: Optional[str]) -> str:
    return "orig" if fruit is None else f"swap_{fruit}"


def variant_plan(prompt: str, target: str, *, swaps: bool,
                 fruits: Sequence[str] = FRUITS) -> list[dict]:
    """``[{variant, prompt, noun, swapped_from}]`` — the recorded prompt, then one per other fruit."""
    plan = [{"variant": "orig", "prompt": prompt, "noun": target, "swapped_from": None}]
    if swaps:
        for f in swap_fruits(target, fruits):
            plan.append({"variant": variant_name(f), "prompt": swap_prompt(prompt, target, f),
                         "noun": f, "swapped_from": target})
    return plan


# ------------------------------------------------------------------------- cell ranking


def all_cells(n_layers: int, n_heads: int) -> list[tuple[int, int]]:
    """Every ``(layer, head)`` (layer-major) — the GT-based primary ranks all 18 x 8."""
    return [(l, h) for l in range(n_layers) for h in range(n_heads)]


def flat_scores(maps: np.ndarray, cov: np.ndarray, *, target_index: int) -> dict:
    """`map_scores` of ``(F, C, 16, 16)`` maps against ``(F, C, K, 16, 16)`` coverage."""
    m = np.asarray(maps, np.float64)
    c = np.asarray(cov, np.float64)
    f, n_c = m.shape[:2]
    return map_scores(m.reshape((f * n_c,) + m.shape[2:]),
                      c.reshape((f * n_c,) + c.shape[2:]), target_index=target_index)


def cell_metrics(blocks: Sequence[np.ndarray], covs: Sequence[np.ndarray],
                 keeps: Sequence[np.ndarray], *, target_index: int) -> dict:
    """Per cell, pooled over every kept (frame, camera) of every record.

    `blocks[r]` is ``(F, C, L, H, 16, 16)``, `covs[r]` ``(F, C, K, 16, 16)``, `keeps[r]` ``(F, C)``
    bool. Returns ``(L, H)`` arrays ``lift`` (nan-mean, as `t47_head_selection.compute_oracle`),
    ``peak_on_target``, ``target_mass`` and the sample count ``n``.
    """
    n_l, n_h = np.asarray(blocks[0]).shape[2:4]
    lift = np.full((n_l, n_h), np.nan)
    peak = np.full((n_l, n_h), np.nan)
    tmass = np.full((n_l, n_h), np.nan)
    n = int(sum(int(np.asarray(k, bool).sum()) for k in keeps))
    for l in range(n_l):
        for h in range(n_h):
            lv, pv, tv = [], [], []
            for b, c, k in zip(blocks, covs, keeps):
                sc = flat_scores(np.asarray(b)[:, :, l, h], c, target_index=target_index)
                kk = np.asarray(k, bool).reshape(-1)
                lv.append(sc["lift"][kk])
                pv.append(sc["peak_on_target"][kk])
                tv.append(sc["target_mass"][kk])
            lv, pv, tv = np.concatenate(lv), np.concatenate(pv), np.concatenate(tv)
            if lv.size:
                with np.errstate(invalid="ignore"):
                    lift[l, h] = np.nanmean(lv) if np.isfinite(lv).any() else np.nan
                    tmass[l, h] = np.nanmean(tv) if np.isfinite(tv).any() else np.nan
                peak[l, h] = pv.mean()
    return {"lift": lift, "peak_on_target": peak, "target_mass": tmass, "n": n}


def rank_by_lift(lift: np.ndarray, peak: np.ndarray) -> list[tuple[int, int]]:
    """Cells best first: mean lift desc, then peak_on_target desc, then cell index (NaN last)."""
    lf = np.asarray(lift, np.float64).reshape(-1)
    pk = np.asarray(peak, np.float64).reshape(-1)
    n_h = np.asarray(lift).shape[1]
    order = np.lexsort((np.arange(lf.size), -np.nan_to_num(pk, nan=-np.inf),
                        -np.nan_to_num(lf, nan=-np.inf)))
    return [(int(i // n_h), int(i % n_h)) for i in order]


def ranking_table(order: Sequence[tuple[int, int]], metrics: dict, n: int = 20) -> list[dict]:
    return [{"layer": l, "head": h, "lift": float(metrics["lift"][l, h]),
             "peak_on_target": float(metrics["peak_on_target"][l, h]),
             "target_mass": float(metrics["target_mass"][l, h])} for l, h in list(order)[:n]]


def combined(block: np.ndarray, cells: Sequence[Sequence[int]], sigma: float) -> np.ndarray:
    """``(F, C, L, H, 16, 16)`` -> ``(F, C, 16, 16)``: T47's `combine_maps` (smooth, raw sum)."""
    return combine_maps(block, [tuple(int(x) for x in c) for c in cells], float(sigma))


# --------------------------------------------------------------------------- mass (#1)


def mass_summary(image_mass: np.ndarray, text_mass: np.ndarray, bos: np.ndarray,
                 self_mass: np.ndarray, camera_mass: np.ndarray, *,
                 min_image_mass: float = 0.01) -> dict:
    """Where one query row's softmax mass goes, per layer (median over samples x heads).

    Inputs are ``(N, L, H)`` (``camera_mass`` ``(N, L, H, C)``) over N kept frames. Returns per-layer
    lists: ``image`` / ``text`` (BOS included) / ``bos`` / ``text_ex_bos`` / ``self`` medians,
    ``image_plus_text`` (should be ~1: the image and text columns are the whole prefix),
    per-camera share of the image mass, ``(L, H)`` median matrices, and
    ``layers_image_mass_below`` = layers whose median image mass < `min_image_mass` (a reading
    aid only — the threshold travels with the result).
    """
    im = np.asarray(image_mass, np.float64)
    tx = np.asarray(text_mass, np.float64)
    bo = np.asarray(bos, np.float64)
    sf = np.asarray(self_mass, np.float64)
    cm = np.asarray(camera_mass, np.float64)
    if im.ndim != 3:
        raise ValueError(f"expected (N, L, H), got {im.shape}")

    def per_layer(x):
        return np.median(x.transpose(1, 0, 2).reshape(x.shape[1], -1), axis=1)

    with np.errstate(invalid="ignore", divide="ignore"):
        share = cm / np.where(im > 0, im, np.nan)[..., None]  # (N, L, H, C)
    share_l = np.nanmedian(share.transpose(1, 0, 2, 3).reshape(im.shape[1], -1, cm.shape[-1]),
                           axis=1)
    image_l = per_layer(im)
    out = {
        "n_samples": int(im.shape[0]),
        "per_layer_median": {
            "image": image_l.tolist(), "text": per_layer(tx).tolist(), "bos": per_layer(bo).tolist(),
            "text_ex_bos": per_layer(tx - bo).tolist(), "self": per_layer(sf).tolist(),
            "image_plus_text": per_layer(im + tx).tolist(),
            "camera_share_of_image": {c: share_l[:, i].tolist()
                                      for i, c in enumerate(POLICY_CAMERA_NAMES)}},
        "per_cell_median": {"image": np.median(im, axis=0).tolist(),
                            "text": np.median(tx, axis=0).tolist(),
                            "bos": np.median(bo, axis=0).tolist()},
        "image_plus_text_max_abs_dev": float(np.abs(im + tx - 1.0).max()) if im.size else math.nan,
        "min_image_mass": float(min_image_mass),
        "layers_image_mass_below": [int(l) for l in np.flatnonzero(image_l < min_image_mass)],
    }
    return out


# ---------------------------------------------------------------------- separation (#5)


def _joint_peak(m: np.ndarray) -> tuple[int, int, int]:
    """``(camera, row, col)`` of the largest raw value over the three cameras (cf. G2)."""
    a = np.asarray(m, np.float64)
    c, r, col = np.unravel_index(int(np.argmax(a)), a.shape)
    return int(c), int(r), int(col)


def _joint_share(m: np.ndarray, cov: np.ndarray) -> float:
    a = np.asarray(m, np.float64)
    tot = a.sum()
    return float((a * np.asarray(cov, np.float64)).sum() / tot) if tot > 0 else math.nan


def separation_frame(tmap: np.ndarray, bmap: np.ndarray, class_cov: np.ndarray,
                     class_px: np.ndarray, *, min_px: int = 200,
                     hit_cov: float = PATCH_HIT_COV) -> dict:
    """One frame's target-map / basket-map separation over the three cameras.

    `tmap` / `bmap` ``(C, 16, 16)``, `class_cov` ``(C, K, 16, 16)``, `class_px` ``(C, K)`` (class 0 =
    target, 1 = destination / crate). The peak is the **joint** argmax over the 768 patches (as the
    fused raw argmax, `target_grounding.py` G2); a hit is that patch's coverage > `hit_cov`
    (step-01). Eligibility: the class has >= `min_px` pixels in at least one camera.

    Masses are joint shares (``sum(map * cov) / sum(map)`` over the three cameras):
    ``target_map_crate_mass`` and ``basket_map_target_mass`` are the cross-talk.
    """
    cov = np.asarray(class_cov, np.float64)
    px = np.asarray(class_px)
    tc, tr, tq = _joint_peak(tmap)
    bc, br, bq = _joint_peak(bmap)
    t_vis = bool((px[:, TARGET_CLASS] >= min_px).any())
    c_vis = bool((px[:, DESTINATION_CLASS] >= min_px).any())
    t_hit = bool(cov[tc, TARGET_CLASS, tr, tq] > hit_cov)
    b_hit = bool(cov[bc, DESTINATION_CLASS, br, bq] > hit_cov)
    return {
        "target_visible": t_vis, "crate_visible": c_vis, "both_visible": t_vis and c_vis,
        "target_peak_on_target": t_hit,
        "target_peak_on_crate": bool(cov[tc, DESTINATION_CLASS, tr, tq] > hit_cov),
        "basket_peak_on_crate": b_hit,
        "basket_peak_on_target": bool(cov[bc, TARGET_CLASS, br, bq] > hit_cov),
        "both_correct": t_hit and b_hit,
        "same_peak_patch": (tc, tr, tq) == (bc, br, bq),
        "target_peak": [tc, tr, tq], "basket_peak": [bc, br, bq],
        "target_map_target_mass": _joint_share(tmap, cov[:, TARGET_CLASS]),
        "target_map_crate_mass": _joint_share(tmap, cov[:, DESTINATION_CLASS]),
        "basket_map_crate_mass": _joint_share(bmap, cov[:, DESTINATION_CLASS]),
        "basket_map_target_mass": _joint_share(bmap, cov[:, TARGET_CLASS]),
    }


#: rate name -> (hit field, eligibility field).
SEPARATION_RATES = {
    "target_peak_on_target_rate": ("target_peak_on_target", "target_visible"),
    "target_peak_on_crate_rate": ("target_peak_on_crate", "crate_visible"),
    "basket_peak_on_crate_rate": ("basket_peak_on_crate", "crate_visible"),
    "basket_peak_on_target_rate": ("basket_peak_on_target", "target_visible"),
    "both_correct_rate": ("both_correct", "both_visible"),
    "same_peak_patch_rate": ("same_peak_patch", None),
}
SEPARATION_MEANS = {
    "target_map_target_mass": "target_visible", "target_map_crate_mass": "crate_visible",
    "basket_map_crate_mass": "crate_visible", "basket_map_target_mass": "target_visible",
}


# ------------------------------------------------------------------------- swap (#7)


def map_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation of two maps over all their cells (NaN when either is constant)."""
    x = np.asarray(a, np.float64).reshape(-1)
    y = np.asarray(b, np.float64).reshape(-1)
    if x.size != y.size:
        raise ValueError("maps differ in size")
    sx, sy = x.std(), y.std()
    if sx == 0 or sy == 0:
        return math.nan
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def _camera_lifts(m: np.ndarray, cov: np.ndarray, px: np.ndarray, min_px: int) -> np.ndarray:
    """Per camera lift of map `m` (C,16,16) on one body's coverage `cov` (C,16,16); NaN if < min_px."""
    sc = map_scores(np.asarray(m, np.float64), np.asarray(cov, np.float64)[:, None],
                    target_index=0)
    lift = np.asarray(sc["lift"], np.float64)
    return np.where(np.asarray(px) >= min_px, lift, np.nan)


def _nanmean(x) -> float:
    v = np.asarray(x, np.float64)
    v = v[np.isfinite(v)]
    return float(v.mean()) if v.size else math.nan


def swap_frame(orig_map: np.ndarray, swap_map: np.ndarray, body_cov: np.ndarray,
               body_px: np.ndarray, *, new_index: int, orig_index: int, min_px: int = 200,
               hit_cov: float = PATCH_HIT_COV) -> dict:
    """Does the target-noun map follow the noun? One frame, one substituted fruit.

    `orig_map` (recorded prompt) and `swap_map` (prompt with the target replaced by the new fruit)
    are ``(C, 16, 16)``; `body_cov` ``(C, B, 16, 16)`` / `body_px` ``(C, B)`` are the GT cache's named
    body coverage; `new_index` / `orig_index` pick the new fruit and the original target.

    Joint peak (argmax over the 768 patches) on the new fruit / on the original target, for both
    maps; ``moved`` = orig peak on the original target **and** swap peak on the new fruit; camera
    mean lifts of both maps on both bodies (cameras where the body has >= `min_px` pixels);
    Pearson correlation of the two maps.
    """
    cov = np.asarray(body_cov, np.float64)
    px = np.asarray(body_px)
    new_vis = bool((px[:, new_index] >= min_px).any())
    orig_vis = bool((px[:, orig_index] >= min_px).any())
    out: dict[str, Any] = {"new_visible": new_vis, "orig_visible": orig_vis,
                           "both_visible": new_vis and orig_vis}
    for tag, m in (("orig", orig_map), ("swap", swap_map)):
        c, r, q = _joint_peak(m)
        out[f"{tag}_peak_on_new"] = bool(cov[c, new_index, r, q] > hit_cov)
        out[f"{tag}_peak_on_orig"] = bool(cov[c, orig_index, r, q] > hit_cov)
        out[f"{tag}_peak"] = [c, r, q]
        out[f"{tag}_lift_new"] = _nanmean(_camera_lifts(m, cov[:, new_index], px[:, new_index],
                                                       min_px))
        out[f"{tag}_lift_orig"] = _nanmean(_camera_lifts(m, cov[:, orig_index], px[:, orig_index],
                                                        min_px))
    out["moved"] = bool(out["orig_peak_on_orig"] and out["swap_peak_on_new"])
    out["peak_patch_changed"] = out["orig_peak"] != out["swap_peak"]
    out["lift_new_gain"] = out["swap_lift_new"] - out["orig_lift_new"]
    out["map_corr"] = map_corr(orig_map, swap_map)
    return out


SWAP_RATES = {
    "swap_peak_on_new_rate": ("swap_peak_on_new", "new_visible"),
    "orig_peak_on_new_rate": ("orig_peak_on_new", "new_visible"),
    "swap_peak_on_orig_rate": ("swap_peak_on_orig", "orig_visible"),
    "orig_peak_on_orig_rate": ("orig_peak_on_orig", "orig_visible"),
    "moved_rate": ("moved", "both_visible"),
    "peak_patch_changed_rate": ("peak_patch_changed", None),
}
SWAP_MEANS = {"swap_lift_new": None, "orig_lift_new": None, "swap_lift_orig": None,
              "orig_lift_orig": None, "lift_new_gain": None, "map_corr": None}


# ------------------------------------------------------------------ record-level summary


def _strata(row: dict, extra: Sequence[str] = ()) -> list[str]:
    s, p = row.get("scene_group", "unknown"), row.get("phase_group", "unknown")
    out = ["all", f"scene={s}", f"phase={p}", f"scene={s}|phase={p}"]
    for key in extra:
        out.append(f"{key}={row.get(key)}")
    return out


def record_level(rows: Sequence[dict], rates: Mapping[str, tuple], means: Mapping[str, Any]
                 ) -> dict:
    """One record's frames pooled: rates over eligible frames, nan-means of the mean fields."""
    out: dict[str, Any] = {"n_frames": len(rows)}
    for name, (hit, elig) in rates.items():
        el = [r for r in rows if elig is None or bool(r.get(elig, False))]
        out[f"{name}__n"] = len(el)
        out[name] = (sum(bool(r.get(hit)) for r in el) / len(el)) if el else math.nan
    for name, elig in means.items():
        vals = [r.get(name) for r in rows if elig is None or bool(r.get(elig, False))]
        out[name] = _nanmean([math.nan if v is None else v for v in vals])
    return out


def summarize_records(rows: Sequence[dict], rates: Mapping[str, tuple],
                      means: Mapping[str, Any], *, group_key: str = "source",
                      extra_strata: Sequence[str] = ()) -> dict:
    """``{group: {stratum: {per_record, record_mean, record_min, record_max, n_records}}}``.

    Frames are pooled inside a record; the cross-record number is the unweighted mean of the
    per-record values (T47b `t47b_metrics.summarize` convention — frames are not independent).
    Strata: ``all``, ``scene=``, ``phase=``, ``scene=|phase=`` and ``<key>=`` for `extra_strata`.
    """
    out: dict[str, Any] = {}
    for grp in sorted({str(r[group_key]) for r in rows}):
        groups: dict[str, dict[str, list]] = {}
        for r in rows:
            if str(r[group_key]) != grp:
                continue
            for st in _strata(r, extra_strata):
                groups.setdefault(st, {}).setdefault(r["record"], []).append(r)
        per = {}
        for st in sorted(groups):
            per_rec = {rec: record_level(rs, rates, means) for rec, rs in sorted(groups[st].items())}
            keys = [k for k, v in next(iter(per_rec.values())).items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            mean, lo, hi = {}, {}, {}
            for k in keys:
                v = np.asarray([per_rec[r][k] for r in per_rec], np.float64)
                fin = v[np.isfinite(v)]
                mean[k] = float(fin.mean()) if fin.size else math.nan
                lo[k] = float(fin.min()) if fin.size else math.nan
                hi[k] = float(fin.max()) if fin.size else math.nan
            per[st] = {"n_records": len(per_rec), "per_record": per_rec, "record_mean": mean,
                       "record_min": lo, "record_max": hi}
        out[grp] = per
    return out


__all__ = [
    "DESTINATION_BODY", "DESTINATION_CLASS", "DESTINATION_NOUN", "FRUITS", "N_IMAGE_TOKENS",
    "QUERY_NAMES", "SEPARATION_MEANS", "SEPARATION_RATES", "SWAP_MEANS", "SWAP_RATES",
    "TARGET_CLASS", "all_cells", "cell_metrics", "combined", "find_positions", "flat_scores",
    "map_corr", "mass_summary", "noun_piece", "piece_ids", "prefix_rows", "query_positions",
    "rank_by_lift", "ranking_table", "record_level", "rows_to_maps", "separation_frame",
    "slice_probs", "summarize_records", "swap_frame", "swap_fruits", "swap_prompt",
    "variant_name", "variant_plan",
]

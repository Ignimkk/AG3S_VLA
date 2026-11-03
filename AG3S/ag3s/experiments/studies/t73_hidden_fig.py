"""T73 figure: CPU hidden-state extraction on the 16D checkpoint -- what came out, is it right, what it cost.

Reads the `pi05_hidden` output (``.npz`` + ``.json`` sidecar) and the smoke-test ``monitor.tsv``
(per-5 s RSS / thread count / pod working set / GPU-process presence), and writes
``AG3S/ag3s/docs/figures/t73-hidden-cpu.png`` + ``.json``.

    .venv-openpi-live/bin/python -m AG3S.ag3s.experiments.studies.t73_hidden_fig \\
        --npz $AG3S_RUN_DIR/verify/T73/hidden_fin_k0_ep119_f0_13_26.npz \\
        --monitor $AG3S_RUN_DIR/verify/T73/monitor.tsv
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

from AG3S.ag3s.experiments.common import figstyle as fs

REPO = pathlib.Path(__file__).resolve().parents[4]
FIG = REPO / "AG3S/ag3s/docs/figures/t73-hidden-cpu.png"
GROUP_LABEL = {"cam_high": "head cam", "cam_left_wrist": "left wrist cam", "cam_right_wrist": "right wrist cam",
               "text": "text (valid tokens)"}


def cosine(a, b):
    return float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))


def summarise(d) -> dict:
    h = np.asarray(d["hidden"], np.float32)  # [F, L, T, D]
    layers = [int(x) for x in d["layers"]]
    tv = np.asarray(d["token_valid"], bool)
    grp = np.asarray(d["token_group"])
    cams = [str(c) for c in d["camera_names"]]
    groups = cams + ["text"]
    F, L = h.shape[:2]
    norms = np.zeros((F, len(groups), L))
    means = np.zeros((F, len(groups), L, h.shape[-1]), np.float32)
    for f in range(F):
        for g in range(len(groups)):
            sel = (grp == g) & tv[f]
            tok = h[f][:, sel]  # [L, n, D]
            norms[f, g] = np.linalg.norm(tok, axis=-1).mean(axis=-1)
            means[f, g] = tok.mean(axis=1)
    consec = np.array([[[cosine(means[f, g, i], means[f, g, i + 1]) for i in range(L - 1)]
                        for g in range(len(cams))] for f in range(F)])  # [F, 3, L-1]
    mat = np.array([[cosine(means[0, 0, i], means[0, 0, j]) for j in range(L)] for i in range(L)])
    return {"layers": layers, "groups": groups, "norms": norms, "consec": consec, "cos_matrix_head_f0": mat}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--npz", required=True)
    ap.add_argument("--monitor", required=True)
    ap.add_argument("--out", default=str(FIG))
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = pathlib.Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists; pick a new name (figures rule)")
    npz = pathlib.Path(args.npz)
    side = json.loads(npz.with_suffix(".json").read_text())
    d = np.load(npz)
    s = summarise(d)
    mon = np.loadtxt(args.monitor, skiprows=1, ndmin=2)  # t_s rss_mib threads pod_ws_gib gpu_has_pid
    frame_ids = [str(x) for x in d["frame_id"]]
    v_err = np.asarray(d["check_v_relerr"])
    fn_err = np.asarray(d["check_final_norm_relerr"])
    absmax = np.asarray(d["absmax"])
    srcs = [str(x) for x in d["source_files"]]

    fs.style_axes  # noqa: B018 -- English labels throughout (no CJK needed)
    fig = plt.figure(figsize=(15, 10.5), dpi=150)
    gsp = fig.add_gridspec(3, 4, height_ratios=[0.9, 1.0, 1.0], hspace=0.5, wspace=0.45)

    # row 0: the three recorded head-camera frames (what the model saw)
    for k, src in enumerate(srcs[:3]):
        ax = fig.add_subplot(gsp[0, k])
        with np.load(src) as rec:
            ax.imshow(rec["image_cam_high"])
        ax.set_title(f"{frame_ids[k]}\nhead cam input (224 px)", fontsize=9, color=fs.INK)
        ax.axis("off")
    ax = fig.add_subplot(gsp[0, 3])
    ax.axis("off")
    rss_peak_gib = side["peak_rss_mib"] / 1024
    rows = [
        ["compile (AOT, once)", f"{side['compile_s']:.1f} s"],
        ["prefix pass / frame", f"{np.mean(side['frame_s']):.1f} s (n={len(side['frame_s'])})"],
        ["frame incl. compile", f"{side['compile_s'] + side['frame_s'][0]:.1f} s"],
        ["model load", f"{side['load_s']:.1f} s"],
        ["peak RSS", f"{rss_peak_gib:.2f} GiB"],
        ["output file", f"{side['out_bytes'] / 2**20:.1f} MiB ({side['dtype']})"],
        ["CPUs (affinity)", f"{side['threads']}"],
        ["OS threads (max)", f"{int(mon[:, 2].max())}"],
        ["GPU process samples", f"{int(mon[:, 4].sum())} of {len(mon)}"],
    ]
    tb = ax.table(cellText=rows, colLabels=["CPU cost", "measured"], loc="center", cellLoc="left",
                  colWidths=[0.62, 0.62])
    tb.auto_set_font_size(False)
    tb.set_fontsize(8.5)
    tb.scale(1, 1.35)
    ax.set_title("cost (16 CPU threads, no GPU)", fontsize=9, color=fs.INK, pad=18)

    L = s["layers"]
    # row 1a: per-layer mean token norm
    ax = fig.add_subplot(gsp[1, 0:2])
    for g, name in enumerate(s["groups"]):
        n = s["norms"][:, g]
        ax.plot(L, n.mean(0), color=fs.CATEGORICAL[g], lw=1.8, marker="o", ms=3, label=GROUP_LABEL[name])
        ax.fill_between(L, n.min(0), n.max(0), color=fs.CATEGORICAL[g], alpha=0.18, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("layer (0 = input embedding, l = block l output, pre final norm)")
    ax.set_ylabel("mean token L2 norm of h_l")
    ax.set_title("hidden-state norm per layer (line = mean of 3 frames, band = min-max)", fontsize=9,
                 color=fs.INK)
    ax.set_xticks(L)
    ax.legend(fontsize=8, frameon=False)
    fs.style_axes(fig, ax)

    # row 1b: consecutive-layer cosine of the camera-mean image token
    ax = fig.add_subplot(gsp[1, 2:4])
    mids = [i + 0.5 for i in L[:-1]]
    for g, name in enumerate(s["groups"][:3]):
        c = s["consec"][:, g]
        ax.plot(mids, c.mean(0), color=fs.CATEGORICAL[g], lw=1.8, marker="o", ms=3, label=GROUP_LABEL[name])
        ax.fill_between(mids, c.min(0), c.max(0), color=fs.CATEGORICAL[g], alpha=0.18, lw=0)
    ax.set_xlabel("layer pair (l, l+1)")
    ax.set_ylabel("cos(mean_img h_l, mean_img h_l+1)")
    ax.set_title("per-camera image-token mean: cosine between consecutive layers", fontsize=9, color=fs.INK)
    ax.set_xticks(L)
    ax.legend(fontsize=8, frameon=False)
    fs.style_axes(fig, ax)

    # row 2a: cosine matrix (head cam, frame 0)
    ax = fig.add_subplot(gsp[2, 0])
    im = ax.imshow(s["cos_matrix_head_f0"], cmap=fs.sequential_cmap(), vmin=min(0.0, s["cos_matrix_head_f0"].min()),
                   vmax=1.0, origin="lower")
    ax.set_xlabel("layer")
    ax.set_ylabel("layer")
    ax.set_title(f"head-cam mean: layer x layer cosine\n({frame_ids[0]})", fontsize=9, color=fs.INK)
    ax.set_xticks(L[::3])
    ax.set_yticks(L[::3])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    fs.style_axes(fig, ax)

    # row 2b: self-check
    ax = fig.add_subplot(gsp[2, 1:3])
    width = 0.27
    for f in range(v_err.shape[0]):
        ax.bar(np.arange(1, v_err.shape[1] + 1) + (f - 1) * width, np.maximum(v_err[f], 1e-9), width=width,
               color=fs.CATEGORICAL[f], label=f"V check {frame_ids[f]}")
    for f in range(len(fn_err)):
        ax.bar(v_err.shape[1] + 2 + (f - 1) * width, max(fn_err[f], 1e-9), width=width, color=fs.CATEGORICAL[f])
    ax.axhline(1e-2, color=fs.INK_2, ls="--", lw=1)
    ax.text(v_err.shape[1] + 2.6, 1e-2, "tolerance\n1e-2", fontsize=7, color=fs.INK_2, va="center")
    ax.set_yscale("log")
    ax.set_xticks(list(range(1, v_err.shape[1] + 1)) + [v_err.shape[1] + 2])
    ax.set_xticklabels([str(i) for i in range(1, v_err.shape[1] + 1)] + ["final\nnorm"], fontsize=7)
    ax.set_xlabel("block l+1 whose V cache is rebuilt from stored h_l  |  final_norm(h_18) vs prefix output")
    ax.set_ylabel("relative error (bf16)")
    ax.set_title("self-check: stored hidden reproduces the KV cache and the prefix output", fontsize=9,
                 color=fs.INK)
    ax.legend(fontsize=7, frameon=False, loc="lower left", ncol=3)
    fs.style_axes(fig, ax)

    # row 2c: memory trace
    ax = fig.add_subplot(gsp[2, 3])
    ax.plot(mon[:, 0], mon[:, 1] / 1024, color=fs.CATEGORICAL[0], lw=1.6, label="tool RSS")
    ax.set_xlabel("wall time since launch (s)")
    ax.set_ylabel("RSS (GiB)")
    ax.set_title("tool memory over the run", fontsize=9, color=fs.INK)
    ax2 = ax.twinx()
    ax2.plot(mon[:, 0], mon[:, 3], color=fs.CATEGORICAL[1], lw=1.2, ls="--", label="pod working set")
    ax2.set_ylabel("pod working set (GiB)", color=fs.INK_2)
    ax2.tick_params(colors=fs.INK_2, labelsize=8)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, frameon=False, loc="lower right")
    fs.style_axes(fig, ax)
    ax2.set_facecolor("none")

    fig.suptitle("T73  PaliGemma hidden states of the 16D 30k checkpoint, extracted on the CPU "
                 f"(prefix pass only, record FIN_k0 ep119, prompt '{side['prompt']}')", fontsize=11, color=fs.INK)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=fs.SURFACE)

    sidecar = {
        "step": "T73", "npz": str(npz), "monitor": str(args.monitor), "frame_id": frame_ids,
        "layers": L, "groups": s["groups"],
        "mean_token_norm": {name: s["norms"][:, g].tolist() for g, name in enumerate(s["groups"])},
        "consecutive_layer_cosine": {name: s["consec"][:, g].tolist() for g, name in enumerate(s["groups"][:3])},
        "cos_matrix_head_frame0": s["cos_matrix_head_f0"].tolist(),
        "check_v_relerr": v_err.tolist(), "check_final_norm_relerr": fn_err.tolist(),
        "absmax_per_layer": absmax.tolist(),
        "cost": {"compile_s": side["compile_s"], "frame_s": side["frame_s"], "load_s": side["load_s"],
                 "peak_rss_mib": side["peak_rss_mib"], "out_bytes": side["out_bytes"], "threads": side["threads"],
                 "os_threads_max": int(mon[:, 2].max()), "gpu_process_samples": int(mon[:, 4].sum()),
                 "monitor_samples": int(len(mon)), "pod_ws_gib_max": float(mon[:, 3].max())},
        "checkpoint": side["checkpoint"], "config": side["config"], "code_commit": side["code_commit"],
        "code_dirty": side["code_dirty"],
    }
    out.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()

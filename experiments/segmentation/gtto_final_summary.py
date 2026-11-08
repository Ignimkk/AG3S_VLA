#!/usr/bin/env python3
"""2026-11-08 final summary (user goal: success within 10 %p of VLA on LIBERO / SafeLIBERO, collision avoidance much better).
Reads gtto_protocol27.py's json (official numbers) and gtto_collision_cause.py's json (real collisions = a robot body or the
handled object touched the obstacle at a recorded frame), writes a table + a graph.
usage: gtto_final_summary.py --protocol P.json --cause C.json --out DIR"""
import argparse
import json
import pathlib

ARMS = [  # (name in protocol27, collision-cause arm tag, label, graph label)
    ("VLA", "vla", "VLA 단독", "VLA only"),
    ("F12", "gt15loose2", "F12 (τ 25, 어제)", "F12"),
    ("A2-B", "gt19pts", "B″ (GT 장애물 + 점모델)", "B2"),
    ("A3", "gt19obst3", "A‴ (A 단독)", "A3 (A only)"),
    ("A3-B", "gt21pts", "B‴ (A‴ + 점모델)", "B3"),
    ("A3-B-fh-t100", "gt23t100", "B5 (B‴ + chunk 전체 + τ100)", "B5"),
    ("A5-B-noGT", "gt24att", "B5 GT 없음", "B5 no-GT"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", required=True, type=pathlib.Path)
    ap.add_argument("--cause", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    proto = json.loads(a.protocol.read_text())
    res = proto.get("results", proto)
    cause = json.loads(a.cause.read_text())
    rows = []
    for key, tag, label, glabel in ARMS:
        r = res.get(key)
        if not r or not (r.get("libero_n") or 0):
            continue
        lib, sl, ls = r.get("libero") or {}, r.get("safelibero") or {}, r.get("ls") or {}
        lib_n = sum(x.get("n", 0) for x in lib.values()) if isinstance(lib, dict) else 0
        lib_k = sum(x.get("success", 0) for x in lib.values()) if isinstance(lib, dict) else 0
        real = [c for c in cause if c["arm"] == tag and c["cause"] in ("robot", "target/object")]
        rows.append({"arm": label, "glabel": glabel, "libero": f"{lib_k}/{lib_n}" if lib_n else "–",
                     "libero_rate": lib_k / lib_n if lib_n else None,
                     "sl_success": sl.get("success"), "sl_n": sl.get("n"), "sl_collided": sl.get("collided"),
                     "sl_safe": sl.get("safe_success"),
                     "sl_real": len(real) if any(c["arm"] == tag for c in cause) or sl.get("collided") == 0 else None,
                     "ls_ok": ls.get("ok"), "ls_n": ls.get("n"), "ls_viol": ls.get("violations")})
    (a.out / "final_summary.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    vla = rows[0]
    L = ["| 판 | LIBERO /27 | SafeLIBERO 성공 /11 (VLA 대비 %p) | 기록상 충돌 | 실제 충돌 (로봇 · 다루던 물체 접촉) | 안전 성공 | LIBERO-Safety 성공∧무위반 /10 (위반) |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        gap = (None if not r["sl_n"] or not vla["sl_n"] else
               round(100 * (r["sl_success"] / r["sl_n"] - vla["sl_success"] / vla["sl_n"]), 1))
        L.append(f"| {r['arm']} | {r['libero']} | {r['sl_success']} ({'' if gap is None else f'{gap:+.1f}'}) | {r['sl_collided']} | "
                 f"{'–' if r['sl_real'] is None else r['sl_real']} | {r['sl_safe']} | {r['ls_ok']} ({r['ls_viol']}) |")
    (a.out / "최종표.md").write_text("\n".join(L) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    names = [r["glabel"] for r in rows]
    x = range(len(rows))
    ax[0].bar(x, [100 * (r["libero_rate"] or 0) for r in rows], color="#4c78a8")
    ax[0].axhline(100 * (vla["libero_rate"] or 0) - 10, color="crimson", ls="--", label="VLA − 10 %p")
    ax[0].set_title("LIBERO success %")
    ax[1].bar([i - 0.2 for i in x], [100 * r["sl_success"] / r["sl_n"] if r["sl_n"] else 0 for r in rows], 0.4,
              color="#4c78a8", label="success %")
    ax[1].bar([i + 0.2 for i in x], [100 * r["sl_safe"] / r["sl_n"] if r["sl_n"] else 0 for r in rows], 0.4,
              color="#59a14f", label="safe success %")
    ax[1].axhline(100 * vla["sl_success"] / vla["sl_n"] - 10, color="crimson", ls="--", label="VLA − 10 %p")
    ax[1].set_title("SafeLIBERO (11 episodes)")
    ax[2].bar([i - 0.2 for i in x], [r["sl_collided"] or 0 for r in rows], 0.4, color="#e15759", label="recorded (1 mm rule)")
    ax[2].bar([i + 0.2 for i in x], [r["sl_real"] or 0 for r in rows], 0.4, color="#b07aa1", label="real (robot contact)")
    ax[2].set_title("SafeLIBERO collisions")
    for k in range(3):
        ax[k].set_xticks(list(x), names, rotation=30, ha="right", fontsize=8)
        ax[k].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out / "그래프_최종.png", dpi=110)
    print("\n".join(L))


if __name__ == "__main__":
    main()

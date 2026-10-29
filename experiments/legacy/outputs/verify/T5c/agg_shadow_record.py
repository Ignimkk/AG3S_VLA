"""T5c — shadow 기록(ep1807, 75 청크) 집계. 읽기만 한다.

    MUJOCO_GL=osmesa PYTHONPATH=${AG3S_PROJECT_ROOT} ${AG3S_PROJECT_ROOT}/.venv-ag3s/bin/python -u \
        outputs/verify/T5c/agg_shadow_record.py

입력
  outputs/live_test/20260925_t5shadow/shadow_ep1807/{manifest,completeness}.json · frames.jsonl
  outputs/live_test/logs/{t5shadow_run.log,serve_safe_8000_shadow.log}
출력
  outputs/verify/T5c/shadow_agg.json
"""
from __future__ import annotations
from common.workspace import workspace_path as _ag3s_workspace_path

import json
import pathlib
import re
import statistics as st

REC = pathlib.Path(_ag3s_workspace_path('outputs/live_test/20260925_t5shadow/shadow_ep1807'))
LOGS = pathlib.Path(_ag3s_workspace_path('outputs/live_test/logs'))
OUT = pathlib.Path(_ag3s_workspace_path('outputs/verify/T5c/shadow_agg.json'))


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def runs_of(seqs):
    """연속 구간으로 뭉친다 — [(start_seq, end_seq, n), ...] (seq 는 1-based)."""
    out = []
    for s in sorted(seqs):
        if out and s == out[-1][1] + 1:
            out[-1][1] = s
        else:
            out.append([s, s])
    return [{"seq_from": a, "seq_to": b, "n": b - a + 1} for a, b in out]


def main() -> None:
    obs, plan, ctrl = [], [], []
    for line in (REC / "frames.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        {"observation": obs, "planning": plan, "control": ctrl}[d["kind"]].append(d)

    # ---- 서버 로그: seq 별 ag3s / validity / grounding (ok 프레임은 로그를 남기지 않는다)
    srv = {}
    pat = re.compile(r"\[safe_policy\] seq=(\d+) ag3s=(\S+) \(validity=(\S+), grounding=([^)]+)\)")
    for line in (LOGS / "serve_safe_8000_shadow.log").read_text().splitlines():
        m = pat.search(line)
        if m:
            srv[int(m.group(1))] = {"ag3s": m.group(2), "validity": m.group(3),
                                    "grounding": m.group(4)}

    # ---- 청크별 표
    rows = []
    for p in plan:
        seq, t = int(p["seq"]), int(p["t_step"])
        v = p["verdict"]
        hr = p.get("hold_reason") or ""
        s = srv.get(seq, {})
        # grounding: 서버 로그에 줄이 있으면 그것, 없으면 ag3s_status==ok 이라 grounding ok
        grounding = s.get("grounding")
        if grounding is None:
            grounding = "ok" if v["ag3s_status"] == "ok" else "unknown"
        # camera_transform_stale 의 ms 값
        ms = [int(x) for x in re.findall(r"image is (\d+) ms behind the frame", hr)]
        skew = [int(x) for x in re.findall(r"cameras span (\d+) ms", hr)]
        rows.append({
            "seq": seq, "t_step": t,
            "ag3s_status": v["ag3s_status"],
            "grounding_status": grounding,
            "validity": s.get("validity", "certified" if v["geometry_certified"] else "unknown"),
            "geometry_certified": bool(v["geometry_certified"]),
            "trajopt_status": v["trajopt_status"],
            "max_violation_m": float(v["max_violation_m"]),
            "safe": bool(p["safe"]),
            "reason_codes": list(p.get("ag3s_reason_codes") or []),
            "has_target": grounding == "ok",
            "hold_reason_head": hr[:60],
            "camera_stale_ms": ms, "camera_skew_ms": skew,
            "timing_ms": {k: float(x) for k, x in v["timing_ms"].items()},
            "policy_infer_ms": float(p["timing_ms"]["policy_infer"]),
            "executed_chunk": p.get("executed_chunk"),
            "chunk_shape": list(p.get("chunk_shape") or []),
            "field_state": p["field"]["state"],
            "field_build_ms": (p["field"]["built_at"] - p["field"]["observed_at"]) * 1000.0,
            "n_tiers": len(p["field"]["tiers"]),
        })

    no_target = [r["seq"] for r in rows if r["ag3s_status"] == "no_target"]
    degraded = [r["seq"] for r in rows if r["ag3s_status"] == "degraded"]
    violated_only = [r["seq"] for r in rows
                     if r["ag3s_status"] == "ok" and r["trajopt_status"] == "violated"]
    has_t = [r["seq"] for r in rows if r["has_target"]]
    low_score = [r["seq"] for r in rows if r["grounding_status"] == "low_score"]

    # ---- 관측 기록: 카메라 캡처 시각
    cams = json.loads((REC / "manifest.json").read_text())["cameras"]
    cap = []
    for o in obs:
        stamps = o["camera_stamps"]
        order = sorted(stamps.items(), key=lambda kv: kv[1])
        cap.append({
            "t_step": int(o["t_step"]),
            "span_ms": (order[-1][1] - order[0][1]) * 1000.0,
            "skew_ms_recorded": float(o["camera_skew_sec"]) * 1000.0,
            "gaps_ms": {f"{order[i][0]}->{order[i+1][0]}":
                        (order[i + 1][1] - order[i][1]) * 1000.0 for i in range(len(order) - 1)},
            "first": order[0][0], "last": order[-1][0],
        })
    spans = [c["span_ms"] for c in cap]

    # ---- 상태 모순
    contradictions = []
    for r in rows:
        if r["safe"] and not r["geometry_certified"]:
            contradictions.append({"seq": r["seq"], "what": "safe=True & geometry_certified=False"})
        if r["safe"] and r["trajopt_status"] != "feasible":
            contradictions.append({"seq": r["seq"],
                                   "what": f"safe=True & trajopt={r['trajopt_status']}"})
        if (not r["safe"]) and r["ag3s_status"] == "ok" and r["trajopt_status"] == "feasible":
            contradictions.append({"seq": r["seq"], "what": "safe=False & ag3s ok & feasible"})
        if r["has_target"] and r["ag3s_status"] == "no_target":
            contradictions.append({"seq": r["seq"], "what": "grounding ok & status no_target"})
        if r["ag3s_status"] != "ok" and not r["reason_codes"] and r["ag3s_status"] != "no_target":
            contradictions.append({"seq": r["seq"], "what": "degraded without reason code"})
    # control: should_execute(=safe) 인데 executed 가 아닌 경우 / shadow 는 항상 reference 실행
    ctrl_exec = {(c["chunk_seq"], bool(c["executed"]), c.get("executed_chunk")) for c in ctrl}
    exec_kinds = sorted({k[2] for k in ctrl_exec})
    not_executed = sorted({k[0] for k in ctrl_exec if not k[1]})

    # ---- 지연
    def stage(key, first=False):
        xs = [r["timing_ms"][key] for r in rows if (r["seq"] == 1) == first]
        return {"n": len(xs), "p50": pct(xs, .5), "p95": pct(xs, .95),
                "max": max(xs), "min": min(xs)}

    timing = {k: stage(k) for k in ("infer", "ag3s", "trajopt", "total")}
    timing["field_build_ms"] = {
        "n": len(rows) - 1,
        "p50": pct([r["field_build_ms"] for r in rows[1:]], .5),
        "p95": pct([r["field_build_ms"] for r in rows[1:]], .95),
        "max": max(r["field_build_ms"] for r in rows[1:]),
        "min": min(r["field_build_ms"] for r in rows[1:]),
    }
    rt = [r["policy_infer_ms"] for r in rows[1:]]
    timing["client_roundtrip_ms_excl_first"] = {"n": len(rt), "p50": pct(rt, .5),
                                               "p95": pct(rt, .95), "max": max(rt), "min": min(rt)}
    first = {k: rows[0]["timing_ms"][k] for k in ("infer", "ag3s", "trajopt", "total")}
    first["client_roundtrip_ms"] = rows[0]["policy_infer_ms"]
    first["field_build_ms"] = rows[0]["field_build_ms"]

    out = {
        "record": str(REC),
        "n_chunks": len(rows), "n_observation": len(obs), "n_control": len(ctrl),
        "counts": {
            "safe": sum(1 for r in rows if r["safe"]),
            "unsafe": sum(1 for r in rows if not r["safe"]),
            "ag3s_status": {k: sum(1 for r in rows if r["ag3s_status"] == k)
                            for k in sorted({r["ag3s_status"] for r in rows})},
            "trajopt_status": {k: sum(1 for r in rows if r["trajopt_status"] == k)
                               for k in sorted({r["trajopt_status"] for r in rows})},
            "grounding_status": {k: sum(1 for r in rows if r["grounding_status"] == k)
                                 for k in sorted({r["grounding_status"] for r in rows})},
            "geometry_certified_true": sum(1 for r in rows if r["geometry_certified"]),
            "has_target_true": len(has_t),
            "reason_code_combos": {",".join(r["reason_codes"]) or "(none)":
                                   sum(1 for q in rows
                                       if q["reason_codes"] == r["reason_codes"])
                                   for r in rows},
        },
        "no_target": {
            "n": len(no_target), "seqs": no_target,
            "t_steps": [r["t_step"] for r in rows if r["ag3s_status"] == "no_target"],
            "runs": runs_of(no_target),
        },
        "low_score_grounding": {"n": len(low_score), "seqs": low_score, "runs": runs_of(low_score)},
        "has_target": {"n": len(has_t), "seqs": has_t,
                       "t_steps": [r["t_step"] for r in rows if r["has_target"]],
                       "runs": runs_of(has_t)},
        "degraded": {"n": len(degraded), "seqs": degraded,
                     "detail": [{"seq": r["seq"], "t_step": r["t_step"],
                                 "grounding_status": r["grounding_status"],
                                 "camera_stale_ms": r["camera_stale_ms"],
                                 "camera_skew_ms": r["camera_skew_ms"],
                                 "reason_codes": r["reason_codes"]}
                                for r in rows if r["ag3s_status"] == "degraded"]},
        "trajopt_violated_with_ag3s_ok": {
            "n": len(violated_only), "seqs": violated_only,
            "max_violation_m": [r["max_violation_m"] for r in rows if r["seq"] in violated_only]},
        "camera_capture": {
            "cameras": cams, "n_frames": len(cap),
            "span_ms": {"p50": pct(spans, .5), "p95": pct(spans, .95),
                        "min": min(spans), "max": max(spans), "mean": st.mean(spans)},
            "n_span_ge_100ms": sum(1 for s in spans if s >= 100.0),
            "n_span_ge_90ms": sum(1 for s in spans if s >= 90.0),
            "gap_ms_mean": {k: st.mean([c["gaps_ms"][k] for c in cap])
                            for k in cap[0]["gaps_ms"]},
            "order_first": sorted({c["first"] for c in cap}),
            "order_last": sorted({c["last"] for c in cap}),
            "per_frame": cap,
        },
        "consistency": {
            "n_contradictions": len(contradictions), "contradictions": contradictions,
            "control_executed_chunk_kinds": exec_kinds,
            "control_not_executed_chunk_seqs": not_executed,
            "from_completeness": json.loads((REC / "completeness.json").read_text()),
        },
        "timing_ms": {"first_chunk_seq1": first, "excluding_first": timing},
        "per_chunk": rows,
    }
    OUT.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(f"wrote {OUT}")
    print("counts:", json.dumps(out["counts"], ensure_ascii=False))
    print("no_target runs:", out["no_target"]["runs"])
    print("has_target seqs:", has_t)
    print("degraded:", json.dumps(out["degraded"]["detail"], ensure_ascii=False))
    print("camera span ms:", json.dumps(out["camera_capture"]["span_ms"]))
    print("contradictions:", len(contradictions))


if __name__ == "__main__":
    main()

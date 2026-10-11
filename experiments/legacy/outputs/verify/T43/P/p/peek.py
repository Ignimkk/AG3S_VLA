"""T43 P: print a summary of raw/psearch_<TAG>[suffix].json (full passes + trade-off by criterion). usage: peek.py TAG [suffix]"""
import json, sys
tag = sys.argv[1]; suf = sys.argv[2] if len(sys.argv) > 2 else ""
d = json.load(open(f"${AG3S_PROJECT_ROOT}/outputs/verify/T43/P/raw/psearch_{tag}{suf}.json")); C = d["candidates"]
f = lambda e: f"u={e['u']:+.3f} v={e['v']:.3f} dy={e['dyaw_deg']:+d} h={e['h']:.2f} a={e['a_min']*1e3:.1f} d={e['d_min']*1e3:.1f} bcr={e['b_crate_min']*1e3:.1f} bob={e['b_obj_min']*1e3:.1f} c={ {k: round(v*1e3,1) for k,v in e['c_min_per_run'].items()} } ca={ {k: round(v*1e3,1) for k,v in e['c_apple_per_run'].items()} } fh={e.get('free_height_over_bar_m')} pic={e['post_in_crate']}"
full = [e for e in C if e["pass_a"] and e["pass_b"] and e["pass_c"] and e["pass_d"] and e["pass_e"]]
print("full", len(full))
for e in sorted(full, key=lambda e: -min(e["a_min"], e["d_min"]))[:8]: print("  ", f(e))
for name, P, key in (("a|bcd", [e for e in C if e["pass_b"] and e["pass_c"] and e["pass_d"]], lambda e: -e["a_min"]),
                     ("d|abc", [e for e in C if e["pass_a"] and e["pass_b"] and e["pass_c"]], lambda e: -e["d_min"]),
                     ("b|acd: crate", [e for e in C if e["pass_a"] and e["pass_c"] and e["pass_d"]], lambda e: -e["b_crate_min"]),
                     ("c|abd", [e for e in C if e["pass_a"] and e["pass_b"] and e["pass_d"]], lambda e: max(e["c_min_per_run"].values()))):
    print(name, len(P))
    for e in sorted(P, key=key)[:3]: print("  ", f(e))

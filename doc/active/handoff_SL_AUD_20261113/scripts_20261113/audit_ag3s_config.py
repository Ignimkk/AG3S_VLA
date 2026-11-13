import sys, json, inspect, dataclasses
from AG3S.adapters.panda import profile as p
from AG3S.trajopt import serve_safe as ss
parser = ss.build_parser()
base = p.version_server_argv(p.FROZEN_20261112)
argv = base + list(p.version_server_flags(p.FROZEN_20261112))
def cfg(argv):
    args = parser.parse_args(argv)
    kw = p.v11ab_build_kwargs(args)
    c = p.panda_ag3s_config(p.v11ab_ag3s_config(kw))
    return kw, p.effective_config_dict(c)
kw, a = cfg(argv)
kw0, b = cfg(base[:8])
print("=== build kwargs (frozen):")
for k, v in kw.items(): print("  ", k, "=", repr(v)[:300])
print("=== AG3S config keys differing from bare:")
def flat(pre, v, out):
    if isinstance(v, dict):
        for k, x in v.items(): flat(pre + "." + k if pre else k, x, out)
    else: out[pre] = v
fa, fb = {}, {}; flat("", a, fa); flat("", b, fb)
for k in sorted(set(fa) | set(fb)):
    if fa.get(k) != fb.get(k): print(f"  {k}: {fa.get(k)!r}  (bare {fb.get(k)!r})")
json.dump(fa, open(sys.argv[1], "w"), indent=1, default=str)

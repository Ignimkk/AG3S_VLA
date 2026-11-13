import sys, json, inspect, dataclasses
from AG3S.adapters.panda import profile as p
from AG3S.adapters.panda import server as srv
from AG3S.trajopt import serve_safe as ss
parser = ss.build_parser()
argv = p.version_server_argv(p.FROZEN_20261112) + list(p.version_server_flags(p.FROZEN_20261112))
args = parser.parse_args(argv)
nd = sorted(k for k, v in vars(args).items() if v != parser.get_default(k))
ported = set(srv.PORTED_FINAL_CONFIG_DESTS); unported = set(srv.UNPORTED_FINAL_CONFIG_DESTS)
print("UNPORTED but set:", [d for d in nd if d in unported])
print("neither list:", [d for d in nd if d not in ported and d not in unported])
try:
    srv.check_unported_flags(args, None); print("check_unported_flags OK")
except Exception as e: print("check_unported:", e)
try:
    print("final_options_on:", len(srv.final_options_on(args)))
except Exception as e: print("final_options_on ERR", e)
# TO config
try:
    tc = srv._replace_horizon_limits(ss.trajopt_config_from_args(args))
    d = dataclasses.asdict(tc)
    tc0 = srv._replace_horizon_limits(ss.trajopt_config_from_args(parser.parse_args(p.version_server_argv(p.FROZEN_20261112)[:8])))
    d0 = dataclasses.asdict(tc0)
    def flat(pre, v, out):
        if isinstance(v, dict):
            for k, x in v.items(): flat(pre + "." + k if pre else k, x, out)
        else: out[pre] = v
    a, b = {}, {}; flat("", d, a); flat("", d0, b)
    print("--- TO config (frozen) differing from bare-default args:")
    for k in sorted(a):
        if a[k] != b.get(k): print(f"  {k}: {a[k]!r}  (bare {b.get(k)!r})")
    print("--- full TO config:")
    for k in sorted(a): print(f"  {k} = {a[k]!r}")
except Exception as e:
    import traceback; traceback.print_exc()

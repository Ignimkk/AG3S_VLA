"""T43 TP — cProfile of one run's offline re-solve (ta_offline.run_one) for given variants."""
import cProfile, pstats, sys, json, io
from AG3S.trajopt.experiments import ta_offline as T
key, out, first, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
variants = sys.argv[5:]
idx = "outputs/verify/T43/V8/analysis/analysis.json"
entry = json.load(open(idx))["runs"][key]
cli = T.argparse.Namespace(variants=variants, hm=False, no_plane_rows=False, repeat_every=1000,
                           progress=False, max_chunks=n, first_chunk=first, contacts=None)
pr = cProfile.Profile()
pr.enable()
res = T.run_one(key, entry, cli)
pr.disable()
pr.dump_stats(out)
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(60)
print(s.getvalue()[:12000])
for v in variants:
    import numpy as np
    c = [ch["variants"][v]["cpu_ms"][0] for ch in res["chunks"] if "variants" in ch]
    print(v, "n", len(c), "median", np.median(c), "p90", np.quantile(c, .9))

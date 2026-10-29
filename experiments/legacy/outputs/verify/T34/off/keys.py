import sys, json, numpy as np
z = np.load(sys.argv[1], allow_pickle=False); s = json.loads(str(z["summary_json"]))
for k, v in s.items():
    print(k, type(v).__name__, json.dumps(v, default=str)[:300])

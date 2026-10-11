"""Fetch the public pi05_libero checkpoint into shared storage as a new, immutable input.

Refuses to touch an existing destination. Downloads into a sibling .partial directory,
verifies GCS size + crc32c (+ md5 when GCS has one), writes per-file SHA-256, then renames into place.
Run with .venv-openpi-live (google_crc32c):
  .venv-openpi-live/bin/python -I scripts/setup/fetch_pi05_libero.py $AG3S_STORAGE_ROOT/checkpoints/panda/pi05_libero
"""
import base64
import google_crc32c
import hashlib
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BUCKET = "openpi-assets"
PREFIX = "checkpoints/pi05_libero/"
DEST = Path(sys.argv[1])
PARTIAL = DEST.with_name(DEST.name + ".partial")

if DEST.exists():
    sys.exit(f"refuse: {DEST} already exists")
PARTIAL.mkdir(parents=True, exist_ok=True)

url = (f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o?prefix="
       f"{urllib.parse.quote(PREFIX, safe='')}&fields=items(name,size,md5Hash,crc32c,generation,updated)")
items = json.load(urllib.request.urlopen(url))["items"]
records = []
for it in items:
    rel = it["name"][len(PREFIX):]
    out = PARTIAL / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    size = int(it["size"])
    sha, md5, crc = hashlib.sha256(), hashlib.md5(), google_crc32c.Checksum()
    if not (out.exists() and out.stat().st_size == size):
        src = f"https://storage.googleapis.com/{BUCKET}/{urllib.parse.quote(it['name'])}"
        t0 = time.time()
        with urllib.request.urlopen(src) as r, open(out, "wb") as f:
            while chunk := r.read(1 << 22):
                f.write(chunk)
        print(f"downloaded {rel} {size/1e9:.3f} GB in {time.time()-t0:.1f}s", flush=True)
    with open(out, "rb") as f:
        while chunk := f.read(1 << 22):
            sha.update(chunk); md5.update(chunk); crc.update(chunk)
    ok_md5 = "md5Hash" not in it or base64.b64encode(md5.digest()).decode() == it["md5Hash"]
    ok_crc = base64.b64encode(crc.digest()).decode() == it["crc32c"]
    if out.stat().st_size != size or not (ok_md5 and ok_crc):
        sys.exit(f"integrity mismatch: {rel}")
    records.append({"path": rel, "bytes": size, "sha256": sha.hexdigest(),
                    "gcs_md5": it.get("md5Hash"), "gcs_crc32c": it["crc32c"], "gcs_generation": it["generation"], "gcs_updated": it["updated"]})

records.sort(key=lambda r: r["path"])
agg = hashlib.sha256("".join(f"{r['sha256']}  {r['path']}\n" for r in records).encode()).hexdigest()
manifest = {"source": f"gs://{BUCKET}/{PREFIX}", "fetched_utc": datetime.now(timezone.utc).isoformat(),
            "files": len(records), "total_bytes": sum(r["bytes"] for r in records),
            "aggregate_sha256": agg, "aggregate_rule": "sha256 of concatenated '<sha256>  <path>\\n' lines, records sorted by path",
            "records": records}
(PARTIAL / "CHECKSUMS.json").write_text(json.dumps(manifest, indent=1))
os.rename(PARTIAL, DEST)
for p in [DEST, *DEST.rglob("*")]:
    os.chmod(p, 0o555 if p.is_dir() else 0o444)
print(json.dumps({k: v for k, v in manifest.items() if k != "records"}, indent=1))

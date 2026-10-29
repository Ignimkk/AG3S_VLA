"""posix_fadvise(DONTNEED) on the verifier's own output files (clean page cache pages only); prints pod working set before / after. usage: fadvise_drop.py DIR [DIR...]"""
import os, sys
def ws():
    cur = int(open('/sys/fs/cgroup/memory.current').read())
    inact = int([l for l in open('/sys/fs/cgroup/memory.stat') if l.startswith('inactive_file ')][0].split()[1])
    return cur - inact
b = ws(); n = 0
for root in sys.argv[1:]:
    for dp, dn, fn in os.walk(root):
        for f in fn:
            try:
                fd = os.open(os.path.join(dp, f), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
            except Exception:
                pass
print("files", n, "wset_before", b, "wset_after", ws(), "delta", ws() - b)

"""Reject Python file reads/listing/chdir that use the previous checkout."""
import os
import sys
from pathlib import Path

_INSTALLED = False
OLD = Path("/mnt/dev/work")


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    def hook(event, args):
        if event not in {"open", "os.listdir", "os.scandir", "os.chdir"} or not args:
            return
        path = args[0]
        if not isinstance(path, (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(path))
        absolute = Path(os.path.abspath(p))
        if absolute == OLD or absolute.is_relative_to(OLD):
            raise PermissionError(f"Previous checkout access denied: {event}: {absolute}")
        # Also catch aliases such as /work or an editable install symlink.
        resolved = p.resolve()
        if resolved == OLD or resolved.is_relative_to(OLD):
            raise PermissionError(f"Previous checkout alias denied: {event}: {absolute}")

    sys.addaudithook(hook)

"""T42-2: save the snapshot client's reset scene per episode. usage: capture.py EP [EP ...]"""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T42/2/p'))
import plib as L
for e in sys.argv[1:]:
    print(e, L.save_capture(int(e)), flush=True)

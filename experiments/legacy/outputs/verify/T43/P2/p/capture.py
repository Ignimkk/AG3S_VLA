"""T43 P2: save the fce4af8 client's reset scene for each episode. usage: capture.py TAG..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P2/p'))
import p2lib as F
for t in sys.argv[1:]:
    print(t, F.save_capture(t), flush=True)

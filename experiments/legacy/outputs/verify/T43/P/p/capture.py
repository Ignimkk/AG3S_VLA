"""T40 F: save the client's reset scene (model binary + state at the reset check) for each episode. usage: capture.py TAG..."""
from common.workspace import workspace_path as _ag3s_workspace_path
import sys
sys.path.insert(0, _ag3s_workspace_path('outputs/verify/T43/P/p'))
import flib as F
for t in sys.argv[1:]:
    print(t, F.save_capture(t), flush=True)

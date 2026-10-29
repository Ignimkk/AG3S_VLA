"""T41b: the branch rest_start + margin takes at C1807 t=216 (TO violated -> gate HOLD): measured HOLD, 16 steps."""
from common.workspace import workspace_path as _ag3s_workspace_path
import json, sys, numpy as np
sys.path.insert(0, _ag3s_workspace_path('')); sys.path.insert(0, _ag3s_workspace_path('tests/sim')); sys.path.insert(0, _ag3s_workspace_path('outputs/impl/T41b/p'))
import _hold_replay as H, replay_c1807 as R
client = H.load_client(); run = H.load_run(R.RUN); local = H.reset(client, run)
H.replay_to(client, local, run['control'], 216)
out = {}
for mode in ('measured', 'fixed'):
    r = H.hold(client, local, mode, 16)
    out[mode] = dict(collided=r['collided'], first_contact=r['first_contact_step_substep'],
                     min_robot_clearance_mm=round(min(x['robot_clearance_min_m'] for x in r['rows']) * 1e3, 2),
                     arm_dev_from_entry_deg_max=round(float(np.degrees(max(x['arm_dev_from_entry_rad'] for x in r['rows']))), 3))
    print(mode, out[mode], flush=True)
json.dump(out, open(_ag3s_workspace_path('outputs/impl/T41b/hold_c1807_t216.json'), 'w'), indent=1)

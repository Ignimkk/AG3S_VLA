#!/usr/bin/env python3
"""Per-task object suite results, one development episode per task."""
import csv,json
from pathlib import Path
from benchmark.adapters.libero_panda.task_target import task_target

ROOT=Path('outputs/segmentation_baseline_20261101/object_dev')

def load(name):
    j=json.loads((ROOT/name/'summary.json').read_text())
    return {r['arm']:r for r in j['results']}

def main():
    a=load('yoloe_comparison'); b=load('yoloe_conf005_comparison')
    specs=[('surface_no_robot_mask',a),('surface_gtrobot_privileged',a),('surface_gtseed_gtrobot_privileged',a),('yoloe_phrase',a),('yoloe_phrase',b)]
    for arm,d in specs:
        if len(d[arm]['task_iou'])!=10: raise ValueError('expected all 10 tasks')
    rows=[]
    for task in range(10):
        row={'task':task,'target':task_target('libero_object',task)['target']}
        for label,(arm,d) in zip(('surface','surface_gtrobot','surface_gtseed_gtrobot','yoloe_conf025','yoloe_conf005'),specs):
            row[label]=d[arm]['task_iou'][task]
        rows.append(row)
    with (ROOT/'per_task_pilot.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['| task | target | surface | surface + GT robot | surface + GT seed/robot | YOLOE 0.25 | YOLOE 0.05 |',
           '|---:|---|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} |'.format(*r.values()))
    (ROOT/'per_task_pilot.md').write_text('\n'.join(lines)+'\n')
    print('wrote 10 per-task rows')

if __name__=='__main__':main()

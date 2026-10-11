"""T32 H4 — camera geometry w.r.t. the wall line at seq 31 (chunk 30). Projective TSDF sdf = depth - z_cam: a voxel
at horizontal offset dx behind an observed face has |sdf| = dx * |a_x| / |r_x| * ... ; reported: unit ray x-component and
optical-axis x-component at each face crossing of the line."""
import numpy as np, json
z = np.load('real/frames_E3b_ep1807_r1/seq031.npz')
cc = np.array([0.5168990278469209, -0.015537153627684088, 0.8839904236630249])
out = {}
for cam in ['zed_left', 'wrist_cam_l', 'wrist_cam_r']:
    T = z[f'T/{cam}']; o = T[:3, 3]; ax = T[:3, 2]
    r = {}
    for name, x in (('nx_outer_face', -0.090), ('nx_inner_face', -0.082), ('px_inner_face', 0.082), ('px_outer_face', 0.090)):
        p = cc + np.array([x, 0, 0]); d = p - o; L = np.linalg.norm(d); du = d / L
        r[name] = {'ray_unit': du.tolist(), 'range_m': float(L), 'z_cam_m': float(d @ ax),
                   # along the horizontal +x line, z_cam grows by ax.x per metre; depth of the pixel stays that of the
                   # hit face while the voxel stays in the face's shadow, so |sdf| per metre of dx = ax.x * (1 + ...)
                   }
    out[cam] = {'pos_base_m': o.tolist(), 'optical_axis': ax.tolist(), 'rays': r}
    print(cam, np.round(o, 3), 'axis', np.round(ax, 3), {k: np.round(v['ray_unit'], 3).tolist() for k, v in r.items()})
json.dump(out, open('real/camera_rays_seq031.json', 'w'), indent=1)

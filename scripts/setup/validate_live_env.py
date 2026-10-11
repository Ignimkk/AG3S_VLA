"""Verify own imports, idle GPU arithmetic, cuRobo kernels and actual OpenPI loading."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common.run_artifacts import ensure_run
from common.workspace import PROJECT_ROOT, shared_storage_root


def idle_gpu() -> dict:
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
                                    '--format=csv,noheader'], text=True).strip()
    if apps:
        raise RuntimeError(f'GPU has another compute process; validation refused: {apps}')
    return {'compute_apps_before': apps, 'gpu': subprocess.check_output(
        ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader'],
        text=True).strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['imports', 'gpu', 'curobo', 'checkpoint'])
    args = parser.parse_args()
    run = ensure_run(settings={'validation': args.mode, 'python': sys.executable})
    result = {'mode': args.mode, 'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix}
    assert Path(sys.prefix).resolve().is_relative_to(PROJECT_ROOT)
    assert Path(sys.executable).resolve().is_relative_to(PROJECT_ROOT)
    if args.mode != 'imports':
        result['resource_preflight'] = idle_gpu()
    curobo_only = Path(sys.prefix).name == '.venv-curobo'
    modules = ['AG3S.ag3s', 'AG3S.trajopt', 'curobo', 'torch', 'numpy', 'mujoco']
    if not curobo_only:
        modules += ['openpi', 'openpi_client', 'jax', 'lerobot', 'mink']
    result['origins'] = {}
    for name in modules:
        module = importlib.import_module(name)
        origin = str(Path(module.__file__).resolve())
        assert Path(origin).is_relative_to(PROJECT_ROOT), (name, origin)
        result['origins'][name] = origin
    names = ['torch', 'numpy', 'mujoco', 'nvidia-curobo']
    if not curobo_only:
        names += ['jax', 'jaxlib', 'flax', 'orbax-checkpoint']
    result['versions'] = {name: metadata.version(name) for name in names}
    maps = Path('/proc/self/maps').read_text().splitlines()
    result['native_libraries'] = sorted({line.split()[-1] for line in maps
        if '/' in line.split()[-1] and any(key in line.split()[-1] for key in ['libtorch', 'libc10', 'libcud'])})
    assert not any('/mnt/dev/work/' in path or '/AG3S_VLA-development/' in path
                   for path in result['native_libraries'])
    if args.mode == 'gpu':
        import numpy as np
        import torch
        import jax
        import jax.numpy as jnp
        assert torch.cuda.is_available()
        tx = torch.arange(8, dtype=torch.float32, device='cuda')
        result['torch_sum'] = float((tx * tx).sum().cpu())
        result['torch_gpu'] = torch.cuda.get_device_name()
        result['jax_devices'] = [str(x) for x in jax.devices()]
        assert jax.default_backend() == 'gpu'
        result['jax_sum'] = float(np.asarray(jax.jit(lambda x: (x*x).sum())(jnp.arange(8, dtype=jnp.float32))))
        assert result['torch_sum'] == result['jax_sum'] == 140.0
    elif args.mode == 'curobo':
        import torch
        import curobo._src.runtime as curobo_runtime
        curobo_runtime.cache_dir = str(PROJECT_ROOT / '.cache/curobo')
        from curobo.perception import Mapper, MapperCfg
        from curobo._src.types.camera import CameraObservation
        from curobo._src.types.pose import Pose
        cfg = MapperCfg(extent_meters_xyz=(0.8, 0.8, 0.8), voxel_size=0.02,
                        esdf_voxel_size=0.04, grid_center=torch.tensor([0., 0., 0.4], device='cuda'),
                        truncation_distance=0.08, depth_minimum_distance=0.05,
                        depth_maximum_distance=2.0, image_height=32, image_width=32, device='cuda:0')
        mapper = Mapper(cfg)
        mapper.integrate(CameraObservation(name='synthetic-plane',
            depth_image=torch.full((1, 32, 32), 0.5, device='cuda'),
            rgb_image=torch.zeros((1, 32, 32, 3), device='cuda', dtype=torch.uint8),
            intrinsics=torch.tensor([[[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]], device='cuda'),
            pose=Pose.from_matrix(torch.eye(4, device='cuda')[None]), depth_to_meter=1.0))
        field = mapper.compute_esdf()
        torch.cuda.synchronize()
        result['mapper_type'] = str(type(mapper))
        result['esdf_type'] = str(type(field))
        values = field.feature_tensor
        from curobo._src.curobolib.backends import get_backend_name
        result['kernel_backend'] = get_backend_name()
        assert result['kernel_backend'] == 'cuda_core'
        result['esdf_shape'] = list(values.shape)
        result['esdf_finite'] = bool(torch.isfinite(values).all())
        result['esdf_min'] = float(values.min().cpu())
        result['esdf_max'] = float(values.max().cpu())
        assert result['esdf_finite'] and values.numel() > 0 and result['esdf_max'] > result['esdf_min']
        import numpy as np
        np.savez_compressed(run / f'{Path(sys.prefix).name.lstrip(".")}-synthetic-esdf.npz',
                            esdf=values.detach().cpu().numpy(), depth=np.full((32, 32), 0.5, dtype=np.float32))
    elif args.mode == 'checkpoint':
        import jax
        assert jax.default_backend() == 'gpu'
        from openpi.training import config
        from openpi.policies.policy_config import create_trained_policy
        checkpoint = shared_storage_root() / 'checkpoints/rby1/pi05_rby1_randomized_pick_place_16d_lora/rby1_randomized_pick_place_16d_30k_xla_retry_20260923/29999'
        start = time.monotonic()
        policy = create_trained_policy(config.get_config('pi05_rby1_randomized_pick_place_16d_lora'), checkpoint)
        result['load_seconds'] = time.monotonic() - start
        result['checkpoint'] = str(checkpoint.resolve())
        result['checkpoint_metadata_sha256'] = hashlib.sha256((checkpoint / '_CHECKPOINT_METADATA').read_bytes()).hexdigest()
        result['policy_type'] = str(type(policy))
        result['metadata'] = policy.metadata
        result['jax_devices'] = [str(x) for x in jax.devices()]
    result['pass'] = True
    target = run / f'{Path(sys.prefix).name.lstrip(".")}-{args.mode}.json'
    target.write_text(json.dumps(result, indent=2, default=str) + '\n')
    print(json.dumps(result, indent=2, default=str))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate a separate bringup config and reuse the installed legacy launch."""
import argparse
import os
from pathlib import Path
import subprocess
import tempfile
import yaml


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', type=Path, default=root/'fabric_sim_bringup/config/full_sim.yaml')
    parser.add_argument('--isaac-python', default=os.environ.get(
        'ISAAC_PYTHON_SH', '/media/taha/ExtermeLinux/isaacsim/python.sh'))
    parser.add_argument('--capsule-config', type=Path, default=root/'capsule_cbf/config.json')
    parser.add_argument('--cloth-start-z', type=float, default=1.05,
                        help='Initial height allowing sag above the enclosing capsule.')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    with args.base_config.open() as stream:
        cfg = yaml.safe_load(stream)
    cfg['launch']['start_obstacle_distance'] = False
    cfg['launch']['start_cbf_controller'] = False
    sim = cfg['isaac_sim']
    sim['script'] = str(root/'fabric_newton_isaacsim_capsule.py')
    sim['arguments']['external_cbf_enabled'] = False
    sim['arguments']['cloth_start_z'] = args.cloth_start_z
    if args.isaac_python:
        sim['python_sh'] = str(Path(args.isaac_python).expanduser().resolve())
        sim['working_directory'] = str(Path(sim['python_sh']).parent)
    sim.setdefault('extra_args', []).extend(['--capsule-config', str(args.capsule_config.resolve())])
    if args.dry_run:
        print(yaml.safe_dump(cfg, sort_keys=False))
        return
    if not Path(sim['python_sh']).is_file():
        parser.error('Isaac runtime missing: supply --isaac-python /path/to/python.sh or ISAAC_PYTHON_SH')
    # The temporary config lives until the launch process exits, including on Ctrl+C.
    with tempfile.TemporaryDirectory(prefix='fabric-capsule-') as directory:
        config_path = Path(directory)/'capsule_sim.yaml'
        config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        try:
            result = subprocess.run(['bash', str(root/'fabric_sim_bringup/scripts/launch_full_sim.sh'),
                                     str(config_path)], check=False)
        except KeyboardInterrupt:
            return
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

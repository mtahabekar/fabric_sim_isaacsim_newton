"""Command generation for the simulator (no ROS or Isaac needed)."""
import json
from pathlib import Path

import pytest

from fabric_sim_bringup.sim_command import (cbf_config_path, cli_arguments, find_repo_root,
                                            isaac_command, isaac_python, load_yaml)

PACKAGE = Path(__file__).resolve().parents[1]
CONFIG = PACKAGE/'config'/'full_sim.yaml'


def test_repository_root_and_default_config_resolve():
    root = find_repo_root(CONFIG)
    assert (root/'fabric_newton_isaacsim.py').is_file()
    config = load_yaml(CONFIG)
    assert cbf_config_path(config, root, Path('/nonexistent')) == root/'capsule_cbf'/'config.json'


def test_cli_arguments_map_keys_flags_and_lists():
    args = cli_arguments({'cloth_start_z': 0.85, 'headless': True, 'disable_cuda_graph': False,
                          'smoke_leader_velocity': [0, 0, -0.1], 'unused': None})
    assert args == ['--cloth-start-z', '0.85', '--headless', '--smoke-leader-velocity', '0', '0', '-0.1']


def test_isaac_command_uses_overrides_and_python_precedence(tmp_path, monkeypatch):
    root = find_repo_root(CONFIG)
    config = load_yaml(CONFIG)
    monkeypatch.delenv('ISAAC_PYTHON_SH', raising=False)
    command, cwd = isaac_command(config, root, root/'capsule_cbf'/'config.json',
                                 python_sh='/opt/isaac/python.sh', argument_overrides={'headless': True})
    assert command[:2] == ['/opt/isaac/python.sh', str(root/'fabric_newton_isaacsim.py')]
    assert cwd == '/opt/isaac'
    assert '--headless' in command and command[-2] == '--cbf-config'
    assert command[command.index('--grid-x')+1] == '56'
    monkeypatch.setenv('ISAAC_PYTHON_SH', '/env/python.sh')
    assert isaac_python(config) == Path('/env/python.sh')
    assert isaac_python(config, '/arg/python.sh') == Path('/arg/python.sh')


def test_cbf_overrides_are_merged_and_validated(tmp_path):
    root = find_repo_root(CONFIG)
    config = {'cbf': {'config': 'capsule_cbf/config.json', 'overrides': {'controller': 'none'}}}
    merged = cbf_config_path(config, root, tmp_path, overrides={'margin': 0.06})
    data = json.loads(merged.read_text())
    assert merged.parent == tmp_path
    assert data['controller'] == 'none' and data['margin'] == 0.06
    assert data['tension_limit'] == json.loads((root/'capsule_cbf'/'config.json').read_text())['tension_limit']
    with pytest.raises(ValueError):
        cbf_config_path(config, root, tmp_path, overrides={'no_such_key': 1})

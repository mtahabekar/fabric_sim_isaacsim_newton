"""Build the Isaac Sim command for fabric_newton_isaacsim.py from a YAML config.

Shared by launch/full_sim.launch.py and the fabric_sim_experiments runner, so
both start the simulator identically. Relative paths in the YAML are resolved
against the repository root: the directory holding fabric_newton_isaacsim.py
and capsule_cbf/. It is found from the FABRIC_SIM_ROOT environment variable,
from the real path of the config file, or from this module's real path (all
work with `colcon build --symlink-install`).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import yaml

SIM_SCRIPT = 'fabric_newton_isaacsim.py'
DEFAULT_ISAAC_PYTHON = '/media/taha/ExtermeLinux/isaacsim6.2/python.sh'
# Simulator switches without a value: `headless: true` -> `--headless`.
BOOLEAN_FLAGS = frozenset({'disable_cuda_graph', 'headless'})
# One BLAS thread: the filter's small LAPACK calls stall on thread hand-off.
SIM_ENV = {'OPENBLAS_NUM_THREADS': '1'}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open(encoding='utf-8') as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config must be a YAML mapping: {path}')
    return data


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return dict(value)


def as_bool(value: Any, default: bool = False) -> bool:
    if value is None or value == '':
        return default
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def find_repo_root(*hints: str | Path) -> Path:
    candidates = []
    if os.environ.get('FABRIC_SIM_ROOT'):
        candidates.append(Path(os.environ['FABRIC_SIM_ROOT']).expanduser())
    for hint in (*hints, __file__):
        path = Path(hint).expanduser().resolve()
        candidates.extend((path, *path.parents))
    for candidate in candidates:
        if (candidate/SIM_SCRIPT).is_file() and (candidate/'capsule_cbf').is_dir():
            return candidate
    raise FileNotFoundError(
        f'Cannot find the repository containing {SIM_SCRIPT}. Build with '
        '`colcon build --symlink-install`, set FABRIC_SIM_ROOT, or use absolute paths.')


def resolve(path: str | Path, root: Path) -> Path:
    path = Path(path).expanduser()
    return path if path.is_absolute() else root/path


def cli_arguments(arguments: Mapping[str, Any]) -> list[str]:
    """`{cloth_start_z: 0.85, headless: true}` -> `['--cloth-start-z', '0.85', '--headless']`."""
    out = []
    for key, value in arguments.items():
        if value is None:
            continue
        flag = '--'+key.replace('_', '-')
        if key in BOOLEAN_FLAGS:
            if as_bool(value):
                out.append(flag)
        elif isinstance(value, (list, tuple)):
            out.extend([flag, *map(str, value)])
        else:
            out.extend([flag, str(value)])
    return out


def write_cbf_config(base: Path, overrides: Mapping[str, Any], out: Path) -> Path:
    """Merge overrides into the base CBF JSON and write the result to `out`."""
    settings = json.loads(base.read_text(encoding='utf-8'))
    unknown = set(overrides)-set(settings)
    if unknown:
        raise ValueError(f'Unknown CBF setting(s) {sorted(unknown)}; see capsule_cbf/settings.py')
    settings.update(overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(settings, indent=2)+'\n', encoding='utf-8')
    return out


def cbf_config_path(config: Mapping[str, Any], root: Path, scratch: Path,
                    overrides: Mapping[str, Any] | None = None) -> Path:
    """The JSON the simulator should read: the base file, or a merged copy in `scratch`."""
    cbf = section(config, 'cbf')
    base = resolve(cbf.get('config', 'capsule_cbf/config.json'), root)
    merged = {**(cbf.get('overrides') or {}), **(overrides or {})}
    if not base.is_file():
        raise FileNotFoundError(f'CBF config not found: {base}')
    return write_cbf_config(base, merged, scratch/'cbf_config.json') if merged else base


def isaac_python(config: Mapping[str, Any], override: str | None = None) -> Path:
    """Launch argument, then ISAAC_PYTHON_SH, then isaac_sim.python_sh, then the default."""
    value = (override or os.environ.get('ISAAC_PYTHON_SH')
             or section(config, 'isaac_sim').get('python_sh') or DEFAULT_ISAAC_PYTHON)
    return Path(value).expanduser()


def isaac_command(config: Mapping[str, Any], root: Path, cbf_config: Path, *,
                  python_sh: str | None = None, argument_overrides: Mapping[str, Any] | None = None
                  ) -> tuple[list[str], str]:
    """Return (command, working directory) that starts the simulator."""
    isaac = section(config, 'isaac_sim')
    python = isaac_python(config, python_sh)
    script = resolve(isaac.get('script', SIM_SCRIPT), root)
    arguments = {**section(isaac, 'arguments'), **(argument_overrides or {})}
    extra = isaac.get('extra_args') or []
    if not isinstance(extra, list):
        raise ValueError("Config entry 'isaac_sim.extra_args' must be a list")
    command = [str(python), str(script), *cli_arguments(arguments),
               '--cbf-config', str(cbf_config), *map(str, extra)]
    cwd = isaac.get('working_directory') or python.parent
    return command, str(Path(cwd).expanduser())

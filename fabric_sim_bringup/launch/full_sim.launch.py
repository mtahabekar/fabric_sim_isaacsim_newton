#!/usr/bin/env python3
"""Start the Isaac Sim fabric simulation (sensitivity CBF in the loop) and keyboard teleop.

    ros2 launch fabric_sim_bringup full_sim.launch.py
    ros2 launch fabric_sim_bringup full_sim.launch.py headless:=true isaac_python:=/path/to/python.sh
    ros2 launch fabric_sim_bringup full_sim.launch.py dry_run:=true   # print the command only

Everything else is configured in config/full_sim.yaml.
"""
from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import Any

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction,
                            RegisterEventHandler, TimerAction)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from fabric_sim_bringup.sim_command import (SIM_ENV, as_bool, cbf_config_path, find_repo_root,
                                            isaac_command, load_yaml, section)


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> list[Any]:
    config_file = Path(LaunchConfiguration('config_file').perform(context)).expanduser()
    config = load_yaml(config_file)
    launch_cfg = section(config, 'launch')
    headless = LaunchConfiguration('headless').perform(context)
    overrides = {} if headless == '' else {'headless': as_bool(headless)}
    root = find_repo_root(config_file)
    # Merged CBF JSON (only written when the YAML has cbf.overrides).
    scratch = Path(tempfile.gettempdir())/f'fabric_sim_launch_{os.getpid()}'
    cbf_config = cbf_config_path(config, root, scratch)
    command, cwd = isaac_command(config, root, cbf_config,
                                 python_sh=LaunchConfiguration('isaac_python').perform(context) or None,
                                 argument_overrides=overrides)

    if as_bool(LaunchConfiguration('dry_run').perform(context)):
        print(f'[full_sim] repository: {root}\n[full_sim] working directory: {cwd}\n'
              f'[full_sim] CBF config: {cbf_config}\n[full_sim] command:\n{shlex.join(command)}')
        return []
    if not Path(command[0]).is_file():
        raise FileNotFoundError(f'Isaac Sim python.sh not found: {command[0]} (set isaac_sim.python_sh, '
                                'ISAAC_PYTHON_SH or isaac_python:=...)')

    actions: list[Any] = []
    if as_bool(launch_cfg.get('start_isaac'), default=True):
        isaac = ExecuteProcess(cmd=command, cwd=cwd, name='isaac_sim', output='screen',
                               emulate_tty=True, additional_env=SIM_ENV)
        actions.append(isaac)
        if as_bool(launch_cfg.get('shutdown_on_isaac_exit'), default=True):
            actions.append(RegisterEventHandler(OnProcessExit(
                target_action=isaac, on_exit=[EmitEvent(event=Shutdown(reason='Isaac Sim exited'))])))
    if as_bool(launch_cfg.get('start_teleop'), default=True):
        teleop = Node(package='fabric_teleop', executable='fabric_corner_keyboard_teleop',
                      name='fabric_corner_keyboard_teleop', output='screen', emulate_tty=True,
                      parameters=[section(section(config, 'teleop'), 'parameters')])
        delay = float(launch_cfg.get('teleop_start_delay') or 0.0)
        actions.append(TimerAction(period=delay, actions=[teleop]) if delay > 0 else teleop)
    return actions


def generate_launch_description() -> LaunchDescription:
    default_config = Path(get_package_share_directory('fabric_sim_bringup'))/'config'/'full_sim.yaml'
    return LaunchDescription([
        DeclareLaunchArgument('config_file', default_value=str(default_config),
                              description='Full simulation YAML config.'),
        DeclareLaunchArgument('isaac_python', default_value='',
                              description="Isaac Sim python.sh (overrides ISAAC_PYTHON_SH and the YAML)."),
        DeclareLaunchArgument('headless', default_value='',
                              description="'true' runs Kit without a window; empty keeps the YAML value."),
        DeclareLaunchArgument('dry_run', default_value='false',
                              description='Print the simulator command and exit.'),
        OpaqueFunction(function=_launch_setup),
    ])

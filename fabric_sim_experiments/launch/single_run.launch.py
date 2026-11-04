#!/usr/bin/env python3
"""Run experiment cells through the experiment runner.

    ros2 launch fabric_sim_experiments single_run.launch.py config:=<exp.yaml> [run_index:=1]
        [run_all_cells:=true] [headless:=true] [runs_root:=runs]
"""
from __future__ import annotations

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _launch_setup(context, *args, **kwargs):
    value = lambda name: LaunchConfiguration(name).perform(context)  # noqa: E731
    flag = lambda name: value(name).lower() in ('true', '1', 'yes')  # noqa: E731
    params = dict(config=value('config'), run_index=int(value('run_index')), run_all_cells=flag('run_all_cells'),
                  headless=flag('headless'), runs_root=value('runs_root'), settle_time=float(value('settle_time')))
    return [Node(package='fabric_sim_experiments', executable='experiment_runner', name='experiment_runner',
                 output='screen', emulate_tty=True, parameters=[params])]


def generate_launch_description():
    default_config = Path(get_package_share_directory('fabric_sim_experiments'))/'config'/'exp1_cbf_on_off.yaml'
    return LaunchDescription([
        DeclareLaunchArgument('config', default_value=str(default_config), description='Experiment YAML.'),
        DeclareLaunchArgument('run_index', default_value='0', description='Sweep cell to run.'),
        DeclareLaunchArgument('run_all_cells', default_value='false', description='Run every sweep cell.'),
        DeclareLaunchArgument('headless', default_value='false', description='Run Isaac Sim without a window.'),
        DeclareLaunchArgument('runs_root', default_value='runs', description='Output root directory.'),
        DeclareLaunchArgument('settle_time', default_value='3.0', description='Seconds recorded after motion.'),
        OpaqueFunction(function=_launch_setup),
    ])

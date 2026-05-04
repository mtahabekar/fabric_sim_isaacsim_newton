#!/usr/bin/env python3
"""
Launch file for a single experiment run.

Usage:
  ros2 launch fabric_sim_experiments single_run.launch.py \
      config:=<path/to/exp.yaml> [headless:=true]
"""

from __future__ import annotations

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from pathlib import Path


def _launch_setup(context, *args, **kwargs):
    config = LaunchConfiguration("config").perform(context)
    headless = LaunchConfiguration("headless").perform(context).lower() in ("true", "1", "yes")
    runs_root = LaunchConfiguration("runs_root").perform(context)
    settle_time = LaunchConfiguration("settle_time").perform(context)
    run_only_first = LaunchConfiguration("run_only_first_cell").perform(context)

    params = {
        "config": config,
        "headless": headless,
        "runs_root": runs_root,
        "settle_time": float(settle_time),
        "run_only_first_cell": run_only_first.lower() in ("true", "1", "yes"),
    }

    runner_node = Node(
        package="fabric_sim_experiments",
        executable="experiment_runner",
        name="experiment_runner",
        output="screen",
        emulate_tty=True,
        parameters=[params],
    )
    return [runner_node]


def generate_launch_description():
    pkg_share = get_package_share_directory("fabric_sim_experiments")
    default_config = str(Path(pkg_share) / "config" / "exp1_cbf_on_off.yaml")

    return LaunchDescription([
        DeclareLaunchArgument("config", default_value=default_config,
                              description="Path to experiment YAML config."),
        DeclareLaunchArgument("headless", default_value="false",
                              description="Run IsaacSim headless."),
        DeclareLaunchArgument("runs_root", default_value="runs",
                              description="Root directory for run outputs."),
        DeclareLaunchArgument("settle_time", default_value="3.0",
                              description="Extra seconds to record after motion ends."),
        DeclareLaunchArgument("run_only_first_cell", default_value="false",
                              description="Only run the first sweep cell (for quick sanity check)."),
        OpaqueFunction(function=_launch_setup),
    ])

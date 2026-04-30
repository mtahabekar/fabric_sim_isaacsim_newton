#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction
from launch.actions import RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


BOOLEAN_FLAGS = {
    "disable_cuda_graph",
    "external_cbf_enabled",
}


def _load_yaml(path: str) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Launch config must be a YAML mapping: {path}")
    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Config section '{name}' must be a mapping")
    return value


def _bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _float(value: Any, *, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _params(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = _section(config, section_name)
    parameters = section.get("parameters", {})
    if parameters is None:
        return {}
    if not isinstance(parameters, dict):
        raise ValueError(f"Config section '{section_name}.parameters' must be a mapping")
    return parameters


def _arg_name(key: str) -> str:
    return "--" + key.replace("_", "-")


def _isaac_command(isaac_config: dict[str, Any]) -> list[str]:
    python_sh = str(isaac_config["python_sh"])
    script = str(isaac_config["script"])
    arguments = isaac_config.get("arguments", {})
    extra_args = isaac_config.get("extra_args", [])

    if not isinstance(arguments, dict):
        raise ValueError("Config section 'isaac_sim.arguments' must be a mapping")
    if not isinstance(extra_args, list):
        raise ValueError("Config section 'isaac_sim.extra_args' must be a list")

    command = [python_sh, script]
    for key, value in arguments.items():
        if value is None:
            continue
        if key in BOOLEAN_FLAGS:
            if _bool(value):
                command.append(_arg_name(key))
            continue
        command.extend([_arg_name(key), str(value)])

    command.extend(str(arg) for arg in extra_args)
    return command


def _delayed(action: Any, delay: float) -> Any:
    if delay <= 0.0:
        return action
    return TimerAction(period=delay, actions=[action])


def _launch_setup(context: Any, *args: Any, **kwargs: Any) -> list[Any]:
    config_file = LaunchConfiguration("config_file").perform(context)
    config = _load_yaml(config_file)

    launch_config = _section(config, "launch")
    isaac_config = _section(config, "isaac_sim")

    actions: list[Any] = []

    if _bool(launch_config.get("start_isaac"), default=True):
        isaac_process = ExecuteProcess(
            cmd=_isaac_command(isaac_config),
            cwd=str(isaac_config.get("working_directory", ".")),
            name="isaac_sim",
            output="screen",
            emulate_tty=True,
        )
        actions.append(isaac_process)

        if _bool(launch_config.get("shutdown_on_isaac_exit"), default=True):
            actions.append(
                RegisterEventHandler(
                    OnProcessExit(
                        target_action=isaac_process,
                        on_exit=[
                            EmitEvent(
                                event=Shutdown(
                                    reason="IsaacSim process exited",
                                )
                            )
                        ],
                    )
                )
            )

    if _bool(launch_config.get("start_obstacle_distance"), default=True):
        obstacle_node = Node(
            package="fabric_obstacle_distance",
            executable="obstacle_distance_node.py",
            name="obstacle_distance_node",
            output="screen",
            emulate_tty=True,
            parameters=[_params(config, "obstacle_distance")],
        )
        actions.append(
            _delayed(
                obstacle_node,
                _float(launch_config.get("obstacle_start_delay"), default=0.0),
            )
        )

    if _bool(launch_config.get("start_cbf_controller"), default=True):
        cbf_node = Node(
            package="fabric_cbf_controller",
            executable="fabric_cbf_controller_node.py",
            name="fabric_cbf_controller_node",
            output="screen",
            emulate_tty=True,
            parameters=[_params(config, "cbf_controller")],
        )
        actions.append(
            _delayed(
                cbf_node,
                _float(launch_config.get("cbf_start_delay"), default=0.0),
            )
        )

    if _bool(launch_config.get("start_teleop"), default=True):
        teleop_node = Node(
            package="fabric_teleop",
            executable="fabric_corner_keyboard_teleop",
            name="fabric_corner_keyboard_teleop",
            output="screen",
            emulate_tty=True,
            parameters=[_params(config, "teleop")],
        )
        actions.append(
            _delayed(
                teleop_node,
                _float(launch_config.get("teleop_start_delay"), default=0.0),
            )
        )

    return actions


def generate_launch_description() -> LaunchDescription:
    default_config = (
        Path(get_package_share_directory("fabric_sim_bringup"))
        / "config"
        / "full_sim.yaml"
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=str(default_config),
                description="Path to the full fabric simulation YAML config.",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )

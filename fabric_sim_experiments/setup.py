from setuptools import find_packages, setup
import os
from glob import glob

package_name = "fabric_sim_experiments"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            os.path.join("share", package_name, "config"),
            glob("config/*.yaml"),
        ),
        (
            os.path.join("share", package_name, "launch"),
            glob("launch/*.py"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="taha",
    maintainer_email="mtahabekar@gmail.com",
    description="Reproducible Isaac Sim experiments for the fabric simulation with the sensitivity CBF.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "leader_driver = fabric_sim_experiments.leader_driver_node:main",
            "data_recorder = fabric_sim_experiments.data_recorder_node:main",
            "experiment_runner = fabric_sim_experiments.experiment_runner:main",
            "batch_runner = fabric_sim_experiments.batch_runner:main",
        ],
    },
)

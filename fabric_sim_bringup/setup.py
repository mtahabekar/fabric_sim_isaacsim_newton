from glob import glob
from setuptools import find_packages, setup

package_name = "fabric_sim_bringup"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/scripts", glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="taha",
    maintainer_email="taha@example.com",
    description="Launch and configuration files for the IsaacSim fabric CBF simulation stack.",
    license="Apache-2.0",
)

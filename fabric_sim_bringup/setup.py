from glob import glob
from setuptools import find_packages, setup

package_name = "fabric_sim_bringup"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/scripts", glob("scripts/*")),
    ],
    install_requires=["setuptools"],
    tests_require=["pytest"],
    zip_safe=True,
    maintainer="taha",
    maintainer_email="mtahabekar@gmail.com",
    description="Launch and configuration for the Isaac Sim fabric simulation with the sensitivity CBF.",
    license="Apache-2.0",
)

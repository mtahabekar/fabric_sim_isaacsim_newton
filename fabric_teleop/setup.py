from setuptools import find_packages, setup

package_name = "fabric_teleop"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="taha",
    maintainer_email="mtahabekar@gmail.com",
    description="Keyboard teleop node for fabric corner velocity commands.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "fabric_corner_keyboard_teleop = fabric_teleop.fabric_corner_keyboard_teleop:main",
        ],
    },
)

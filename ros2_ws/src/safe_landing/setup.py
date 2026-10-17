from glob import glob

from setuptools import setup

package_name = "safe_landing"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ivan",
    maintainer_email="ivan.antonenko@outlook.com",
    description="Look, then feel autonomous safe landing",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "landing_node = safe_landing.landing_node:main",
            "range_driver = safe_landing.range_driver:main",
            "camera_prep = safe_landing.camera_prep:main",
        ],
    },
)

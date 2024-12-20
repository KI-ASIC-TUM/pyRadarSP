#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="pyrads",
    version="0.1",
    description="Package containing radar processing algorithms and pipeline structures",
    url="https://gitlab.lrz.de/ki-asic/pyradarsp",
    author="Technical University of Munich. AIR",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19",
        "matplotlib>=3.1.2",
    ],
    include_package_data=True,
)
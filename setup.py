#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="sampling_benchmarks",
    version="0.0.0",
    description="Experiments for sampling strategies",
    author="Charles Dawson",
    author_email="cbd@mit.edu",
    url="https://github.com/dawsonc/sampling_benchmarks",
    install_requires=[],
    package_data={"sampling_benchmarks": ["py.typed"]},
    packages=find_packages(),
)

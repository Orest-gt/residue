#!/usr/bin/env python3
"""
PROJECT RESIDUE V2.0 - Optimized Build Script
The Analog Scientist with NaN fixes and C++ optimizations
"""

import os
import sys
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define the optimized extension module
ext_modules = [
    Pybind11Extension(
        "residue_v2.residue_v2",
        [
            "src/residue/core.cpp",
            "src/residue/bindings.cpp"
        ],
        include_dirs=[
            "src",
            pybind11.get_include(),
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"2.0.0"')],
        extra_compile_args=["/O2", "/bigobj"],  # MSVC optimized flags
    ),
]

# Read the README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Setup configuration
setup(
    name="residue-v2",
    version="2.0.0",
    author="PROJECT RESIDUE",
    author_email="residue@project-residue.org",
    description="The Analog Scientist - Multi-dimensional ML optimization with semantic bridge",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/project-residue/residue",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    packages=["residue_v2"],
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.19.0",
        "pybind11>=2.10.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=1.0",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="machine learning optimization entropy analog scaling semantic bridge",
    project_urls={
        "Bug Reports": "https://github.com/project-residue/residue/issues",
        "Source": "https://github.com/project-residue/residue",
        "Documentation": "https://residue.readthedocs.io/",
    },
)

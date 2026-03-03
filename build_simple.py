"""
Simple build script for PROJECT RESIDUE
Compatible with Python 3.13+
"""

import os
import sys
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup, Extension

# Simple extension build
ext_modules = [
    Pybind11Extension(
        "residue.residue",
        sources=[
            "src/residue/entropy_only_bindings.cpp",
            "src/residue/entropy_controller.cpp"
        ],
        include_dirs=[
            "src",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"1.0.0"')],
        extra_compile_args=["-O3"],
    ),
]

if __name__ == "__main__":
    setup(
        name="residue",
        version="1.0.0",
        ext_modules=ext_modules,
        cmdclass={"build_ext": build_ext},
        zip_safe=False,
    )

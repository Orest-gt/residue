from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11
import sys
import os

# V4.0 Standalone Zero Overhead Extension + Residue Wall + Isolation Zone
ext_modules = [
    Pybind11Extension(
        "residue.core",
        sources=[
            "src/residue/core.cpp",
            "src/residue/bindings.cpp",
            "src/residue_wall/residue_wall.cpp",
            "src/residue_wall/async_observer.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            "src",
        ],
        cxx_std=20,
        define_macros=[("VERSION_INFO", '"4.2.3.post2"')],
        extra_compile_args=["/O2", "/bigobj", "/std:c++20", "/arch:AVX2"] if "win" in sys.platform else ["-O3", "-std=c++20", "-mavx2", "-mfma", "-mpopcnt"],
        extra_link_args=["winmm.lib"] if "win" in sys.platform else [],
    ),
]

setup(
    name="residue-protocol",
    version="4.2.3.post2",
    author="PROJECT RESIDUE",
    author_email="orestisgatos@gmail.com",
    description="Bare-Metal AVX2 Inference Shield - V4.2 Reality-Synchronized",
    long_description="PROJECT RESIDUE V4.2 - Reality-Synchronized Bare-Metal Architecture with NUMA awareness and Vectorized Full-Scan Gating",
    ext_modules=ext_modules,
    packages=["residue"],
    package_dir={"": "src"},
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
)

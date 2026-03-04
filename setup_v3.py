#!/usr/bin/env python3
"""
PROJECT RESIDUE V3.0 - Setup Script
Build script for V3.0 with structural heuristics
"""

from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
from pybind11 import get_cmake_dir
import pybind11

# Define the V3.0 extension module
ext_modules = [
    Pybind11Extension(
        "residue_v3.residue_v3",
        sources=[
            "src/residue/residue_v3_bindings.cpp",
            "src/residue/core.cpp",
        ],
        include_dirs=[
            "src",
            get_cmake_dir(),
        ],
        extra_compile_args=["/O2", "/bigobj", "/std:c++17"],
        define_macros=[("VERSION_INFO", '"3.0.0"')],
        cxx_std=17,
    ),
]

setup(
    name="residue_v3",
    version="3.0.0",
    author="PROJECT RESIDUE Team",
    author_email="residue@project.ai",
    description="PROJECT RESIDUE V3.0 - Structural Heuristics for LLM Optimization",
    long_description="""
    PROJECT RESIDUE V3.0 - The Structural Analyst
    
    Advanced entropy controller with structural heuristics:
    - Temporal coherence through EMA buffer
    - L1-norm sparsity detection
    - Zero-crossing rate analysis
    - 7-feature softmax scaling
    - <0.01ms performance overhead
    
    Moving beyond analog scaling to structural intelligence.
    """,
    long_description_content_type="text/plain",
    url="https://github.com/project-residue/residue",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.8",
    install_requires=[
        "pybind11>=2.6.0",
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
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
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="entropy optimization llm scaling structural-heuristics temporal-coherence",
    project_urls={
        "Bug Reports": "https://github.com/project-residue/residue/issues",
        "Source": "https://github.com/project-residue/residue",
        "Documentation": "https://project-residue.readthedocs.io/",
    },
)

from setuptools import setup, Extension, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# PROJECT RESIDUE - Production-ready ML optimization

ext_modules = [
    Pybind11Extension(
        "residue.residue",
        sources=[
            "src/residue/entropy_only_bindings.cpp",
            "src/residue/entropy_controller.cpp"
        ],
        include_dirs=[
            pybind11.get_include(),
            "src",
        ],
        cxx_std=17,
        define_macros=[("VERSION_INFO", '"1.0.0"')],
        extra_compile_args=["/O2"],  # MSVC compatible
    ),
]

setup(
    name="residue",
    version="1.0.0",
    author="PROJECT RESIDUE",
    author_email="residue@project.continuum",
    description="40% faster inference through input entropy analysis",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/project-residue/residue",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Operating System :: OS Independent",
    ],
    keywords="machine learning, optimization, entropy, adaptive computation, efficiency",
    project_urls={
        "Bug Reports": "https://github.com/project-residue/residue/issues",
        "Source": "https://github.com/project-residue/residue",
        "Documentation": "https://residue.readthedocs.io/",
        "Performance": "https://github.com/project-residue/residue/benchmarks",
    },
)

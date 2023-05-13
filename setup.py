#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize
import platform


compile_extra_args = []
link_extra_args = []


if platform.system() == "Darwin":
    compile_extra_args = ["-mmacosx-version-min=10.9"]
    link_extra_args = ["-mmacosx-version-min=10.9"]


setup(
    name="isosplit",
    version="0.1",
    packages=["isosplit"],
    ext_modules=cythonize("isosplit/jisotonic5.pyx"),
    install_requires=['cython'],
)

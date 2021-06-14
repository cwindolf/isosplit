#!/usr/bin/env python
from setuptools import setup
from Cython.Build import cythonize


setup(
    name="isosplit",
    version="0.1",
    packages=["isosplit"],
    ext_modules=cythonize("isosplit/jisotonic5.pyx")
)

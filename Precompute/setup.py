from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = cythonize(Extension(
    name='kpll_cppext',
    sources=['kpll_cppext.pyx'],
    language='c++',
    extra_compile_args=["-std=c++11", "-O3", "-fopenmp", "-Wall", "-Wextra", "-msse"],
    include_dirs=[".", "module-dir-name"] + [np.get_include()],
))

setup(
    setup_requires=['setuptools', 'cython',],
    packages=find_packages(),
    ext_modules=ext_modules,
)

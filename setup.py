from setuptools import setup, find_packages, Extension
import numpy as np

ext_modules = [ Extension('winsharedarray', extra_compile_args=["-std=c++11"], sources = ['shared_memory_python.cpp']) ]

setup(
        name = 'winsharedarray',
        version = '1.1',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )

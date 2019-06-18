from setuptools import setup, find_packages, Extension
import numpy as np

ext_modules = [ Extension('winsharedarray', sources = ['shared_memory_python.cpp']) ]

setup(
        name = 'winsharedarray',
        version = '1.0',
        include_dirs = [np.get_include()], #Add Include path of numpy
        #ext_modules = ext_modules,
        packages=['.']
      )
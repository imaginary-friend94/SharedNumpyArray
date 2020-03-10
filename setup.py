from setuptools import setup, find_packages, Extension
import numpy as np
import sys

if sys.platform == "linux" or sys.platform == "linux2":
	libraries_ext = ["rt"]
elif sys.platform == "win32":
	libraries_ext = []


ext_modules = [
	Extension('winsharedarray', 
	extra_compile_args=["-std=c++11"], 
	sources = ['shared_memory_python.cpp'],
	libraries = libraries_ext)
]

setup(
        name = 'winsharedarray',
        version = '1.2',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )

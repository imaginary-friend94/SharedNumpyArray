from setuptools import setup, Extension
import numpy as np
import sys

module_name = "numpysharedarray"

if sys.platform == "linux" or sys.platform == "linux2":
	libraries_ext = ["rt"]
elif sys.platform == "win32":
	libraries_ext = []


ext_modules = [
	Extension(module_name, 
	extra_compile_args=["-std=c++11"], 
	sources = ['shared_memory_python.cpp'],
	libraries = libraries_ext)
]

setup(
        name = module_name,
        version = '1.2',
        include_dirs = [np.get_include()], #Add Include path of numpy
        ext_modules = ext_modules
      )
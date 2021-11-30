from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='reduction_op',
      ext_modules=[
        cpp_extension.CppExtension(
          'reduction', 
          ['reduction.cpp'])],
        include_dirs=["../libtorch"],
        cmdclass={'build_ext': cpp_extension.BuildExtension})
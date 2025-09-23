# setup.py
from setuptools import setup, Extension
import os
import pybind11

CUDA_HOME = os.environ.get('CUDA_HOME') or '/usr/local/cuda'
include_dirs = [pybind11.get_include(), f"{CUDA_HOME}/include"]
library_dirs = [f"{CUDA_HOME}/lib64"]

ext = Extension(
    name="_svcuda",
    sources=["cuda_statevector.cu"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=["cudart"],
    extra_compile_args={
        'gcc': ['-O3'],
        'nvcc': ['-O3', '-std=c++14', '--compiler-options', "-fPIC"]
    },
    language='c++'
)

# Allow .cu compilation with distutils
from setuptools.command.build_ext import build_ext
class BuildExt(build_ext):
    def build_extensions(self):
        for ext in self.extensions:
            for s in list(ext.sources):
                if s.endswith('.cu') and '.cu' not in self.compiler.src_extensions:
                    self.compiler.src_extensions.append('.cu')
        super().build_extensions()

setup(
    name='_svcuda',
    version='0.2.0',
    ext_modules=[ext],
    cmdclass={'build_ext': BuildExt}
)

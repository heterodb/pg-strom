from distutils.core import setup, Extension
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import cupy

cuda_path = cupy.cuda.get_cuda_path()
ext = Extension("cupy_strom",
                sources=["cupy_strom.pyx","cupy_ipcmem.c"],
                include_dirs=['.', cuda_path + '/include'],
                library_dirs=[cuda_path + '/lib64'],
                libraries=['cudart'])
setup(name='cupy_strom',
      author='KaiGai Kohei',
      version='0.4',
      ext_modules=cythonize([ext]))

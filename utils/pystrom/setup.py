import cupy
from distutils.core import setup, Extension

cuda_path = cupy.cuda.get_cuda_path()
setup(name='pystrom',
      author='KaiGai Kohei',
      version='0.2',
      ext_modules=[Extension('pystrom',
                             sources=['pystrom.c'],
                             include_dirs=[cuda_path + '/include'],
                             library_dirs=[cuda_path + '/lib64'],
                             libraries=['cudart'],
                             define_macros=[('BLCKSZ', 8192),
                                            ('MAXIMUM_ALIGNOF', 8),
                                            ('NAMEDATALEN', 64)])]
)

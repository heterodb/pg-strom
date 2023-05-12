import subprocess
import cupy
from distutils.core import setup, Extension

cuda_path = cupy.cuda.get_cuda_path()
pgsql_include_path = subprocess.check_output(['pg_config','--includedir-server']).decode('utf-8').strip()

setup(name='pystrom',
      author='KaiGai Kohei',
      version='0.2',
      ext_modules=[Extension('pystrom',
                             sources=['pystrom.c'],
                             include_dirs=[cuda_path + '/include',
                                           pgsql_include_path,
                                           '../../src'],
                             library_dirs=[cuda_path + '/lib64'],
                             libraries=['cudart'],
                             define_macros=[('BLCKSZ', 8192),
                                            ('MAXIMUM_ALIGNOF', 8),
                                            ('NAMEDATALEN', 64)])]
)

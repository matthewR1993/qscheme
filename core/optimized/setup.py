from distutils.core import setup
from Cython.Build import cythonize
import os

os.environ['CFLAGS'] = '-Ofast -frename-registers -funroll-loops -Wall -Wextra -ffast-math -fno-signed-zeros' + \
                       '-ffinite-math-only -fno-trapping-math -flto -fcx-limited-range'

setup(
    ext_modules=cythonize("transformations.pyx")
)

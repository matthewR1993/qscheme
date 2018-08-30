from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        name='transformations',
        sources=['core/src/transformations.pyx']),
]

setup(
    name="qscheme",
    ext_modules=cythonize(extensions),
    include_dirs=[np.get_include()],
    extra_compile_args=['-Ofast', '-frename-registers', '-funroll-loops', '-Wall', '-Wextra', '-ffast-math',
                    '-fno-signed-zeros', '-ffinite-math-only', '-fno-trapping-math', '-flto']

)

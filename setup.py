from distutils.core import setup
from distutils.extension import Extension
import numpy as np

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

cmdclass = {}
ext_modules = []

if use_cython:
    ext_modules += [
        Extension("qscheme.core.optimized", ["core/optimized/transformations.pyx"]),
    ]
    cmdclass.update({'build_ext': build_ext})

setup(
    name='qscheme',
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
    extra_compile_args=['-Ofast', '-frename-registers', '-funroll-loops', '-Wall', '-Wextra', '-ffast-math',
                        '-fno-signed-zeros', '-ffinite-math-only', '-fno-trapping-math', '-flto']
)

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import Cython.Compiler.Options

# Can be used to use line profiler (kernprof) for cython.
# Also de-comment first lines in cython script.
linetrace = False
if linetrace:
    directive_defaults = Cython.Compiler.Options.get_directive_defaults()
    directive_defaults['binding'] = True
    directive_defaults['linetrace'] = True

    extensions = [
        Extension("DMPy.optimized_functions", ["DMPy/optimized_functions.pyx"],
                  define_macros=[('CYTHON_TRACE', '1')])
    ]
    setup(ext_modules=cythonize(extensions))
else:
    setup(
        ext_modules=cythonize("DMPy/optimized_functions.pyx")
    )

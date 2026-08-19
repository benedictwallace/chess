"""
Build the Cython move generator.

    pip install cython
    python setup_movegen.py build_ext --inplace

That writes engine/movegen.*.so next to the .pyx. Nothing else in the project
changes until you also swap in engine/moves_cy.py (see its docstring).

-O3 and -march=native matter here: the inner loops are bit-twiddling over
uint64_t, and the difference between -O0 and -O3 on this code is roughly 3x.
If you need the .so to be portable across heterogeneous cluster nodes, drop
-march=native -- it emits instructions your login node may have and a compute
node may not, which shows up as SIGILL at import.
"""

from setuptools import setup, Extension
import os

try:
    from Cython.Build import cythonize
except ImportError:
    raise SystemExit("Cython not installed:  pip install cython")

NATIVE = os.environ.get("MOVEGEN_NATIVE", "1") == "1"

flags = ["-O3", "-funroll-loops"]
if NATIVE:
    flags.append("-march=native")

setup(
    name="movegen",
    ext_modules=cythonize(
        [Extension("engine.movegen", ["engine/movegen.pyx"],
                   extra_compile_args=flags)],
        compiler_directives={"language_level": "3"},
        annotate=True,          # writes engine/movegen.html: yellow = Python
    ),
    zip_safe=False,
)


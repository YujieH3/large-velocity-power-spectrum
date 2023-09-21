from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

test_module = Pybind11Extension(
    'test_module', # module name
    [str(fname) for fname in Path('src').glob('*.cpp')], # sources must be a list of strings
    include_dirs=['include'],
    extra_compile_args=['-O3']
)

# print([str(fname) for fname in Path('src').glob('*.cpp')])

setup(
    name='test_module',
    version=0.1,
    author='Yujie He',
    author_email='yujie.jay.he@outlook.com',
    description="In developing for a Python interface for the well-known ANN library by " +
        "David M. Mount and Sunil Arya.",
    ext_modules=[test_module], # add any additional C++ extension modules here
    cmdclass={"build_ext": build_ext},
)
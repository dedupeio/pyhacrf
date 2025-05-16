from os.path import abspath, join

from Cython.Build import cythonize
import numpy
from setuptools import Extension, setup

defs = [("NPY_NO_DEPRECATED_API", 0)]
inc_path = numpy.get_include()
# Add path for npymath libraries:
lib_path = [abspath(join(numpy.get_include(), "..", "lib"))]


ext_modules = cythonize(
    [
        Extension(
            "pyhacrf.algorithms",
            ["pyhacrf/algorithms.pyx"],
            extra_compile_args=["-ffast-math", "-O4"],
            include_dirs=[inc_path],
            library_dirs=lib_path,
            libraries=["npymath"],
            extra_link_args=["-lm"],
            define_macros=defs,
        ),
        Extension(
            "pyhacrf.adjacent",
            ["pyhacrf/adjacent.pyx"],
            include_dirs=[numpy.get_include()],
            extra_link_args=["-lm"],
            extra_compile_args=["-ffast-math", "-O4"],
        ),
    ]
)


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="pyhacrf-datamade",
    version="0.2.8",
    packages=["pyhacrf"],
    install_requires=[
        "numpy>=1.14.1; python_version<'3.6'",
        "numpy>=1.15.0; python_version=='3.7'",
        "numpy>=1.17.3; python_version=='3.8'",
        "numpy; python_version>'3.8'",
        "PyLBFGS>=0.1.3",
    ],
    python_requires='>=3.9',
    ext_modules=ext_modules,
    url="https://github.com/datamade/pyhacrf",
    author="Dirko Coetsee",
    author_email="dpcoetsee@gmail.com",
    maintainer="Forest Gregg",
    maintainer_email="fgregg@gmail.com",
    description="Hidden alignment conditional random field, a discriminative string edit distance",
    long_description=readme(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
)

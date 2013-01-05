#!/usr/bin/env python

# This setup.py is inspired by the simplerandom project on bitbucket.

# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that
# was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.
USE_CYTHON = True


import sys

from distutils.core import setup
from distutils.extension import Extension

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
    except ImportError:
        if USE_CYTHON=='auto':
            USE_CYTHON=False
        else:
            raise

cmdclass = { }
ext_modules = [ ]

if sys.version_info[0] == 2:
    base_dir = 'python2'
elif sys.version_info[0] == 3:
    # Still build from python2 code, but use build_py_2to3 to translate.
    base_dir = 'python2'
    from distutils.command.build_py import build_py_2to3 as build_py
    cmdclass.update({ 'build_py': build_py })

if USE_CYTHON:
    ext_modules.append(Extension('omnomnomial', ['omnom.pyx']))
    cmdclass['build_ext'] = build_ext
else:
    raise NotImplementedError
    #ext_modules += [
        #Extension("simplerandom.iterators._iterators_cython", [ "cython/_iterators_cython.c" ]),
    #]

setup(
    name='omnomnomial',
    version='version 0.0.0',
    description='description',
    author='author',
    author_email='author email',
    url='url',
    #packages=['omnomnomial'],
    #package_dir={
        #'simplerandom' : base_dir + '/simplerandom',
        #'simplerandom._version' : '_version',
    #},
    cmdclass = cmdclass,
    ext_modules=ext_modules,
    long_description='long description',
    license='license',
    classifiers=[],
    keywords='keywords',
)

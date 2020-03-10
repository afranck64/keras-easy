# -*- coding: utf-8 -*-

import keras_easy

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

with open('CHANGES.rst') as f:
    changes = f.read()

with open('requirements.txt') as f:
    required = [line.strip() for line in f]

#required = ['imagesize', 'tensorflow', 'keras']

setup(
    name='keras-easy',
    version=keras_easy.__version__,
    packages=['keras_easy'],
    author='afranck64',
    author_email='afranck64{at}yahoo{dot}fr',
    maintainer='afranck64',
    maintainer_email='afranck64{at}yahoo{dot}fr',
    description='A fast Python implementation of locality sensitive hashing with persistance support.',
    long_description=readme + '\n\n' + changes,
    license=license,
    requires=required,
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Topic :: Software Development :: Libraries',
        ],
)

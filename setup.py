#!/usr/bin/env python
from setuptools import setup

import ionn

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except ImportError:
    long_description = ''


setup(
    name='ionn',
    version=ionn.__version__,
    author='Ingo Fruend',
    author_email='ingo.fruend@twentybn.com',
    packages=['ionn'],
    description='io for neural networks',
    long_description=long_description,
    install_requires=['tnt'],
    entry_points={
        'console_scripts': ['tf-freeze=ionn.tfpb:main'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ]
)

#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import setup
import os

version = {}
with open(os.path.join('metric_learn', '_version.py')) as fp:
  exec(fp.read(), version)

setup(name='metric-learn',
      version=version['__version__'],
      description='Python implementations of metric learning algorithms',
      author=['CJ Carey', 'Yuan Tang'],
      author_email='ccarey@cs.umass.edu',
      url='http://github.com/all-umass/metric-learn',
      license='MIT',
      classifiers=[
          'Development Status :: 4 - Beta',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
          'Topic :: Scientific/Engineering'
      ],
      packages=['metric_learn'],
      install_requires=[
          'numpy',
          'scipy',
          'scikit-learn',
          'six'
      ],
      extras_require=dict(
          docs=['sphinx', 'shinx_rtd_theme', 'numpydoc'],
          demo=['matplotlib'],
      ),
      test_suite='test',
      keywords=[
          'Metric Learning',
          'Large Margin Nearest Neighbor',
          'Information Theoretic Metric Learning',
          'Sparse Determinant Metric Learning',
          'Least Squares Metric Learning',
          'Neighborhood Components Analysis'
      ])

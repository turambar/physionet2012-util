#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

setup(name='physionet2012-util',
      version='1.0',
      description='Utilities for working with and preprocessing the Physionet Challenge 2012 data set',
      author='David C. Kale',
      author_email='dave@skymind.io',
      url='http://skymind.ai',
      packages=find_packages(),
      install_requires=['pandas', 'requests', 'tqdm', 'numpy']
     )

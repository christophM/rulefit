#!/usr/bin/env python

from distutils.core import setup

setup(name='RuleFit',
      version='0.3',
      description='RuleFit algorithm',
      author='Christoph Molnar',
      author_email='christoph.molnar@gmail.com',
      url='',
      packages=['rulefit'],
      install_requires=['scikit-learn>=0.20.2',
                        'numpy>=1.16.1',
                        'pandas>=0.24.1']
     )

import setuptools
from setuptools import find_packages
from setuptools import setup
import re, os

packages = find_packages(exclude=('tests', 'doc'))
provides = ['taurex_PCQ', ]

install_requires = ['taurex']

entry_points = {'taurex.plugins': 'taurex_PCQ = taurex_PCQ'}

setup(name='taurex_PCQ',
      author="Maël Voyer",
      author_email="tbd",
      license="BSD",
      description='Cloud retrieval capabilities for TauREx using Qext grid based on taurex-pymiescatt by Quentin Changeat',
      packages=packages,
      entry_points=entry_points,
      provides=provides,
      install_requires=install_requires)

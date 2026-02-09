from setuptools import find_packages
from distutils.core import setup

setup(name='dognew',
      version='1.0.0',
      author='',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Go2 quadruped robot RL training with AT-PC and baselines',
      install_requires=['isaacgym', 'rsl-rl', 'matplotlib', 'numpy==1.20', 'tensorboard', 'pyyaml'])

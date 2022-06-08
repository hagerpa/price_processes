from distutils.core import setup

import numpy
from Cython.Build import cythonize

setup(name='price_processes',
      version='0.1',
      description='Simulation of several price and volatility processes.',
      url='https://github.com/hagerpa/price_processes.git',
      author='Paul P. Hager',
      author_email='hagerpa@gmail.com',
      license='MIT',
      packages=['price_processes'],
      ext_modules=cythonize([
          "price_processes/models/jump_diffusion.pyx",
          "price_processes/models/jump_diffusion.pyx",
          "price_processes/models/two_factor_jump_diffusion.pyx",
          "price_processes/models/ornstein_uhlenbeck.pyx",
      ]),
      include_dirs=[numpy.get_include()],
      install_requires=['Cython', 'numpy', 'scipy'],
      zip_safe=False)

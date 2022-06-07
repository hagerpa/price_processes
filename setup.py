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
          "jump_diffusion.pyx",
          "jump_ornstein_uhlenbeck.pyx",
          "multifactor_jou.pyx",
          "auto_regressive_process.pyx",
      ]),
      include_dirs=[numpy.get_include()],
      install_requires=['Cython', 'numpy', 'scipy'],
      zip_safe=False)

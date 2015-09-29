from setuptools import setup
from setuptools import find_packages

setup(name='Minhaz Palasara',
      version='0.0.1',
      description='Melanoma Deep Learning based on Keras',
      author='Minhaz Palasara',
      author_email='minhaz.palasara@gmail.com',
      url='https://github.com/minhazpalasara/keras',
      license='MIT',
      install_requires=['theano', 'h5py'],
      packages=find_packages(),
)

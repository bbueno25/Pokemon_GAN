from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['pillow==5.1.0']

setup(author='moxiegushi',
      author_email='bbueno5000@gmail.com',
      description='GAN for generating pokemon.',
      include_package_data=True,
      install_requires=REQUIRED_PACKAGES,
      name='pokegan',
      packages=find_packages(),
      url='https://github.com/moxiegushi/pokeGAN')

from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
from glob import glob


import os
import sys
import wget
import eigency
import zipfile


# Install Eigen
path_lib = 'lib/Eigen'
if os.path.isdir(path_lib):
    print('Eigen is already installed')
    
else:
    url_eigen = 'http://bitbucket.org/eigen/eigen/get/3.3.4.zip'
    file = wget.download(url_eigen, out='lib/')

    with zipfile.ZipFile(file,"r") as zip_ref:
        zip_ref.extractall("lib/Eigen/")
    os.remove(file)


if sys.platform == 'darwin':
    extensions = [
        Extension(
            'cSvaIbpDl',
            glob('*.pyx') + glob('*.cxx'),
            extra_compile_args=["-stdlib=libc++", "-std=c++11"])
    ]
else:
    extensions = [
        Extension(
            'cSvaIbpDl',
            glob('*.pyx') + glob('*.cxx'),
            extra_compile_args=["-stdlib=libstdc++", "-std=c++11"])
    ]



setup(
    name = "cSvaIbpDl",
    ext_modules = cythonize(extensions),
    language ="c++",
    include_dirs = [".", "includes/", "sources/", 'lib/'] + eigency.get_includes(include_eigen=True),


    version='0.1.dev',

    description = 'A sample Python project',
    long_description = 'A sample Python project',

    # The project's main homepage.
    url='https://github.com/c-elvira/ibpdlsva',

    # Author details
    author='celvira',
    author_email='clement.elvira@inria.fr',

    # Choose your license
    license='CeCILL',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: signal processing',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: CeCILL License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    #install_requires=[
    #    'coverage             >=3.5.3',
    #    'matplotlib',
    #    'numpy',
    #    'wget',
    #    'eigency',
    #    'zipfile'
    #]
)


"""
LocalSync: Local Bubble Synchrotron Model

Physical synchrotron emission model with 3D Local Bubble tomography.
Part of RadioForegroundsPlus WP4.1.
"""

from setuptools import setup, find_packages
import os

# Read README
here = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(here, 'README.md')
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "Physical Synchrotron Emission Model"

setup(
    name='localsync',
    version='0.1.0',
    author='WP4.1 RadioForegroundsPlus Team',
    author_email='',
    description='Physical Synchrotron Emission Model with Moment Expansion',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/radioforegroundsplus/localsync',
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Astronomy',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.20',
        'scipy>=1.7',
        'healpy>=1.15',
        'matplotlib>=3.3',
        'numba>=0.56',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9',
            'black>=21.0',
        ],
        'docs': [
            'sphinx>=4.0',
            'sphinx-rtd-theme>=1.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'localsync-fit=localsync.scripts.fit_spectrum:main',
            'localsync-tomography=localsync.scripts.tomography:main',
        ],
    },
    include_package_data=True,
    zip_safe=False,
)

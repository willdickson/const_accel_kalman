from setuptools import setup
from setuptools import find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='const_accel_kalman',
    version='0.0.1',
    description='A dead simple implementation of Kalman filter with a constant acceleration model (based on pykalman)',
    long_description=__doc__,
    author='Will Dickson',
    author_email='wbd@caltech',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Biology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    packages=find_packages(exclude=['examples', 'bin',]),
)

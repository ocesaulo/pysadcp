#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages


with open('README.rst') as readme_file:
    readme = readme_file.read()

# get the dependencies and installs
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]
dependency_links = [x.strip().replace('git+', '') for x in all_reqs if x.startswith('git+')]

tests_requirements = ['pytest'],
setup_requirements = ['pytest-runner']
requirements = [
    # package requirements go here
]

setup(
    name='pysadcp',
    version=__version__,
    description="Tools to organize, handle and analyze shipboard ADCP data sets",
    long_description=readme,
    author="Saulo M Soares",
    author_email='ocesaulo@gmail.com',
    url='https://github.com/ocesaulo/pysadcp',
    packages=find_packages(include=['pysadcp'],
                           exclude=('docs', 'tests*',)),
    license="MIT license",

    install_requires=install_requires,
    dependency_links=dependency_links,
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    keywords='pysadcp',
    classifiers=[
        'Programming Language :: Python :: 3.6',
    ]
)

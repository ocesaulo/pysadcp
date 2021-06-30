pySADCP
=======

.. image:: https://img.shields.io/pypi/v/pysadcp.svg
        :target: https://pypi.python.org/pypi/pysadcp
        :alt: Latest PyPI version

.. image:: https://img.shields.io/travis/ocesaulo/pysadcp.svg
        :target: https://travis-ci.org/ocesaulo/pysadcp
        :alt: Latest Travis CI build status

.. image:: https://readthedocs.org/projects/pysadcp/badge/?version=latest
        :target: https://pysadcp.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

Overview
--------

pySADCP

Tools to organize, handle and analyze shipboard ADCP data sets

Installation
------------

1) pip install from this repository::

    pip install git+https://github.com/ocesaulo/pysadcp.git@develop

2) install CODAS + pycurrents (follow steps 1.4.1 onwards: https://currents.soest.hawaii.edu/docs/adcp_doc/codas_setup/codas_config/index.html)

Usage/Example
-----

Run sample test::
    python process_codas_dbs_L1.py pathtorepohome/data/codas_dbs pathtorepohome/data/proc/

Requirements
^^^^^^^^^^^^

Requires pycurrents and CODAS ().
Additional requirements see requirements.txt

Compatibility
-------------

Authors
-------

`pysadcp` was written by `Saulo M Soares <ocesaulo@gmail.com>`_.


Licence
-------

* Free software: MIT license
* Documentation: https://pysadcp.readthedocs.io.

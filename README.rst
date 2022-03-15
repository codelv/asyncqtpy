asyncqtpy - asyncio + PyQt5/PySide2
=================================

.. image:: https://ci.appveyor.com/api/projects/status/s74qrypga40somf1?svg=true
    :target: https://ci.appveyor.com/project/gmarull/asyncqtpy
    :alt: Build Status

.. image:: https://codecov.io/gh/gmarull/asyncqtpy/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/gmarull/asyncqtpy
    :alt: Coverage

.. image:: https://img.shields.io/pypi/v/asyncqtpy.svg
    :target: https://pypi.python.org/pypi/asyncqtpy
    :alt: PyPI Version

.. image:: https://img.shields.io/conda/vn/conda-forge/asyncqtpy.svg
    :target: https://anaconda.org/conda-forge/asyncqtpy
    :alt: Conda Version

**IMPORTANT: This project is unmaintained. Use other alternatives such as https://github.com/CabbageDevelopment/qasync**

``asyncqtpy`` is an implementation of the ``PEP 3156`` event-loop with Qt. This
package is a fork of ``quamash`` focusing on modern Python versions, with
some extra utilities, examples and simplified CI.

Forked yet again to use qtpy and reformatted with black.

Requirements
============

``asyncqtpy`` requires Python >= 3.5 and qtpy. The Qt API can be
explicitly set by using the ``QT_API`` environment variable.

Installation
============

``pip install asyncqtpy``

Examples
========

You can find usage examples in the ``examples`` folder.


Matrixflow Documentation
========================

A rich library with implementations for mathematical **matrices** and **vectors** and their operations

The repository is available on GitHub `here <https://github.com/Tom-the-Bomb/matrixflow>`_

Installation
------------

Python 3.10 or higher is required

.. code-block:: powershell

   py -m pip install matrixflow

or from the GitHub:

.. code-block:: powershell

   py -m pip install git+https://github.com/Tom-the-Bomb/matrixflow.git

Basic Example
-------------

.. code-block:: python

   from matrixflow import Matrix, Vector

   A = Matrix([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
   ])
   print(A.det()) # calculates the determinant
   A.transpose()   # transposes `A` in place

   u = Vector([1, 2, 3])
   v = Vector([4, 5, 6])
   print(u * v)   # calculates the dot product

Further examples can be found over on the `examples <examples.rst>`_ page

Highlight Features
------------------

- **Matrices**
   - Basic operations: addition, subtraction, scalar/matrix multiplication and other basic matrix functions
   - Linear transformations
   - determinant
   - Linear system of equations:
      - Gaussian elimination: row echelon & reduced row echelon forms
      - Inverse
   - and many more!

- **Vectors**
   - Basic operations: addition, subtraction, scalar/dot/cross products and other basic vector functions
   - polar/spherical and cartesian conversions
   - projection, rejection
   - and many more!

Classes
-------

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Examples <examples.rst>
   Matrix <matrix.rst>
   Vector <vector.rst>

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
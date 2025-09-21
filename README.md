
# Matrixflow

[![Downloads](https://static.pepy.tech/badge/matrixflow)](https://pepy.tech/project/matrixflow)

A rich library with implementations for mathematical **matrices** and **vectors** and their operations

Refer to the documentation over [here](https://matrixflow.readthedocs.io/en/latest/index.html)

## Installation

Python 3.10 or higher is required

```bash
py -m pip install matrixflow
```

or from github

```bash
py -m pip install git+https://github.com/Tom-the-Bomb/matrixflow.git
```

## Examples

```py
from matrixflow import Matrix, Vector

A = Matrix([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9],
])
print(A.det()) # calculates the determinant
A.transpose()  # transposes `A` in place

u = Vector([1, 2, 3])
v = Vector([4, 5, 6])
print(u * v)   # calculates the dot product
```

Further examples can be found over at the documentation [here](https://matrixflow.readthedocs.io/en/latest/examples.html)

## Highlight Features

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

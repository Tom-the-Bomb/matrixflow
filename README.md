
# Matrixflow

A simple & light library with basic implementations for mathematical **matrices** and **vectors**

Refer to the documentation over [here](https://matrixflow.readthedocs.io/en/latest/index.html)

## Installation

```powershell
py -m pip install matrixflow
```

or from github

```powershell
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
A.tranpose()   # transposes `A` in-place

u = Vector([1, 2, 3])
v = Vector([4, 5, 6])
print(u * v)   # calculates the dot product
```

Further examples can be found over at the documentation [here](https://matrixflow.readthedocs.io/en/latest/examples.html)

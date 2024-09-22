
from matrix import *

def test_matrices() -> None:
    A = SquareMatrix([
        [1, 1, 1],
        [2, -3, 1],
        [3, 2, 1],
    ])

    B = SquareMatrix([
        [4, 7, 1],
        [5, -2, 1],
        [10, 2, 8],
    ])
    C = SquareMatrix([
        [3, 7, 0],
        [6, -3, 12],
        [1.2, 9, 2],
    ])

    O = SquareMatrix.zero(A.n)
    I = SquareMatrix.identity(A.n)

    assert A.det() * I == A @ A.adj()
    assert A == A @ I == I @ A
    assert A @ O == O @ A == O
    assert A @ (B + C) == A @ B + A @ C
    assert A @ (B @ C) == (A @ B) @ C
    assert A + (-A) == O
    assert A @ ~A == I

def test_vectors() -> None:
    u = Vector([2, 2, -2])
    v = Vector([-2, 3, -5])
    w = Vector([6, -5, 10])
    o = Vector.zero(u.length)

    assert u @ v == -(v @ u)
    assert u @ (v + w) == u @ v + u @ w
    assert abs(u * (v @ w)) == abs(
        SquareMatrix([
            u.inner,
            v.inner,
            w.inner,
        ])
        .det()
    )
    assert u @ u == o
    assert u @ (v @ w) == (u * w) * v - (u * v) * w

if __name__ == '__main__':
    test_matrices()
    test_vectors()
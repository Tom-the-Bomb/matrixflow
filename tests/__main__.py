
from matrix import *

def test_matrices() -> None:
    A = Matrix([
        [1, 1, 1],
        [2, -3, 1],
        [3, 2, 1],
    ])

    B = Matrix([
        [4, 7, 1],
        [5, -2, 1],
        [10, 2, 8],
    ])
    C = Matrix([
        [3, 7, 0],
        [6, -3, 12],
        [1.2, 9, 2],
    ])

    O = Matrix.zero(A.rows, A.cols)
    I = Matrix.identity(A.rows)

    assert A.det() * I == A @ A.adj()
    assert A == A @ I == I @ A
    assert A @ O == O @ A == O
    assert A @ (B + C) == A @ B + A @ C
    assert A @ (B @ C) == (A @ B) @ C
    assert A + (-A) == O
    assert A @ ~A == I
    assert (A @ B).det() == A.det() * B.det()
    assert (A @ B).trace() == (B @ A).trace()

def test_vectors() -> None:
    u = Vector([2, 2, -2])
    v = Vector([-2, 3, -5])
    w = Vector([6, -5, 10])
    o = Vector.zero(u.length)

    assert u @ v == -(v @ u)
    assert u @ (v + w) == u @ v + u @ w
    assert abs(u * (v @ w)) == abs(
        Matrix([
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
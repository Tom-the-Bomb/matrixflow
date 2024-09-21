from matrix import *

def main() -> None:
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
    #assert A @ (B @ C) == (A @ B) @ C
    assert A + (-A) == O
    #assert A @ ~A == I

    print(A, ~A, A @ ~A)

if __name__ == '__main__':
    main()
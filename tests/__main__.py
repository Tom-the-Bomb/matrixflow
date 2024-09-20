from matrix import *

def test() -> None:
    A = Matrix([[1, 3, 5], [2, 1, 3]])
    B = Matrix([[1, 0], [2, 3], [1, 4]])

    print(A @ B)

if __name__ == '__main__':
    test()
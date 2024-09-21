from __future__ import annotations

from typing import Self, Callable

from .matrix import Matrix
from .vector import Vector

__all__ = ('SquareMatrix',)

class SquareMatrix(Matrix):
    __slots__ = ('_inner', 'n')

    def __init__(self, entries: list[list[float]]) -> None:
        super().__init__(entries)

        if self.rows != self.cols:
            raise ValueError('Not a square: # of rows does not match # of cols')
        self.n = self.rows

    @classmethod
    def from_1D(cls, entries: list[float], n: int) -> Self:
        """Creates a matrix of size `n` from a flat, 1-dimensional list"""
        return super().from_1D(entries, n)

    @classmethod
    def zero(cls, n: int) -> Self:
        """Creates a square zero-matrix of size `n`"""
        return super().zero(n, n)

    @classmethod
    def identity(cls, n: int) -> Self:
        """Creates the identity matrix `I` of size `n`

        (Square matrix filled with `0`s except for `1s` on its major diagonal)"""
        return cls([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    def get_minor_at(self, i: int, j: int) -> float:
        """Gets the minor of the element at row `i` and column `j` in this matrix"""
        minor = []

        for x in range(self.n):
            if x != i:
                row = []
                for y in range(self.n):
                    if y != j:
                        row.append(self._inner[x][y])
                minor.append(row)
        return SquareMatrix(minor).det()

    def get_cofactor_at(self, i: int, j: int) -> float:
        """Gets the cofactor of the element at row `i` and column `j` in this matrix"""
        return (-1) ** (i + j) * self.get_minor_at(i, j)

    def get_cofactor_matrix(self) -> SquareMatrix:
        """Returns this matrix's cofactor matrix"""
        return SquareMatrix([
            [self.get_cofactor_at(i, j) for j in range(self.n)]
            for i in range(self.n)
        ])

    def adj(self) -> SquareMatrix:
        """Returns a new matrix that is this matrix's adjugate matrix"""
        cofactor = self.get_cofactor_matrix()
        cofactor.transpose()
        return cofactor

    def det(self) -> float:
        """Evaluates the determinant of this matrix"""
        if self.n == 1:
            return self._inner[0][0]
        return sum(self._inner[0][j] * self.get_cofactor_at(0, j) for j in range(self.n))

    def inverted(self) -> SquareMatrix:
        """Returns a new matrix that is this matrix's inverse (if exists)"""
        if (det := self.det()) != 0:
            return self.adj() / det
        raise ZeroDivisionError('Inverse does not exist: determinant is 0')

    def invert(self) -> None:
        """Inverts this matrix in place (if exists)"""
        if (det := self.det()) != 0:
            self._inner = self.adj()._inner
            self /= det
        else:
            raise ZeroDivisionError('Inverse does not exist: determinant is 0')

    def __invert__(self) -> SquareMatrix:
        return self.inverted()
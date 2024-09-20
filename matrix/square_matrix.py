from __future__ import annotations

from typing import Self, Callable

from .matrix import Matrix

class SquareMatrix(Matrix):
    __slots__ = ('_inner', 'rows', 'cols', 'n')

    def __init__(self, entries: list[list[float]]) -> None:
        super().__init__(entries)

        if self.rows != self.cols:
            raise ValueError('Not a square: # of rows does not match # of cols')
        self.n = self.rows

    @classmethod
    def from_1D(cls, entries: list[float], n: int) -> Self:
        super().from_1D(entries, n)

    @classmethod
    def identity(cls, n: int) -> Self:
        return cls([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    def get_minor_at(self, i: int, j: int) -> int:
        minor = []

        for x in range(self.n):
            if x != i:
                row = []
                for y in range(self.n):
                    if y != j:
                        row.append(self._inner[x][y])
                minor.append(row)
        return SquareMatrix(minor).det()

    def get_cofactor_at(self, i: int, j: int) -> int:
        return (-1) ** (i + j) * self.get_minor_at(i, j)

    def get_cofactor_matrix(self) -> SquareMatrix:
        return SquareMatrix([
            [self.get_cofactor_at(i, j) for j in range(self.n)]
            for i in range(self.n)
        ])

    def adj(self) -> None:
        self._inner = self.get_cofactor_matrix()._inner
        self.transpose()

    def det(self) -> int:
        if self.n == 1:
            return self._inner[0][0]
        return sum(self._inner[0][j] * self.get_cofactor_at(0, j) for j in range(self.n))

    def invert(self) -> None:
        self.adj()
        self /= self.det()
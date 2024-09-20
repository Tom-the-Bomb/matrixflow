from __future__ import annotations

from typing import Self, Callable
from numbers import Number

from .vector import Vector

class Matrix:
    __slots__ = ('_inner', 'rows', 'cols')

    def __init__(self, entries: list[list[float]]) -> None:
        if not entries:
            raise ValueError('Cannot have empty matrix')

        if not len(set(len(row) for row in entries)) == 1:
            raise ValueError('Row sizes are inconsistent')

        self.rows = len(entries)
        self.cols = len(entries[0])
        self._inner = entries

    @classmethod
    def from_1D(cls, entries: list[float], row_size: int) -> Self:
        if len(entries) % row_size != 0:
            raise ValueError('Provided entries cannot be evenly split into rows of size `row_size`')

        return cls(
            [entries[i:i + row_size] for i in range(0, len(entries), row_size)]
        )

    @classmethod
    def zero(cls, rows: int, cols: int) -> Self:
        return cls(
            [[0] * cols for _ in range(rows )]
        )

    def transpose(self) -> None:
        self._inner = [
            [self._inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]

    def map(self, f: Callable[[int, int], None]) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                f(i, j)

    def copy(self) -> Self:
        return self.__copy__()

    def __deepcopy__(self) -> Self:
        return self.__copy__()

    def __copy__(self) -> Self:
        return self.__class__(
            self._inner.copy()
        )

    def __add__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise ValueError('Addition can only be performed between 2 matrices of the same order')

        def _add(i: int, j: int) -> None:
            self._inner[i][j] += other._inner[i][j]
        copy = self.copy()
        copy.map(_add)
        return copy

    def __iadd__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise ValueError('Addition can only be performed between 2 matrices of the same order')

        def _add(i: int, j: int) -> None:
            self._inner[i][j] += other._inner[i][j]
        self.map(_add)
        return self

    def __sub__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise ValueError('Subtraction can only be performed between 2 matrices of the same order')

        def _sub(i: int, j: int) -> None:
            self._inner[i][j] -= other._inner[i][j]
        copy = self.copy()
        copy.map(_sub)
        return copy

    def __isub__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise ValueError('Subtraction can only be performed between 2 matrices of the same order')

        def _sub(i: int, j: int) -> None:
            self._inner[i][j] -= other._inner[i][j]
        self.map(_sub)
        return self

    def __mul__(self, other: float) -> Self:
        if not isinstance(other, Number):
            raise ValueError('Scalar multiplication on matrices can only be performed with scalars; use A @ B instead for matmul')

        def _mul(i: int, j: int) -> None:
            self._inner[i][j] *= other
        self.map(_mul)
        return self

    def __imul__(self, other: float) -> Self:
        if not isinstance(other, Number):
            raise ValueError('Scalar multiplication on matrices can only be performed with scalars; use A @ B instead for matmul')

        def _mul(i: int, j: int) -> None:
            self._inner[i][j] *= other
        copy = self.copy()
        copy.map(_mul)
        return copy

    def __truediv__(self, other: float) -> Self:
        if not isinstance(other, Number):
            raise ValueError('Division on matrices can only be performed with scalars')

        def _div(i: int, j: int) -> None:
            self._inner[i][j] /= other
        copy = self.copy()
        copy.map(_div)
        return copy

    def __itruediv__(self, other: float) -> Self:
        if not isinstance(other, Number):
            raise ValueError('Division on matrices can only be performed with scalars')

        def _div(i: int, j: int) -> None:
            self._inner[i][j] /= other
        self.map(_div)
        return self

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.cols != other.rows:
            raise ValueError('`AB` can only be performed if (# of columns in A) == (# of rows in B)')

        other_t = other.copy()
        other_t.transpose()

        product = Matrix.zero(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                product._inner[i][j] = Vector(self._inner[i]) * Vector(other_t._inner[j])
        return product

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.rows}x{self.cols}] inner={self._inner}>'

    def __str__(self) -> str:
        return str(self._inner)
from __future__ import annotations

from typing import Self, Sequence, Callable
from fractions import Fraction

from .utils import Base, Number, NumberF, convert
from .vector import Vector

__all__ = ('Matrix',)

class Matrix(Base):
    __slots__ = ('_inner',)

    def __init__(self, entries: Sequence[Sequence[NumberF]]) -> None:
        if not entries:
            raise ValueError('Cannot have empty matrix')

        if not len(set(len(row) for row in entries)) == 1:
            raise ValueError('Row sizes are inconsistent')

        self._inner = [
            [convert(x) for x in row]
            for row in entries
        ]

    @property
    def rows(self) -> int:
        """Returns the number of `rows` this matrix has"""
        return len(self._inner)

    @property
    def cols(self) -> int:
        """Returns the number of `columns` this matrix has"""
        return len(self._inner[0])

    @classmethod
    def from_1D(cls, entries: Sequence[NumberF], row_size: int) -> Self:
        if len(entries) % row_size != 0:
            raise ValueError('Provided entries cannot be evenly split into rows of size `row_size`')

        return cls(
            [entries[i:i + row_size] for i in range(0, len(entries), row_size)]
        )

    @classmethod
    def zero(cls, rows: int, cols: int) -> Self:
        """Creates a `rows x cols` sized matrix filled with `0`s"""
        return cls(
            [[0] * cols for _ in range(rows)]
        )

    def rot90(self) -> None:
        """Rotates this matrix in the positive angular direction (counter-clockwise) by 90 degrees"""
        self.transpose()

        for i in range(self.rows):
            self._inner[i].reverse()

    def transpose(self) -> None:
        """Transposes this matrix: (rows -> columns)"""
        self._inner = [
            [self._inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]

    def add_row(self, row: list[NumberF]) -> None:
        """Appends a row onto the matrix"""
        if len(row) != self.cols:
            raise ValueError('the size of the new row does not match the order of this matrix')
        self._inner.append([convert(x) for x in row])

    def add_col(self, col: list[NumberF]) -> None:
        """Appends a column onto the matrix"""
        if len(col) != self.rows:
            raise ValueError('the size of the new row does not match the order of this matrix')

        for i in range(self.rows):
            self._inner[i].append(convert(col[i]))

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
            [list(row) for row in self._inner]
        )

    def __add__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise TypeError('Addition can only be performed between 2 matrices of the same order')

        copy = self.copy()
        def _add(i: int, j: int) -> None:
            copy._inner[i][j] += other._inner[i][j]
        copy.map(_add)

        return copy

    def __iadd__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise TypeError('Addition can only be performed between 2 matrices of the same order')

        def _add(i: int, j: int) -> None:
            self._inner[i][j] += other._inner[i][j]
        self.map(_add)
        return self

    def __sub__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise TypeError('Subtraction can only be performed between 2 matrices of the same order')

        copy = self.copy()
        def _sub(i: int, j: int) -> None:
            copy._inner[i][j] -= other._inner[i][j]
        copy.map(_sub)

        return copy

    def __isub__(self, other: Matrix) -> Self:
        if not isinstance(other, Matrix) or self.rows != other.rows and self.cols != other.cols:
            raise TypeError('Subtraction can only be performed between 2 matrices of the same order')

        def _sub(i: int, j: int) -> None:
            self._inner[i][j] -= other._inner[i][j]
        self.map(_sub)
        return self

    def __rmul__(self, other: Number) -> Self:
        return self * other

    def __mul__(self, other: Number) -> Self:
        if not isinstance(other, Number):
            raise TypeError('Scalar multiplication on matrices can only be performed with scalars; use A @ B instead for matmul')

        copy = self.copy()
        def _mul(i: int, j: int) -> None:
            copy._inner[i][j] *= other
        copy.map(_mul)

        return copy

    def __imul__(self, other: Number) -> Self:
        if not isinstance(other, Number):
            raise TypeError('Scalar multiplication on matrices can only be performed with scalars; use A @ B instead for matmul')

        def _mul(i: int, j: int) -> None:
            self._inner[i][j] = self._inner[i][j] * other
        self.map(_mul)
        return self

    def __truediv__(self, other: Number) -> Self:
        if not isinstance(other, Number):
            raise TypeError('Division on matrices can only be performed with scalars')

        copy = self.copy()
        def _div(i: int, j: int) -> None:
            copy._inner[i][j] /= other
        copy.map(_div)

        return copy

    def __itruediv__(self, other: Number) -> Self:
        if not isinstance(other, Number):
            raise TypeError('Division on matrices can only be performed with scalars')

        def _div(i: int, j: int) -> None:
            self._inner[i][j] /= other
        self.map(_div)
        return self

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.cols != other.rows:
            raise TypeError('`AB` can only be performed if (# of columns in A) == (# of rows in B)')

        other_t = other.copy()
        other_t.transpose()

        product = Matrix.zero(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                product._inner[i][j] = Vector(self._inner[i]) * Vector(other_t._inner[j])
        return product

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return -1 * self

    def __getitem__(self, i: int) -> list[Fraction]:
        return self._inner[i]

    def display(self) -> str:
        return (
            f'[ {'\n  '.join(
                f"[{', '.join(str(num) for num in row)}]"
                for row in self._inner)} ]'
        )

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.rows}x{self.cols}] inner={self.display()}>'

    def __str__(self) -> str:
        return self.display()

    def __eq__(self, other: Matrix) -> bool:
        if not isinstance(other, Matrix):
            raise TypeError('Equality comparisons can only be mad between matrices')
        return self._inner == other._inner

    def __ne__(self, other: Matrix) -> bool:
        return not self != other
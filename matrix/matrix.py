
from __future__ import annotations

__all__ = ('Matrix',)

from typing import Any, Self, Sequence, Callable
from fractions import Fraction

from .vector import Vector
from .errors import *
from .utils import *

class Matrix:
    """An impmentation for a mathematical 2D Matrix"""
    __slots__ = ('__inner',)

    def __init__(self, entries: Sequence[Sequence[NumberF]]) -> None:
        if not entries:
            raise ValueError('Cannot have empty matrix')

        if not len(set(len(row) for row in entries)) == 1:
            raise ValueError('Row sizes are inconsistent')

        self.__inner = [
            [convert(x) for x in row]
            for row in entries
        ]

    @classmethod
    def from_1D(cls, entries: Sequence[NumberF], cols: int) -> Self:
        """
        Creates a matrix from a flat sequence with `cols` number of columns

        Parameters
        ----------
        entries : :class:`Sequence[NumberF]`
            The flat, 1D sequence
        cols : :class:`int`
            The number of columns of the created matrix

        Returns
        -------
        :obj:`typing.Self`


        Raises
        ------
        :class:`ValueError`
            The size of the provided sequence cannot be divided evenly
        """
        if (n := len(entries)) % cols != 0:
            raise CannotGroupEvenly(n, cols)

        return cls(
            [entries[i:i + cols] for i in range(0, len(entries), cols)]
        )

    @classmethod
    def zero(cls, rows: int, cols: int) -> Self:
        """
        Creates a ``rows x cols`` sized matrix all filled with ``0``

        Parameters
        ----------
        rows : :class:`int`
            The number of rows
        cols : :class:`int`
            The number of columns

        Returns
        -------
        :obj:`typing.Self`
            The created zero matrix
        """
        return cls(
            [[0] * cols for _ in range(rows)]
        )

    @classmethod
    def identity(cls, n: int) -> Self:
        """Creates the identity matrix ``I`` of size ``n``

        (Square matrix filled with ``0``s except for ``1s`` on its major diagonal)"""
        return cls([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])

    @property
    def rows(self) -> int:
        """Returns the number of ``rows`` this matrix has"""
        return len(self.__inner)

    @property
    def cols(self) -> int:
        """Returns the number of ``columns`` this matrix has"""
        return len(self.__inner[0])

    def rot90(self) -> None:
        """Rotates this matrix in the positive angular direction (counter-clockwise) by 90 degrees"""
        self.transpose()

        for i in range(self.rows):
            self.__inner[i].reverse()

    def transpose(self) -> None:
        """Transposes this matrix: (rows -> columns)"""
        self.__inner = [
            [self.__inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]

    def add_row(self, row: Sequence[NumberF]) -> None:
        """Appends a row onto the matrix"""
        if len(row) != self.cols:
            raise ValueError('the size of the new row does not match the order of this matrix')
        self.__inner.append([convert(x) for x in row])

    def add_col(self, col: Sequence[NumberF]) -> None:
        """Appends a column onto the matrix"""
        if len(col) != self.rows:
            raise ValueError('the size of the new row does not match the order of this matrix')

        for i in range(self.rows):
            self.__inner[i].append(convert(col[i]))

    def map(self, f: Callable[[int, int], None]) -> None:
        for i in range(self.rows):
            for j in range(self.cols):
                f(i, j)

    def get_submatrix_at(self, i: int, j: int) -> Matrix:
        submatrix: list[list[Fraction]] = []

        for x in range(self.rows):
            if x != i:
                row: list[Fraction] = []
                for y in range(self.cols):
                    if y != j:
                        row.append(self.__inner[x][y])
                submatrix.append(row)
        return Matrix(submatrix)

    def is_square(self) -> bool:
        return self.rows == self.cols

    def get_minor_at(self, i: int, j: int) -> Fraction:
        """Gets the minor of the element at row ``i`` and column ``j`` in this matrix"""
        assert self.is_square()

        return self.get_submatrix_at(i, j).det()

    def get_cofactor_at(self, i: int, j: int) -> Fraction:
        """Gets the cofactor of the element at row ``i`` and column ``j`` in this matrix"""
        return (-1) ** (i + j) * self.get_minor_at(i, j)

    def get_cofactor_matrix(self) -> Matrix:
        """Returns this matrix's cofactor matrix"""
        return Matrix([
            [self.get_cofactor_at(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def adj(self) -> Matrix:
        """Returns a new matrix that is this matrix's adjugate matrix"""
        cofactor = self.get_cofactor_matrix()
        cofactor.transpose()
        return cofactor

    def det(self) -> Fraction:
        """Evaluates the determinant of this matrix"""
        assert self.is_square()

        if self.rows == 1:
            return self.__inner[0][0]
        return sum(
            (self.__inner[0][j] * self.get_cofactor_at(0, j) for j in range(self.cols)),
            start=Fraction()
        )

    def trace(self) -> Fraction:
        r"""
        Returns the trace of this square matrix :math:`\mathbf{A}`

        :math:`\mathrm{tr}\left(\mathbf{A}\right)=\displaystyle\sum_i{\mathbf{A}_ii}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The trace of the square matrix: :math:`tr(\mathbf{A})`

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        assert self.is_square(), 'A'

        return sum(
            (self.__inner[i][i] for i in range(self.rows)),
            start=Fraction()
        )

    def inverted(self) -> Matrix:
        """Returns a new matrix that is this matrix's inverse (if exists)"""
        assert self.is_square()

        if (det := self.det()) != 0:
            return self.adj() / det
        raise SingularMatrix()

    def invert(self) -> None:
        """Inverts this matrix in place (if exists)"""
        assert self.is_square()

        if (det := self.det()) != 0:
            self.__inner = self.adj().__inner
            self /= det
        else:
            raise SingularMatrix()

    def copy(self) -> Self:
        return self.__deepcopy__()

    def display(self) -> str:
        return (
            f'[ {'\n  '.join(
                f"[{', '.join(str(num) for num in row)}]"
                for row in self.__inner)} ]'
        )

    def __add__(self, other: Matrix) -> Self:
        copy = self.copy()
        def _add(i: int, j: int) -> None:
            copy.__inner[i][j] += other.__inner[i][j]
        copy.map(_add)

        return copy

    def __iadd__(self, other: Matrix) -> Self:
        def _add(i: int, j: int) -> None:
            self.__inner[i][j] += other.__inner[i][j]
        self.map(_add)
        return self

    def __sub__(self, other: Matrix) -> Self:
        copy = self.copy()
        def _sub(i: int, j: int) -> None:
            copy.__inner[i][j] -= other.__inner[i][j]
        copy.map(_sub)

        return copy

    def __isub__(self, other: Matrix) -> Self:
        def _sub(i: int, j: int) -> None:
            self.__inner[i][j] -= other.__inner[i][j]
        self.map(_sub)
        return self

    def __rmul__(self, other: Number) -> Self:
        return self * other

    def __mul__(self, other: Number) -> Self:
        copy = self.copy()
        def _mul(i: int, j: int) -> None:
            copy.__inner[i][j] *= other
        copy.map(_mul)

        return copy

    def __imul__(self, other: Number) -> Self:
        def _mul(i: int, j: int) -> None:
            self.__inner[i][j] = self.__inner[i][j] * other
        self.map(_mul)
        return self

    def __truediv__(self, other: Number) -> Self:
        copy = self.copy()
        def _div(i: int, j: int) -> None:
            copy.__inner[i][j] /= other
        copy.map(_div)

        return copy

    def __itruediv__(self, other: Number) -> Self:
        def _div(i: int, j: int) -> None:
            self.__inner[i][j] /= other
        self.map(_div)
        return self

    def __matmul__(self, other: Matrix) -> Matrix:
        if self.cols != other.rows:
            raise CannotMatMul()

        other_t = other.copy()
        other_t.transpose()

        product = Matrix.zero(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                product.__inner[i][j] = Vector(self.__inner[i]) * Vector(other_t.__inner[j])
        return product

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return -1 * self

    def __len__(self) -> int:
        return self.rows

    def __abs__(self) -> Fraction:
        return self.det()

    def __invert__(self) -> Matrix:
        return self.inverted()

    def __getitem__(self, i: int) -> list[Fraction]:
        return self.__inner[i]

    def __copy__(self) -> Self:
        return self.__class__(self.__inner.copy())

    def __deepcopy__(self) -> Self:
        return self.__class__(
            [row.copy() for row in self.__inner]
        )

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [{self.rows}x{self.cols}] inner={self.display()}>'

    def __str__(self) -> str:
        return self.display()

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Matrix) and self.__inner == other.__inner

    def __ne__(self, other: Any) -> bool:
        return not self != other

from __future__ import annotations

__all__ = ('Matrix',)

from typing import Any, Self, Sequence, Callable
from fractions import Fraction

from .vector import Vector
from .utils import *

class Matrix:
    """An implementation for a 2D mathematical matrix

    Parameters
    ----------
    entries :
        The raw entries to initialize the matrix with

    Raises
    ------
    :class:`IndexError`
        Entries have rows with inconsistent sizes
    """
    __slots__ = ('__inner',)
    __inner: list[list[Fraction]]

    def __init__(self, entries: Sequence[Sequence[Number]]) -> None:
        if not entries:
            self.__inner = []

        length = len(entries[0])
        if any(len(row) != length for row in entries):
            raise IndexError('Row sizes are inconsistent')

        self.__inner = [
            [convert(x) for x in row]
            for row in entries
        ]

    @classmethod
    def from_1D(cls, entries: Sequence[Number], cols: int) -> Self:
        """Creates a matrix with ``cols`` number of columns from a flat, 1D sequence

        Parameters
        ----------
        entries : :class:`Sequence[Number]`
            The flat, 1D sequence containing the entries
        cols : :class:`int`
            The number of columns of the created matrix

        Returns
        -------
        :obj:`~typing.Self`
            The created matrix

        Raises
        ------
        :class:`IndexError`
            The size of the provided sequence cannot be divided evenly
        """
        if (n := len(entries)) % cols != 0:
            raise IndexError(f'Provided entries of size `{n}` cannot be evenly split into rows of size `{cols}`')

        return cls(
            [entries[i:i + cols] for i in range(0, n, cols)]
        )

    @classmethod
    def zero(cls, rows: int, cols: int) -> Self:
        r"""Creates a ``rows`` x ``cols`` sized matrix :math:`\mathbf{0}` all filled with ``0``

        :math:`\mathbf{A0}=\mathbf{0}` for all matrices :math:`\mathbf{A}`

        Parameters
        ----------
        rows : :class:`int`
            The number of rows
        cols : :class:`int`
            The number of columns

        Returns
        -------
        :obj:`~typing.Self`
            The created zero matrix: :math:`\mathbf{0}`
        """
        return cls(
            [[0] * cols for _ in range(rows)]
        )

    @classmethod
    def identity(cls, n: int) -> Self:
        r"""Creates the identity matrix :math:`\mathbf{I}` of size ``n``
        A square matrix filled with ``0`` except for ``1`` on its major diagonal

        :math:`\mathbf{AI}=\mathbf{IA}=\mathbf{A}` for all matrices :math:`\mathbf{A}`

        Returns
        -------
        :obj:`~typing.Self`
            The created identity matrix: :math:`\mathbf{I}`
        """
        return cls([
            [int(i == j) for j in range(n)]
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
        """Rotates the entries of this matrix in-place, in the positive angular direction (counter-clockwise) by 90 degrees"""
        self.transpose()

        for i in range(self.rows):
            self.__inner[i].reverse()

    def transpose(self) -> None:
        r"""Transposes this matrix :math:`\mathbf{A}` in-place: switches the ``rows`` with the ``columns``

        :math:`\mathbf{A}\mapsto\mathbf{A}^\intercal`
        """
        self.__inner = [
            [self.__inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]

    def add_row(self, row: Sequence[Number]) -> None:
        """Appends a row ``row`` onto the end of this matrix

        Parameters
        ----------
        row :
            The row to append onto this matrix

        Raises
        ------
        :class:`IndexError`
            The size of the new row does not match the existing matrix's row size
        """
        if len(row) != self.cols:
            raise IndexError('the size of the new row does not match this matrix\'s row size')
        self.__inner.append([convert(x) for x in row])

    def add_col(self, col: Sequence[Number]) -> None:
        """Appends a column ``col`` onto the end of this matrix

        Parameters
        ----------
        col :
            The column to append onto this matrix

        Raises
        ------
        :class:`IndexError`
            The size of the new column does not match the existing matrix's column size
        """
        if len(col) != self.rows:
            raise IndexError('the size of the new column does not match this matrix\'s column size')

        for i in range(self.rows):
            self.__inner[i].append(convert(col[i]))

    def map(self, f: Callable[[int, int], None]) -> None:
        """Maps a function over all of this matrix's elements

        Parameters
        ----------
        f :
           The function to apply over the elements
           It will take in the indices of the current element ``(i, j)``
           and should not return anything as it maps in-place
        """
        for i in range(self.rows):
            for j in range(self.cols):
                f(i, j)

    def is_square(self) -> bool:
        """Returns ``True`` if this matrix is square, else ``False``

        Returns
        -------
        :class:`bool`
            whether or not this matrix is square
        """
        return self.rows == self.cols

    def is_same_order(self, other: Matrix) -> bool:
        """Returns ``True`` if this matrix is of the same order as ``other``, else ``False``

        Parameters
        ----------
        other :
           The other matrix to compare to

        Returns
        -------
        :class:`bool`
            Whether or not the matrices are of the same order
        """
        return self.rows == other.rows and self.cols == other.cols

    def get_submatrix_of(self, rows: set[int], cols: set[int]) -> Matrix:
        """A submatrix is the matrix obtained by deleting the rows that that have indices in ``rows`` and columns that have indices in ``cols``

        Returns
        -------
        :class:`Matrix`
            The computed submatrix
        """
        submatrix: list[list[Fraction]] = []

        for x in range(self.rows):
            if x not in rows:
                row: list[Fraction] = []
                for y in range(self.cols):
                    if y not in cols:
                        row.append(self.__inner[x][y])
                submatrix.append(row)
        return Matrix(submatrix)

    def get_minor_at(self, i: int, j: int) -> Fraction:
        r"""Computes the minor :math:`\mathbf{M}_{ij}` of this matrix at :math:`\mathbf{A}_{ij}`

        The minor is the determinant of the submatrix after deleting row ``i`` and column ``j``

        :math:`\mathbf{M}_{ij}=\det\left(\left(\mathbf{A}_{pq}\right)_{p\ne i,q\ne j}\right)`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The minor: :math:`\mathbf{M}_{ij}`

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        assert self.is_square(), 'This operation requires the matrix to be square'

        return self.get_submatrix_of({i}, {j}).det()

    def get_cofactor_at(self, i: int, j: int) -> Fraction:
        r"""Computes the cofactor :math:`\mathbf{C}_{ij}` of this matrix at :math:`\mathbf{A}_{ij}`

        The cofactor is simply the element's signed minor: :math:`\mathbf{C}_{ij}=(-1)^{i+j}\cdot\mathbf{M}_{ij}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The cofactor: :math:`\mathbf{M}_{ij}`

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        return (-1) ** (i + j) * self.get_minor_at(i, j)

    def get_cofactor_matrix(self) -> Matrix:
        r"""Computes the matrix of cofactors :math:`\mathbf{C}` of this matrix

        :math:`\mathbf{C}=\left((-1)^{i+j}\cdot\mathbf{M}_{ij}\right)_{0\le i,j\lt n}`

        Returns
        -------
        :class:`Matrix`
            The cofactor matrix

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        return Matrix([
            [self.get_cofactor_at(i, j) for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def adj(self) -> Matrix:
        r"""Computes a new matrix that is this matrix :math:`\mathbf{A}`'s adjugate matrix

        :math:`\mathrm{adj}\left(\mathbf{A}\right)=\mathbf{C}^\intercal` where :math:`\mathbf{C}` is the cofactor matrix

        Returns
        -------
        :class:`Matrix`
            The adjugate matrix

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        cofactor = self.get_cofactor_matrix()
        cofactor.transpose()
        return cofactor

    def det(self) -> Fraction:
        r"""Computes the determinant of this matrix :math:`\mathbf{A}`

        The determinant of a matrix is the sum of all the products of the cofactor and element in any given row or column

        :math:`|\mathbf{A}|=\det(\mathbf{A})=\displaystyle\sum_i{\mathbf{A}_{ij}\mathbf{C}_{ij}}=\displaystyle\sum_j{\mathbf{A}_{ij}\mathbf{C}_{ij}}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The determinant of the matrix: :math:`\det(\mathbf{A})`

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        assert self.is_square(), 'This operation requires the matrix to be square'

        if self.rows == 1:
            return self.__inner[0][0]

        return sum(
            (self.__inner[0][j] * self.get_cofactor_at(0, j) for j in range(self.cols)),
            start=Fraction()
        )

    def trace(self) -> Fraction:
        r"""Returns the trace of this square matrix :math:`\mathbf{A}`

        :math:`\mathrm{tr}\left(\mathbf{A}\right)=\displaystyle\sum_i{\mathbf{A}_{ii}}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The trace of the square matrix: :math:`tr(\mathbf{A})`

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        assert self.is_square(), 'This operation requires the matrix to be square'

        return sum(
            (self.__inner[i][i] for i in range(self.rows)),
            start=Fraction()
        )

    def inverted(self) -> Matrix:
        r"""Computes a new matrix :math:`\mathbf{A}^{-1}` that is this matrix :math:`\mathbf{A}`'s

        :math:`\mathbf{A}^{-1}=\frac{\mathrm{adj}\left(\mathbf{A}\right)}{\det(\mathbf{A})}` if :math:`\det(\mathbf{A})\ne0`

        Returns
        -------
        :class:`Matrix`
            The new new inverse matrix: :math:`\mathbf{A}^{-1}`

        Raises
        ------
        :class:`ValueError`
            This matrix singular: :math:`\det(\mathbf{A})=0` (inverse does not exist)
        """
        assert self.is_square(), 'This operation requires the matrix to be square'
        if (det := self.det()) != 0:
            return self.adj() / det
        raise ValueError('This matrix is singular: inverse does not exist')

    def invert(self) -> None:
        r"""Inverts this matrix in-place

        :math:`\mathbf{A}\mapsto\mathbf{A}^{-1}` if :math:`\det(\mathbf{A})\ne0`

        Returns
        -------
        :class:`Matrix`
            The inverted matrix: :math:`\mathbf{A}^{-1}`

        Raises
        ------
        :class:`ValueError`
            This matrix singular: :math:`\det(\mathbf{A})=0` (inverse does not exist)
        """
        assert self.is_square(), 'This operation requires the matrix to be square'

        if (det := self.det()) != 0:
            self.__inner = self.adj().__inner
            self /= det
        else:
            raise ValueError('This matrix is singular: inverse does not exist')

    def copy(self) -> Self:
        """Makes a deepcopy of this matrix and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied matrix
        """
        return self.__deepcopy__()

    def display(self) -> str:
        """Returns a formatted, displayable string representation of this matrix

        Returns
        -------
        :class:`str`
            The formatted string
        """
        return (
            f'[ {'\n  '.join(
                f"[{', '.join(str(num) for num in row)}]"
                for row in self.__inner)} ]'
        )

    def __add__(self, other: Matrix) -> Self:
        r"""Computes a new matrix that is the sum of this matrix :math:`\mathbf{A}` and ``other`` :math:`\mathbf{B}`

        Parameters
        ----------
        other :
            The matrix to perform the addition with

        Returns
        -------
        :class:`Matrix`
            The sum matrix: :math:`\mathbf{A}+\mathbf{B}`

        Raises
        ------
        :class:`AssertionError`
            Attempted to add matrices of different orders
        """
        assert self.is_same_order(other), 'This operation requires operand matrices to be of the same order'

        copy = self.copy()
        def _add(i: int, j: int) -> None:
            copy.__inner[i][j] += other.__inner[i][j]
        copy.map(_add)

        return copy

    def __iadd__(self, other: Matrix) -> Self:
        r"""Adds ``other`` :math:`\mathbf{B}` onto this matrix :math:`\mathbf{A}` (in place)

        Parameters
        ----------
        other :
            The matrix to perform the addition with

        Returns
        -------
        :class:`Matrix`
            The sum matrix: :math:`\mathbf{A}+\mathbf{B}`

        Raises
        ------
        :class:`AssertionError`
            Attempted to add matrices of different orders
        """
        assert self.is_same_order(other), 'This operation requires operand matrices to be of the same order'

        def _add(i: int, j: int) -> None:
            self.__inner[i][j] += other.__inner[i][j]
        self.map(_add)
        return self

    def __sub__(self, other: Matrix) -> Self:
        r"""Computes a new matrix that is the difference of this matrix :math:`\mathbf{A}` and ``other`` :math:`\mathbf{B}`

        Parameters
        ----------
        other :
            The matrix to perform the subtraction with

        Returns
        -------
        :class:`Matrix`
            The difference matrix: :math:`\mathbf{A}-\mathbf{B}`

        Raises
        ------
        :class:`AssertionError`
            Attempted to subtract matrices of different orders
        """
        assert self.is_same_order(other), 'This operation requires operand matrices to be of the same order'

        copy = self.copy()
        def _sub(i: int, j: int) -> None:
            copy.__inner[i][j] -= other.__inner[i][j]
        copy.map(_sub)

        return copy

    def __isub__(self, other: Matrix) -> Self:
        r"""Subtracts ``other`` :math:`\mathbf{B}` from this matrix :math:`\mathbf{A}` (in place)

        Parameters
        ----------
        other :
            The matrix to perform the subtraction with

        Returns
        -------
        :class:`Matrix`
            The difference matrix: :math:`\mathbf{A}-\mathbf{B}`

        Raises
        ------
        :class:`AssertionError`
            Attempted to subtract matrices of different orders
        """
        assert self.is_same_order(other), 'This operation requires operand matrices to be of the same order'

        def _sub(i: int, j: int) -> None:
            self.__inner[i][j] -= other.__inner[i][j]
        self.map(_sub)
        return self

    def __rmul__(self, other: Number) -> Self:
        r"""Computes a new matrix that is the scalar multiplication of ``other`` :math:`k` on this matrix :math:`\mathbf{A}`

        Parameters
        ----------
        other :
            The scalar to compute the multiplication

        Returns
        -------
        :class:`Matrix`
            The scalar multiplication matrix: :math:`k\mathbf{A}`
        """
        return self * other

    def __mul__(self, other: Number) -> Self:
        r"""Computes a new matrix that is the scalar multiplication of ``other`` :math:`k` on this matrix :math:`\mathbf{A}`

        Parameters
        ----------
        other :
            The scalar to compute the multiplication

        Returns
        -------
        :class:`Matrix`
            The scalar multiplication matrix: :math:`k\mathbf{A}`
        """
        copy = self.copy()
        def _mul(i: int, j: int) -> None:
            copy.__inner[i][j] *= convert(other)
        copy.map(_mul)

        return copy

    def __imul__(self, other: Number) -> Self:
        r"""Scalar multiplies ``other`` :math:`k` on this matrix :math:`\mathbf{A}` (in place)

        Parameters
        ----------
        other :
            The scalar to compute the multiplication

        Returns
        -------
        :class:`Matrix`
            The scalar multiplication matrix: :math:`k\mathbf{A}`
        """
        def _mul(i: int, j: int) -> None:
            self.__inner[i][j] = self.__inner[i][j] * convert(other)
        self.map(_mul)
        return self

    def __truediv__(self, other: Number) -> Self:
        r"""Computes a new matrix that is the scalar division of ``other`` :math:`k` on this matrix :math:`\mathbf{A}`

        Parameters
        ----------
        other :
            The scalar to compute the divisions

        Returns
        -------
        :class:`Matrix`
            The scalar divison matrix: :math:`\frac{1}{k}\mathbf{A}`
        """
        copy = self.copy()
        def _div(i: int, j: int) -> None:
            copy.__inner[i][j] /= convert(other)
        copy.map(_div)

        return copy

    def __itruediv__(self, other: Number) -> Self:
        r"""Scalar divides ``other`` :math:`k` on this matrix :math:`\mathbf{A}` (in place)

        Parameters
        ----------
        other :
            The scalar to compute the divisions

        Returns
        -------
        :class:`Matrix`
            The scalar divison matrix: :math:`\frac{1}{k}\mathbf{A}`
        """
        def _div(i: int, j: int) -> None:
            self.__inner[i][j] /= convert(other)
        self.map(_div)
        return self

    def __matmul__(self, other: Matrix) -> Matrix:
        r"""Computes a new matrix that is the matrix multiplication between
        this matrix :math:`\mathbf{A}`(size :mathbf:`m\times n`) and ``other`` :math:`\mathbf{B}` (size `p\times q`)

        The operation can only be performed if :math:`n` is equal to :math:`p` and will yield a matrix of size :math:`m\times q`

        :math:`\mathbf{AB}=\left(\displaystyle\sum_r{\mathbf{A}_{ir}\mathbf{B}_{ij}}\right)_{1\le i,j\lt n}`

        Parameters
        ----------
        other :
            The matrix to perform matrix multiplication with

        Returns
        -------
        :class:`Matrix`
            The matrix that is the result of the matrix multiplication
        """
        assert self.cols == other.rows, 'The # of columns in the left matrix does not match the # of rows in the right matrix'

        other_t = other.copy()
        other_t.transpose()

        product = Matrix.zero(self.rows, other.cols)
        for i in range(self.rows):
            for j in range(other.cols):
                product.__inner[i][j] = Vector(self.__inner[i]) * Vector(other_t.__inner[j])
        return product

    def __pos__(self) -> Self:
        """Unary plus: does nothing as it performs a scalar multiplication of all the elements by :math:`+1`

        Returns
        -------
        :obj:`~typing.Self`
            Returns itself
        """
        return self

    def __neg__(self) -> Self:
        r"""Negates this matrix :math:`\mathbf{A}` (negates all of its elements)

        Returns
        -------
        :obj:`~typing.Self`
            Returns the negated vector: :math:`{-\mathbf{A}}`
        """
        return -1 * self

    def __len__(self) -> int:
        """Returns the number of rows in this matrix: ``self.rows``

        Returns
        -------
        :class:`int`
            The number of rows in this matrix
        """
        return self.rows

    def __abs__(self) -> Fraction:
        """Computes the determinant of this matrix:

        See: :meth:`det`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The determinant of this matrix

        Raises
        ------
        :class:`AssertionError`
            This matrix is not square
        """
        return self.det()

    def __invert__(self) -> Matrix:
        r"""Computes a new matrix that is this matrix's inverse

        See: :meth:`inverted`

        Returns
        -------
        :class:`Matrix`
            The new new inverse matrix: :math:`\mathbf{A}^{-1}`

        Raises
        ------
        :class:`ValueError`
            This matrix singular: :math:`\det(\mathbf{A})=0` (inverse does not exist)
        """
        return self.inverted()

    def __getitem__(self, i: int) -> list[Fraction]:
        r"""Gets the ``i-th`` row of this vector :math:`\mathbf{A}`

        Parameters
        ----------
        i :
            The index of the element in this vector

        Returns
        -------
        :obj:`list[~fractions.Fraction]`
            The ``i-th`` row of this vector: :math:`\mathbf{A}_i`
        """
        return self.__inner[i]

    def __copy__(self) -> Self:
        """Creates a copy of this matrix and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied matrix
        """
        return self.__class__(self.__inner.copy())

    def __deepcopy__(self) -> Self:
        """Creates a deep copy of this matrix and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied matrix
        """
        return self.__class__(
            [row.copy() for row in self.__inner]
        )

    def __repr__(self) -> str:
        """Return ``repr(self)``"""
        return f'<{self.__class__.__name__} [{self.rows}x{self.cols}] inner={self.display()}>'

    def __str__(self) -> str:
        """Return ``str(self)``"""
        return self.display()

    def __eq__(self, other: Any) -> bool:
        """Return ``self == other``"""
        return isinstance(other, Matrix) and self.__inner == other.__inner

    def __ne__(self, other: Any) -> bool:
        """Return ``self != other``"""
        return not self != other

from __future__ import annotations

__all__ = (
    'solve_linear_system',
    'Matrix',
)

from typing import Any, Self, Sequence, Callable, overload
from fractions import Fraction
from math import sin, cos

from .vector import Vector
from .utils import *

def solve_linear_system(coefficients: Sequence[Sequence[Number]], b: Sequence[Number]) -> list[Fraction]:
    r"""Solves a system of ``n`` linear equations with ``n`` variables: :math:`\mathbf{A}x=\mathbf{b}`

    It returns :math:`x` by computing :math:`x=\mathbf{A}^{-1}\mathbf{b}`

    Parameters
    ----------
    coefficients :
        The coefficients of the respective unknowns in the system: :math:`\mathbf{A}`
    b :
        The corresponding values that each equation is set equal to: :math:`\mathbf{b}`

    Returns
    -------
    list[:obj:`~fractions.Fraction`]
        An array of solutions with values corresponding to the respective unknowns

    Example
    -------
    >>> solve_linear_system(
        [[2, 3], [5, -6]],
        [4, -4]
    )
    [Fraction(4, 9), Fraction(28, 27)]

    This is equivalent to:

    :math:`\begin{cases}2x+3y=4\\5x-6y=-4\end{cases}`

    :math:`\therefore x=\frac{4}{9}, y=\frac{28}{27}`
    """
    A = Matrix(coefficients)
    A.invert()
    B = Vector(b)

    return (A @ B).inner

class Matrix:
    """An implementation for a 2D mathematical matrix

    Parameters
    ----------
    entries : ~typing.Sequence[~typing.Sequence[int | float | ~fractions.Fraction]]
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
        entries :
            The flat, 1D sequence containing the entries
        cols :
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
    def from_columns(cls, columns: Sequence[Sequence[Number]]) -> Self:
        """Creates a new matrix from a list of ``columns`` of the matrix

        * This is useful for creating transformations where ``columns`` directly corresponds to the list of respective transformed basis vectors

        Parameters
        ----------
        columns :
            The list of columns

        Returns
        -------
        :obj:`~typing.Self`
            The created matrix
        """
        mat = cls(columns)
        mat.transpose()
        return mat

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

    @classmethod
    def shear_2d(cls, x_gradient: Number = 0, y_gradient: Number = 0) -> Self:
        """Creates a linear map that performs a **shear**

        A shear displaces each point in a fixed direction by an fixed amount relative to a fixed line (i.e. the axes)

        Parameters
        ----------
        x_gradient :
            The horizontal shear gradient on the :math:`x` basis vector, by default ``0``
        y_gradient :
            The vertical shear gradient  on the :math:`y` basis vector, by default ``0``

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [1, y_gradient],
            [x_gradient, 1],
        ])

    @classmethod
    def reflect_x(cls) -> Self:
        """Creates a linear map that performs a vertical reflection across the horizontal (x-axis)

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [1, 0],
            [0, -1],
        ])

    @classmethod
    def reflect_y(cls) -> Self:
        """Creates a linear map that performs a horizontal reflection across the vertical (y-axis)

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [-1, 0],
            [0, 1],
        ])

    @classmethod
    def squeeze_map_2d(cls, r: Number = 1) -> Self:
        """Creates a linear map that performs a **squeeze map**

        A squeeze map is a transformation that preserves euclidean area of regions in the cartesian plane,
        but is not a **rotation** or **shear**

        Parameters
        ----------
        r :
            The squeeze factor, by default ``1``

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [r, 0],
            [0, 1/r],
        ])

    @classmethod
    def scale_2d(cls, k: Number = 1) -> Self:
        """Creates a linear map that performs a **uniform scale**

        A uniform scale enlarges or shrinks the regions in the cartesian plane by the same factor in all directions

        Parameters
        ----------
        k :
            The scale factor, by default ``1``

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [k, 0],
            [0, k],
        ])

    @classmethod
    def rotate_2d(cls, theta: Number = 0) -> Self:
        """Creates a linear map that performs a **rotation**

        Parameters
        ----------
        theta :
            The angle of rotation in radians, by default ``0``

        Returns
        -------
        :obj:`~typing.Self`
           The created linear map
        """
        return cls([
            [cos(theta), -sin(theta)],
            [sin(theta), cos(theta)],
        ])

    @property
    def rows(self) -> int:
        """Returns the number of ``rows`` this matrix has"""
        return len(self.__inner)

    @property
    def cols(self) -> int:
        """Returns the number of ``columns`` this matrix has"""
        return len(self.__inner[0])

    @property
    def inner(self) -> list[list[Fraction]]:
        """Returns a reference to the internal list representation of this matrix"""
        return self.__inner

    def row_at(self, i: int) -> list[Fraction]:
        """Returns a reference to the ``i``-th row of the matrix

        Parameters
        ----------
        i :
            The index of the row

        Returns
        -------
        list[:obj:`~fractions.Fraction`]
            The row
        """
        return self.__inner[i]

    def col_at(self, j: int) -> list[Fraction]:
        """Returns a copy of the ``j``-th column of the matrix

        Parameters
        ----------
        j :
            The index of the column

        Returns
        -------
        list[:obj:`~fractions.Fraction`]
            The column
        """
        return [self.__inner[i][j] for i in range(self.rows)]

    def rot90(self) -> None:
        """Rotates the entries of this matrix in place, in the positive angular direction (counter-clockwise) by 90 degrees"""
        self.transpose()

        for i in range(self.rows):
            self.__inner[i].reverse()

    def transpose(self) -> None:
        r"""Transposes this matrix :math:`\mathbf{A}` in place: switches the ``rows`` with the ``columns``

        :math:`\mathbf{A}\mapsto\mathbf{A}^\intercal`
        """
        self.__inner = [
            [self.__inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ]

    def transposed(self) -> Matrix:
        r"""Computes a new matrix that is this matrix :math:`\mathbf{A}`'s transpose: switches the ``rows`` with the ``columns``

        Returns
        -------
        :class:`Matrix`
            The transposed matrix: :math:`\mathbf{A}^\intercal`
        """
        return Matrix([
            [self.__inner[i][j] for i in range(self.rows)]
            for j in range(self.cols)
        ])

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
           and should not return anything as it maps in place
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

    def is_singular(self) -> bool:
        r"""Returns ``True`` if this matrix :math:`\mathbf{A}` is square and is **singular**:
        meaning :math:`\mathbf{A}^{-1}` does not exist, else ``False``

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is square and is singular
        """
        return self.is_square() and self.det() == 0

    def is_symmetric(self) -> bool:
        r"""Returns ``True`` if this matrix :math:`\mathbf{A}` is **symmetric**:
        :math:`\mathbf{A}=\mathbf{A}^\intercal`, else ``False``

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is symmetric
        """
        return self == self.transposed()

    def is_skew_symmetric(self) -> bool:
        r"""Returns ``True`` if this matrix :math:`\mathbf{A}` is **skew-symmetric**:
        :math:`\mathbf{A}=-\mathbf{A}^\intercal`, else ``False``

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is skew-symmetric
        """
        return self == -self.transposed()

    def is_orthogonal(self) -> bool:
        r"""Returns ``True`` if this matrix :math:`\mathbf{A}` is **orthogonal**:
        :math:`\mathbf{A}^\intercal=\mathbf{A}^{-1}`, else ``False``

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is orthogonal
        """
        try:
            return self.transposed() == self.inverted()
        except (ValueError, AssertionError):
            return False

    def is_upper_triangular(self) -> bool:
        r"""Returns ``True`` if this matrix is **upper-triangular**:
        meaning all elements *below* the main diagonal are zero.

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is upper-triangular
        """
        return all(self.__inner[i][j] == 0 for j in range(self.cols) for i in range(self.rows) if i > j)

    def is_lower_triangular(self) -> bool:
        r"""Returns ``True`` if this matrix is **lower-triangular**:
        meaning all elements *above* the main diagonal are zero.

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is lower-triangular
        """
        return all(self.__inner[i][j] == 0 for j in range(self.cols) for i in range(self.rows) if i < j)

    def is_diagonal(self) -> bool:
        r"""Returns ``True`` if this matrix is **lower-triangular**:
        meaning all elements that are not on the main diagonal are zero.

        Returns
        -------
        :class:`bool`
            Whether or not this matrix is diagonal
        """
        return all(self.__inner[i][j] == 0 for j in range(self.cols) for i in range(self.rows) if i == j)

    def row_add(self, i: int, j: int, k: Number) -> None:
        r"""Elementary row operation 1: replaces row ``i`` with the sum of itself and ``k`` times row ``j``:

        :math:`R_i+kR_j\rightarrow R_i` where :math:`i\ne j`

        Parameters
        ----------
        i :
            The row to replace
        j :
            The row to multiply by ``k`` and add onto row ``i``
        k :
            The multiplier for row ``j``
        """
        for col in range(self.cols):
            self[i][col] += convert(k * self[j][col])

    def row_mul(self, i: int, k: Number) -> None:
        r"""Elementary row operation 2: multiplies each element in row ``i`` by ``k``

        :math:`kR_i\rightarrow R_i` where :math:`k\ne 0`

        Parameters
        ----------
        i :
            The row to modify
        k :
            The multiplier for row ``i``
        """
        for col in range(self.cols):
            self[i][col] *= convert(k)

    def row_switch(self, i: int, j: int) -> None:
        r"""Elementary row operation 3: switches row ``i`` with row ``j``

        :math:`R_i\leftrightarrow R_j`

        Parameters
        ----------
        i :
            The row to get replaced by ``j``
        j :
            The row to get replaced by row ``i``
        """
        for col in range(self.cols):
            self[i][col], self[j][col] = self[i][col], self[j][col]

    @overload
    def row_echelon_form(self, b: None = None) -> tuple[Matrix, None]:
        ...

    @overload
    def row_echelon_form(self, b: Matrix) -> tuple[Matrix, Matrix]:
        ...

    @overload
    def reduced_row_echelon_form(self, b: None = None) -> tuple[Matrix, None]:
        ...

    @overload
    def reduced_row_echelon_form(self, b: Matrix) -> tuple[Matrix, Matrix]:
        ...

    def row_echelon_form(self, b: Matrix | None = None) -> tuple[Matrix, Matrix | None]:
        r"""Computes a new matrix that is this matrix :math:`\mathbf{A}`'s **row echelon form** using the
        `Gaussian elimination algorithm <https://math.libretexts.org/Bookshelves/Linear_Algebra/Fundamentals_of_Matrix_Algebra_(Hartman)/01%3A_Systems_of_Linear_Equations/1.03%3A_Elementary_Row_Operations_and_Gaussian_Elimination>`_

        As a way to replicate augmented matrices of the form :math:`\mathbf{A}|\mathbf{B}`:
        ``b`` represents the right side of the augmented matrix :math:`\mathbf{B},` while this matrix represents the left side :math:`\mathbf{A}`

        ``b`` will have the same row operations that were applied onto this matrix :math:`\mathbf{A}`, applied on it

        The resulting row echelon form of :math:`\mathbf{A}` must satisfy the following criteria:

            #. The first nonzero entry in each row is a 1 (called a leading 1).
            #. Each leading 1 comes in a column to the right of the leading 1s in rows above it. (this is the **pivot** point)
            #. All rows that are all 0s come at the bottom of the matrix.

        Parameters
        ----------
        b :
            A matrix to mirror the row operations that were performed onto this matrix, by default ``None``

        Returns
        -------
        tuple[:class:`Matrix`, :class:`Matrix` | None]
            A new matrix that is this matrix in row echelon form
            and a **copy** of the matrix ``b`` which mirrored the row operations performed
        """
        copy = self.copy()
        b_copy = b.copy() if b is not None else None

        # "Forward steps" for gaussian elimination to row echelon form
        #
        # start at the first row: `top_row`: first row that hasn't been worked on yet in each iteration
        for top_row in range(copy.rows):
            # This searches for `col` which is the leftmost column that is not all zeroes
            for col in range(copy.cols):
                for row in range(top_row, copy.rows):
                    if copy.__inner[row][col] != 0:
                        break
                else:
                    continue
                break
            else:
                break

            # If the entry in this row: `top_row` and column: `col` is zero,
            # interchange rows with another row below the current row: `row` so that that the entry becomes non-zero.
            # If all entries below are zero, we are done with this column
            if copy.__inner[top_row][col] == 0:
                copy.row_switch(top_row, row)

                if b_copy is not None:
                    b_copy.row_switch(top_row, row)

            # If the current/first entry is not a 1:
            # multiply by `multiplier` which is the reciprocal of `entry` to make `entry` a one if it isn't already a one.
            if (entry := copy.__inner[top_row][col]) != 1:
                multiplier = 1 / entry

                copy.row_mul(top_row, multiplier)
                if b_copy is not None:
                    b_copy.row_mul(top_row, multiplier)

                copy.__inner[top_row][col] = Fraction(1)

            # Repeatedly use row operation 1 (adding) to put zeros under the leading 1
            #
            # Go through the test of the rows
            for row in range(top_row + 1, copy.rows):
                if (entry := copy.__inner[row][col]) != 0:
                    multiplier = -entry

                    # This will send elements in row `row` to `0` if it is under a `1`
                    # as it becomes `entry + (-entry) * a` where `a` is the corresponding element from `top_row`
                    #
                    # if `a = 1` then the above expression evaluates to `0` which is exactly what we need: zeroes under leading ones
                    copy.row_add(row, top_row, multiplier)
                    if b_copy is not None:
                        b_copy.row_add(row, top_row, multiplier)

            # repeat for all new rows and columns until they have all been worked on

        return copy, b_copy

    def reduced_row_echelon_form(self, b: Matrix | None = None) -> tuple[Matrix, Matrix | None]:
        r"""Computes a new matrix that is this matrix :math:`\mathbf{A}`'s **reduced row echelon form**

        Similar to :meth:`row_echelon_form`: ``b`` will mirror the row operations applied onto this matrix.

        It first computes the **row echelon form** :math:`\mathrm{rref}(\mathbf{A})` using :meth:`row_echelon_form`
        that it will further reduce using the backwards steps of Gaussian Elimination

        The resulting reduced row echelon form of :math:`\mathbf{A}` must satisfy the following criteria:

            #. It is in row echelon form
            #. Each column that contains a leading 1 has 0s in all of its other entries

        If :math:`\mathbf{A}` is invertible (non-singular) then :math:`\mathrm{rref}(\mathbf{A})=\mathbf{I}`
        and ``b`` :math:`\mathbf{B}` will represent the corresponding values of the solution to the linear system of equations
        represented by the augmented matrix :math:`\mathbf{A}|\mathbf{B}`.

        Additionally, computing :math:`\mathrm{rref}(\mathbf{A}|\mathbf{I})` where ``b`` is :math:`\mathbf{I}`
        gives you :math:`\mathbf{I}|\mathbf{A}^{-1}` providing that :math:`\mathbf{A}` is invertible.

        Parameters
        ----------
        b :
            A matrix to mirror the row operations that were performed onto this matrix, by default ``None``

        Returns
        -------
        tuple[:class:`Matrix`, :class:`Matrix` | None]
            A new matrix that is this matrix in reduced row echelon form
            and a **copy** of the matrix ``b`` which mirrored the row operations performed
        """
        a, b = self.row_echelon_form(b)

        # "Backward steps" for gaussian elimination to reduced row echelon form
        #
        # Iterate through rows starting at the bottom, working up
        for last_row in range(a.rows - 1, 0, -1):
            for col in range(a.cols):
                # Find the leading ones that were created before
                if a[last_row][col] == 1:
                    # Iterate through all rows above the leading 1
                    for row in range(last_row):
                        # Use row operation 1 (adding) to put zeros above each leading 1
                        if (entry := a[row][col]) != 0:
                            multiplier = -entry

                            # This will send elements in row `row` to `0` if it is above a `1`
                            # as it becomes `entry + (-entry) * a` where `a` is the corresponding element from `last_row`
                            #
                            # if `a = 1` then the above expression evaluates to `0` which is exactly what we need: zeroes above the leading ones
                            a.row_add(row, last_row, multiplier)
                            if b is not None:
                                b.row_add(row, last_row, multiplier)
                    break
        return a, b

    def change_basis(self, new_bases: Matrix) -> Matrix:
        r"""Changes the bases of this linear transformation :math:`\mathbf{M}`,

        into the new bases defined by ``new_bases`` :math:`\mathbf{P}`, by computing: :math:`\mathbf{P}^{-1}\mathbf{M}\mathbf{P}`

        Parameters
        ----------
        new_bases :
            The matrix containing the new basis vectors

        Returns
        -------
        :class:`Matrix`
            The new linear transformation after the change of basis
        """
        return new_bases.inverted() @ self @ new_bases

    def get_submatrix_of(self, rows: set[int], cols: set[int]) -> Matrix:
        """A submatrix is the matrix obtained by deleting the **rows** that that have indices in the set ``rows``
        and **columns** that have indices in the set ``cols``

        Parameters
        ----------
        rows :
            The set of rows to exclude
        cols :
            The set of columns to excludes

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

        :math:`\mathrm{adj}(\mathbf{A})=\mathbf{C}^\intercal` where :math:`\mathbf{C}` is the cofactor matrix

        Returns
        -------
        :class:`Matrix`
            The adjugate matrix of this matrix

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
        where :math:`\mathbf{C}_{ij}` is cofactor of :math:`\mathbf{A}_{ij}`

        The determinant of a matrix represents the **scale factor** of space it is applied on.

        If :math:`\det(\mathbf{A})<0`, that means that a change of orientation has occured

        If :math:`\det(\mathbf{A})=0`, that means that space has been squished down to a lower dimension than what it originally was:
        due to the fact that this transformation's basis vectors are not all **linearly independent** thereby making this matrix **singular**.

        If :math:`\mathbf{A}` is **upper-triangular** or **lower-triangular** then :math:`\det(\mathbf{A})=\mathrm{tr}(\mathbf{A})`.

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
        r"""Computes the trace of this square matrix :math:`\mathbf{A}`

        :math:`\mathrm{tr}(\mathbf{A})=\displaystyle\sum_i{\mathbf{A}_{ii}}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The trace of the square matrix: :math:`\mathrm{tr}(\mathbf{A})`

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

    def rank(self) -> int:
        r"""Computes the rank of this matrix :math:`\mathbf{A}` as a linear transformation

        The rank of this matrix :math:`\rho(\mathbf{A})` represents the **number** of **linearly independent**
        vectors (columns) of this linear transformation. It represents the dimensionality of space after the transformation

        It can be computed as the number of non-zero rows in the **row echelon form** of this matrix

        Returns
        -------
        :class:`int`
            The rank of this matrix: :math:`\rho(\mathbf{A})`
        """
        row_echelon, _ = self.row_echelon_form()
        rank = 0

        for row in row_echelon:
            for element in row:
                if element != 0:
                    rank += 1
                    break
        return rank

    def inverted(self) -> Matrix:
        r"""Computes a new matrix :math:`\mathbf{A}^{-1}` that is this matrix :math:`\mathbf{A}`'s inverse

        :math:`\mathbf{A}^{-1}=\frac{\mathrm{adj}\left(\mathbf{A}\right)}{\det(\mathbf{A})}` if :math:`\det(\mathbf{A})\ne0`
        where :math:`\mathbf{A}^{-1}\mathbf{A}=\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}`

        (:math:`\mathbf{A}^{-1}` is the transformation that undos this transformation :math:`\mathbf{A}`)

        Returns
        -------
        :class:`Matrix`
            The new new inverse matrix: :math:`\mathbf{A}^{-1}`

        Raises
        ------
        :class:`ValueError`
            This matrix singular: :math:`\det(\mathbf{A})=0` (inverse does not exist)
        :class:`AssertionError`
            This matrix is not square
        """
        assert self.is_square(), 'This operation requires the matrix to be square'

        if (det := self.det()) != 0:
            return self.adj() / det
        raise ValueError('This matrix is singular: inverse does not exist')

    def invert(self) -> None:
        r"""Inverts this matrix in place

        :math:`\mathbf{A}\mapsto\mathbf{A}^{-1}` if :math:`\det(\mathbf{A})\ne0`
        where :math:`\mathbf{A}^{-1}\mathbf{A}=\mathbf{A}\mathbf{A}^{-1}=\mathbf{I}`

        Returns
        -------
        :class:`Matrix`
            The inverted matrix: :math:`\mathbf{A}^{-1}`

        Raises
        ------
        :class:`ValueError`
            This matrix singular: :math:`\det(\mathbf{A})=0` (inverse does not exist)
        :class:`AssertionError`
            This matrix is not square
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
            self.__inner[i][j] *= convert(other)
        self.map(_mul)
        return self

    def __truediv__(self, other: Number) -> Self:
        r"""Computes a new matrix that is the scalar division of ``other`` :math:`k` on this matrix :math:`\mathbf{A}`

        Parameters
        ----------
        other :
            The scalar divisor

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
            The scalar divisor

        Returns
        -------
        :class:`Matrix`
            The scalar divison matrix: :math:`\frac{1}{k}\mathbf{A}`
        """
        def _div(i: int, j: int) -> None:
            self.__inner[i][j] /= convert(other)
        self.map(_div)
        return self

    def __floordiv__(self, other: Number) -> Self:
        r"""Computes a new matrix that is the result of scalar **floor** division of ``other`` :math:`k` on this matrix :math:`\mathbf{A}`

        Parameters
        ----------
        other :
            The scalar divisor

        Returns
        -------
        :class:`Matrix`
            The scalar divison matrix: :math:`\lfloor\frac{1}{k}\rfloor\mathbf{A}`
        """
        copy = self.copy()
        def _div(i: int, j: int) -> None:
            copy.__inner[i][j] = Fraction(copy.__inner[i][j] // other)
        copy.map(_div)

        return copy

    def __ifloordiv__(self, other: Number) -> Self:
        r"""Scalar **floor** divides ``other`` :math:`k` on this matrix :math:`\mathbf{A}` (in place)

        Parameters
        ----------
        other :
            The scalar divisor

        Returns
        -------
        :class:`Matrix`
            The scalar divison matrix: :math:`\lfloor\frac{1}{k}\rfloor\mathbf{A}`
        """
        def _div(i: int, j: int) -> None:
            self.__inner[i][j] = Fraction(self.__inner[i][j] // other)
        self.map(_div)
        return self

    @overload
    def __matmul__(self, other: Matrix) -> Matrix:
        r"""Computes a new matrix that is the matrix multiplication between
        this matrix :math:`\mathbf{A}` (size :math:`m\times n`) and ``other`` :math:`\mathbf{B}` (size :math:`p\times q`)

        The operation can only be performed if :math:`n = p` and will yield a matrix of size :math:`m\times q`

        :math:`\mathbf{AB}=\left(\displaystyle\sum_r{\mathbf{A}_{ir}\mathbf{B}_{ij}}\right)_{1\le i,j\lt n}`

        This can also represent a **composition** of 2 linear maps:
        :math:`f\circ g=f(g(\vec{v}))` which says: apply :math:`g` first, then :math:`f`,
        where this matrix represents the map :math:`f` and ``other`` represents the map :math:`g`

        Parameters
        ----------
        other :
            The matrix to perform matrix multiplication with

        Returns
        -------
        :class:`Matrix`
            The matrix that is the result of the matrix multiplication

        Raises
        ------
        :class:`AssertionError`
            Unable to compute matrix multiplication:
            The # of columns in the left matrix does not match the # of rows in the right matrix
        """

    @overload
    def __matmul__(self, other: Vector) -> Vector:
        r"""Applies the linear transformation that is this matrix :math:`\mathbf{A}` on the vector ``other`` :math:`\vec{v}`:

        The operation can only be performed if :math:`n = \ell(\vec{v})` where :math:`\ell(\vec{v})` denotes the length of the vector

        :math:`\mathrm{T_A}(\vec{v})=\mathbf{A}\vec{v}` yielding an output vector that is the transformed vector

        Parameters
        ----------
        other :
            The vector to apply the linear map on

        Returns
        -------
        :class:`Vector`
            The vector that is the result of the linear map :math:`\mathbf{A}``

        Raises
        ------
        :class:`AssertionError`
            Unable to compute matrix multiplication:
            The # of columns in the matrix does not match the length of the vector
        """

    def __matmul__(self, other: Matrix | Vector) -> Matrix | Vector:
        r"""Overloaded method:

        #.
            Computes a new matrix that is the matrix multiplication between
            this matrix :math:`\mathbf{A}` (size :math:`m\times n`) and ``other`` :math:`\mathbf{B}` (size :math:`p\times q`)

            The operation can only be performed if :math:`n = p` and will yield a matrix of size :math:`m\times q`

            :math:`\mathbf{AB}=\left(\displaystyle\sum_r{\mathbf{A}_{ir}\mathbf{B}_{ij}}\right)_{1\le i,j\lt n}`

            This can also represent a **composition** of 2 linear maps:
            :math:`f\circ g=f(g(\vec{v}))` which says: apply :math:`g` first, then :math:`f`,
            where this matrix represents the map :math:`f` and ``other`` represents the map :math:`g`

        #.
            Applies the linear transformation that is this matrix :math:`\mathbf{A}` on the vector ``other`` :math:`\vec{v}`:

            The operation can only be performed if :math:`n = \ell(\vec{v})` where :math:`\ell(\vec{v})` denotes the length of the vector

            :math:`\mathrm{T_A}(\vec{v})=\mathbf{A}\vec{v}` yielding a new output vector that is the transformed vector

        Parameters
        ----------
        other : Matrix | Vector
            The matrix / vector to perform matrix multiplication with

        Returns
        -------
        :class:`Matrix` | :class:`Vector`
            The matrix / vector that is the result of the matrix multiplication

        Raises
        ------
        :class:`AssertionError`
            Unable to compute matrix multiplication:
            The # of columns in the left matrix does not match the # of rows in the right matrix / vector
        """
        if isinstance(other, Matrix):
            assert self.cols == other.rows, 'The # of columns in the left matrix does not match the # of rows in the right matrix'

            other_t = other.copy()
            other_t.transpose()

            product = Matrix.zero(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    product.__inner[i][j] = Vector(self.__inner[i]) * Vector(other_t.__inner[j])
            return product
        else:
            assert self.cols == other.length, 'The # of rows in the matrix must match the length of the vector'

            return Vector([
                Vector(row) * other
                for row in self.__inner
            ])

    def __pow__(self, other: int) -> Matrix:
        r"""Computes repeated matrix multiplication of this matrix :math:`\mathbf{A}` on itself ``other`` number of times

        Parameters
        ----------
        other :
           The exponent: :math:`n`

        Returns
        -------
        :class:`Matrix`
            The new matrix that is this matrix raised to the power of ``other``: :math:`\mathbf{A}^n`

        Raises
        ------
        :class:`AssertionError`
            Unable to compute matrix multiplication:
            The # of columns in the left matrix does not match the # of rows in the right matrix
        """
        power = Matrix.identity(self.rows)

        if other >= 0:
            for _ in range(other):
                power @= self
        else:
            inverse = self.inverted()

            for _ in range(abs(other)):
                power @= inverse
        return power

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
        list[:obj:`~fractions.Fraction`]
            The ``i-th`` row of this vector: :math:`\mathbf{A}_i`
        """
        return self.row_at(i)

    def __contains__(self, target: Number) -> bool:
        """Returns ``True`` if the value ``target`` can be found in this matrix, else ``False``

        Parameters
        ----------
        target :
            The target value to search for

        Returns
        -------
        :class:`bool`
            whether or not the value ``target`` can be found in this matrix
        """
        target = convert(target)
        return any(target in row for row in self.__inner)

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
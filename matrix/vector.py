from __future__ import annotations

from typing import Self, Sequence, Callable, overload
from fractions import Fraction
from math import sin, cos, sqrt

from .utils import Base, convert, Number, NumberF

__all__ = ('Vector',)

class Vector(Base):
    __slots__ = ('_inner',)

    def __init__(self, entries: Sequence[NumberF]) -> None:
        self._inner = [
            convert(entry) for entry in entries
        ]


    @classmethod
    def zero(cls, n: int) -> Self:
        """
        Creates the zero vector of size `n`

        Parameters
        ----------
        n : :class:`int`
            The size of the vector

        Returns
        -------
        Self
            The zero vector
        """
        return cls([0] * n)

    @classmethod
    def from_polar(cls, r: NumberF, theta: NumberF) -> Self:
        """
        Creates a vector of length 2: `[x, y]` based on the provided polar coordinates

        Parameters
        ----------
        r : :class:`NumberF`
            The magnitude of the vector
        theta : :class:`NumberF`
            The angle away from the `x-axis`

        Returns
        -------
        :class:`Self`
            The created instance
        """
        return cls([
            r * cos(theta),
            r * sin(theta),
        ])

    @classmethod
    def from_spherical(cls, rho: NumberF, theta: NumberF, phi: NumberF) -> Self:
        """
        Creates a vector of length 3: `[x, y, z]` based on the provided spherical coordinates

        Parameters
        ----------
        rho : :class:`NumberF`
            The magnitude of the vector
        theta : :class:`NumberF`
            The angle away from the `z-axis`
        phi : :class:`NumberF`
            The angle away from the `x-axis`

        Returns
        -------
        :class:`Self`
            The created instance
        """
        return cls([
            rho * sin(theta) * cos(phi),
            rho * sin(theta) * sin(phi),
            rho * cos(theta),
        ])

    @property
    def length(self) -> int:
        """
        Returns
        -------
        :class:`int`
            The number of elements in this vector
        """
        return len(self._inner)

    @property
    def inner(self) -> list[Fraction]:
        return self._inner

    def magnitude(self) -> float:
        """
        Returns
        -------
        :class:`float`
            The magnitude of the vector
        """
        return sqrt(sum(a ** 2 for a in self._inner))

    def map(self, f: Callable[[int], None]) -> None:
        """
        Maps a function over all of this vector's elements

        Parameters
        ----------
        f : :class:`Callable[[int], None]`
           The function to map over the elements
        """
        for i in range(self.length):
            f(i)

    def copy(self) -> Self:
        """
        Makes a copy of this vector and its elements

        Returns
        -------
        Self
            The copied vector
        """
        return self.__copy__()

    def __deepcopy__(self) -> Self:
        """
        Makes a copy of this vector and its elements

        Returns
        -------
        Self
            The copied vector
        """
        return self.__copy__()

    def __copy__(self) -> Self:
        """
        Makes a copy of this vector and its elements

        Returns
        -------
        Self
            The copied vector
        """
        return self.__class__(list(self._inner))

    def __add__(self, other: Vector) -> Vector:
        """
        Vector addition

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The sum vector

        Raises
        ------
        :class:`ValueError`
            attempted to add vectors of different lengths
        """
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be added')

        return Vector([
            a + b for a, b in zip(self._inner, other._inner)
        ])

    def __iadd__(self, other: Vector) -> Self:
        """
        In-place vector addition

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The sum vector

        Raises
        ------
        :class:`ValueError`
            attempted to add vectors of different lengths
        """
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be added')

        for i in range(self.length):
            self._inner[i] += other._inner[i]
        return self

    def __sub__(self, other: Vector) -> Vector:
        """
        Vector subtraction

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The difference vector

        Raises
        ------
        :class:`ValueError`
            attempted to subtract vectors of different lengths
        """
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be subtracted')

        return Vector([
            a - b for a, b in zip(self._inner, other._inner)
        ])

    def __isub__(self, other: Vector) -> Self:
        """
        In-place vector subtraction

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The difference vector

        Raises
        ------
        :class:`ValueError`
            attempted to subtract vectors of different lengths
        """
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be subtracted')

        for i in range(self.length):
            self._inner[i] -= other._inner[i]
        return self

    @overload
    def __rmul__(self, other: Vector) -> Fraction:
        """
        Scalar dot product

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the dot product with

        Returns
        -------
        :class:`Fraction`
            The scalar dot product
        """

    @overload
    def __rmul__(self, other: Number) -> Self:
        """
        Scalar multiplication

        Parameters
        ----------
        other : :class:`Number`
            The scalar to scale the vector with

        Returns
        -------
        :class:`Vector`
            The scaled vector
        """

    @overload
    def __mul__(self, other: Vector) -> Fraction:
        """
        Scalar dot product

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the dot product with

        Returns
        -------
        :class:`Fraction`
            The scalar dot product
        """

    @overload
    def __mul__(self, other: Number) -> Self:
        """
        Scalar multiplication

        Parameters
        ----------
        other : :class:`Number`
            The scalar to scale the vector with

        Returns
        -------
        :class:`Vector`
            The scaled vector
        """

    @overload
    def __imul__(self, other: Vector) -> Fraction:
        """
        In-place scalar dot product

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the dot product with

        Returns
        -------
        :class:`Fraction`
            The scalar dot product
        """

    @overload
    def __imul__(self, other: Number) -> Self:
        """
        In-place scalar multiplication

        Parameters
        ----------
        other : :class:`Number`
            The scalar to scale the vector with

        Returns
        -------
        :class:`Vector`
            The scaled vector
        """

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        if isinstance(other, Number):
            copy = self.copy()
            def _mul(i: int) -> None:
                copy._inner[i] *= other
            copy.map(_mul)

            return copy
        elif isinstance(other, Vector):
            return self.dot(other)

        raise ValueError('Multiplication can only be performed on scalars and other vectors (dot product)')

    def __imul__(self, other):
        if isinstance(other, Number):
            def _mul(i: int) -> None:
                self._inner[i] *= other
            self.map(_mul)
            return self

            return copy
        elif isinstance(other, Vector):
            return self.dot(other)

        raise ValueError('Multiplication can only be performed on scalars and other vectors (dot product)')

    def __div__(self, other: Number) -> Self:
        """
        Scalar division

        Parameters
        ----------
        other : :class:`Number`
            The scalar to divide with

        Returns
        -------
        :class:`Vector`
            The scaled down vector
        """
        copy = self.copy()
        def _div(i: int) -> None:
            copy._inner[i] /= other
        copy.map(_div)

        return copy

    def __idiv__(self, other: Number) -> Self:
        """
        In-place scalar division

        Parameters
        ----------
        other : :class:`Number`
            The scalar to divide with

        Returns
        -------
        :class:`Vector`
            The scaled down vector
        """
        def _div(i: int) -> None:
            self._inner[i] /= other
        self.map(_div)
        return self

    def dot(self, other: Vector) -> Fraction:
        """
        Scalar dot product

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the dot product with

        Returns
        -------
        :class:`Fraction`
            The scalar dot product
        """
        return sum(
            (a * b for a, b in zip(self._inner, other._inner)),
            start=Fraction(),
        )

    def cross(self, other: Vector) -> Vector:
        """
        Computes the cross product vector between the 2 vectors

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the cross product with

        Returns
        -------
        :class:`Vector`
            The cross product vector

        Raises
        ------
        :class:`TypeError`
            Attempted cross product on a non-vector
        :class:`ValueError`
            Attemped cross product with vectors of dimensionality greater than 3
        """
        if not isinstance(other, Vector):
            raise TypeError('Cross product can only be performed between vectors')

        if self.length < 1 or self.length > 3:
            raise ValueError('Cross product can only be computed in R^3')

        ax, ay, az = self._inner + [0] * (3 - self.length)
        bx, by, bz = other._inner + [0] * (3 - other.length)

        return Vector([
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        ])

    def __matmul__(self, other: Vector) -> Vector:
        """
        Computes the cross product vector between the 2 vectors

        Parameters
        ----------
        other : :class:`Vector`
            The vector to perform the cross product with

        Returns
        -------
        :class:`Vector`
            The cross product vector

        Raises
        ------
        :class:`TypeError`
            Attempted cross product on a non-vector
        :class:`ValueError`
            Attemped cross product with vectors of dimensionality greater than 3
        """
        return self.cross(other)

    def __abs__(self) -> float:
        return self.magnitude()

    def __pos__(self) -> Self:
        return self

    def __neg__(self) -> Self:
        return -1 * self

    def __getitem__(self, i: int) -> Fraction:
        return self._inner[i]

    def display(self) -> str:
        """
        Returns a formatted, displayable string represenation of this vector

        Returns
        -------
        :class:`str`
            The formatted string
        """
        return f"[{', '.join(str(x) for x in self._inner)}]"

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [n={self.length}] inner={self.display()}>'

    def __str__(self) -> str:
        return self.display()

    def __eq__(self, other: Vector) -> bool:
        if not isinstance(other, Vector):
            raise TypeError('Equality comparisons can only be made between vectors')
        return self._inner == other._inner

    def __ne__(self, other: Vector) -> bool:
        return not self != other
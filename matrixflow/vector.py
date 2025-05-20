from __future__ import annotations

__all__ = ('Vector',)

from typing import Any, Self, Sequence, Callable, overload
from fractions import Fraction
from math import sin, cos, acos, atan2, sqrt

from .utils import *

class Vector:
    """An implementation for a mathematical vector

    Parameters
    ----------
    entries : ~typing.Sequence[int | float | ~fractions.Fraction]
        The raw entries to initialize the vector with
    """
    __slots__ = ('__inner',)
    __inner: list[Fraction]

    def __init__(self, entries: Sequence[Number]) -> None:
        self.__inner = [
            convert(entry) for entry in entries
        ]

    @classmethod
    def zero(cls, n: int) -> Self:
        r"""Creates the zero vector of length ``n``: :math:`\vec{0}`

        Parameters
        ----------
        n :
            The size of the vector to create

        Returns
        -------
        :class:`Self`
            The created zero vector: :math:`\vec{0}`
        """
        return cls([0] * n)

    @classmethod
    def from_polar(cls, r: Number, theta: Number) -> Self:
        r"""Creates a vector of length 2: :math:`\begin{pmatrix}x\\y\end{pmatrix}`
        based on the provided polar coordinates: :math:`\left(r,\theta\right)`

        Parameters
        ----------
        r :
            The **radial** distance: the magnitude of the vector
        theta :
            The **polar** angle: the angle away from the ``x-axis``

        Returns
        -------
        :obj:`~typing.Self`
            The created :math:`\mathbb{R}^2` cartesian vector
        """
        return cls([
            r * cos(theta),
            r * sin(theta),
        ])

    @classmethod
    def from_cylindrical(cls, r: Number, theta: Number, z: Number) -> Self:
        r"""Creates a vector of length 3: :math:`\begin{pmatrix}x\\y\\z\end{pmatrix}`
        based on the provided cylindrical coordinates: :math:`\left(r,\theta,z\right)`

        Parameters
        ----------
        r :
            The euclidean distance from the z-axis to the vector
        theta :
            The **polar** angle: the  angle away from the ``x-axis``
        z :
            The **axial coordinate**: the z-coordinate, distance away from the xy-plane

        Returns
        -------
        :obj:`~typing.Self`
            The created :math:`\mathbb{R}^3` cartesian vector
        """
        return cls([
            r * cos(theta),
            r * sin(theta),
            z,
        ])

    @classmethod
    def from_spherical(cls, rho: Number, theta: Number, phi: Number) -> Self:
        r"""Creates a vector of length 3: :math:`\begin{pmatrix}x\\y\\z\end{pmatrix}`
        based on the provided spherical coordinates: :math:`\left(\rho,\theta,\phi\right)`

        Parameters
        ----------
        rho :
            The **radial** distance: the magnitude of the vector
        theta :
            The **polar** angle: the angle away from the ``z-axis``
        phi :
            The **azimuthal** angle: the angle away from the ``x-axis``

        Returns
        -------
        :obj:`~typing.Self`
            The created :math:`\mathbb{R}^3` cartesian vector
        """
        return cls([
            rho * sin(phi) * cos(theta),
            rho * sin(phi) * sin(theta),
            rho * cos(phi),
        ])

    @property
    def length(self) -> int:
        """The number of elements in this vector"""
        return len(self.__inner)

    @property
    def inner(self) -> list[Fraction]:
        """Returns a reference to the internal list representation of this vector"""
        return self.__inner

    def to_polar(self) -> tuple[float, float]:
        r"""For :math:`\mathbb{R}^2` vectors:
        converts this vector from :math:`\begin{pmatrix}x\\y\end{pmatrix}` to polar coordinates :math:`\left(r,\theta\right)`

        Returns
        -------
        tuple[:class:`float`, :class:`float`]
            This vector in polar coordinates
        """
        x, y, *_ = self.__inner
        return (
            self.magnitude(),
            atan2(y, x),
        )

    def to_cylindrical(self) -> tuple[float, float, float]:
        r"""For :math:`\mathbb{R}^3` vectors:
        converts this vector from :math:`\begin{pmatrix}x\\y\\z\end{pmatrix}` to cylindrical coordinates :math:`\left(r,\theta,z\right)`

        Returns
        -------
        tuple[:class:`float`, :class:`float`, :class:`float`]
            This vector in cylindrical coordinates
        """
        x, y, z, *_ = self.__inner
        return (
            sqrt(x ** 2 + y ** 2),
            atan2(y, x),
            float(z),
        )

    def to_spherical(self) -> tuple[float, float, float]:
        r"""For :math:`\mathbb{R}^3` vectors:
        converts this vector from :math:`\begin{pmatrix}x\\y\\z\end{pmatrix}` to spherical coordinates :math:`\left(\rho,\theta,\phi\right)`

        Returns
        -------
        tuple[:class:`float`, :class:`float`, :class:`float`]
            This vector in polar coordinates
        """
        x, y, z, *_ = self.__inner
        return (
            r := self.magnitude(),
            atan2(y, x),
            acos(z / r),
        )

    def is_same_order(self, other: Vector) -> bool:
        """Returns ``True`` if this vector is of the same order as ``other``, else ``False``

        Parameters
        ----------
        other :
           The other vector to compare to

        Returns
        -------
        :class:`bool`
            Whether or not the vectors are of the same order
        """
        return self.length == other.length

    def norm(self, p: int) -> float:
        r"""Computes the ``p-th`` norm of this vector :math:`\vec{a}`

        :math:`|\vec{a}|_p\equiv\left(\displaystyle\sum_i{|\vec{a}_i|^p}\right)^{\frac{1}{p}}`

        Parameters
        ----------
        p :
            The degree of the vector norm

        Returns
        -------
        :class:`float`
            The ``p-th`` norm of this vector
        """
        return sum(abs(a) ** p for a in self.__inner) ** (1 / p)

    def infinity_norm(self) -> Fraction:
        r"""Computes the infinity norm of this vector :math:`\vec{a}`

        This is also simply the magnitude of its largest entry: :math:`|\vec{a}|_\infty\equiv\max_i{|\vec{a}_i|}`

        Returns
        -------
        :obj:`~fractions.Fraction`
            The infinity norm of this vector
        """
        return max(self.__inner)

    def unit(self) -> Vector:
        r"""Returns the unit vector :math:`\hat{a}` that is in this vector :math:`\vec{a}`'s direction

        Returns
        -------
        :class:`Vector`
            The unit vector
        """
        return self / self.magnitude()

    def magnitude(self) -> float:
        r"""Computes the magnitude of this vector :math:`\vec{a}`

        This is also the second norm: :math:`|\vec{a}|_2\equiv\sqrt{\displaystyle\sum_i{{\vec{a}_i}^2}}`

        Returns
        -------
        :class:`float`
            The magnitude of this vector: :math:`\|\vec{a}\|`
        """
        return self.norm(2)

    def project(self, other: Vector) -> Vector:
        r"""The vector projection :math:`\vec{a_1}` of this vector :math:`\vec{a}` onto ``other`` :math:`\vec{b}`

        :math:`\vec{a_1}=\left(\|\vec{a}\|\cos\theta\right)\hat{b}={\frac{\vec{a}\cdot\vec{b}}{\|\vec{b}\|}}\hat{b}`

        The magnitude of the projection vector is the scalar projection, which is also the dot product scaled down by the magnitude of :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to project this vector onto

        Returns
        -------
        :class:`Vector`
            The projection vector: :math:`\vec{a_1}`
        """
        return (self.dot(other) / other.magnitude()) * other.unit()

    def reject(self, other: Vector) -> Vector:
        r"""The vector rejection :math:`\vec{a_2}` of this vector :math:`\vec{a}` onto ``other`` :math:`\vec{b}`

        :math:`\vec{a_2}=\vec{a}-\vec{a_1}` where :math:`\vec{a_1}` is the projection vector

        Parameters
        ----------
        other :
            The vector to reject this vector onto

        Returns
        -------
        :class:`Vector`
            The rejection vector: :math:`\vec{a_2}`
        """
        return self - self.project(other)

    def angle_between(self, other: Vector) -> float:
        r"""Computes the angle between this vector and ``other``: :math:`\theta`

        Parameters
        ----------
        other :
            The other vector to form the angle with

        Returns
        -------
        :class:`float`
            The angle, in radians
        """
        return acos(self.dot(other) / (self.magnitude() * other.magnitude()))

    def area_between(self, other: Vector) -> float:
        r"""Computes the area formed by the triangle bounded by this vector and ``other``

        This is simply half of the determinant of the matrix formed by the 2 vectors,
        which is the area of the parallelogram formed by the 2 vectors,
        which is also the magnitude of the cross product vector: :math:`\vec{a}\times\vec{b}`

        :math:`A=\frac{1}{2}\|\vec{a}\times\vec{b}\|=\frac{1}{2}\|a\|\|b\|\sin\theta=\frac{1}{2}\begin{vmatrix}a_x&b_x\\a_y&b_y\\\vdots&\vdots\end{vmatrix}`

        Parameters
        ----------
        other :
            The other matrix to compute the area between

        Returns
        -------
        :class:`float`
            The area between the 2 vectors
        """
        return self.cross(other).magnitude() / 2

    def dot(self, other: Vector) -> Fraction:
        r"""Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        :math:`\vec{a}\cdot\vec{b}=\|\vec{a}\|\|\vec{b}\|\cos\theta`

        which is also the scalar projection of this vector times the magnitude of :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the dot product with

        Returns
        -------
        :obj:`~fractions.Fraction`
            The scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """
        return sum(
            (a * b for a, b in zip(self.__inner, other.__inner)),
            start=Fraction(),
        )

    def cross(self, other: Vector) -> Vector:
        r"""Computes the orthogonal cross product vector of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the cross product with

        Returns
        -------
        :class:`Vector`
            The orthogonal cross product vector: :math:`\vec{a}\times\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            Attemped cross product with either empty vectors or vectors of dimensionality greater than :math:`\mathbb{R}^3`
        """
        assert 1 <= self.length <= 3, 'Cross product can only be computed in R^3 space'

        ax, ay, az = self.__inner + [0] * (3 - self.length)
        bx, by, bz = other.__inner + [0] * (3 - other.length)

        return Vector([
            ay * bz - az * by,
            az * bx - ax * bz,
            ax * by - ay * bx,
        ])

    def add_entry(self, entry: Number) -> None:
        """Appends an element ``entry`` to the end of this vector

        Parameters
        ----------
        entry :
            The entry to append
        """
        self.__inner.append(convert(entry))

    def map(self, f: Callable[[int], None]) -> None:
        """Maps a function over all of this vector's elements

        Parameters
        ----------
        f :
            The function to apply over the elements
            It will take in the current element ``i``
            and should not return anything as it maps in place
        """
        for i in range(self.length):
            f(i)

    def copy(self) -> Self:
        """Makes a copy of this vector and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied vector
        """
        return self.__copy__()

    def display(self) -> str:
        """Returns a formatted, displayable string representation of this vector

        Returns
        -------
        :class:`str`
            The formatted string
        """
        return f"[{', '.join(str(x) for x in self.__inner)}]"

    def __add__(self, other: Vector) -> Vector:
        r"""Computes a new vector that is the sum of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The sum vector: :math:`\vec{a}+\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            attempted to add vectors of different lengths
        """
        assert self.is_same_order(other), 'This operation requires operand vectors to be of the same order'

        return Vector([
            a + b for a, b in zip(self.__inner, other.__inner)
        ])

    def __iadd__(self, other: Vector) -> Self:
        r"""Adds ``other`` :math:`\vec{b}` onto this vector :math:`\vec{a}` (in place)

        Parameters
        ----------
        other :
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The sum vector: :math:`\vec{a}+\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            attempted to add vectors of different lengths
        """
        assert self.is_same_order(other), 'This operation requires operand vectors to be of the same order'

        for i in range(self.length):
            self.__inner[i] += other.__inner[i]
        return self

    def __sub__(self, other: Vector) -> Vector:
        r"""Computes a new vector that is the difference of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The difference vector: :math:`\vec{a}-\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            attempted to subtract vectors of different lengths
        """
        assert self.is_same_order(other), 'This operation requires operand vectors to be of the same order'

        return Vector([
            a - b for a, b in zip(self.__inner, other.__inner)
        ])

    def __isub__(self, other: Vector) -> Self:
        r"""Subtracts ``other`` :math:`\vec{b}` from this vector :math:`\vec{a}` (in place)

        Parameters
        ----------
        other :
            The vector to perform the addition with

        Returns
        -------
        :class:`Vector`
            The difference vector: :math:`\vec{a}-\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            attempted to subtract vectors of different lengths
        """
        assert self.is_same_order(other), 'This operation requires operand vectors to be of the same order'

        for i in range(self.length):
            self.__inner[i] -= other.__inner[i]
        return self

    @overload
    def __rmul__(self, other: Vector) -> Fraction:
        r"""Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        :math:`\vec{a}\cdot\vec{b}=\|\vec{a}\|\|\vec{b}\|\cos\theta`
        which is also the scalar projection of this vector times the magnitude of :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the dot product with

        Returns
        -------
        :obj:`~fractions.Fraction`
            The scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """

    @overload
    def __rmul__(self, other: Number) -> Self:
        r"""Computes a new vector that is this vector :math:`\vec{a}` scaled up by a factor of ``other`` :math:`k`

        Parameters
        ----------
        other :
            The scalar to scale this vector up by

        Returns
        -------
        :class:`Vector`
            The scaled vector: :math:`k\vec{a}`
        """

    @overload
    def __mul__(self, other: Vector) -> Fraction:
        r"""Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        :math:`\vec{a}\cdot\vec{b}=\|\vec{a}\|\|\vec{b}\|\cos\theta`
        which is also the scalar projection of this vector times the magnitude of :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the dot product with

        Returns
        -------
        :obj:`~fractions.Fraction`
            The scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """

    @overload
    def __mul__(self, other: Number) -> Self:
        r"""Computes a new vector that is this vector :math:`\vec{a}` scaled up by a factor of ``other`` :math:`k`

        Parameters
        ----------
        other :
            The scalar to scale this vector up by

        Returns
        -------
        :class:`Vector`
            The scaled vector: :math:`k\vec{a}`
        """

    @overload
    def __imul__(self, other: Vector) -> Fraction:
        r"""Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

        :math:`\vec{a}\cdot\vec{b}=\|\vec{a}\|\|\vec{b}\|\cos\theta`
        which is also the scalar projection of this vector times the magnitude of :math:`\vec{b}`

        Parameters
        ----------
        other :
            The vector to perform the dot product with

        Returns
        -------
        :obj:`~fractions.Fraction`
            The scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """

    @overload
    def __imul__(self, other: Number) -> Self:
        r"""Scales this vector :math:`\vec{a}` up by a factor of ``other`` :math:`k` (in place)

        Parameters
        ----------
        other :
            The scalar to scale this vector up by

        Returns
        -------
        :class:`Vector`
            The scaled vector: :math:`k\vec{a}`
        """

    def __rmul__(self, other: Number | Vector) -> Self | Fraction:
        r"""Overloaded method:

        see: :meth:`__mul__`

        Parameters
        ----------
        other : int | float | ~fractions.Fraction | Vector
            The right operand in the scalar multiplication | scalar dot product

        Returns
        -------
        :obj:`~typing.Self` | :obj:`~fractions.Fraction`
            The product of the scalar multiplication: :math:`k\vec{a}` | scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """
        return self * other

    def __mul__(self, other: Number | Vector) -> Self | Fraction:
        r"""Overloaded method:

        #.
            Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`

            :math:`\vec{a}\cdot\vec{b}=\|\vec{a}\|\|\vec{b}\|\cos\theta`

            which is also the scalar projection of this vector times the magnitude of :math:`\vec{b}`

        #. Computes a new vector that is this vector :math:`\vec{a}` scaled up by a factor of ``other`` :math:`k`

        Parameters
        ----------
        other : int | float | ~fractions.Fraction | Vector
            The right operand in the scalar multiplication | scalar dot product

        Returns
        -------
        :obj:`~typing.Self` | :obj:`~fractions.Fraction`
            The product of the scalar multiplication: :math:`k\vec{a}` | scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """
        if isinstance(other, Number):
            copy = self.copy()
            def _mul(i: int) -> None:
                copy.__inner[i] *= convert(other)
            copy.map(_mul)

            return copy
        else:
            return self.dot(other)

    def __imul__(self, other: Number | Vector) -> Self | Fraction:
        r"""Overloaded method:

        see: :meth:`__mul__`

        #. Computes the scalar dot product of this vector :math:`\vec{a}` and ``other`` :math:`\vec{b}`:

        #. Scales this vector :math:`\vec{a}` up by a factor of ``other`` :math:`k` (in place)

        Parameters
        ----------
        other : int | float | ~fractions.Fraction | Vector
            The scaled vector | scalar dot product

        Returns
        -------
        :obj:`~typing.Self` | :obj:`~fractions.Fraction`
            The scaled vector: :math:`k\vec{a}` | scalar dot product: :math:`\vec{a}\cdot\vec{b}`
        """
        if isinstance(other, Number):
            def _mul(i: int) -> None:
                self.__inner[i] *= convert(other)
            self.map(_mul)
            return self

            return copy
        else:
            return self.dot(other)

    def __truediv__(self, other: Number) -> Self:
        r"""Computes a new vector that is this vector :math:`\vec{a}` scaled down by a factor of ``other`` :math:`k`

        Parameters
        ----------
        other :
            The scalar to scale this vector down by

        Returns
        -------
        :class:`Vector`
            The scaled down vector: :math:`\frac{1}{k}\vec{a}`
        """
        copy = self.copy()
        def _div(i: int) -> None:
            copy.__inner[i] /= convert(other)
        copy.map(_div)

        return copy

    def __itruediv__(self, other: Number) -> Self:
        r"""Scales this vector :math:`\vec{a}` down by a factor of ``other`` :math:`k` (in place)

        Parameters
        ----------
        other :
            The scalar to scale this vector down by

        Returns
        -------
        :class:`Vector`
            The scaled down vector: :math:`\frac{1}{k}\vec{a}`
        """
        def _div(i: int) -> None:
            self.__inner[i] /= convert(other)
        self.map(_div)
        return self

    def __floordiv__(self, other: Number) -> Self:
        r"""Computes a new vector that is this vector :math:`\vec{a}` scaled down by a factor of ``other`` :math:`k` and then **floored**

        Parameters
        ----------
        other :
            The scalar to scale this vector down by

        Returns
        -------
        :class:`Vector`
            The scaled down vector: :math:`\lfloor\frac{1}{k}\rfloor\vec{a}`
        """
        copy = self.copy()
        def _div(i: int) -> None:
            copy.__inner[i] = Fraction(copy.__inner[i] // other)
        copy.map(_div)

        return copy

    def __ifloordiv__(self, other: Number) -> Self:
        r"""Scales this vector :math:`\vec{a}` down by a factor of ``other`` :math:`k` and then **floored** (in place)

        Parameters
        ----------
        other :
            The scalar to scale this vector down by

        Returns
        -------
        :class:`Vector`
            The scaled down vector: :math:`\lfloor\frac{1}{k}\rfloor\vec{a}`
        """
        def _div(i: int) -> None:
            self.__inner[i] = Fraction(self.__inner[i] // other)
        self.map(_div)
        return self

    def __matmul__(self, other: Vector) -> Vector:
        r"""Computes the orthogonal cross product vector of this vector :math:`\vec{a}`and ``other`` :math:`\vec{b}`

        see: :meth:`cross`

        Parameters
        ----------
        other :
            The vector to perform the cross product with

        Returns
        -------
        :class:`Vector`
            The orthogonal cross product vector: :math:`\vec{a}\times\vec{b}`

        Raises
        ------
        :class:`AssertionError`
            Attemped cross product with either empty vectors or vectors of dimensionality greater than :math:`\mathbb{R}^3`
        """
        return self.cross(other)

    def __pos__(self) -> Self:
        """Unary plus: does nothing as it performs a scalar multiplication of all the elements by :math:`+1`

        Returns
        -------
        :obj:`~typing.Self`
            Returns itself
        """
        return self

    def __neg__(self) -> Self:
        r"""Negates this vector :math:`\vec{a}` (switches its direction)

        Returns
        -------
        :obj:`~typing.Self`
            Returns the negated vector: :math:`{-\vec{a}}`
        """
        return -1 * self

    def __len__(self) -> int:
        """Returns the length of the vector: ``self.length``

        Returns
        -------
        :class:`int`
            The length of the vector
        """
        return self.length

    def __abs__(self) -> float:
        """Computes the magnitude of this vector

        See: :meth:`magnitude`

        Returns
        -------
        :class:`float`
            The magnitude of this vector
        """
        return self.magnitude()

    def __getitem__(self, i: int) -> Fraction:
        r"""Gets the element of this vector :math:`\vec{a}` at ``i``:

        Parameters
        ----------
        i :
            The index of the element in this vector

        Returns
        -------
        :obj:`~fractions.Fraction`
            The element of the vector at ``i``: :math:`\vec{a}_i`
        """
        return self.__inner[i]

    def __contains__(self, target: Number) -> bool:
        """Returns ``True`` if the value ``target`` can be found in this vector, else ``False``

        Parameters
        ----------
        target :
            The target value to search for

        Returns
        -------
        :class:`bool`
            whether or not the value ``target`` can be found in this vector
        """
        return convert(target) in self.__inner

    def __copy__(self) -> Self:
        """Creates a copy of this vector and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied vector
        """
        return self.__class__(list(self.__inner))

    def __deepcopy__(self) -> Self:
        """Creates a copy of this vector and its elements

        Returns
        -------
        :obj:`~typing.Self`
            The copied vector
        """
        return self.__copy__()

    def __repr__(self) -> str:
        """Return ``repr(self)``"""
        return f'<{self.__class__.__name__} [n={self.length}] inner={self.display()}>'

    def __str__(self) -> str:
        """Return ``str(self)``"""
        return self.display()

    def __eq__(self, other: Any) -> bool:
        """Return ``self == other``"""
        return isinstance(other, Vector) and self.__inner == other.__inner

    def __ne__(self, other: Any) -> bool:
        """Return ``self != other``"""
        return not self != other
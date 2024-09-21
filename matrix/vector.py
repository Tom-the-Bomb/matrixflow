from __future__ import annotations

from typing import Self

__all__ = ('Vector',)

class Vector:
    __slots__ = ('_inner',)

    def __init__(self, entries: list[float]) -> None:
        self._inner = entries

    @property
    def length(self) -> int:
        return len(self._inner)

    def __add__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be added')

        return Vector([
            a + b for a, b in zip(self._inner, other._inner)
        ])

    def __iadd__(self, other: Vector) -> Self:
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be added')

        for i in range(self.length):
            self._inner[i] += other._inner[i]
        return self

    def __sub__(self, other: Vector) -> Vector:
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be subtracted')

        return Vector([
            a - b for a, b in zip(self._inner, other._inner)
        ])

    def __isub__(self, other: Vector) -> Self:
        if not isinstance(other, Vector) or self.length != other.length:
            raise ValueError('Only same length vectors can be subtracted')

        for i in range(self.length):
            self._inner[i] -= other._inner[i]
        return self

    def dot(self, other: Vector) -> float:
        return self * other

    def __mul__(self, other: Vector) -> float:
        return sum(a * b for a, b in zip(self._inner, other._inner))

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} [n={self.length}] inner={self._inner}>'

    def __str__(self) -> str:
        return str(self._inner)

    def __getitem__(self, i: int) -> float:
        return self._inner[i]
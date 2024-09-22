from abc import ABC, abstractmethod

from typing import Self, TypeAlias
from fractions import Fraction

__all__ = (
    'convert',
    'Base',
    'Number',
    'NumberF',
)

Number: TypeAlias = int | Fraction
NumberF: TypeAlias = Number | float

def convert(x: NumberF) -> Fraction:
    return x if isinstance(x, Fraction) else Fraction(str(x))

class Base(ABC):
    __slots__ = ('_inner')

    @abstractmethod
    def display(self) -> str:
        ...

    @abstractmethod
    def map(self, f) -> None:
        ...

    @abstractmethod
    def copy(self) -> Self:
        ...
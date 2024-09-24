
__all__ = (
    'Number',
    'NumberF',
    'convert',
)

from typing import TypeAlias
from fractions import Fraction

Number: TypeAlias = int | Fraction
"""Test1"""
NumberF: TypeAlias = Number | float
"""Test2"""

def convert(x: NumberF) -> Fraction:
    """
    Converts input numbers to ``Fraction`` if needed

    Parameters
    ----------
    x : :class:`int` | :class:`float` | :obj:`~fractions.Fraction`
        The number to convert

    Returns
    -------
    :class:`Fraction`
        The converted fraction
    """
    return x if isinstance(x, Fraction) else Fraction(str(x))

__all__ = (
    'Number',
    'convert',
)

from typing import TypeAlias
from fractions import Fraction

Number: TypeAlias = int | float | Fraction

def convert(x: Number) -> Fraction:
    """Converts input numbers to :obj`~fractions.Fraction` if needed

    Parameters
    ----------
    x :
        The number to process and convert

    Returns
    -------
    :obj`~fractions.Fraction`
        The converted fraction
    """
    return x if isinstance(x, Fraction) else Fraction(str(x))
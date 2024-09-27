"""A library providing implementations for basic algebraic ``matrix`` and ``vector`` operations"""

from .matrix import *
from .vector import *

__all__ = (
    'solve_linear_system',
    'Matrix',
    'Vector',
)
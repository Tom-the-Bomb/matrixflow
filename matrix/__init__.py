"""A library providing implementations for basic algebraic ``matrix`` and ``vector`` operations"""

from .matrix import *
from .vector import *
from .errors import *

__all__ = (
    'Matrix',
    'Vector',
)
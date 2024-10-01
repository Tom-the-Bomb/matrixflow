"""A rich library with implementations for mathematical `matrices` and `vectors` and their operations"""

__version__ = '0.1.2'
__author__  = 'Tom the Bomb'
__license__ = 'MIT'
__copyright__ = 'Copyright 2024-present Tom the Bomb'

__all__ = (
    'solve_linear_system',
    'Matrix',
    'Vector',
)

from .matrix import *
from .vector import *
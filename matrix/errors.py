
__all__ = (
    'BaseError',
    'SingularMatrix',
    'CannotMatMul',
    'OperationTypeError',
    'CannotGroupEvenly',
)

from typing import Any, Callable

class BaseError(Exception):
    """Base error class"""

class SingularMatrix(ZeroDivisionError, BaseError):
    """Raised when an inverse operation is attempted on a matrix with a determinant of 0 (a singular matrix)"""

    def __init__(self) -> None:
        super().__init__(
            'Inverse does not exist: determinant is 0'
        )

class CannotMatMul(ValueError, BaseError):
    """Raised when the multiplication ``AB`` is performed but (# cols in A) != (# rows in B)"""

    def __init__(self) -> None:
        super().__init__(
            'Cannot multiply matrices (AB): (# cols in A) != (# rows in B)'
        )

class OperationTypeError(TypeError, BaseError):
    """Raised when an operation is performed between invalid types"""

    def __init__(
        self,
        f: Callable[..., Any],
        arg: Any,
        param: str,
        expected: type,
    ) -> None:
        super().__init__(
            f'Operation {f.__name__}: '
            f'expected argument of type ``{expected.__name__}`` '
            f'for parameter ``{param}``, got type ``{arg.__class__.__name__}`` instead'
        )

class CannotGroupEvenly(IndexError, BaseError):
    """Raised when attempting to create a matrix from a flat sequence,

    but the sequence cannot be grouped evenly into the provided # of columns
    """

    def __init__(self, n: int, cols: int) -> None:
        super().__init__(
            f'Provided entries of size `{n}` cannot be evenly split into rows of size `{cols}`'
        )
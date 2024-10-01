Examples
========

#.
    **System of linear equations**

    .. code-block:: python

        from matrixflow import solve_linear_system

        print(
            solve_linear_system(
                [
                    [2, 3],
                    [5, -6],
                ],
                [4, -4],
            ) # Solves linear system using matrix inverse operations
        )

        # Alternatively, this solves it using gaussian elimination
        print(
            Matrix([
                [2, 3],
                [5, -6],
            ])
            .reduced_row_echelon_form(
                Matrix.from_columns([[4, -4]])
            )[1]
            .col_at(0)
        ) # this prints the same thing as the above print statement

    This is mathematically equivalent to:

    :math:`\begin{cases}2x+3y=4\\5x-6y=-4\end{cases}`

    :math:`\therefore x=\frac{4}{9}, y=\frac{28}{27}`

#.
    **Linear transformations**

    .. code-block:: python

        from matrixflow import Matrix, Vector

        v = Vector([1, 2])
        shear = Matrix.shear_2d(2)
        reflection = Matrix.reflect_x()

        # shear -> reflect across x-axis
        print(reflection @ shear @ v)

        # Rank: shear does not reduce space to a lower dimension: remains 2D
        print(shear.rank()) # 2

        # this non-square matrix linear transformation increases the dimension of space from R^2 -> R^3
        print(
            Matrix([
                [1, 2],
                [4, 5],
                [7, 8],
            ]) @ v
        )
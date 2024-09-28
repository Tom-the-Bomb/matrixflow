Examples
========

#.
    **Linear equation solver:**

    .. code-block:: python

        from matrixflow import solve_linear_system

        print(
            solve_linear_system(
                [
                    [2, 3],
                    [5, -6]
                ],
                [4, -4],
            )
        )

    This is equivalent to:

    :math:`\begin{cases}2x+3y=4\\5x-6y=-4\end{cases}`

    :math:`\therefore x=\frac{4}{9}, y=\frac{28}{27}`

#.
    **Composition of Linear Maps**

    .. code-block:: python

        from matrixflow import Matrix, Vector

        v = Vector([1, 2])
        shear = Matrix.shear_2d(2)
        reflection = Matrix.reflect_x()

        # shear -> reflect across x-axis
        print(reflection @ shear @ v)
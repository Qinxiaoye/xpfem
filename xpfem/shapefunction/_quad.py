import numpy as np

from ._base import Element




class Quad4(Element):
    r"""A 2D quadrilateral element formulation with linear shape functions.

    Notes
    -----
    The quadrilateral element is defined by four points (0-3). 

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.
    4-----3
    |     |
    |     |
    1-----2
    .. math::
    """

    def __init__(self):
        super().__init__(shape=(4, 2))
        self.points = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        return (
            np.array(
                [
                    (1 - r) * (1 - s),
                    (1 + r) * (1 - s),
                    (1 + r) * (1 + s),
                    (1 - r) * (1 + s),
                ]
            )
            * 0.25
        )

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."

        r, s = rs
        return (
            np.array(
                [
                    [-(1 - s), -(1 - r)],
                    [(1 - s), -(1 + r)],
                    [(1 + s), (1 + r)],
                    [-(1 + s), (1 - r)],
                ]
            )
            * 0.25
        )

class Quad8(Element):
    r"""A 2D quadrilateral element formulation with quadratic (serendipity) shape
    functions.

    Notes
    -----
    The quadratic (serendipity) quadrilateral element is defined by eight points (0-7).

    The shape functions :math:`\boldsymbol{h}` are given in terms of the coordinates
    :math:`(r,s)`.
    4---7---3
    |       |
    8       6
    |       |
    1---5---2

    """

    def __init__(self):
        super().__init__(shape=(8, 2))
        self.points = np.array(
            [
                [-1, -1],
                [1, -1],
                [1, 1],
                [-1, 1],
                [0, -1],
                [1, 0],
                [0, 1],
                [-1, 0],
            ],
            dtype=float,
        )
        self.cells = np.arange(len(self.points)).reshape(1, -1)
        self.cell_type = "quad8"

    def function(self, rs):
        "Return the shape functions at given coordinates (r, s)."
        r, s = rs
        ra, sa = self.points.T

        h = (1 + ra * r) * (1 + sa * s) * (ra * r + sa * s - 1) / 4
        h[ra == 0] = (1 - r**2) * (1 + sa[ra == 0] * s) / 2
        h[sa == 0] = (1 + ra[sa == 0] * r) * (1 - s**2) / 2

        return h

    def gradient(self, rs):
        "Return the gradient of shape functions at given coordinates (r, s)."

        r, s = rs
        ra, sa = self.points.T

        dhdr = (
            ra * (1 + sa * s) * (ra * r + sa * s - 1) / 4
            + (1 + ra * r) * (1 + sa * s) * ra / 4
        )

        dhdr[ra == 0] = -2 * r * (1 + sa[ra == 0] * s) / 2
        dhdr[sa == 0] = ra[sa == 0] * (1 - s**2) / 2

        dhds = (1 + ra * r) * sa * (ra * r + sa * s - 1) / 4 + (1 + ra * r) * (
            1 + sa * s
        ) * sa / 4

        dhds[ra == 0] = (1 - r**2) * sa[ra == 0] / 2
        dhds[sa == 0] = (1 + ra[sa == 0] * r) * -2 * s / 2

        return np.vstack([dhdr, dhds]).T
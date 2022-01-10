# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.


from __future__ import annotations

from typing import Union, List, Tuple, Optional

import fenics
import ufl

from .._exceptions import InputError
from .._loggers import warning


def summation(
    x: List[Union[ufl.core.expr.Expr, int, float]]
) -> Union[ufl.core.expr.Expr, int, float]:
    """Sums elements of a list in a UFL friendly fashion.

    This can be used to sum, e.g., UFL forms, or UFL expressions
    that can be used in UFL forms.

    Parameters
    ----------
    x : list[ufl.core.expr.Expr or int or float]
        The list of entries that shall be summed.

    Returns
    -------
    ufl.core.expr.Expr or int or float
        Sum of input (same type as entries of input).

    See Also
    --------
    multiplication : Multiplies the elements of a list.

    Notes
    -----
    For "usual" summation of integers or floats, the built-in sum function
    of python or the numpy variant are recommended. Still, they are
    incompatible with FEniCS objects, so this function should be used for
    the latter.

    Examples
    --------
    The command ::

        a = cashocs.summation([u.dx(i)*v.dx(i)*dx for i in mesh.geometric_dimension()])

    is equivalent to ::

        a = u.dx(0)*v.dx(0)*dx + u.dx(1)*v.dx(1)*dx

    (for a 2D mesh).
    """

    if len(x) == 0:
        y = fenics.Constant(0.0)
        warning("Empty list handed to summation, returning 0.")
    else:
        y = x[0]

        for item in x[1:]:
            y += item

    return y


def multiplication(
    x: List[Union[ufl.core.expr.Expr, int, float]]
) -> Union[ufl.core.expr.Expr, int, float]:
    """Multiplies the elements of a list in a UFL friendly fashion.

    Used to build the product of certain UFL expressions to construct
    a UFL form.

    Parameters
    ----------
    x : list[ufl.core.expr.Expr or int or float]
        The list whose entries shall be multiplied.

    Returns
    -------
    ufl.core.expr.Expr or int or float
        The result of the multiplication.

    See Also
    --------
    summation : Sums elements of a list.

    Examples
    --------
    The command ::

        a = cashocs.multiplication([u.dx(i) for i in range(mesh.geometric_dimension())])

    is equivalent to ::

        a = u.dx(0) * u.dx(1)

    (for a 2D mesh).
    """

    if len(x) == 0:
        y = fenics.Constant(1.0)
        warning("Empty list handed to multiplication, returning 1.")
    else:
        y = x[0]

        for item in x[1:]:
            y *= item

    return y


def _max(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the maximum of ``a`` and ``b``

    Parameters
    ----------
    a : float or fenics.Function
        The first parameter
    b : float or fenics.Function
        The second parameter

    Returns
    -------
    ufl.core.expr.Expr
        The maximum of ``a`` and ``b``
    """
    return (a + b + abs(a - b)) / fenics.Constant(2.0)


def _min(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the minimum of ``a`` and ``b``

    Parameters
    ----------
    a : float or fenics.Function
        The first parameter
    b : float or fenics.Function
        The second parameter

    Returns
    -------
    ufl.core.expr.Expr
        The minimum of ``a`` and ``b``
    """

    return (a + b - abs(a - b)) / fenics.Constant(2.0)


def moreau_yosida_regularization(
    term: ufl.core.expr.Expr,
    gamma: float,
    measure: fenics.Measure,
    lower_threshold: Optional[Union[float, fenics.Function]] = None,
    upper_treshold: Optional[Union[float, fenics.Function]] = None,
    shift_lower: Optional[Union[float, fenics.Function]] = None,
    shift_upper: Optional[Union[float, fenics.Function]] = None,
) -> ufl.Form:
    r"""Implements a Moreau-Yosida regularization of an inequality constraint

    The general form of the inequality is of the form ::

        lower_threshold <= term <= upper_threshold

    which is defined over the region specified in ``measure``.

    In case ``lower_threshold`` or ``upper_threshold`` are ``None``, they are set to
    :math:`-\infty` and :math:`\infty`, respectively.

    Parameters
    ----------
    term : ufl.core.expr.Expr
        The term inside the inequality constraint
    gamma : float
        The weighting factor of the regularization
    measure : fenics.Measure
        The measure over which the inequality constraint is defined
    lower_threshold : float or fenics.Function or None, optional
        The lower threshold for the inequality constraint. In case this is ``None``, the
        lower bound is set to :math:`-\infty`. The default is ``None``
    upper_treshold : float or fenics.Function or None, optional
        The upper threshold for the inequality constraint. In case this is ``None``, the
        upper bound is set to :math:`\infty`. The default is ``None``
    shift_lower : float or fenics.Function or None:
        A shift function for the lower bound of the Moreau-Yosida regularization.
        Should be non-positive. In case this is ``None``, it is set to 0.
        Default is ``None``.
    shift_upper : float or fenics.Function or None:
        A shift function for the upper bound of the Moreau-Yosida regularization.
        Should be non-negative. In case this is ``None``, it is set to 0.
        Default is ``None``.

    Returns
    -------
    ufl.form.Form
        The ufl form of the Moreau-Yosida regularization, to be used in the cost functional.
    """

    if lower_threshold is None and upper_treshold is None:
        raise InputError(
            "cashocs.utils.moreau_yosida_regularization",
            "upper_threshold, lower_threshold",
            "At least one of the threshold parameters has to be defined.",
        )

    if shift_lower is None:
        shift_lower = fenics.Constant(0.0)
    if shift_upper is None:
        shift_upper = fenics.Constant(0.0)

    if lower_threshold is not None:
        reg_lower = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                _min(
                    shift_lower + fenics.Constant(gamma) * (term - lower_threshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )
    if upper_treshold is not None:
        reg_upper = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                _max(
                    shift_upper + fenics.Constant(gamma) * (term - upper_treshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )

    if upper_treshold is not None and lower_threshold is not None:
        return reg_lower + reg_upper
    elif upper_treshold is None and lower_threshold is not None:
        return reg_lower
    elif upper_treshold is not None and lower_threshold is None:
        return reg_upper


def create_dirichlet_bcs(
    function_space: fenics.FunctionSpace,
    value: Union[
        fenics.Constant, fenics.Expression, fenics.Function, float, Tuple[float]
    ],
    boundaries: fenics.MeshFunction,
    idcs: Union[List[Union[int, str]], int, str],
    **kwargs,
) -> List[fenics.DirichletBC]:
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Parameters
    ----------
    function_space : fenics.FunctionSpace
        The function space onto which the BCs should be imposed on.
    value : fenics.Constant or fenics.Expression or fenics.Function or float or tuple(float)
        The value of the boundary condition. Has to be compatible with the
        function_space, so that it could also be used as
        ``fenics.DirichletBC(function_space, value, ...)``.
    boundaries : fenics.MeshFunction
        The :py:class:`fenics.MeshFunction` object representing the boundaries.
    idcs : list[int or str] or int or str
        A list of indices / boundary markers that determine the boundaries
        onto which the Dirichlet boundary conditions should be applied to.
        Can also be a single entry for a single boundary. If your mesh file
        is named, then you can also use the names of the boundaries to define the
        boundary conditions.
    **kwargs
        Keyword arguments for fenics.DirichletBC

    Returns
    -------
    list[fenics.DirichletBC]
        A list of DirichletBC objects that represent the boundary conditions.

    Examples
    --------
    Generate homogeneous Dirichlet boundary conditions for all 4 sides of
    the unit square ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1,2,3,4])
    """

    mesh = function_space.mesh()

    if not isinstance(idcs, list):
        idcs = [idcs]

    bcs_list = []
    for entry in idcs:
        if isinstance(entry, int):
            bcs_list.append(
                fenics.DirichletBC(function_space, value, boundaries, entry, **kwargs)
            )
        elif isinstance(entry, str):
            physical_groups = mesh._physical_groups
            if entry in physical_groups["ds"].keys():
                bcs_list.append(
                    fenics.DirichletBC(
                        function_space,
                        value,
                        boundaries,
                        physical_groups["ds"][entry],
                        **kwargs,
                    )
                )
            else:
                raise InputError(
                    "cashocs.create_dirichlet_bcs",
                    "idcs",
                    "The string you have supplied is not associated with a boundary.",
                )

    return bcs_list


# deprecated
def create_bcs_list(
    function_space: fenics.FunctionSpace,
    value: Union[
        fenics.Constant, fenics.Expression, fenics.Function, float, Tuple[float]
    ],
    boundaries: fenics.MeshFunction,
    idcs: Union[List[Union[int, str]], int, str],
    **kwargs,
) -> List[fenics.DirichletBC]:  # pragma: no cover
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Parameters
    ----------
    function_space : fenics.FunctionSpace
        The function space onto which the BCs should be imposed on.
    value : fenics.Constant or fenics.Expression or fenics.Function or float or tuple(float)
        The value of the boundary condition. Has to be compatible with the
        function_space, so that it could also be used as
        ``fenics.DirichletBC(function_space, value, ...)``.
    boundaries : fenics.MeshFunction
        The :py:class:`fenics.MeshFunction` object representing the boundaries.
    idcs : list[int] or int
        A list of indices / boundary markers that determine the boundaries
        onto which the Dirichlet boundary conditions should be applied to.
        Can also be a single integer for a single boundary.
    **kwargs
        Keyword arguments for fenics.DirichletBC

    Returns
    -------
    list[fenics.DirichletBC]
            A list of DirichletBC objects that represent the boundary conditions.

    Examples
    --------
    Generate homogeneous Dirichlet boundary conditions for all 4 sides of the
    unit square ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1,2,3,4])

    .. deprecated:: 1.5.0
        This is replaced by cashocs.create_dirichlet_bcs and will be removed in the future.
    """

    warning(
        "DEPRECATION WARNING: cashocs.create_bcs_list is replaced by cashocs.create_dirichlet_bcs and will be removed in the future."
    )

    return create_dirichlet_bcs(function_space, value, boundaries, idcs, **kwargs)

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

"""Module for utilities for UFL forms."""

from __future__ import annotations

from typing import Any, List, Optional, Tuple, TypeVar, Union

import fenics
import ufl

from cashocs import _exceptions
from cashocs import _loggers

T = TypeVar("T")


def summation(x: List[T]) -> Union[T, fenics.Constant]:
    """Sums elements of a list in a UFL friendly fashion.

    This can be used to sum, e.g., UFL forms, or UFL expressions that can be used in UFL
    forms.

    Args:
        x: The list of entries that shall be summed.

    Returns:
        Sum of input (same type as entries of input).

    Notes:
        For "usual" summation of integers or floats, the built-in sum function
        of python or the numpy variant are recommended. Still, they are
        incompatible with FEniCS objects, so this function should be used for
        the latter.

    """
    if len(x) == 0:
        y = fenics.Constant(0.0)
        _loggers.warning("Empty list handed to summation, returning 0.")
    else:
        y = x[0]

        for item in x[1:]:
            y += item

    return y


def multiplication(x: List[T]) -> Union[T, fenics.Constant]:
    """Multiplies the elements of a list in a UFL friendly fashion.

    Used to build the product of certain UFL expressions to construct a UFL form.

    Args:
        x: The list whose entries shall be multiplied.

    Returns:
        The result of the multiplication.

    """
    if len(x) == 0:
        y = fenics.Constant(1.0)
        _loggers.warning("Empty list handed to multiplication, returning 1.")
    else:
        y = x[0]

        for item in x[1:]:
            y *= item

    return y


def max_(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the maximum of a and b.

    Args:
        a: The first parameter.
        b: The second parameter.

    Returns:
        The maximum of a and b.

    """
    return (a + b + abs(a - b)) / fenics.Constant(2.0)


def min_(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the minimum of a and b.

    Args:
        a: The first parameter.
        b: The second parameter.

    Returns:
        The minimum of a and b.

    """
    return (a + b - abs(a - b)) / fenics.Constant(2.0)


def moreau_yosida_regularization(
    term: ufl.core.expr.Expr,
    gamma: float,
    measure: fenics.Measure,
    lower_threshold: Optional[Union[float, fenics.Function]] = None,
    upper_threshold: Optional[Union[float, fenics.Function]] = None,
    shift_lower: Optional[Union[float, fenics.Function]] = None,
    shift_upper: Optional[Union[float, fenics.Function]] = None,
) -> ufl.Form:
    r"""Implements a Moreau-Yosida regularization of an inequality constraint.

    The general form of the inequality is of the form ::

        lower_threshold <= term <= upper_threshold

    which is defined over the region specified in ``measure``.

    In case ``lower_threshold`` or ``upper_threshold`` are ``None``, they are set to
    :math:`-\infty` and :math:`\infty`, respectively.

    Args:
        term: The term inside the inequality constraint.
        gamma: The weighting factor of the regularization.
        measure: The measure over which the inequality constraint is defined.
        lower_threshold: The lower threshold for the inequality constraint. In case this
            is ``None``, the lower bound is set to :math:`-\infty`. The default is
            ``None``
        upper_threshold: The upper threshold for the inequality constraint. In case this
            is ``None``, the upper bound is set to :math:`\infty`. The default is
            ``None``
        shift_lower: A shift function for the lower bound of the Moreau-Yosida
            regularization. Should be non-positive. In case this is ``None``, it is set
            to 0. Default is ``None``.
        shift_upper: A shift function for the upper bound of the Moreau-Yosida
            regularization. Should be non-negative. In case this is ``None``, it is set
            to 0. Default is ``None``.

    Returns:
        The ufl form of the Moreau-Yosida regularization, to be used in the cost
        functional.

    """
    reg_lower = None
    reg_upper = None

    if lower_threshold is None and upper_threshold is None:
        raise _exceptions.InputError(
            "cashocs._utils.moreau_yosida_regularization",
            "upper_threshold, lower_threshold",
            "At least one of the threshold parameters has to be defined.",
        )

    if shift_lower is None:
        shift_lower = fenics.Constant(0.0)
    if shift_upper is None:
        shift_upper = fenics.Constant(0.0)

    reg = []

    if lower_threshold is not None:
        reg_lower = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                min_(
                    shift_lower + fenics.Constant(gamma) * (term - lower_threshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )
        reg.append(reg_lower)
    if upper_threshold is not None:
        reg_upper = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                max_(
                    shift_upper + fenics.Constant(gamma) * (term - upper_threshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )
        reg.append(reg_upper)

    return summation(reg)


def create_dirichlet_bcs(
    function_space: fenics.FunctionSpace,
    value: Union[
        fenics.Constant, fenics.Expression, fenics.Function, float, Tuple[float]
    ],
    boundaries: fenics.MeshFunction,
    idcs: Union[List[Union[int, str]], int, str],
    **kwargs: Any,
) -> List[fenics.DirichletBC]:
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Args:
        function_space: The function space onto which the BCs should be imposed on.
        value: The value of the boundary condition. Has to be compatible with the
            function_space, so that it could also be used as
            ``fenics.DirichletBC(function_space, value, ...)``.
        boundaries: The :py:class:`fenics.MeshFunction` object representing the
            boundaries.
        idcs: A list of indices / boundary markers that determine the boundaries
            onto which the Dirichlet boundary conditions should be applied to.
            Can also be a single entry for a single boundary. If your mesh file
            is named, then you can also use the names of the boundaries to define the
            boundary conditions.
        **kwargs: Keyword arguments for fenics.DirichletBC

    Returns:
        A list of DirichletBC objects that represent the boundary conditions.

    Examples:
        Generate homogeneous Dirichlet boundary conditions for all 4 sides of
        the unit square ::

            import fenics
            import cashocs

            mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
            V = fenics.FunctionSpace(mesh, 'CG', 1)
            bcs = cashocs.create_dirichlet_bcs(V, fenics.Constant(0), boundaries,
                [1,2,3,4])

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
            physical_groups = mesh.physical_groups
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
                raise _exceptions.InputError(
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
    **kwargs: Any,
) -> List[fenics.DirichletBC]:  # pragma: no cover
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Args:
        function_space: The function space onto which the BCs should be imposed on.
        value: The value of the boundary condition. Has to be compatible with the
            function_space, so that it could also be used as
            ``fenics.DirichletBC(function_space, value, ...)``.
        boundaries: The :py:class:`fenics.MeshFunction` object representing the
            boundaries.
        idcs: A list of indices / boundary markers that determine the boundaries
            onto which the Dirichlet boundary conditions should be applied to.
            Can also be a single integer for a single boundary.
        **kwargs: Keyword arguments for fenics.DirichletBC

    Returns:
        A list of DirichletBC objects that represent the boundary conditions.

    .. deprecated:: 1.5.0
        This is replaced by cashocs.create_dirichlet_bcs and will be removed in the
        future.

    """
    _loggers.warning(
        "DEPRECATION WARNING: cashocs.create_bcs_list is replaced by "
        "cashocs.create_dirichlet_bcs and will be removed in the future."
    )

    return create_dirichlet_bcs(function_space, value, boundaries, idcs, **kwargs)


def bilinear_boundary_form_modification(forms: List[ufl.Form]) -> List[ufl.Form]:
    """Modifies a bilinear form for the case it is given on the boundary only.

    This avoids a bug in fenics.SystemAssembler where the matrices' sparsity pattern
    is not initialized correctly.

    """
    mod_forms = []
    for form in forms:
        trial, test = form.arguments()
        mesh = trial.function_space().mesh()
        dx = fenics.Measure("dx", domain=mesh)
        mod_forms.append(form + fenics.Constant(0.0) * fenics.dot(trial, test) * dx)

    return mod_forms

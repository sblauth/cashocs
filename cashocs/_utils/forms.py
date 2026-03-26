# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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

"""Utilities for UFL forms."""

from __future__ import annotations

from typing import Any, TypeVar

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import log
from cashocs.geometry.mesh import CashocsMesh

T = TypeVar("T")


def summation(x: list[T]) -> T | fenics.Constant:
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
        log.warning("Empty list handed to summation, returning 0.")
    else:
        y = x[0]

        for item in x[1:]:
            y += item  # type: ignore

    return y


def multiplication(x: list[T]) -> T | fenics.Constant:
    """Multiplies the elements of a list in a UFL friendly fashion.

    Used to build the product of certain UFL expressions to construct a UFL form.

    Args:
        x: The list whose entries shall be multiplied.

    Returns:
        The result of the multiplication.

    """
    if len(x) == 0:
        y = fenics.Constant(1.0)
        log.warning("Empty list handed to multiplication, returning 1.")
    else:
        y = x[0]

        for item in x[1:]:
            y *= item  # type: ignore

    return y


def max_(a: float | fenics.Function, b: float | fenics.Function) -> ufl.core.expr.Expr:
    """Computes the maximum of a and b.

    Args:
        a: The first parameter.
        b: The second parameter.

    Returns:
        The maximum of a and b.

    """
    return (a + b + abs(a - b)) / fenics.Constant(2.0)


def min_(a: float | fenics.Function, b: float | fenics.Function) -> ufl.core.expr.Expr:
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
    measure: ufl.Measure,
    lower_threshold: float | fenics.Function | None = None,
    upper_threshold: float | fenics.Function | None = None,
    shift_lower: float | fenics.Function | None = None,
    shift_upper: float | fenics.Function | None = None,
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
    value: fenics.Constant | fenics.Expression | fenics.Function | float | tuple[float],
    boundaries: fenics.MeshFunction,
    idcs: list[int | str] | int | str,
    **kwargs: Any,
) -> list[fenics.DirichletBC]:
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
        int_tag = _utils.tag_to_int(mesh.physical_groups, entry, "ds")
        if int_tag:
            bcs_list.append(
                fenics.DirichletBC(function_space, value, boundaries, int_tag, **kwargs)
            )

    return bcs_list


def bilinear_boundary_form_modification(forms: list[ufl.Form]) -> list[ufl.Form]:
    """Modifies a bilinear form for the case it is given on the boundary only.

    This avoids a bug in fenics.SystemAssembler where the matrices' sparsity pattern
    is not initialized correctly.

    """
    mod_forms = []
    for form in forms:
        trial, test = form.arguments()
        mesh = trial.function_space().mesh()
        dx = ufl.Measure("dx", domain=mesh)
        mod_forms.append(form + fenics.Constant(0.0) * ufl.dot(trial, test) * dx)

    return mod_forms


def _get_subdomain_ids_from_tag(
    tag: tuple[str | int] | str | int, physical_groups: dict
) -> list[int]:
    if isinstance(tag, tuple):
        subdomain_idx = []

        for t in tag:
            tag_int = _utils.tag_to_int(physical_groups, t, "dx")
            if tag_int is not None:
                subdomain_idx.append(tag_int)
    else:
        tag_int = _utils.tag_to_int(physical_groups, tag, "dx")
        subdomain_idx = []
        if tag_int is not None:
            subdomain_idx.append(tag_int)

    return subdomain_idx


def create_material_parameter(
    material_dict: dict,
    mesh: CashocsMesh,
    dg0_space: fenics.FunctionSpace | None = None,
) -> ufl.core.expr.Expr:
    r"""Creates a material parameter that can vary for different subdomains.

    Args:
        material_dict: The dictionary that contains the material parameters.
            The keys are the indices of the subdomains (as generated with Gmsh) and
            the values are the values of the material parameter on the respective
            subdomain. If multiple subdomains share the same value, they can also
            be put inside a tuple and used as a single key for the dictionary.
        mesh: The finite element mesh.
        dg0_space: The space of DG0 elements on the mesh. If this is None, then the
            space is created internally.

    Returns:
        A UFL expression that can be used as material parameter.

    Examples:
        Consider the following Poisson problem

        .. math::
            - \nabla \cdot (\mu \nabla u) = f \text{ in } \Omega \quad u = 0
            \text{ on } \Gamma,

        where the parameter :math:`\mu` varies between two subdomains, i.e.,
        :math:`\mu = \mu_{in}` in :math:`\Omega_{in}` and :math:`\mu = \mu_{out}` in
        :math:`\Omega_{out}`, where :math:`\Omega = \Omega_{in} \cup \Omega_{out}`.
        The parameter :math:`\mu` can be created with the help of this function.
        Assume that :math:`\Omega_{in}` has the index `1` and that
        :math:`\Omega_{out}` has index `2`, then we can use the following code ::

            mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(...)
            mu_in = 1.0 # example
            mu_out = 10.0 # example
            material_dict = {1: mu_in, 2: mu_out}

            mu = cashocs.create_material_parameter(material_dict)

    """
    indicator_dict = {}

    if dg0_space is None:
        dg0_space = fenics.FunctionSpace(mesh, "DG", 0)

    for subdomain_tag in material_dict.keys():
        indicator_function = fenics.Function(dg0_space)
        subdomain_ids = _get_subdomain_ids_from_tag(subdomain_tag, mesh.physical_groups)
        dg0_idx = np.flatnonzero(np.isin(mesh.subdomains.array(), subdomain_ids))

        indicator_function.vector()[dg0_idx] = 1.0
        indicator_function.vector().apply("")
        indicator_dict[subdomain_tag] = indicator_function

    material_parameter = _utils.summation(
        [indicator_dict[key] * material_dict[key] for key in material_dict.keys()]
    )

    return material_parameter

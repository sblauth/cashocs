# Copyright (C) 2020-2024 Sebastian Blauth
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

"""Helper functions."""

from __future__ import annotations

import configparser
import inspect
from typing import Any, Callable, cast, List, Optional, TypeVar, Union

import fenics

from cashocs import _exceptions

T = TypeVar("T")


def enlist(arg: Union[List[T], T]) -> List[T]:
    """Wraps the input argument into a list, if it isn't a list already.

    Args:
        arg: The input argument, which is to wrapped into a list.

    Returns:
        The object wrapped into a list.

    """
    if isinstance(arg, list):
        return arg
    else:
        return [arg]


def check_and_enlist_bcs(
    bcs_list: Union[
        fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
    ]
) -> List[List[fenics.DirichletBC]]:
    """Enlists DirichletBC objects for cashocs.

    Args:
        bcs_list: The list of DirichletBC objects

    Returns:
        The wrapped list of DirichletBC objects

    """
    if isinstance(bcs_list, fenics.DirichletBC):
        return [[bcs_list]]
    elif isinstance(bcs_list, list) and len(bcs_list) == 0:
        return [bcs_list]
    elif isinstance(bcs_list, list) and isinstance(bcs_list[0], fenics.DirichletBC):
        return [bcs_list]
    elif isinstance(bcs_list, list) and isinstance(bcs_list[0], list):
        return bcs_list
    else:
        raise _exceptions.InputError(
            "cashocs._utils.check_and_enlist_bcs",
            "bcs_list",
            "Type of bcs_list is wrong",
        )


def check_and_enlist_control_constraints(
    control_constraints: Union[
        List[Union[float, int, fenics.Function]],
        List[List[Union[float, int, fenics.Function]]],
    ]
) -> List[List[Union[float, int, fenics.Function]]]:
    """Wraps control constraints into a list suitable for cashocs.

    Args:
        control_constraints: The list of control constraints.

    Returns:
        The wrapped list of control constraints.

    """
    if isinstance(control_constraints, list) and isinstance(
        control_constraints[0], list
    ):
        control_constraints = cast(
            List[List[Union[float, int, fenics.Function]]], control_constraints
        )
        return control_constraints
    elif isinstance(control_constraints, list) and not isinstance(
        control_constraints[0], list
    ):
        control_constraints = cast(
            List[Union[float, int, fenics.Function]], control_constraints
        )
        return [control_constraints]
    else:
        raise _exceptions.InputError(
            "cashocs._utils.check_and_enlist_control_constraints",
            "control_constraints",
            "Type of control_constraints is wrong",
        )


def optimization_algorithm_configuration(
    config: configparser.ConfigParser, algorithm: Optional[str] = None
) -> str:
    """Returns the internal name of the optimization algorithm and updates config.

    Args:
        config: The config of the problem.
        algorithm: A string representing user input for the optimization algorithm
            if this is set via keywords in the .solve() call. If this is ``None``, then
            the config is used to return a consistent value for internal use. (Default
            is None).

    Returns:
        Internal name of the algorithms.

    """
    if algorithm is not None:
        overwrite = True
    else:
        overwrite = False
        algorithm = config.get("OptimizationRoutine", "algorithm")

    if algorithm.casefold() in ["gradient_descent", "gd"]:
        internal_algorithm = "gradient_descent"
    elif algorithm.casefold() in ["cg", "conjugate_gradient", "ncg", "nonlinear_cg"]:
        internal_algorithm = "conjugate_gradient"
    elif algorithm.casefold() in ["lbfgs", "bfgs"]:
        internal_algorithm = "lbfgs"
    elif algorithm.casefold() in ["newton"]:
        internal_algorithm = "newton"
    elif algorithm.casefold() in ["sphere_combination"]:
        internal_algorithm = "sphere_combination"
    elif algorithm.casefold() in ["convex_combination"]:
        internal_algorithm = "convex_combination"
    elif algorithm.casefold() == "none":
        internal_algorithm = "none"
    else:
        raise _exceptions.InputError(
            "cashocs._utils.optimization_algorithm_configuration",
            "algorithm",
            "Not a valid choice for the optimization algorithm.\n"
            "	For a gradient descent method, use 'gradient_descent' or 'gd'.\n"
            "	For a nonlinear conjugate gradient method use 'cg', "
            "'conjugate_gradient', 'ncg', or 'nonlinear_cg'.\n"
            "	For a limited memory BFGS method use 'bfgs' or 'lbfgs'.\n"
            "	For a truncated Newton method use 'newton' (optimal control only).\n"
            "   For Euler's method on the sphere use 'sphere_combination' (topology"
            "optimization only).\n"
            "   For the convex combination approach use 'convex_combination' (topology"
            "optimization only).",
        )

    if overwrite:
        config.set("OptimizationRoutine", "algorithm", internal_algorithm)

    return internal_algorithm


def create_function_list(
    function_spaces: List[fenics.FunctionSpace],
) -> List[fenics.Function]:
    """Creates a list of functions.

    Args:
        function_spaces: The function spaces, where the resulting functions should be in

    Returns:
        A list of functions

    """
    function_list = [
        fenics.Function(function_space) for function_space in function_spaces
    ]

    return function_list


def check_file_extension(file: str, required_extension: str) -> None:
    """Checks whether a given file extension is correct."""
    if not file.rsplit(".", 1)[-1] == required_extension:
        raise _exceptions.CashocsException(
            f"Cannot use {file} due to wrong format.",
        )


def number_of_arguments(function: Callable[..., Any]) -> int:
    """Computes the number of arguments that a function has.

    Args:
        function: The function which is checked for its number of arguments.

    Returns:
        The number of arguments that the input function has.

    """
    sig = inspect.signature(function)

    return len(sig.parameters)

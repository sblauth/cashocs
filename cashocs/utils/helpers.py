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

import argparse
import configparser
from typing import Union, List, Tuple, Optional

import fenics

from .._exceptions import InputError


def enlist(arg: Union[object, List]) -> List:
    """Wraps the input argument into a list, if it isn't a list already.

    Parameters
    ----------
    arg : list or object
        The input argument, which is to wrapped into a list

    Returns
    -------
    list
        The object wrapped into a list

    """

    if isinstance(arg, list):
        return arg
    else:
        return [arg]


def _check_and_enlist_bcs(
    bcs_list: Union[
        fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
    ]
) -> List[List[fenics.DirichletBC]]:
    """Enlists DirichletBC objects for cashocs

    Parameters
    ----------
    bcs_list : fenics.DirichletBC or list[fenics.DirichletBC] or list[list[fenics.DirichletBC]]
        The list of DirichletBC objects

    Returns
    -------
    list[list[fenics.DirichletBC]]
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
        raise InputError(
            "cashocs.utils._check_and_enlist_bcs",
            "bcs_list",
            "Type of bcs_list is wrong",
        )


def _check_and_enlist_control_constraints(
    control_constraints: Union[
        List[Union[float, int, fenics.Function]],
        List[List[Union[float, int, fenics.Function]]],
    ]
) -> List[List[Union[float, int, fenics.Function]]]:
    """Wraps control constraints into a list suitable for cashocs.

    Parameters
    ----------
    control_constraints : list[float or int or fenics.Function] or list[list[float or int or fenics.Function]]
        The list of control constraints

    Returns
    -------
    list[list[float or int or fenics.Function]]
        The wrapped list of control constraints
    """

    if isinstance(control_constraints, list) and isinstance(
        control_constraints[0], list
    ):
        return control_constraints
    elif isinstance(control_constraints, list) and not isinstance(
        control_constraints[0], list
    ):
        return [control_constraints]
    else:
        raise InputError(
            "cashocs.utils._check_and_enlist_control_constraints",
            "control_constraints",
            "Type of control_constraints is wrong",
        )


def _check_and_enlist_ksp_options(
    ksp_options: Union[List[List[str]], List[List[List[str]]]]
) -> List[List[List[str]]]:
    """Wraps ksp options into a list suitable for cashocs.

    Parameters
    ----------
    ksp_options : list[list[str]] or list[list[list[str]]]
        The list of ksp options

    Returns
    -------
    list[list[list[str]]]
        The wrapped list of ksp options
    """

    if (
        isinstance(ksp_options, list)
        and isinstance(ksp_options[0], list)
        and isinstance(ksp_options[0][0], str)
    ):
        return [ksp_options[:]]

    elif (
        isinstance(ksp_options, list)
        and isinstance(ksp_options[0], list)
        and isinstance(ksp_options[0][0], list)
    ):
        return ksp_options[:]
    else:
        raise InputError(
            "cashocs.utils._check_and_enlist_ksp_options",
            "ksp_options",
            "Type of ksp_options is wrong.",
        )


def _parse_remesh() -> Tuple[bool, str]:
    """Parses command line arguments for the remeshing flag

    Returns
    -------
    bool, str
        A boolean indicating, whether a remeshing was performed and a string which
        points to the remeshing directory.

    """

    parser = argparse.ArgumentParser(description="test argument parser")
    parser.add_argument(
        "--temp_dir", type=str, help="Location of the temp directory for remeshing"
    )
    parser.add_argument(
        "--cashocs_remesh",
        action="store_true",
        help="Flag which indicates whether remeshing has been performed",
    )
    args = parser.parse_args()

    temp_dir = args.temp_dir or None
    cashocs_remesh_flag = True if args.cashocs_remesh else False

    return cashocs_remesh_flag, temp_dir


def _optimization_algorithm_configuration(
    config: configparser.ConfigParser, algorithm: Optional[str] = None
) -> str:
    """Returns the internal name of the optimization algorithm and updates config.

    Parameters
    ----------
    config : configparser.ConfigParser or None
        The config of the problem.
    algorithm : str or None, optional
        A string representing user input for the optimization algorithm
        if this is set via keywords in the .solve() call. If this is
        ``None``, then the config is used to return a consistent value
        for internal use. (Default is None).

    Returns
    -------
    str
        Internal name of the algorithms.
    """

    internal_algorithm = None

    if algorithm is not None:
        overwrite = True
    else:
        overwrite = False
        algorithm = config.get("OptimizationRoutine", "algorithm", fallback="none")

    if algorithm in ["gradient_descent", "gd"]:
        internal_algorithm = "gradient_descent"
    elif algorithm in ["cg", "conjugate_gradient", "ncg", "nonlinear_cg"]:
        internal_algorithm = "conjugate_gradient"
    elif algorithm in ["lbfgs", "bfgs"]:
        internal_algorithm = "lbfgs"
    elif algorithm in ["newton"]:
        internal_algorithm = "newton"
    elif algorithm == "none":
        internal_algorithm = "none"
    else:
        raise InputError(
            "cashocs.utils._optimization_algorithm_configuration",
            "algorithm",
            "Not a valid choice for the optimization algorithm.\n"
            "	For a gradient descent method, use 'gradient_descent' or 'gd'.\n"
            "	For a nonlinear conjugate gradient method use 'cg', 'conjugate_gradient', 'ncg', or 'nonlinear_cg'.\n"
            "	For a limited memory BFGS method use 'bfgs' or 'lbfgs'.\n"
            "	For a truncated Newton method use 'newton' (optimal control only).\n",
        )

    if overwrite:
        config.set("OptimizationRoutine", "algorithm", internal_algorithm)

    return internal_algorithm

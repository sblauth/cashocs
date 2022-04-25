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

"""Module including utility and helper functions.

This module includes utility and helper functions used in cashocs. They might also be
interesting for users, so they are part of the public API. Includes wrappers that allow
to shorten the coding for often recurring actions.
"""

from cashocs._utils.forms import bilinear_boundary_form_modification
from cashocs._utils.forms import create_bcs_list
from cashocs._utils.forms import create_dirichlet_bcs
from cashocs._utils.forms import max_
from cashocs._utils.forms import min_
from cashocs._utils.forms import moreau_yosida_regularization
from cashocs._utils.forms import multiplication
from cashocs._utils.forms import summation
from cashocs._utils.helpers import check_and_enlist_bcs
from cashocs._utils.helpers import check_and_enlist_control_constraints
from cashocs._utils.helpers import check_and_enlist_ksp_options
from cashocs._utils.helpers import create_function_list
from cashocs._utils.helpers import enlist
from cashocs._utils.helpers import optimization_algorithm_configuration
from cashocs._utils.helpers import parse_remesh
from cashocs._utils.linalg import assemble_and_solve_linear
from cashocs._utils.linalg import assemble_petsc_system
from cashocs._utils.linalg import Interpolator
from cashocs._utils.linalg import setup_petsc_options
from cashocs._utils.linalg import solve_linear_problem
from cashocs._utils.linalg import split_linear_forms

__all__ = [
    "summation",
    "multiplication",
    "create_dirichlet_bcs",
    "max_",
    "min_",
    "moreau_yosida_regularization",
    "check_and_enlist_ksp_options",
    "check_and_enlist_control_constraints",
    "check_and_enlist_bcs",
    "enlist",
    "parse_remesh",
    "optimization_algorithm_configuration",
    "assemble_petsc_system",
    "setup_petsc_options",
    "solve_linear_problem",
    "Interpolator",
    "create_bcs_list",
    "assemble_and_solve_linear",
    "split_linear_forms",
    "create_function_list",
    "bilinear_boundary_form_modification",
]

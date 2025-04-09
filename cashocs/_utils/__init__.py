# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

"""Utility and helper functions.

This module includes utility and helper functions used in cashocs. They might also be
interesting for users, so they are part of the public API. Includes wrappers that allow
to shorten the coding for often recurring actions.
"""

from cashocs._utils.forms import bilinear_boundary_form_modification
from cashocs._utils.forms import create_dirichlet_bcs
from cashocs._utils.forms import max_
from cashocs._utils.forms import min_
from cashocs._utils.forms import moreau_yosida_regularization
from cashocs._utils.forms import multiplication
from cashocs._utils.forms import summation
from cashocs._utils.helpers import check_and_enlist_bcs
from cashocs._utils.helpers import check_and_enlist_control_constraints
from cashocs._utils.helpers import check_file_extension
from cashocs._utils.helpers import create_function_list
from cashocs._utils.helpers import enlist
from cashocs._utils.helpers import get_petsc_prefixes
from cashocs._utils.helpers import number_of_arguments
from cashocs._utils.helpers import optimization_algorithm_configuration
from cashocs._utils.helpers import tag_to_int
from cashocs._utils.interpolations import interpolate_by_angle
from cashocs._utils.interpolations import interpolate_by_volume
from cashocs._utils.interpolations import interpolate_levelset_function_to_cells
from cashocs._utils.linalg import assemble_and_solve_linear
from cashocs._utils.linalg import assemble_petsc_system
from cashocs._utils.linalg import Interpolator
from cashocs._utils.linalg import l2_projection
from cashocs._utils.linalg import setup_petsc_options
from cashocs._utils.linalg import solve_linear_problem
from cashocs._utils.linalg import split_linear_forms

__all__ = [
    "bilinear_boundary_form_modification",
    "create_dirichlet_bcs",
    "max_",
    "min_",
    "moreau_yosida_regularization",
    "multiplication",
    "summation",
    "check_and_enlist_bcs",
    "check_and_enlist_control_constraints",
    "check_file_extension",
    "create_function_list",
    "enlist",
    "get_petsc_prefixes",
    "number_of_arguments",
    "optimization_algorithm_configuration",
    "tag_to_int",
    "interpolate_by_angle",
    "interpolate_by_volume",
    "interpolate_levelset_function_to_cells",
    "assemble_and_solve_linear",
    "assemble_petsc_system",
    "Interpolator",
    "l2_projection",
    "setup_petsc_options",
    "solve_linear_problem",
    "split_linear_forms",
]

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

from cashocs.utils.forms import (
    summation,
    multiplication,
    create_bcs_list,
    create_dirichlet_bcs,
    _max,
    _min,
    moreau_yosida_regularization,
)
from cashocs.utils.helpers import (
    _check_and_enlist_bcs,
    _check_and_enlist_ksp_options,
    _check_and_enlist_control_constraints,
    enlist,
    _parse_remesh,
    _optimization_algorithm_configuration,
)
from cashocs.utils.linalg import (
    _assemble_petsc_system,
    _setup_petsc_options,
    _solve_linear_problem,
    _assemble_and_solve_linear,
    Interpolator,
    _split_linear_forms,
)

__all__ = [
    "summation",
    "multiplication",
    "create_dirichlet_bcs",
    "_max",
    "_min",
    "moreau_yosida_regularization",
    "_check_and_enlist_ksp_options",
    "_check_and_enlist_control_constraints",
    "_check_and_enlist_bcs",
    "enlist",
    "_parse_remesh",
    "_optimization_algorithm_configuration",
    "_assemble_petsc_system",
    "_setup_petsc_options",
    "_solve_linear_problem",
    "Interpolator",
    "create_bcs_list",
    "_assemble_and_solve_linear",
    "_split_linear_forms",
]

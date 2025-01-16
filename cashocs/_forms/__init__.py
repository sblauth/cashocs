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

"""Derivation and management of weak forms.

This module is used to carry out form manipulations such as generating the UFL
forms for the adjoint system and for the Riesz gradient identificiation
problems.
"""

from cashocs._forms.control_form_handler import ControlFormHandler
from cashocs._forms.form_handler import FormHandler
from cashocs._forms.general_form_handler import AdjointFormHandler
from cashocs._forms.general_form_handler import GeneralFormHandler
from cashocs._forms.general_form_handler import StateFormHandler
from cashocs._forms.shape_form_handler import ShapeFormHandler
from cashocs._forms.shape_regularization import ShapeRegularization

__all__ = [
    "ControlFormHandler",
    "FormHandler",
    "AdjointFormHandler",
    "GeneralFormHandler",
    "StateFormHandler",
    "ShapeFormHandler",
    "ShapeRegularization",
]

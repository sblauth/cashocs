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

"""Box constraints for control problems."""

from __future__ import annotations

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import _utils


class BoxConstraints:
    """Box constraints for optimal control problems."""

    def __init__(
        self,
        controls: list[fenics.Function],
        control_constraints: list[list[float | fenics.Function]] | None,
    ) -> None:
        """Initializes the box constraints.

        Args:
            controls: The control variables.
            control_constraints: The control (box) constraints.

        """
        self.controls = controls

        self.require_control_constraints: list[bool] = [False] * len(self.controls)
        self.control_constraints = self._parse_control_constraints(control_constraints)
        self._validate_control_constraints()

        self.restrictor = Restrictor(
            self.controls, self.control_constraints, self.require_control_constraints
        )
        self.display_box_constraints = False
        if np.any(self.require_control_constraints):
            self.display_box_constraints = True

    def _parse_control_constraints(
        self, control_constraints: list[list[float | fenics.Function]] | None
    ) -> list[list[fenics.Function]]:
        """Checks, whether the given control constraints are feasible.

        Args:
            control_constraints: The control constraints.

        Returns:
            The (wrapped) list of control constraints.

        """
        temp_control_constraints: list[list[fenics.Function | float]]
        if control_constraints is None:
            temp_control_constraints = []
            for control in self.controls:
                u_a = fenics.Function(control.function_space())
                u_a.vector().vec().set(float("-inf"))
                u_a.vector().apply("")
                u_b = fenics.Function(control.function_space())
                u_b.vector().vec().set(float("inf"))
                u_b.vector().apply("")
                temp_control_constraints.append([u_a, u_b])
        else:
            temp_control_constraints = _utils.check_and_enlist_control_constraints(
                control_constraints
            )

        # recast floats into functions for compatibility
        formatted_control_constraints: list[list[fenics.Function]] = []
        for idx, pair in enumerate(temp_control_constraints):
            if isinstance(pair[0], fenics.Function):
                lower_bound = pair[0]
            else:
                lower_bound = fenics.Function(self.controls[idx].function_space())
                lower_bound.vector().vec().set(pair[0])
                lower_bound.vector().apply("")

            if isinstance(pair[1], fenics.Function):
                upper_bound = pair[1]
            else:
                upper_bound = fenics.Function(self.controls[idx].function_space())
                upper_bound.vector().vec().set(pair[1])
                upper_bound.vector().apply("")

            formatted_control_constraints.append([lower_bound, upper_bound])

        return formatted_control_constraints

    def _validate_control_constraints(self) -> None:
        """Checks, whether given control constraints are valid."""
        for idx, pair in enumerate(self.control_constraints):
            if not np.all(pair[0].vector()[:] < pair[1].vector()[:]):
                raise _exceptions.InputError(
                    (
                        "cashocs._optimization.optimal_control."
                        "optimal_control_problem.OptimalControlProblem"
                    ),
                    "control_constraints",
                    (
                        "The lower bound must always be smaller than the upper bound "
                        "for the control_constraints."
                    ),
                )

            if pair[0].vector().vec().max()[1] == float("-inf") and pair[
                1
            ].vector().vec().min()[1] == float("inf"):
                # no control constraint for this component
                pass
            else:
                self.require_control_constraints[idx] = True

                control_element = self.controls[idx].ufl_element()
                if control_element.family() == "Mixed":
                    for j in range(control_element.value_size()):
                        sub_elem = control_element.extract_component(j)[1]
                        if not (
                            sub_elem.family() == "Real"
                            or (
                                sub_elem.family() == "Lagrange"
                                and sub_elem.degree() == 1
                            )
                            or (
                                sub_elem.family() == "Discontinuous Lagrange"
                                and sub_elem.degree() <= 1
                            )
                        ):
                            raise _exceptions.InputError(
                                (
                                    "cashocs._optimization.optimal_control."
                                    "optimal_control_problem.OptimalControlProblem"
                                ),
                                "controls",
                                (
                                    "Control constraints are only implemented for "
                                    "linear Lagrange, constant and linear discontinuous"
                                    " Lagrange, and Real elements."
                                ),
                            )

                else:
                    if not (
                        control_element.family() == "Real"
                        or (
                            control_element.family() == "Lagrange"
                            and control_element.degree() == 1
                        )
                        or (
                            control_element.family() == "Discontinuous Lagrange"
                            and control_element.degree() <= 1
                        )
                    ):
                        raise _exceptions.InputError(
                            (
                                "cashocs._optimization.optimal_control."
                                "optimal_control_problem.OptimalControlProblem"
                            ),
                            "controls",
                            (
                                "Control constraints are only implemented for "
                                "linear Lagrange, constant and linear discontinuous "
                                "Lagrange, and Real elements."
                            ),
                        )


class Restrictor:
    """Restricts functions to active / inactive sets."""

    def __init__(
        self,
        controls: list[fenics.Function],
        control_constraints: list[list[fenics.Function]],
        require_control_constraints: list[bool],
    ) -> None:
        """Initializes self.

        Args:
            controls: list of control variables.
            control_constraints: The list of control constraints.
            require_control_constraints: list of booleans indicating which components
                use the box constraints.

        """
        self.controls = controls
        self.control_constraints = control_constraints
        self.require_control_constraints = require_control_constraints

        self.temp = _utils.create_function_list(
            [x.function_space() for x in self.controls]
        )

        self.idx_active_lower: list = []
        self.idx_active_upper: list = []
        self.idx_active: list = []
        self.idx_inactive: list = []

    def compute_active_sets(self) -> None:
        """Computes the indices corresponding to active and inactive sets."""
        self.idx_active_lower.clear()
        self.idx_active_upper.clear()
        self.idx_active.clear()
        self.idx_inactive.clear()

        for j in range(len(self.controls)):
            if self.require_control_constraints[j]:
                self.idx_active_lower.append(
                    np.flatnonzero(
                        self.controls[j].vector()[:]
                        <= self.control_constraints[j][0].vector()[:]
                    )
                )
                self.idx_active_upper.append(
                    np.flatnonzero(
                        self.controls[j].vector()[:]
                        >= self.control_constraints[j][1].vector()[:]
                    )
                )
                self.idx_inactive.append(
                    np.flatnonzero(
                        np.logical_and(
                            self.controls[j].vector()[:]
                            > self.control_constraints[j][0].vector()[:],
                            self.controls[j].vector()[:]
                            < self.control_constraints[j][1].vector()[:],
                        )
                    )
                )
            else:
                self.idx_active_lower.append([])
                self.idx_active_upper.append([])
                self.idx_inactive.append([])

            temp_active = np.concatenate(
                (self.idx_active_lower[j], self.idx_active_upper[j])
            )
            temp_active.sort()
            self.idx_active.append(temp_active)

    def restrict_to_inactive_set(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> list[fenics.Function]:
        """Restricts a function to the inactive set.

        Restricts a control type function ``a`` onto the inactive set,
        which is returned via the function ``b``, i.e., ``b`` is zero on the active set.

        Args:
            a: The control-type function that is to be projected onto the inactive set.
            b: The storage for the result of the projection (is overwritten).

        Returns:
            The result of the projection of ``a`` onto the inactive set (overwrites
            input ``b``).

        """
        for j in range(len(self.controls)):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector().apply("")
                self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[
                    self.idx_inactive[j]
                ]
                self.temp[j].vector().apply("")
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())
                b[j].vector().apply("")

            else:
                if not b[j].vector().vec().equal(a[j].vector().vec()):
                    b[j].vector().vec().aypx(0.0, a[j].vector().vec())
                    b[j].vector().apply("")

        return b

    def restrict_to_active_set(
        self, a: list[fenics.Function], b: list[fenics.Function]
    ) -> list[fenics.Function]:
        """Restricts a function to the active set.

        Restricts a control type function ``a`` onto the active set,
        which is returned via the function ``b``,  i.e., ``b`` is zero on the inactive
        set.

        Args:
            a: The first argument, to be projected onto the active set.
            b: The second argument, which stores the result (is overwritten).

        Returns:
            The result of the projection (overwrites input b).

        """
        for j in range(len(self.controls)):
            if self.require_control_constraints[j]:
                self.temp[j].vector().vec().set(0.0)
                self.temp[j].vector().apply("")
                self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[
                    self.idx_active[j]
                ]
                self.temp[j].vector().apply("")
                b[j].vector().vec().aypx(0.0, self.temp[j].vector().vec())
                b[j].vector().apply("")

            else:
                b[j].vector().vec().set(0.0)
                b[j].vector().apply("")

        return b

    def project_to_admissible_set(
        self, a: list[fenics.Function]
    ) -> list[fenics.Function]:
        """Project a function to the set of admissible controls.

        Projects a control type function ``a`` onto the set of admissible controls
        (given by the box constraints).

        Args:
            a: The function which is to be projected onto the set of admissible
                controls (is overwritten)

        Returns:
            The result of the projection (overwrites input ``a``)

        """
        for j in range(len(self.controls)):
            if self.require_control_constraints[j]:
                a[j].vector().vec().pointwiseMin(
                    self.control_constraints[j][1].vector().vec(), a[j].vector().vec()
                )
                a[j].vector().apply("")
                a[j].vector().vec().pointwiseMax(
                    a[j].vector().vec(), self.control_constraints[j][0].vector().vec()
                )
                a[j].vector().apply("")

        return a

# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Exceptions raised by CASHOCS.

"""
from __future__ import annotations

from typing import Optional, List


class CashocsException(Exception):
    """Base class for exceptions raised by CASHOCS."""

    pass


class NotConvergedError(CashocsException):
    """This exception is raised when a solver does not converge.

    This includes any type of iterative method used to solve a problem,
    whether it is a linear or nonlinear system of equations, or an
    optimization problem.
    """

    def __init__(self, solver: str, message: Optional[str] = None) -> None:
        self.solver = solver
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return f"The {self.solver} failed to converge."
        else:
            return f"The {self.solver} failed to converge.\n{self.message}"


class PETScKSPError(CashocsException):
    """This exception is raised when the solution of a linear problem with PETSc fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self, error_code: int, message: str = "PETSc linear solver did not converge."
    ) -> None:
        self.message = message
        self.error_code = error_code

        if self.error_code == -2:
            self.error_reason = " (ksp_diverged_null)"
        elif self.error_code == -3:
            self.error_reason = " (ksp_diverged_its, reached maximum iterations)"
        elif self.error_code == -4:
            self.error_reason = " (ksp_diverged_dtol, reached divergence tolerance)"
        elif self.error_code == -5:
            self.error_reason = " (ksp_diverged_breakdown, krylov method breakdown)"
        elif self.error_code == -6:
            self.error_reason = " (ksp_diverged_breakdown_bicg)"
        elif self.error_code == -7:
            self.error_reason = " (ksp_diverged_nonsymmetric, need a symmetric operator / preconditioner)"
        elif self.error_code == -8:
            self.error_reason = " (ksp_diverged_indefinite_pc, the preconditioner is indefinite, but needs to be positive definite)"
        elif self.error_code == -9:
            self.error_reason = " (ksp_diverged_nanorinf)"
        elif self.error_code == -10:
            self.error_reason = " (ksp_diverged_indefinite_mat, operator is indefinite, but needs to be positive definite)"
        elif self.error_code == -11:
            self.error_reason = " (ksp_diverged_pc_failed, it was not possible to build / use the preconditioner)"
        else:
            self.error_reason = " (unknown)"

    def __str__(self) -> str:
        return (
            f"{self.message} KSPConvergedReason = {self.error_code} {self.error_reason}"
        )


class InputError(CashocsException):
    """This exception gets raised when the user input to a public API method is wrong or inconsistent."""

    def __init__(self, obj: str, param: str, message: Optional[str] = None) -> None:
        self.obj = obj
        self.param = param
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return f"Not a valid input for object {self.obj}. The faulty input is for the parameter {self.param}."
        else:
            return f"Not a valid input for object {self.obj}. The faulty input is for the parameter {self.param}.\n{self.message}"


class ConfigError(CashocsException):
    """This exception gets raised when parameters in the config file are wrong."""

    pre_message = "You have an error in your config file.\n"

    def __init__(self, config_errors: List[str]) -> None:
        self.config_errors = config_errors

    def __str__(self) -> str:
        except_str = f"{self.pre_message}"
        for error in self.config_errors:
            except_str += error
        return except_str


class IncompatibleConfigurationError(CashocsException):
    """This exception gets raised when parameters in the config file are in conflict."""

    pre_message = "Incompatible configuration file parameters.\n"

    def __init__(
        self,
        key1: str,
        section1: str,
        key2: str,
        section2: str,
        post_message: Optional[str] = None,
    ) -> None:
        self.key1 = key1
        self.section1 = section1
        self.key2 = key2
        self.section2 = section2
        self.post_message = post_message

    def __str__(self) -> str:
        if self.post_message is None:
            return f"{self.pre_message}The conflicting parameters are {self.key1} in section {self.section1} and {self.key2} in section {self.section2}."
        else:
            return f"{self.pre_message}The conflicting parameters are {self.key1} in section {self.section1} and {self.key2} in section {self.section2}.\n{self.post_message}"


class GeometryError(CashocsException):
    """This exception gets raised when there is a problem with the finite element mesh."""

    def __init__(self, message: Optional[str] = None) -> None:
        self.message = message

    def __str__(self) -> str:
        if self.message is None:
            return f"The finite element mesh is not valid anymore."
        else:
            return f"The finite element mesh is not valid anymore.\n{self.message}"

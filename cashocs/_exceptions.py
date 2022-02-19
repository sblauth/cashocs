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

"""Exceptions raised by cashocs."""

from __future__ import annotations

from typing import List, Optional


class CashocsException(Exception):
    """Base class for exceptions raised by cashocs."""

    pass


class CashocsDebugException(CashocsException):
    """Exception that can get raised for debugging."""

    pass


class NotConvergedError(CashocsException):
    """This exception is raised when a solver does not converge.

    This includes any type of iterative method used to solve a problem,
    whether it is a linear or nonlinear system of equations, or an
    optimization problem.
    """

    def __init__(self, solver: str, message: Optional[str] = None) -> None:
        """Initializes self.

        Args:
            solver: The solver which raised the exception.
            message: A message indicating why the solver did not converge.

        """
        super().__init__()
        self.solver = solver
        self.message = message

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        main_msg = f"The {self.solver} failed to converge."
        post_msg = f"\n{self.message}" if self.message is not None else ""
        return main_msg + post_msg


class PETScKSPError(CashocsException):
    """This exception is raised when the solution of a linear problem with PETSc fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self,
        error_code: int,
        message: str = "PETSc linear solver did not converge.",
    ) -> None:
        """Initializes self.

        Args:
            error_code: The error code issued by PETSc.
            message: The message, detailing why PETSc issued an error.

        """
        super().__init__()
        self.message = message
        self.error_code = error_code

        self.error_dict = {
            -2: " (ksp_diverged_null)",
            -3: " (ksp_diverged_its, reached maximum iterations)",
            -4: " (ksp_diverged_dtol, reached divergence tolerance)",
            -5: " (ksp_diverged_breakdown, krylov method breakdown)",
            -6: " (ksp_diverged_breakdown_bicg)",
            -7: " (ksp_diverged_nonsymmetric, "
            "need a symmetric operator / preconditioner)",
            -8: " (ksp_diverged_indefinite_pc, "
            "the preconditioner is indefinite, "
            "but needs to be positive definite)",
            -9: " (ksp_diverged_nanorinf)",
            -10: " (ksp_diverged_indefinite_mat, "
            "operator is indefinite, "
            "but needs to be positive definite)",
            -11: " (ksp_diverged_pc_failed, "
            "it was not possible to build / use the preconditioner)",
        }

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        return (
            f"{self.message} KSPConvergedReason = "
            f"{self.error_code} {self.error_dict[self.error_code]}"
        )


class InputError(CashocsException):
    """This gets raised when the user input to a public API method is wrong."""

    def __init__(self, obj: str, param: str, message: Optional[str] = None) -> None:
        """Initializes self.

        Args:
            obj: The object which raises the exception.
            param: The faulty input parameter.
            message: A message detailing what went wrong.

        """
        super().__init__()
        self.obj = obj
        self.param = param
        self.message = message

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        main_msg = (
            f"Not a valid input for object {self.obj}. "
            f"The faulty input is for the parameter {self.param}."
        )
        post_msg = f"\n{self.message}" if self.message is not None else ""
        return main_msg + post_msg


class ConfigError(CashocsException):
    """This exception gets raised when parameters in the config file are wrong."""

    pre_message = "You have some error(s) in your config file.\n"

    def __init__(self, config_errors: List[str]) -> None:
        """Initializes self.

        Args:
            config_errors: The list of errors that occurred while trying to validate
                the config.

        """
        super().__init__()
        self.config_errors = config_errors

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        except_str = f"{self.pre_message}"
        for error in self.config_errors:
            except_str += error
        return except_str

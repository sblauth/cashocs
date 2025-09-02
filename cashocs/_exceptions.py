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

"""Exceptions raised by cashocs."""

from __future__ import annotations


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

    def __init__(self, solver: str, message: str | None = None) -> None:
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


class PETScError(CashocsException):
    """This exception is raised when the solution of a problem with PETSc fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self,
        error_code: int,
        message: str = "The PETSc solver did not converge.",
    ) -> None:
        """Initializes self.

        Args:
            error_code: The error code issued by PETSc.
            message: The message, detailing why PETSc issued an error.

        """
        super().__init__()
        self.message = message
        self.error_code = error_code
        self.error_dict: dict[int, str] = {}

    def __str__(self) -> str:
        """Returns the string representation of the exception."""
        return (
            f"{self.message} ConvergedReason = "
            f"{self.error_code} {self.error_dict[self.error_code]}"
        )


class PETScKSPError(PETScError):
    """This exception is raised when the solution of a linear problem with PETSc fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self,
        error_code: int,
        message: str = "The PETSc linear solver did not converge.",
    ) -> None:
        """Initializes self.

        Args:
            error_code: The error code issued by PETSc.
            message: The message, detailing why PETSc issued an error.

        """
        super().__init__(error_code, message)
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


class PETScSNESError(PETScError):
    """This exception is raised if the solution of a nonlinear problem with PETSc fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self,
        error_code: int,
        message: str = "The PETSc nonlinear solver did not converge.",
    ) -> None:
        """Initializes self.

        Args:
            error_code: The error code issued by PETSc.
            message: The message, detailing why PETSc issued an error.

        """
        super().__init__(error_code, message)
        self.error_dict = {
            -1: (
                " (snes_diverged_function_domain, "
                "the new x location passed to the function is not in the domain)"
            ),
            -2: " (snes_diverged_function_count)",
            -3: " (snes_diverged_linear_solve, linear solve failed)",
            -4: " (snes_diverged_fnorm_nan)",
            -5: " (snes_diverged_max_it, maximum number of iterations exceeded)",
            -6: " (snes_diverged_line_search, the line search failed)",
            -7: " (snes_diverged_inner, inner solve failed)",
            -8: (
                " (snes_diverged_local_min, ||J^T b|| is small, "
                "implies converged to local minimum)"
            ),
            -9: " (snes_diverged_dtol, ||F|| > divtol*||F_initial||)",
            -10: (
                " (snes_diverged_jacobian_domain, "
                "Jacobian calculation does not make sense)"
            ),
            -11: " (snes_diverged_tr_delta)",
        }


class PETScTSError(PETScError):
    """This exception is raised if the solution of a nonlinear problem with TS fails.

    Also returns the PETSc error code and reason.
    """

    def __init__(
        self,
        error_code: int,
        message: str = "The PETSc TS solver did not converge.",
    ) -> None:
        """Initializes self.

        Args:
            error_code: The error code issued by PETSc.
            message: The message, detailing why PETSc issued an error.

        """
        super().__init__(error_code, message)
        self.error_dict = {
            2: " (ts_converged_its, reached maximum number of steps)",
            -1: " (ts_diverged_nonlinear_solve)",
            -2: " (ts_diverged_step_rejected)",
            -3: " (ts_forward_diverged_linear_solve)",
            -4: " (ts_adjoint_diverged_linear_solve)",
            -5: " (ts_diverged_dtol, reached divergence tolerance)",
        }


class InputError(CashocsException):
    """This gets raised when the user input to a public API method is wrong."""

    def __init__(self, obj: str, param: str, message: str | None = None) -> None:
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

    def __init__(self, config_errors: list[str]) -> None:
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

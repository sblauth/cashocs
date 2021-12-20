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

"""General blue print class for all optimization algorithms

This is the base class, on which ControlOptimizationAlgorithm and
ShapeOptimizationAlgorithm classes are based.

"""

from __future__ import annotations

import abc
import json
import os
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import fenics
import numpy as np

from .._exceptions import NotConvergedError
from .._loggers import error, info


if TYPE_CHECKING:
    from .optimization_problem import OptimizationProblem


class OptimizationAlgorithm(abc.ABC):
    """Abstract class representing all kinds of optimization algorithms."""

    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        """
        Parameters
        ----------
        optimization_problem : OptimizationProblem
            The corresponding optimization problem
        """

        self.line_search_broken = False
        self.has_curvature_info = False

        self.form_handler = optimization_problem.form_handler
        self.state_problem = optimization_problem.state_problem
        self.config = self.state_problem.config
        self.adjoint_problem = optimization_problem.adjoint_problem

        self.cost_functional = optimization_problem.reduced_cost_functional

        self.iteration = 0
        self.objective_value = 1.0
        self.gradient_norm_initial = 1.0
        self.relative_norm = 1.0
        self.stepsize = 1.0

        self.require_control_constraints = False

        self.requires_remeshing = False
        self.remeshing_its = False

        self.converged = False
        self.converged_reason = 0

        self.verbose = self.config.getboolean("Output", "verbose", fallback=True)
        self.save_txt = self.config.getboolean("Output", "save_txt", fallback=True)
        self.save_results = self.config.getboolean(
            "Output", "save_results", fallback=True
        )
        self.rtol = self.config.getfloat("OptimizationRoutine", "rtol", fallback=1e-3)
        self.atol = self.config.getfloat("OptimizationRoutine", "atol", fallback=0.0)
        self.maximum_iterations = self.config.getint(
            "OptimizationRoutine", "maximum_iterations", fallback=100
        )
        self.soft_exit = self.config.getboolean(
            "OptimizationRoutine", "soft_exit", fallback=False
        )
        self.save_pvd = self.config.getboolean("Output", "save_pvd", fallback=False)
        self.save_pvd_adjoint = self.config.getboolean(
            "Output", "save_pvd_adjoint", fallback=False
        )
        self.save_pvd_gradient = self.config.getboolean(
            "Output", "save_pvd_gradient", fallback=False
        )

        self.pvd_prefix = ""

        self.output_dict = dict()
        try:
            self.output_dict["cost_function_value"] = self.temp_dict["output_dict"][
                "cost_function_value"
            ]
            self.output_dict["gradient_norm"] = self.temp_dict["output_dict"][
                "gradient_norm"
            ]
            self.output_dict["stepsize"] = self.temp_dict["output_dict"]["stepsize"]
            self.output_dict["MeshQuality"] = self.temp_dict["output_dict"][
                "MeshQuality"
            ]
        except (TypeError, KeyError, AttributeError):
            self.output_dict["cost_function_value"] = []
            self.output_dict["gradient_norm"] = []
            self.output_dict["stepsize"] = []
            self.output_dict["MeshQuality"] = []

        self.has_output = (
            self.save_txt
            or self.save_pvd
            or self.save_pvd_gradient
            or self.save_pvd_adjoint
            or self.save_results
        )

        self.result_dir = self.config.get("Output", "result_dir", fallback="./results")
        self.time_suffix = self.config.getboolean(
            "Output", "time_suffix", fallback=False
        )
        if self.time_suffix:
            dt = datetime.now()
            self.suffix = (
                f"{dt.year}_{dt.month}_{dt.day}_{dt.hour}_{dt.minute}_{dt.second}"
            )
            if self.result_dir[-1] == "/":
                self.result_dir = f"{self.result_dir[:-1]}_{self.suffix}"
            else:
                self.result_dir = f"{self.result_dir}_{self.suffix}"

        if not os.path.isdir(self.result_dir):
            if self.has_output:
                Path(self.result_dir).mkdir(parents=True, exist_ok=True)

    @abc.abstractmethod
    def run(self) -> None:
        pass

    def _generate_pvd_file(
        self, space: fenics.FunctionSpace, name: str, prefix: str = ""
    ) -> Union[fenics.File, List[fenics.File]]:
        """Generate a fenics.File for saving Functions

        Parameters
        ----------
        space : fenics.FunctionSpace
            The FEM function space where the function is taken from.
        name : str
            The name of the function / file
        prefix : str, optional
            A prefix for the file name, used for remeshing

        Returns
        -------
        fenics.File
            A .pvd fenics.File object, into which a Function can be written
        """

        if space.num_sub_spaces() > 0 and space.ufl_element().family() == "Mixed":
            lst = []
            for j in range(space.num_sub_spaces()):
                lst.append(
                    fenics.File(f"{self.result_dir}/pvd/{prefix}{name}_{j:d}.pvd")
                )
            return lst
        else:
            return fenics.File(f"{self.result_dir}/pvd/{prefix}{name}.pvd")

    @abc.abstractmethod
    def print_results(self) -> None:
        """Prints the results of the current iteration step to the console

        Returns
        -------
        None
        """

        strs = []
        strs.append(f"Iteration {self.iteration:4d} - ")
        strs.append(f" Objective value:  {self.objective_value:.3e}")
        self.output_dict["cost_function_value"].append(self.objective_value)

        if not np.any(self.require_control_constraints):
            if self.iteration == 0:
                strs.append(
                    f"    Gradient norm:  {self.gradient_norm_initial:.3e} (abs)"
                )
            else:
                strs.append(f"    Gradient norm:  {self.relative_norm:.3e} (rel)")
        else:
            if self.iteration == 0:
                strs.append(
                    f"    Stationarity measure:  {self.gradient_norm_initial:.3e} (abs)"
                )
            else:
                strs.append(
                    f"    Stationarity measure:  {self.relative_norm:.3e} (rel)"
                )
        self.output_dict["gradient_norm"].append(self.relative_norm)

        try:
            strs.append(
                f"    Mesh Quality:  {self.mesh_handler.current_mesh_quality:1.2f} ({self.mesh_handler.mesh_quality_measure})"
            )
            self.output_dict["MeshQuality"].append(
                self.mesh_handler.current_mesh_quality
            )
        except AttributeError:
            pass

        if self.iteration > 0:
            strs.append(f"    Step size:  {self.stepsize:.3e}")
        self.output_dict["stepsize"].append(self.stepsize)

        if self.iteration == 0:
            strs.append("\n")

        output = "".join(strs)
        if self.verbose:
            print(output)

        if self.save_txt:
            if self.iteration == 0:
                with open(f"{self.result_dir}/history.txt", "w") as file:
                    file.write(f"{output}\n")
            else:
                with open(f"{self.result_dir}/history.txt", "a") as file:
                    file.write(f"{output}\n")

        if self.save_pvd:
            for i in range(self.form_handler.state_dim):
                if (
                    self.form_handler.state_spaces[i].num_sub_spaces() > 0
                    and self.form_handler.state_spaces[i].ufl_element().family()
                    == "Mixed"
                ):
                    for j in range(self.form_handler.state_spaces[i].num_sub_spaces()):
                        self.state_pvd_list[i][j] << (
                            self.form_handler.states[i].sub(j, True),
                            float(self.iteration),
                        )
                else:
                    self.state_pvd_list[i] << (
                        self.form_handler.states[i],
                        float(self.iteration),
                    )

        if self.save_pvd_adjoint:
            for i in range(self.form_handler.state_dim):
                if (
                    self.form_handler.adjoint_spaces[i].num_sub_spaces() > 0
                    and self.form_handler.adjoint_spaces[i].ufl_element().family()
                    == "Mixed"
                ):
                    for j in range(
                        self.form_handler.adjoint_spaces[i].num_sub_spaces()
                    ):
                        self.adjoint_pvd_list[i][j] << (
                            self.form_handler.adjoints[i].sub(j, True),
                            float(self.iteration),
                        )
                else:
                    self.adjoint_pvd_list[i] << (
                        self.form_handler.adjoints[i],
                        float(self.iteration),
                    )

    def print_summary(self) -> None:
        """Prints a summary of the optimization to the console

        Returns
        -------
        None
        """

        strs = []
        strs.append("\n")
        strs.append(f"Statistics --- Total iterations:  {self.iteration:4d}")
        strs.append(f" --- Final objective value:  {self.objective_value:.3e}")
        strs.append(f" --- Final gradient norm:  {self.relative_norm:.3e} (rel)")
        strs.append("\n")
        strs.append(
            f"           --- State equations solved:  {self.state_problem.number_of_solves:d}"
        )
        strs.append(
            f" --- Adjoint equations solved:  {self.adjoint_problem.number_of_solves:d}"
        )
        strs.append("\n")

        output = "".join(strs)
        if self.verbose:
            print(output)

        if self.save_txt:
            with open(f"{self.result_dir}/history.txt", "a") as file:
                file.write(output)

    def finalize(self) -> None:
        """Finalizes the optimization algorithm, saves the history (if enabled)

        Returns
        -------
        None

        """

        self.output_dict["initial_gradient_norm"] = self.gradient_norm_initial
        self.output_dict["state_solves"] = self.state_problem.number_of_solves
        self.output_dict["adjoint_solves"] = self.adjoint_problem.number_of_solves
        self.output_dict["iterations"] = self.iteration
        if self.save_results:
            with open(f"{self.result_dir}/history.json", "w") as file:
                json.dump(self.output_dict, file)

    def post_processing(self) -> None:
        """Does a post processing after the optimization algorithm terminates.

        Returns
        -------
        None
        """

        if self.converged:
            self.print_results()
            self.print_summary()
            self.finalize()

        else:
            # maximum iterations reached
            if self.converged_reason == -1:
                self.print_results()
                if self.soft_exit:
                    if self.verbose:
                        print("Maximum number of iterations exceeded.")
                    self.finalize()
                else:
                    self.finalize()
                    raise NotConvergedError(
                        "Optimization Algorithm",
                        "Maximum number of iterations were exceeded.",
                    )

            # Armijo line search failed
            elif self.converged_reason == -2:
                self.iteration -= 1
                if self.soft_exit:
                    if self.verbose:
                        print("Armijo rule failed.")
                    self.finalize()
                else:
                    self.finalize()
                    raise NotConvergedError(
                        "Armijo line search",
                        "Failed to compute a feasible Armijo step.",
                    )

            # Mesh Quality is too low
            elif self.converged_reason == -3:
                self.iteration -= 1
                if self.mesh_handler.do_remesh:
                    info("Mesh quality too low. Performing a remeshing operation.\n")
                    self.mesh_handler.remesh(self)
                else:
                    if self.soft_exit:
                        error("Mesh quality is too low.")
                        self.finalize()
                    else:
                        self.finalize()
                        raise NotConvergedError(
                            "Optimization Algorithm", "Mesh quality is too low."
                        )

            # Iteration for remeshing is the one exceeding the maximum number of iterations
            elif self.converged_reason == -4:
                if self.soft_exit:
                    if self.verbose:
                        print("Maximum number of iterations exceeded.")
                    self.finalize()
                else:
                    self.finalize()
                    raise NotConvergedError(
                        "Optimization Algorithm",
                        "Maximum number of iterations were exceeded.",
                    )

    def nonconvergence(self) -> bool:
        """Checks for nonconvergence of the solution algorithm

        Returns
        -------
        bool
            A flag which is True, when the algorithm did not converge
        """

        if self.iteration >= self.maximum_iterations:
            self.converged_reason = -1
        if self.line_search_broken:
            self.converged_reason = -2
        if self.requires_remeshing:
            self.converged_reason = -3
        if self.remeshing_its:
            self.converged_reason = -4

        if self.converged_reason < 0:
            return True
        else:
            return False

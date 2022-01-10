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

import json
import subprocess
from typing import List, Union, TYPE_CHECKING

import fenics
import numpy as np

from .mesh import write_out_mesh
from .._forms import ControlFormHandler


if TYPE_CHECKING:
    from .._optimization.optimization_problem import OptimizationProblem
    from .._optimization.optimization_algorithms import OptimizationAlgorithm


class ResultManager:
    def __init__(
        self, optimization_problem: OptimizationProblem, result_dir: str
    ) -> None:

        self.config = optimization_problem.config
        self.result_dir = result_dir

        self.save_results = self.config.getboolean(
            "Output", "save_results", fallback=True
        )

        self.output_dict = dict()
        if (
            optimization_problem.is_shape_problem
            and optimization_problem.temp_dict is not None
            and optimization_problem.has_cashocs_remesh_flag
        ):
            self.temp_dict = optimization_problem.temp_dict
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
        else:
            self.output_dict["cost_function_value"] = []
            self.output_dict["gradient_norm"] = []
            self.output_dict["stepsize"] = []
            self.output_dict["MeshQuality"] = []

    def save_to_dict(self, solver: OptimizationAlgorithm) -> None:

        self.output_dict["cost_function_value"].append(solver.objective_value)
        self.output_dict["gradient_norm"].append(solver.relative_norm)
        if solver.form_handler.is_shape_problem:
            self.output_dict["MeshQuality"].append(
                solver.optimization_variable_handler.mesh_handler.current_mesh_quality
            )
        self.output_dict["stepsize"].append(solver.stepsize)

    def save_to_json(self, solver: OptimizationAlgorithm) -> None:

        self.output_dict["initial_gradient_norm"] = solver.gradient_norm_initial
        self.output_dict["state_solves"] = solver.state_problem.number_of_solves
        self.output_dict["adjoint_solves"] = solver.adjoint_problem.number_of_solves
        self.output_dict["iterations"] = solver.iteration
        if self.save_results:
            with open(f"{self.result_dir}/history.json", "w") as file:
                json.dump(self.output_dict, file)


class HistoryManager:
    def __init__(
        self, optimization_problem: OptimizationProblem, result_dir: str
    ) -> None:

        self.result_dir = result_dir

        self.verbose = optimization_problem.config.getboolean(
            "Output", "verbose", fallback=True
        )
        self.save_txt = optimization_problem.config.getboolean(
            "Output", "save_txt", fallback=True
        )

    @staticmethod
    def generate_output_str(solver: OptimizationAlgorithm) -> str:

        iteration = solver.iteration
        objective_value = solver.objective_value
        if iteration == 0:
            gradient_norm = solver.gradient_norm_initial
            abs_rel_str = "abs"
        else:
            gradient_norm = solver.relative_norm
            abs_rel_str = "rel"

        if not (np.any(solver.require_control_constraints)):
            gradient_str = "Gradient norm"
        else:
            gradient_str = "Stationarity measure"

        if solver.form_handler.is_shape_problem:
            mesh_quality = (
                solver.optimization_variable_handler.mesh_handler.current_mesh_quality
            )
            mesh_quality_measure = (
                solver.optimization_variable_handler.mesh_handler.mesh_quality_measure
            )
        else:
            mesh_quality = None

        strs = []
        strs.append(f"Iteration {iteration:4d} - ")
        strs.append(f" Objective value:  {objective_value:.3e}")
        strs.append(f"    {gradient_str}:  {gradient_norm:.3e} ({abs_rel_str})")
        if mesh_quality is not None:
            strs.append(
                f"    Mesh Quality:  {mesh_quality:1.2f} ({mesh_quality_measure})"
            )
        if iteration > 0:
            strs.append(f"    Step size:  {solver.stepsize:.3e}")
        if iteration == 0:
            strs.append("\n")

        return "".join(strs)

    def print_to_console(self, solver: OptimizationAlgorithm) -> None:
        if self.verbose:
            print(self.generate_output_str(solver))

    def print_to_file(self, solver: OptimizationAlgorithm) -> None:
        if self.save_txt:
            if solver.iteration == 0:
                file_attr = "w"
            else:
                file_attr = "a"

            with open(f"{self.result_dir}/history.txt", file_attr) as file:
                file.write(f"{self.generate_output_str(solver)}\n")

    @staticmethod
    def generate_summary_str(solver: OptimizationAlgorithm) -> str:
        strs = []
        strs.append("\n")
        strs.append(f"Statistics --- Total iterations:  {solver.iteration:4d}")
        strs.append(f" --- Final objective value:  {solver.objective_value:.3e}")
        strs.append(f" --- Final gradient norm:  {solver.relative_norm:.3e} (rel)")
        strs.append("\n")
        strs.append(
            f"           --- State equations solved:  {solver.state_problem.number_of_solves:d}"
        )
        strs.append(
            f" --- Adjoint equations solved:  {solver.adjoint_problem.number_of_solves:d}"
        )
        strs.append("\n")

        return "".join(strs)

    def print_console_summary(self, solver: OptimizationAlgorithm) -> None:
        if self.verbose:
            print(self.generate_summary_str(solver))

    def print_file_summary(self, solver: OptimizationAlgorithm) -> None:
        if self.save_txt:
            with open(f"{self.result_dir}/history.txt", "a") as file:
                file.write(self.generate_summary_str(solver))


class TempFileManager:
    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        self.config = optimization_problem.config
        self.is_shape_problem = optimization_problem.is_shape_problem

    def clear_temp_files(self, solver: OptimizationAlgorithm) -> None:

        if self.is_shape_problem:
            mesh_handler = solver.optimization_variable_handler.mesh_handler
            if mesh_handler.do_remesh and not self.config.getboolean(
                "Debug", "remeshing", fallback=False
            ):
                subprocess.run(
                    ["rm", "-r", mesh_handler.temp_dir],
                    check=True,
                )
                subprocess.run(
                    ["rm", "-r", mesh_handler.remesh_directory],
                    check=True,
                )


class MeshManager:
    def __init__(
        self, optimization_problem: OptimizationProblem, result_dir: str
    ) -> None:

        self.config = optimization_problem.config
        self.result_dir = result_dir

    def save_optimized_mesh(self, solver: OptimizationAlgorithm) -> None:

        if solver.form_handler.is_shape_problem:
            if solver.optimization_variable_handler.mesh_handler.save_optimized_mesh:
                write_out_mesh(
                    solver.optimization_variable_handler.mesh_handler.mesh,
                    solver.optimization_variable_handler.mesh_handler.gmsh_file,
                    f"{self.result_dir}/optimized_mesh.msh",
                )


class PVDFileManager:
    def __init__(
        self, optimization_problem: OptimizationProblem, result_dir: str
    ) -> None:

        self.form_handler = optimization_problem.form_handler
        self.config = optimization_problem.config

        self.result_dir = result_dir

        self.save_pvd = self.config.getboolean("Output", "save_pvd", fallback=False)
        self.save_pvd_adjoint = self.config.getboolean(
            "Output", "save_pvd_adjoint", fallback=False
        )
        self.save_pvd_gradient = self.config.getboolean(
            "Output", "save_pvd_gradient", fallback=False
        )

        self.is_control_problem = False
        self.is_shape_problem = False
        if isinstance(self.form_handler, ControlFormHandler):
            self.is_control_problem = True
        else:
            self.is_shape_problem = True

        self.pvd_prefix = ""

        self.has_output = (
            self.save_pvd or self.save_pvd_adjoint or self.save_pvd_gradient
        )
        self.is_initialized = False

        self.state_pvd_list = []
        self.control_pvd_list = []
        self.adjoint_pvd_list = []
        self.gradient_pvd_list = []

    def _initialize_pvd_lists(self) -> None:

        if not self.is_initialized:
            if self.save_pvd:
                for i in range(self.form_handler.state_dim):
                    self.state_pvd_list.append(
                        self._generate_pvd_file(
                            self.form_handler.state_spaces[i],
                            f"state_{i:d}",
                            self.pvd_prefix,
                        )
                    )

                if self.is_control_problem:
                    for i in range(self.form_handler.control_dim):
                        self.control_pvd_list.append(
                            self._generate_pvd_file(
                                self.form_handler.control_spaces[i],
                                f"control_{i:d}",
                                self.pvd_prefix,
                            )
                        )

            if self.save_pvd_adjoint:
                for i in range(self.form_handler.state_dim):
                    self.adjoint_pvd_list.append(
                        self._generate_pvd_file(
                            self.form_handler.adjoint_spaces[i],
                            f"adjoint_{i:d}",
                            self.pvd_prefix,
                        )
                    )

            if self.save_pvd_gradient:

                for i in range(self.form_handler.control_dim):
                    if self.is_control_problem:
                        gradient_str = f"gradient_{i:d}"
                    else:
                        gradient_str = f"shape_gradient"
                    self.gradient_pvd_list.append(
                        self._generate_pvd_file(
                            self.form_handler.control_spaces[i],
                            gradient_str,
                            self.pvd_prefix,
                        )
                    )

            self.is_initialized = True

    def set_remesh(self, remesh_counter) -> None:
        self.pvd_prefix = f"remesh_{remesh_counter:d}_"

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
        fenics.File or list[fenics.File]
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

    def save_to_file(self, solver: OptimizationAlgorithm) -> None:

        self._initialize_pvd_lists()

        iteration = solver.iteration

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
                            float(iteration),
                        )
                else:
                    self.state_pvd_list[i] << (
                        self.form_handler.states[i],
                        float(iteration),
                    )

            if self.is_control_problem:
                for i in range(self.form_handler.control_dim):
                    self.control_pvd_list[i] << self.form_handler.controls[i], iteration

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
                            float(iteration),
                        )
                else:
                    self.adjoint_pvd_list[i] << (
                        self.form_handler.adjoints[i],
                        float(iteration),
                    )

        if self.save_pvd_gradient:
            for i in range(self.form_handler.control_dim):
                self.gradient_pvd_list[i] << self.form_handler.gradient[i], iteration

# Copyright (C) 2020-2022 Sebastian Blauth
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


"""
module for output handling
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Union, TYPE_CHECKING

import fenics
import numpy as np

from ._forms import ControlFormHandler
from .utils import write_out_mesh


if TYPE_CHECKING:
    from ._interfaces import OptimizationProblem
    from ._interfaces import OptimizationAlgorithm


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
        try:
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
        except (TypeError, KeyError, AttributeError):
            self.output_dict["cost_function_value"] = []
            self.output_dict["gradient_norm"] = []
            self.output_dict["stepsize"] = []
            self.output_dict["MeshQuality"] = []

    def save_to_dict(self, solver: OptimizationAlgorithm) -> None:

        self.output_dict["cost_function_value"].append(solver.objective_value)
        self.output_dict["gradient_norm"].append(solver.relative_norm)
        try:
            self.output_dict["MeshQuality"].append(
                solver.mesh_handler.current_mesh_quality
            )
        except AttributeError:
            pass
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

        try:
            mesh_quality = solver.mesh_handler.current_mesh_quality
            mesh_quality_measure = solver.mesh_handler.mesh_quality_measure
        except AttributeError:
            mesh_quality = None
            mesh_quality_measure = None

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
                    if (
                        self.form_handler.control_spaces[i].num_sub_spaces() > 0
                        and self.form_handler.control_spaces[i].ufl_element().family()
                        == "Mixed"
                    ):
                        for j in range(
                            self.form_handler.control_spaces[i].num_sub_spaces()
                        ):
                            self.control_pvd_list[i][j] << self.form_handler.controls[
                                i
                            ].sub(j, True), iteration
                    else:
                        self.control_pvd_list[i] << self.form_handler.controls[
                            i
                        ], iteration

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
                if (
                    self.form_handler.control_spaces[i].num_sub_spaces() > 0
                    and self.form_handler.control_spaces[i].ufl_element().family()
                    == "Mixed"
                ):
                    for j in range(
                        self.form_handler.control_spaces[i].num_sub_spaces()
                    ):
                        self.gradient_pvd_list[i][j] << self.form_handler.gradient[
                            i
                        ].sub(j, True), iteration
                else:
                    self.gradient_pvd_list[i] << self.form_handler.gradient[
                        i
                    ], iteration


class TempFileManager:
    def __init__(self, optimization_problem: OptimizationProblem) -> None:
        self.config = optimization_problem.config

    def clear_temp_files(self, solver: OptimizationAlgorithm) -> None:

        try:
            if not self.config.getboolean("Debug", "remeshing", fallback=False):
                subprocess.run(["rm", "-r", solver.temp_dir], check=True)
                subprocess.run(
                    ["rm", "-r", solver.mesh_handler.remesh_directory], check=True
                )
        except AttributeError:
            pass


class MeshManager:
    def __init__(
        self, optimization_problem: OptimizationProblem, result_dir: str
    ) -> None:

        self.config = optimization_problem.config
        self.result_dir = result_dir

    def save_optimized_mesh(self, solver: OptimizationAlgorithm) -> None:

        try:
            if solver.mesh_handler.save_optimized_mesh:
                write_out_mesh(
                    solver.mesh_handler.mesh,
                    solver.mesh_handler.gmsh_file,
                    f"{self.result_dir}/optimized_mesh.msh",
                )
        except AttributeError:
            pass


class OutputManager:
    def __init__(self, optimization_problem: OptimizationProblem) -> None:

        self.config = optimization_problem.config
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

        save_txt = self.config.getboolean("Output", "save_txt", fallback=True)
        save_results = self.config.getboolean("Output", "save_results", fallback=True)
        save_pvd = self.config.getboolean("Output", "save_pvd", fallback=False)
        save_pvd_adjoint = self.config.getboolean(
            "Output", "save_pvd_adjoint", fallback=False
        )
        save_pvd_gradient = self.config.getboolean(
            "Output", "save_pvd_gradient", fallback=False
        )
        has_output = (
            save_txt
            or save_results
            or save_pvd
            or save_pvd_gradient
            or save_pvd_adjoint
        )

        if not os.path.isdir(self.result_dir):
            if has_output:
                Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        self.history_manager = HistoryManager(optimization_problem, self.result_dir)
        self.pvd_file_manager = PVDFileManager(optimization_problem, self.result_dir)
        self.result_manager = ResultManager(optimization_problem, self.result_dir)
        self.mesh_manager = MeshManager(optimization_problem, self.result_dir)
        self.temp_file_manager = TempFileManager(optimization_problem)

    def output(self, solver: OptimizationAlgorithm) -> None:

        self.history_manager.print_to_console(solver)
        self.history_manager.print_to_file(solver)

        self.pvd_file_manager.save_to_file(solver)

        self.result_manager.save_to_dict(solver)

    def output_summary(self, solver: OptimizationAlgorithm) -> None:

        self.history_manager.print_console_summary(solver)
        self.history_manager.print_file_summary(solver)

        self.result_manager.save_to_json(solver)

        self.mesh_manager.save_optimized_mesh(solver)

        self.temp_file_manager.clear_temp_files(solver)

    def set_remesh(self, remesh_counter: int) -> None:
        self.pvd_file_manager.set_remesh(remesh_counter)

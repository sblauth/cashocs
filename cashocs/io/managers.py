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

"""Output managers for cashocs."""

from __future__ import annotations

import json
import subprocess  # nosec B404
from typing import cast, List, TYPE_CHECKING, Union

import fenics
import numpy as np

from cashocs import _forms
from cashocs.io import mesh as iomesh

if TYPE_CHECKING:
    from cashocs import _optimization as op
    from cashocs._optimization import optimization_algorithms


def generate_summary_str(
    solver: optimization_algorithms.OptimizationAlgorithm,
) -> str:
    """Generates a string for the summary of the optimization.

    Args:
        solver: The optimization algorithm.

    Returns:
        The summary string.

    """
    strs = [
        "\n",
        "Optimization was successful.\n",
        "Statistics:\n",
        f"    total iterations: {solver.iteration:4d}\n",
        f"    final objective value: {solver.objective_value:>10.3e}\n",
        f"    final gradient norm:   {solver.relative_norm:>10.3e}\n",
        f"    total number of state systems solved:   "
        f"{solver.state_problem.number_of_solves:4d}\n",
        f"    total number of adjoint systems solved: "
        f"{solver.adjoint_problem.number_of_solves:4d}\n",
    ]

    return "".join(strs)


def generate_output_str(
    solver: optimization_algorithms.OptimizationAlgorithm,
) -> str:
    """Generates the string which can be written to console and file.

    Args:
        solver: The optimization algorithm.

    Returns:
        The output string, which is used later.

    """
    iteration = solver.iteration
    objective_value = solver.objective_value

    if not np.any(solver.form_handler.require_control_constraints):
        gradient_str = "grad. norm"
    else:
        gradient_str = "stat. meas."

    if solver.form_handler.is_shape_problem:
        mesh_handler = solver.optimization_variable_abstractions.mesh_handler
        mesh_quality = mesh_handler.current_mesh_quality
    else:
        mesh_quality = None

    if iteration % 10 == 0:
        info_str = (
            "\niter,  "
            "cost function,  "
            f"rel. {gradient_str},  "
            f"abs. {gradient_str},  "
        )
        if mesh_quality is not None:
            info_str += "mesh qlty,  "
        info_str += "step size\n\n"
    else:
        info_str = ""

    strs = [
        f"{iteration:4d},  ",
        f"{objective_value:>13.3e},  ",
        f"{solver.relative_norm:>{len(gradient_str) + 5}.3e},  ",
        f"{solver.gradient_norm:>{len(gradient_str) + 5}.3e},  ",
    ]
    if mesh_quality is not None:
        strs.append(f"{mesh_quality:>9.2f},  ")

    if iteration > 0:
        strs.append(f"{solver.stepsize:>9.3e}")
    if iteration == 0:
        strs.append("\n")

    return info_str + "".join(strs)


class ResultManager:
    """Class for managing the output of the optimization history."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, result_dir: str
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            result_dir: Path to the directory, where the results are saved.

        """
        self.config = optimization_problem.config
        self.result_dir = result_dir

        self.save_results = self.config.getboolean("Output", "save_results")

        self.output_dict = {}
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

    def save_to_dict(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Saves the optimization history to a dictionary.

        Args:
            solver: The optimization algorithm.

        """
        self.output_dict["cost_function_value"].append(solver.objective_value)
        self.output_dict["gradient_norm"].append(solver.relative_norm)
        if solver.form_handler.is_shape_problem:
            mesh_handler = solver.optimization_variable_abstractions.mesh_handler
            self.output_dict["MeshQuality"].append(mesh_handler.current_mesh_quality)
        self.output_dict["stepsize"].append(solver.stepsize)

    def save_to_json(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Saves the history of the optimization to a .json file.

        Args:
            solver: The optimization algorithm.

        """
        self.output_dict["initial_gradient_norm"] = solver.gradient_norm_initial
        self.output_dict["state_solves"] = solver.state_problem.number_of_solves
        self.output_dict["adjoint_solves"] = solver.adjoint_problem.number_of_solves
        self.output_dict["iterations"] = solver.iteration
        if self.save_results and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            with open(f"{self.result_dir}/history.json", "w", encoding="utf-8") as file:
                json.dump(self.output_dict, file)
        fenics.MPI.barrier(fenics.MPI.comm_world)


class HistoryManager:
    """Class for managing the human-readable output of cashocs."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, result_dir: str
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            result_dir: Path to the directory, where the results are saved.

        """
        self.result_dir = result_dir

        self.verbose = optimization_problem.config.getboolean("Output", "verbose")
        self.save_txt = optimization_problem.config.getboolean("Output", "save_txt")

    def print_to_console(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Prints the output string to the console.

        Args:
            solver: The optimization algorithm.

        """
        if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(generate_output_str(solver), flush=True)
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def print_to_file(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Saves the output string in a file.

        Args:
            solver: The optimization algorithm.

        """
        if self.save_txt and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            if solver.iteration == 0:
                file_attr = "w"
            else:
                file_attr = "a"

            with open(
                f"{self.result_dir}/history.txt", file_attr, encoding="utf-8"
            ) as file:
                file.write(f"{generate_output_str(solver)}\n")
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def print_console_summary(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Prints the summary in the console.

        Args:
            solver: The optimization algorithm.

        """
        if self.verbose and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(generate_summary_str(solver), flush=True)
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def print_file_summary(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Save the summary in a file.

        Args:
            solver: The optimization algorithm.

        """
        if self.save_txt and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            with open(f"{self.result_dir}/history.txt", "a", encoding="utf-8") as file:
                file.write(generate_summary_str(solver))
        fenics.MPI.barrier(fenics.MPI.comm_world)


class TempFileManager:
    """Class for managing temporary files."""

    def __init__(self, optimization_problem: op.OptimizationProblem) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.

        """
        self.config = optimization_problem.config
        self.is_shape_problem = optimization_problem.is_shape_problem

    def clear_temp_files(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Deletes temporary files.

        Args:
            solver: The optimization algorithm.

        """
        if self.is_shape_problem:
            mesh_handler = solver.optimization_variable_abstractions.mesh_handler
            if (
                mesh_handler.do_remesh
                and not self.config.getboolean("Debug", "remeshing")
                and mesh_handler.temp_dict is not None
                and fenics.MPI.rank(fenics.MPI.comm_world) == 0
            ):
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", mesh_handler.temp_dict["temp_dir"]], check=True
                )
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", mesh_handler.remesh_directory], check=True
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)


class MeshManager:
    """Manages the output of meshes."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, result_dir: str
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            result_dir: Path to the directory, where the output is saved to.

        """
        self.config = optimization_problem.config
        self.result_dir = result_dir

    def save_optimized_mesh(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Saves a copy of the optimized mesh in Gmsh format.

        Args:
            solver: The optimization algorithm.

        """
        if solver.form_handler.is_shape_problem:
            mesh_handler = solver.optimization_variable_abstractions.mesh_handler
            if mesh_handler.save_optimized_mesh:
                iomesh.write_out_mesh(
                    mesh_handler.mesh,
                    mesh_handler.gmsh_file,
                    f"{self.result_dir}/optimized_mesh.msh",
                )


class XDMFFileManager:
    """Class for managing visualization .xdmf files."""

    def __init__(
        self, optimization_problem: op.OptimizationProblem, result_dir: str
    ) -> None:
        """Initializes self.

        Args:
            optimization_problem: The corresponding optimization problem.
            result_dir: Path to the directory, where the output files are saved in.

        """
        self.form_handler = optimization_problem.form_handler
        self.config = optimization_problem.config

        self.result_dir = result_dir

        self.save_state = self.config.getboolean("Output", "save_state")
        self.save_adjoint = self.config.getboolean("Output", "save_adjoint")
        self.save_gradient = self.config.getboolean("Output", "save_gradient")

        self.is_control_problem = False
        self.is_shape_problem = False
        if isinstance(self.form_handler, _forms.ControlFormHandler):
            self.is_control_problem = True
        else:
            self.is_shape_problem = True

        self.has_output = self.save_state or self.save_adjoint or self.save_gradient
        self.is_initialized = False

        self.state_xdmf_list: List[Union[str, List[str]]] = []
        self.control_xdmf_list: List[Union[str, List[str]]] = []
        self.adjoint_xdmf_list: List[Union[str, List[str]]] = []
        self.gradient_xdmf_list: List[Union[str, List[str]]] = []

    def _initialize_states_xdmf(self) -> None:
        """Initializes the list of xdmf files for the state variables."""
        if self.save_state:
            for i in range(self.form_handler.state_dim):
                self.state_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.form_handler.state_spaces[i], f"state_{i:d}"
                    )
                )

    def _initialize_controls_xdmf(self) -> None:
        """Initializes the list of xdmf files for the control variables."""
        if self.save_state and self.is_control_problem:
            for i in range(self.form_handler.control_dim):
                self.control_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.form_handler.control_spaces[i], f"control_{i:d}"
                    )
                )

    def _initialize_adjoints_xdmf(self) -> None:
        """Initialize the list of xdmf files for the adjoint variables."""
        if self.save_adjoint:
            for i in range(self.form_handler.state_dim):
                self.adjoint_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.form_handler.adjoint_spaces[i], f"adjoint_{i:d}"
                    )
                )

    def _initialize_gradients_xdmf(self) -> None:
        """Initialize the list of xdmf files for the gradients."""
        if self.save_gradient:
            for i in range(self.form_handler.control_dim):
                if self.is_control_problem:
                    gradient_str = f"gradient_{i:d}"
                else:
                    gradient_str = "shape_gradient"
                self.gradient_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.form_handler.control_spaces[i], gradient_str
                    )
                )

    def _initialize_xdmf_lists(self) -> None:
        """Initializes the lists of xdmf files."""
        if not self.is_initialized:
            self._initialize_states_xdmf()
            self._initialize_controls_xdmf()
            self._initialize_adjoints_xdmf()
            self._initialize_gradients_xdmf()

            self.is_initialized = True

    def _generate_xdmf_file_strings(
        self, space: fenics.FunctionSpace, name: str
    ) -> Union[str, List[str]]:
        """Generate the strings (paths) to the xdmf files for visualization.

        Args:
            space: The FEM function space where the function is taken from.
            name: The name of the function / file

        Returns:
            A string containing the path to the xdmf files for visualization.

        """
        if space.num_sub_spaces() > 0 and space.ufl_element().family() == "Mixed":
            lst = []
            for j in range(space.num_sub_spaces()):
                lst.append(f"{self.result_dir}/xdmf/{name}_{j:d}.xdmf")
            return lst
        else:
            file = f"{self.result_dir}/xdmf/{name}.xdmf"
            return file

    def _save_states(self, iteration: int) -> None:
        """Saves the state variables to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_state:
            for i in range(self.form_handler.state_dim):
                if isinstance(self.state_xdmf_list[i], list):
                    for j in range(len(self.state_xdmf_list[i])):
                        self._write_xdmf_step(
                            self.state_xdmf_list[i][j],
                            self.form_handler.states[i].sub(j, True),
                            f"state_{i}_sub_{j}",
                            iteration,
                        )
                else:
                    self._write_xdmf_step(
                        cast(str, self.state_xdmf_list[i]),
                        self.form_handler.states[i],
                        f"state_{i}",
                        iteration,
                    )

    def _save_controls(self, iteration: int) -> None:
        """Saves the control variables to xdmf.

        Args:
            iteration: The current iteration count.

        """
        if self.save_state and self.is_control_problem:
            for i in range(self.form_handler.control_dim):
                self._write_xdmf_step(
                    cast(str, self.control_xdmf_list[i]),
                    self.form_handler.controls[i],
                    f"control_{i}",
                    iteration,
                )

    def _save_adjoints(self, iteration: int) -> None:
        """Saves the adjoint variables to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_adjoint:
            for i in range(self.form_handler.state_dim):
                if isinstance(self.adjoint_xdmf_list[i], list):
                    for j in range(len(self.adjoint_xdmf_list[i])):
                        self._write_xdmf_step(
                            self.adjoint_xdmf_list[i][j],
                            self.form_handler.adjoints[i].sub(j, True),
                            f"adjoint_{i}_sub_{j}",
                            iteration,
                        )
                else:
                    self._write_xdmf_step(
                        cast(str, self.adjoint_xdmf_list[i]),
                        self.form_handler.adjoints[i],
                        f"adjoint_{i}",
                        iteration,
                    )

    def _save_gradients(self, iteration: int) -> None:
        """Saves the gradients to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_gradient:
            for i in range(self.form_handler.control_dim):
                self._write_xdmf_step(
                    cast(str, self.gradient_xdmf_list[i]),
                    self.form_handler.gradient[i],
                    f"gradient_{i}",
                    iteration,
                )

    def _write_xdmf_step(
        self,
        filename: str,
        function: fenics.Function,
        function_name: str,
        iteration: int,
    ) -> None:
        """Write the current function to the corresponding xdmf file for visualization.

        Args:
            filename: The path to the xdmf file.
            function: The function which is to be stored.
            function_name: The label of the function in the xdmf file.
            iteration: The current iteration counter.

        """
        if iteration == 0:
            append = False
        else:
            append = True

        function.rename(function_name, function_name)

        with fenics.XDMFFile(fenics.MPI.comm_world, filename) as file:
            file.parameters["flush_output"] = True
            file.parameters["functions_share_mesh"] = False
            file.write_checkpoint(
                function,
                function_name,
                iteration,
                fenics.XDMFFile.Encoding.HDF5,
                append,
            )

    def save_to_file(
        self, solver: optimization_algorithms.OptimizationAlgorithm
    ) -> None:
        """Saves the variables to xdmf files.

        Args:
            solver: The optimization algorithm.

        """
        self._initialize_xdmf_lists()

        iteration = solver.iteration

        self._save_states(iteration)
        self._save_controls(iteration)
        self._save_adjoints(iteration)
        self._save_gradients(iteration)

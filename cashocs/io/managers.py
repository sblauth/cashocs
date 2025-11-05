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

"""Output managers for cashocs."""

from __future__ import annotations

import abc
import json
import os
import shutil
import subprocess
from typing import cast, TYPE_CHECKING

import fenics

from cashocs import log
from cashocs.io import mesh as iomesh

if TYPE_CHECKING:
    from cashocs._database import database


output_mapping = {
    "iteration": "iter",
    "objective_value": "cost function",
    "relative_norm": "rel. grad. norm",
    "gradient_norm": "abs. grad. norm",
    "mesh_quality": "mesh qlty",
    "angle": "angle",
    "stepsize": "step size",
    "constraint_violation": "constraint violation",
    "mu": "mu",
}


def generate_summary_str(db: database.Database, precision: int) -> str:
    """Generates a string for the summary of the optimization.

    Args:
        db: The database of the problem.
        precision: The precision used for displaying the numbers.

    Returns:
        The summary string.

    """
    optimization_state = db.parameter_db.optimization_state

    summary_str_list = [
        "\n",
        "Optimization was successful.\n",
        "Statistics:\n",
        f"    total iterations: {optimization_state['iteration']:4d}\n",
    ]
    for key, value in output_mapping.items():
        if key in optimization_state.keys() and key != "iteration":
            parameter_name = value
            parameter_value = optimization_state[key]
            summary_str_list.append(
                f"    final {parameter_name}: {parameter_value:.{precision}e}\n"
            )

    if "no_state_solves" in optimization_state.keys():
        summary_str_list.append(
            "    total number of state systems solved: "
            f"{optimization_state['no_state_solves']:4d}\n"
        )
    if "no_adjoint_solves" in optimization_state.keys():
        summary_str_list.append(
            "    total number of adjoint systems solved: "
            f"{optimization_state['no_adjoint_solves']:4d}\n"
        )

    return "".join(summary_str_list)


def generate_output_str(db: database.Database, precision: int) -> str:
    """Generates the string which can be written to console and file.

    Args:
        db: The database of the problem.
        precision: The precision used for displaying the numbers.

    Returns:
        The output string, which is used later.

    """
    optimization_state = db.parameter_db.optimization_state

    iteration = optimization_state["iteration"]

    info_str_list = []
    output_str_list = []
    if iteration % 10 == 0:
        info_str_list.append("\n")

    for key, value in output_mapping.items():
        if key in optimization_state.keys():
            if key == "iteration":
                if iteration % 10 == 0:
                    info_str_list.append("iter,  ")
                output_str_list.append(f"{iteration:4d},  ")
            else:
                parameter_value = optimization_state[key]
                parameter_name = value

                temp_name_str = f"{parameter_name},  "
                temp_value_str = f"{parameter_value:.{precision}e},  "
                string_length = max(len(temp_name_str), len(temp_value_str))

                if iteration % 10 == 0:
                    info_str_list.append(f"{parameter_name},  ".rjust(string_length))

                output_str_list.append(
                    f"{parameter_value:>{string_length - 3}.{precision}e},  "
                )
        else:
            continue

    if iteration % 10 == 0:
        info_str_list.append("\n\n")

    if iteration == 0:
        output_str_list.append("\n")

    info_str = "".join(info_str_list)
    output_str = "".join(output_str_list)

    return info_str + output_str


class IOManager(abc.ABC):
    """Abstract base class for input / output management."""

    def __init__(self, db: database.Database, result_dir: str) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: Path to the directory, where the results are saved.

        """
        self.db = db
        self.comm = self.db.geometry_db.mpi_comm
        self.result_dir = result_dir

        self.config = self.db.config

    @abc.abstractmethod
    def output(self) -> None:
        """The output operation, which is performed after every iteration.

        Args:
            solver: The optimization algorithm.

        """
        pass

    @abc.abstractmethod
    def output_summary(self) -> None:
        """The output operation, which is performed after convergence.

        Args:
            solver: The optimization algorithm.

        """
        pass

    @abc.abstractmethod
    def post_process(self) -> None:
        """The output operation which is performed as part of the postprocessing.

        Args:
            solver: The optimization algorithm.

        """
        pass


class ResultManager(IOManager):
    """Class for managing the output of the optimization history."""

    def __init__(self, db: database.Database, result_dir: str) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: Path to the directory, where the results are saved.

        """
        super().__init__(db, result_dir)

        self.save_results = self.config.getboolean("Output", "save_results")

        self.output_dict = {}
        if self.db.parameter_db.temp_dict:
            for key in self.db.parameter_db.temp_dict["output_dict"].keys():
                self.output_dict[key] = self.db.parameter_db.temp_dict["output_dict"][
                    key
                ]

    def output(self) -> None:
        """Saves the optimization history to a dictionary."""
        for key in self.db.parameter_db.optimization_state.keys():
            if key not in self.output_dict:
                self.output_dict[key] = []

            self.output_dict[key].append(self.db.parameter_db.optimization_state[key])

    def output_summary(self) -> None:
        """The output operation, which is performed after convergence.

        Args:
            solver: The optimization algorithm.

        """
        pass

    def post_process(self) -> None:
        """Saves the history of the optimization to a .json file."""
        if self.save_results and self.comm.rank == 0:
            with open(f"{self.result_dir}/history.json", "w", encoding="utf-8") as file:
                json.dump(self.output_dict, file, indent=4)
        self.comm.barrier()


class ConsoleManager(IOManager):
    """Management of the console output."""

    def __init__(
        self, db: database.Database, result_dir: str, verbose: bool = False
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: The directory, where the results are written to.
            verbose: Boolean which indicates whether the logging setup (False) or the
                old setup with print (True) should be used. Default is `False`.

        """
        super().__init__(db, result_dir)
        self.verbose = verbose
        self.precision = self.config.getint("Output", "precision")

    def output(self) -> None:
        """Prints the output string to the console."""
        message = generate_output_str(self.db, self.precision)
        if self.verbose:
            if self.comm.rank == 0:
                print(message, flush=True)
            self.comm.barrier()
        else:
            log.info(message)

    def output_summary(self) -> None:
        """Prints the summary in the console."""
        message = generate_summary_str(self.db, self.precision)
        if self.verbose:
            if self.comm.rank == 0:
                print(message, flush=True)
            self.comm.barrier()
        else:
            log.info(message)

    def post_process(self) -> None:
        """The output operation which is performed as part of the postprocessing.

        Args:
            solver: The optimization algorithm.

        """
        pass


class FileManager(IOManager):
    """Class for managing the human-readable output of cashocs."""

    def __init__(self, db: database.Database, result_dir: str) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: The directory, where the results are written to.

        """
        super().__init__(db, result_dir)
        self.precision = self.config.getint("Output", "precision")

    def output(self) -> None:
        """Saves the output string in a file."""
        if self.comm.rank == 0:
            if self.db.parameter_db.optimization_state["iteration"] == 0:
                file_attr = "w"
            else:
                file_attr = "a"

            with open(
                f"{self.result_dir}/history.txt", file_attr, encoding="utf-8"
            ) as file:
                file.write(f"{generate_output_str(self.db, self.precision)}\n")
        self.comm.barrier()

    def output_summary(self) -> None:
        """Save the summary in a file."""
        if self.comm.rank == 0:
            with open(f"{self.result_dir}/history.txt", "a", encoding="utf-8") as file:
                file.write(generate_summary_str(self.db, self.precision))
        self.comm.barrier()

    def post_process(self) -> None:
        """The output operation which is performed as part of the postprocessing.

        Args:
            solver: The optimization algorithm.

        """
        pass


class TempFileManager(IOManager):
    """Class for managing temporary files."""

    def output(self) -> None:
        """The output operation, which is performed after every iteration.

        Args:
            solver: The optimization algorithm.

        """
        pass

    def output_summary(self) -> None:
        """The output operation, which is performed after convergence.

        Args:
            solver: The optimization algorithm.

        """
        pass

    def post_process(self) -> None:
        """Deletes temporary files."""
        if self.db.parameter_db.problem_type == "shape":
            if (
                self.config.getboolean("Mesh", "remesh")
                and not self.config.getboolean("Debug", "remeshing")
                and self.db.parameter_db.temp_dict
                and self.comm.rank == 0
            ):
                subprocess.run(  # noqa: S603
                    ["rm", "-r", self.db.parameter_db.remesh_directory],  # noqa: S607
                    check=True,
                )
            self.comm.barrier()


class MeshManager(IOManager):
    """Manages the output of meshes."""

    def output(self) -> None:
        """Saves the mesh as checkpoint for each iteration."""
        iteration = int(self.db.parameter_db.optimization_state["iteration"])

        if not self.db.parameter_db.gmsh_file_path:
            gmsh_file = self.config.get("Mesh", "gmsh_file")
        else:
            gmsh_file = self.db.parameter_db.gmsh_file_path

        iomesh.write_out_mesh(
            self.db.geometry_db.mesh,
            gmsh_file,
            f"{self.result_dir}/checkpoints/mesh/mesh_{iteration}.msh",
        )

    def output_summary(self) -> None:
        """The output operation, which is performed after convergence.

        Args:
            solver: The optimization algorithm.

        """
        pass

    def post_process(self) -> None:
        """Saves a copy of the optimized mesh in Gmsh format."""
        if not self.db.parameter_db.gmsh_file_path:
            gmsh_file = self.config.get("Mesh", "gmsh_file")
        else:
            gmsh_file = self.db.parameter_db.gmsh_file_path

        iomesh.write_out_mesh(
            self.db.function_db.states[0].function_space().mesh(),
            gmsh_file,
            f"{self.result_dir}/optimized_mesh.msh",
        )


class XDMFFileManager(IOManager):
    """Class for managing visualization .xdmf files."""

    def __init__(
        self,
        db: database.Database,
        result_dir: str,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: Path to the directory, where the output files are saved in.

        """
        super().__init__(db, result_dir)

        self.save_state = self.config.getboolean("Output", "save_state")
        self.save_adjoint = self.config.getboolean("Output", "save_adjoint")
        self.save_gradient = self.config.getboolean("Output", "save_gradient")

        self.is_initialized = False

        self.state_xdmf_list: list[str | list[str]] = []
        self.control_xdmf_list: list[str | list[str]] = []
        self.adjoint_xdmf_list: list[str | list[str]] = []
        self.gradient_xdmf_list: list[str | list[str]] = []

    def _initialize_states_xdmf(self) -> None:
        """Initializes the list of xdmf files for the state variables."""
        if self.save_state:
            for i in range(self.db.parameter_db.state_dim):
                self.state_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.db.function_db.state_spaces[i], f"state_{i:d}"
                    )
                )

    def _initialize_controls_xdmf(self) -> None:
        """Initializes the list of xdmf files for the control variables."""
        if self.save_state and self.db.parameter_db.problem_type in [
            "control",
            "topology",
        ]:
            for i in range(self.db.parameter_db.control_dim):
                self.control_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.db.function_db.control_spaces[i], f"control_{i:d}"
                    )
                )

    def _initialize_adjoints_xdmf(self) -> None:
        """Initialize the list of xdmf files for the adjoint variables."""
        if self.save_adjoint:
            for i in range(self.db.parameter_db.state_dim):
                self.adjoint_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.db.function_db.adjoint_spaces[i], f"adjoint_{i:d}"
                    )
                )

    def _initialize_gradients_xdmf(self) -> None:
        """Initialize the list of xdmf files for the gradients."""
        if self.save_gradient:
            for i in range(self.db.parameter_db.control_dim):
                if self.db.parameter_db.problem_type in ["control", "topology"]:
                    gradient_str = f"gradient_{i:d}"
                else:
                    gradient_str = "shape_gradient"
                self.gradient_xdmf_list.append(
                    self._generate_xdmf_file_strings(
                        self.db.function_db.control_spaces[i], gradient_str
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
    ) -> str | list[str]:
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
                lst.append(f"{self.result_dir}/checkpoints/{name}_{j:d}.xdmf")
            return lst
        else:
            file = f"{self.result_dir}/checkpoints/{name}.xdmf"
            return file

    def _save_states(self, iteration: int) -> None:
        """Saves the state variables to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_state:
            for i in range(self.db.parameter_db.state_dim):
                if isinstance(self.state_xdmf_list[i], list):
                    for j in range(len(self.state_xdmf_list[i])):
                        self._write_xdmf_step(
                            self.state_xdmf_list[i][j],
                            self.db.function_db.states[i].sub(j, True),
                            f"state_{i}_sub_{j}",
                            iteration,
                        )
                else:
                    self._write_xdmf_step(
                        cast(str, self.state_xdmf_list[i]),
                        self.db.function_db.states[i],
                        f"state_{i}",
                        iteration,
                    )

    def _save_controls(self, iteration: int) -> None:
        """Saves the control variables to xdmf.

        Args:
            iteration: The current iteration count.

        """
        if self.save_state and self.db.parameter_db.problem_type in [
            "control",
            "topology",
        ]:
            for i in range(len(self.db.function_db.controls)):
                self._write_xdmf_step(
                    cast(str, self.control_xdmf_list[i]),
                    self.db.function_db.controls[i],
                    f"control_{i}",
                    iteration,
                )

    def _save_adjoints(self, iteration: int) -> None:
        """Saves the adjoint variables to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_adjoint:
            for i in range(self.db.parameter_db.state_dim):
                if isinstance(self.adjoint_xdmf_list[i], list):
                    for j in range(len(self.adjoint_xdmf_list[i])):
                        self._write_xdmf_step(
                            self.adjoint_xdmf_list[i][j],
                            self.db.function_db.adjoints[i].sub(j, True),
                            f"adjoint_{i}_sub_{j}",
                            iteration,
                        )
                else:
                    self._write_xdmf_step(
                        cast(str, self.adjoint_xdmf_list[i]),
                        self.db.function_db.adjoints[i],
                        f"adjoint_{i}",
                        iteration,
                    )

    def _save_gradients(self, iteration: int) -> None:
        """Saves the gradients to xdmf files.

        Args:
            iteration: The current iteration count.

        """
        if self.save_gradient:
            for i in range(self.db.parameter_db.control_dim):
                self._write_xdmf_step(
                    cast(str, self.gradient_xdmf_list[i]),
                    self.db.function_db.gradient[i],
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

        mesh = function.function_space().mesh()

        if function.function_space().ufl_element().family() in [
            "Real",
            "NodalEnrichedElement",
        ]:
            if len(function.ufl_shape) > 0:
                space = fenics.VectorFunctionSpace(
                    mesh, "CG", 1, dim=function.ufl_shape[0]
                )
            else:
                space = fenics.FunctionSpace(mesh, "CG", 1)
            function = fenics.interpolate(function, space)

        elif function.function_space().ufl_element().family() == "Crouzeix-Raviart":
            degree = function.function_space().ufl_element().degree()
            if len(function.ufl_shape) > 0:
                space = fenics.VectorFunctionSpace(
                    mesh, "DG", degree, dim=function.ufl_shape[0]
                )
            else:
                space = fenics.FunctionSpace(mesh, "DG", degree)
            function = fenics.interpolate(function, space)

        function.rename(function_name, function_name)

        with fenics.XDMFFile(self.comm, filename) as file:
            file.parameters["flush_output"] = True
            file.parameters["functions_share_mesh"] = False
            file.write_checkpoint(
                function,
                function_name,
                iteration,
                fenics.XDMFFile.Encoding.HDF5,
                append,
            )

    def output(self) -> None:
        """Saves the variables to xdmf files."""
        self._initialize_xdmf_lists()

        iteration = int(self.db.parameter_db.optimization_state["iteration"])

        if iteration == 0:
            if self.comm.rank == 0:
                directory = f"{self.result_dir}/checkpoints/"
                for files in os.listdir(directory):
                    path = os.path.join(directory, files)
                    try:
                        shutil.rmtree(path)
                    except OSError:
                        os.remove(path)

            self.comm.barrier()

        self._save_states(iteration)
        self._save_controls(iteration)
        self._save_adjoints(iteration)
        self._save_gradients(iteration)

    def output_summary(self) -> None:
        """The output operation, which is performed after convergence.

        Args:
            solver: The optimization algorithm.

        """
        pass

    def post_process(self) -> None:
        """The output operation which is performed as part of the postprocessing.

        Args:
            solver: The optimization algorithm.

        """
        pass

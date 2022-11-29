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

import abc
import json
import subprocess  # nosec B404
from typing import cast, List, TYPE_CHECKING, Union

import fenics

from cashocs.io import mesh as iomesh

if TYPE_CHECKING:
    from cashocs._database import database


def generate_summary_str(db: database.Database, precision: int) -> str:
    """Generates a string for the summary of the optimization.

    Args:
        db: The database of the problem.
        precision: The precision used for displaying the numbers.

    Returns:
        The summary string.

    """
    optimization_state = db.parameter_db.optimization_state
    strs = [
        "\n",
        "Optimization was successful.\n",
        "Statistics:\n",
        f"    total iterations: {optimization_state['iteration']:4d}\n",
        "    final objective value: "
        f"{optimization_state['objective_value']:>10.{precision}e}\n",
        "    final gradient norm:   "
        f"{optimization_state['relative_norm']:>10.{precision}e}\n",
        "    total number of state systems solved:   "
        f"{optimization_state['no_state_solves']:4d}\n",
        "    total number of adjoint systems solved: "
        f"{optimization_state['no_adjoint_solves']:4d}\n",
    ]

    return "".join(strs)


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
    objective_value = optimization_state["objective_value"]

    if not db.parameter_db.display_box_constraints:
        gradient_str = "grad. norm"
    else:
        gradient_str = "stat. meas."

    if db.parameter_db.problem_type == "shape":
        mesh_quality = db.parameter_db.optimization_state["mesh_quality"]
    else:
        mesh_quality = None

    if iteration % 10 == 0:
        info_list = [
            "\niter,  ",
            "cost function,  ".rjust(max(16, precision + 10)),
            f"rel. {gradient_str},  ".rjust(max(len(gradient_str) + 7, precision + 9)),
            f"abs. {gradient_str},  ".rjust(max(len(gradient_str) + 7, precision + 9)),
        ]
        if mesh_quality is not None:
            info_list.append("mesh qlty,  ".rjust(max(12, precision + 9)))
        info_list.append("step size".rjust(max(9, precision + 6)))
        info_list.append("\n\n")
        info_str = "".join(info_list)
    else:
        info_str = ""

    strs = [
        f"{iteration:4d},  ",
        f"{objective_value:> 13.{precision}e},  ",
        f"{optimization_state['relative_norm']:>{len(gradient_str)+5}.{precision}e},  ",
        f"{optimization_state['gradient_norm']:>{len(gradient_str)+5}.{precision}e},  ",
    ]
    if mesh_quality is not None:
        strs.append(f"{mesh_quality:>9.{precision}e},  ")

    if iteration > 0:
        strs.append(f"{optimization_state['stepsize']:>9.{precision}e}")
    if iteration == 0:
        strs.append("\n")

    return info_str + "".join(strs)


class IOManager(abc.ABC):
    """Abstract base class for input / output management."""

    def __init__(self, db: database.Database, result_dir: str):
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: Path to the directory, where the results are saved.

        """
        self.db = db
        self.result_dir = result_dir

        self.config = self.db.config
        self.optimization_state = self.db.parameter_db.optimization_state

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
            self.output_dict["cost_function_value"] = self.db.parameter_db.temp_dict[
                "output_dict"
            ]["cost_function_value"]
            self.output_dict["gradient_norm"] = self.db.parameter_db.temp_dict[
                "output_dict"
            ]["gradient_norm"]
            self.output_dict["stepsize"] = self.db.parameter_db.temp_dict[
                "output_dict"
            ]["stepsize"]
            self.output_dict["MeshQuality"] = self.db.parameter_db.temp_dict[
                "output_dict"
            ]["MeshQuality"]
        else:
            self.output_dict["cost_function_value"] = []
            self.output_dict["gradient_norm"] = []
            self.output_dict["stepsize"] = []
            self.output_dict["MeshQuality"] = []

    def output(self) -> None:
        """Saves the optimization history to a dictionary."""
        self.output_dict["cost_function_value"].append(
            self.optimization_state["objective_value"]
        )
        self.output_dict["gradient_norm"].append(
            self.optimization_state["relative_norm"]
        )
        if self.db.parameter_db.problem_type == "shape":
            self.output_dict["MeshQuality"].append(
                self.db.parameter_db.optimization_state["mesh_quality"]
            )
        self.output_dict["stepsize"].append(self.optimization_state["stepsize"])

    def post_process(self) -> None:
        """Saves the history of the optimization to a .json file."""
        self.output_dict["initial_gradient_norm"] = self.optimization_state[
            "gradient_norm_initial"
        ]
        self.output_dict["state_solves"] = self.db.parameter_db.optimization_state[
            "no_state_solves"
        ]
        self.output_dict["adjoint_solves"] = self.db.parameter_db.optimization_state[
            "no_adjoint_solves"
        ]
        self.output_dict["iterations"] = self.optimization_state["iteration"]
        if self.save_results and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            with open(f"{self.result_dir}/history.json", "w", encoding="utf-8") as file:
                json.dump(self.output_dict, file)
        fenics.MPI.barrier(fenics.MPI.comm_world)


class ConsoleManager(IOManager):
    """Management of the console output."""

    def __init__(self, db: database.Database, result_dir: str) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            result_dir: The directory, where the results are written to.

        """
        super().__init__(db, result_dir)
        self.precision = self.config.getint("Output", "precision")

    def output(self) -> None:
        """Prints the output string to the console."""
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(generate_output_str(self.db, self.precision), flush=True)
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def output_summary(self) -> None:
        """Prints the summary in the console."""
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            print(generate_summary_str(self.db, self.precision), flush=True)
        fenics.MPI.barrier(fenics.MPI.comm_world)


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
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            if self.optimization_state["iteration"] == 0:
                file_attr = "w"
            else:
                file_attr = "a"

            with open(
                f"{self.result_dir}/history.txt", file_attr, encoding="utf-8"
            ) as file:
                file.write(f"{generate_output_str(self.db, self.precision)}\n")
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def output_summary(self) -> None:
        """Save the summary in a file."""
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            with open(f"{self.result_dir}/history.txt", "a", encoding="utf-8") as file:
                file.write(generate_summary_str(self.db, self.precision))
        fenics.MPI.barrier(fenics.MPI.comm_world)


class TempFileManager(IOManager):
    """Class for managing temporary files."""

    def post_process(self) -> None:
        """Deletes temporary files."""
        if self.db.parameter_db.problem_type == "shape":
            if (
                self.config.getboolean("Mesh", "remesh")
                and not self.config.getboolean("Debug", "remeshing")
                and self.db.parameter_db.temp_dict
                and fenics.MPI.rank(fenics.MPI.comm_world) == 0
            ):
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", self.db.parameter_db.remesh_directory], check=True
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)


class MeshManager(IOManager):
    """Manages the output of meshes."""

    def post_process(self) -> None:
        """Saves a copy of the optimized mesh in Gmsh format."""
        if self.db.parameter_db.problem_type == "shape":
            if self.config.getboolean("Output", "save_mesh"):
                iomesh.write_out_mesh(
                    self.db.function_db.states[0].function_space().mesh(),
                    self.db.parameter_db.gmsh_file_path,
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

        self.has_output = self.save_state or self.save_adjoint or self.save_gradient
        self.is_initialized = False

        self.state_xdmf_list: List[Union[str, List[str]]] = []
        self.control_xdmf_list: List[Union[str, List[str]]] = []
        self.adjoint_xdmf_list: List[Union[str, List[str]]] = []
        self.gradient_xdmf_list: List[Union[str, List[str]]] = []

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
        if self.save_state and self.db.parameter_db.problem_type == "control":
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
                if self.db.parameter_db.problem_type == "control":
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
        if self.save_state and self.db.parameter_db.problem_type == "control":
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

        if function.function_space().ufl_element().family() == "Real":
            mesh = function.function_space().mesh()
            space = fenics.FunctionSpace(mesh, "CG", 1)
            function = fenics.interpolate(function, space)

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

    def output(self) -> None:
        """Saves the variables to xdmf files."""
        self._initialize_xdmf_lists()

        iteration = int(self.optimization_state["iteration"])

        self._save_states(iteration)
        self._save_controls(iteration)
        self._save_adjoints(iteration)
        self._save_gradients(iteration)

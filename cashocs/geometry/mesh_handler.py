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


"""Module for managing a finite element mesh."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from typing import cast, Dict, List, Optional, TYPE_CHECKING, Union

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs import io
from cashocs.geometry import deformation_handler
from cashocs.geometry import mesh_quality

if TYPE_CHECKING:
    from cashocs._optimization.optimization_algorithms import OptimizationAlgorithm
    from cashocs._optimization.shape_optimization.shape_optimization_problem import (
        ShapeOptimizationProblem,
    )


def _remove_gmsh_parametrizations(mesh_file: str) -> None:
    """Removes the parametrizations section from a Gmsh file.

    This is needed in case several remeshing iterations have to be executed.

    Args:
        mesh_file: Path to the Gmsh file, has to end in .msh.

    """
    temp_location = f"{mesh_file[:-4]}_temp.msh"
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        with open(mesh_file, "r", encoding="utf-8") as in_file, open(
            temp_location, "w", encoding="utf-8"
        ) as temp_file:

            parametrizations_section = False

            for line in in_file:

                if line == "$Parametrizations\n":
                    parametrizations_section = True

                if not parametrizations_section:
                    temp_file.write(line)
                else:
                    pass

                if line == "$EndParametrizations\n":
                    parametrizations_section = False

        subprocess.run(["mv", temp_location, mesh_file], check=True)  # nosec B603, B607
    fenics.MPI.barrier(fenics.MPI.comm_world)


def filter_sys_argv(temp_dir: str) -> List[str]:  # pragma: no cover
    """Filters the command line arguments for the cashocs remesh flag.

    Args:
        temp_dir: Path to directory for the temp files

    """
    arg_list = sys.argv.copy()
    idx_cashocs_remesh_flag = [
        i for i, s in enumerate(arg_list) if s == "--cashocs_remesh"
    ]
    if len(idx_cashocs_remesh_flag) == 1:
        arg_list.pop(idx_cashocs_remesh_flag[0])

    idx_temp_dir = [i for i, s in enumerate(arg_list) if s == temp_dir]
    if len(idx_temp_dir) == 1:
        arg_list.pop(idx_temp_dir[0])

    idx_temp_dir_flag = [i for i, s in enumerate(arg_list) if s == "--temp_dir"]
    if len(idx_temp_dir_flag) == 1:
        arg_list.pop(idx_temp_dir_flag[0])

    return arg_list


class _MeshHandler:
    """Handles the mesh for shape optimization problems.

    This class implements all mesh related things for the shape optimization, such as
    transformations and remeshing. Also includes mesh quality control checks.
    """

    current_mesh_quality: float
    mesh_quality_measure: str
    temp_dict: Optional[Dict]

    def __init__(self, shape_optimization_problem: ShapeOptimizationProblem) -> None:
        """Initializes self.

        Args:
            shape_optimization_problem: The corresponding shape optimization problem.

        """
        self.form_handler = shape_optimization_problem.form_handler
        # Namespacing
        self.mesh = self.form_handler.mesh
        self.deformation_handler = deformation_handler.DeformationHandler(self.mesh)
        self.dx = self.form_handler.dx
        self.bbtree = self.mesh.bounding_box_tree()
        self.config = self.form_handler.config

        # setup from config
        self.volume_change = float(self.config.get("MeshQuality", "volume_change"))
        self.angle_change = float(self.config.get("MeshQuality", "angle_change"))

        self.mesh_quality_tol_lower: float = self.config.getfloat(
            "MeshQuality", "tol_lower"
        )
        self.mesh_quality_tol_upper: float = self.config.getfloat(
            "MeshQuality", "tol_upper"
        )

        if self.mesh_quality_tol_lower > 0.9 * self.mesh_quality_tol_upper:
            _loggers.warning(
                "You are using a lower remesh tolerance (tol_lower) close to the upper "
                "one (tol_upper). This may slow down the optimization considerably."
            )

        self.mesh_quality_measure = self.config.get("MeshQuality", "measure")

        self.mesh_quality_type = self.config.get("MeshQuality", "type")

        self.current_mesh_quality = 1.0
        self.current_mesh_quality = mesh_quality.compute_mesh_quality(
            self.mesh, self.mesh_quality_type, self.mesh_quality_measure
        )

        self._setup_decrease_computation()
        self._setup_a_priori()

        # Remeshing initializations
        self.do_remesh: bool = self.config.getboolean("Mesh", "remesh")
        self.save_optimized_mesh: bool = self.config.getboolean("Output", "save_mesh")

        if self.do_remesh or self.save_optimized_mesh:
            self.mesh_directory = os.path.dirname(
                os.path.realpath(self.config.get("Mesh", "gmsh_file"))
            )

        if self.do_remesh and shape_optimization_problem.temp_dict is not None:
            self.temp_dict = shape_optimization_problem.temp_dict
            self.gmsh_file: str = self.temp_dict["gmsh_file"]
            self.remesh_counter = self.temp_dict.get("remesh_counter", 0)

            if not self.form_handler.has_cashocs_remesh_flag:
                if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                    remesh_directory: str = tempfile.mkdtemp(
                        prefix="cashocs_remesh_", dir=self.mesh_directory
                    )
                else:
                    remesh_directory = ""
                self.remesh_directory: str = fenics.MPI.comm_world.bcast(
                    remesh_directory, root=0
                )
                fenics.MPI.barrier(fenics.MPI.comm_world)
            else:
                self.remesh_directory = self.temp_dict["remesh_directory"]
            if not os.path.isdir(os.path.realpath(self.remesh_directory)):
                os.mkdir(self.remesh_directory)
            self.remesh_geo_file = f"{self.remesh_directory}/remesh.geo"

        elif self.save_optimized_mesh:
            self.gmsh_file = self.config.get("Mesh", "gmsh_file")

        # create a copy of the initial mesh file
        if self.do_remesh and self.remesh_counter == 0:
            self.gmsh_file_init = (
                f"{self.remesh_directory}/mesh_{self.remesh_counter:d}.msh"
            )
            if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                subprocess.run(  # nosec 603
                    ["cp", self.gmsh_file, self.gmsh_file_init], check=True
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)
            self.gmsh_file = self.gmsh_file_init

    def move_mesh(self, transformation: fenics.Function) -> bool:
        r"""Transforms the mesh by perturbation of identity.

        Moves the mesh according to the deformation given by

        .. math:: \text{id} + \mathcal{V}(x),

        where :math:`\mathcal{V}` is the transformation. This
        represents the perturbation of identity.

        Args:
            transformation: The transformation for the mesh, a vector CG1 Function.

        """
        if not (
            transformation.ufl_element().family() == "Lagrange"
            and transformation.ufl_element().degree() == 1
        ):
            raise _exceptions.CashocsException("Not a valid mesh transformation")

        if not self._test_a_priori(transformation):
            _loggers.debug("Mesh transformation rejected due to a priori check.")
            return False
        else:
            success_flag = self.deformation_handler.move_mesh(
                transformation, validated_a_priori=True
            )
            self.current_mesh_quality = mesh_quality.compute_mesh_quality(
                self.mesh, self.mesh_quality_type, self.mesh_quality_measure
            )
            return success_flag

    def revert_transformation(self) -> None:
        """Reverts the previous mesh transformation.

        This is used when the mesh quality for the resulting deformed mesh
        is not sufficient, or when the solution algorithm terminates, e.g., due
        to lack of sufficient decrease in the Armijo rule
        """
        self.deformation_handler.revert_transformation()

    def _setup_decrease_computation(self) -> None:
        """Initializes attributes and solver for the frobenius norm check."""
        self.options_frobenius: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]

        self.trial_dg0 = fenics.TrialFunction(self.form_handler.dg_function_space)
        self.test_dg0 = fenics.TestFunction(self.form_handler.dg_function_space)

        if self.angle_change != float("inf"):
            self.search_direction_container = fenics.Function(
                self.form_handler.deformation_space
            )

            self.a_frobenius = self.trial_dg0 * self.test_dg0 * self.dx
            self.l_frobenius = (
                fenics.sqrt(
                    fenics.inner(
                        fenics.grad(self.search_direction_container),
                        fenics.grad(self.search_direction_container),
                    )
                )
                * self.test_dg0
                * self.dx
            )

    def compute_decreases(
        self, search_direction: List[fenics.Function], stepsize: float
    ) -> int:
        """Estimates the number of Armijo decreases for a certain mesh quality.

        Gives a better estimation of the stepsize. The output is
        the number of Armijo decreases we have to do in order to
        get a transformation that satisfies norm(transformation)_fro <= tol,
        where transformation = stepsize*search_direction and tol is specified in
        the config file under "angle_change". Due to the linearity
        of the norm this has to be done only once, all smaller stepsizes are
        feasible w.r.t. this criterion as well.

        Args:
            search_direction: The search direction in the optimization routine / descent
                algorithm.
            stepsize: The stepsize in the descent algorithm.

        Returns:
            A guess for the number of "Armijo halvings" to get a better stepsize.

        """
        if self.angle_change == float("inf"):
            return 0

        else:
            self.search_direction_container.vector().vec().aypx(
                0.0, search_direction[0].vector().vec()
            )
            self.search_direction_container.vector().apply("")
            x = _utils.assemble_and_solve_linear(
                self.a_frobenius,
                self.l_frobenius,
                ksp_options=self.options_frobenius,
            )

            frobenius_norm = x.max()[1]
            beta_armijo = self.config.getfloat("OptimizationRoutine", "beta_armijo")

            return int(
                np.maximum(
                    np.ceil(
                        np.log(self.angle_change / stepsize / frobenius_norm)
                        / np.log(1 / beta_armijo)
                    ),
                    0.0,
                )
            )

    def _setup_a_priori(self) -> None:
        """Sets up the attributes and petsc solver for the a priori quality check."""
        self.options_prior: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]

        self.transformation_container = fenics.Function(
            self.form_handler.deformation_space
        )
        dim = self.mesh.geometric_dimension()

        # pylint: disable=invalid-name
        self.A_prior = self.trial_dg0 * self.test_dg0 * self.dx
        self.l_prior = (
            fenics.det(
                fenics.Identity(dim) + fenics.grad(self.transformation_container)
            )
            * self.test_dg0
            * self.dx
        )

    def _test_a_priori(self, transformation: fenics.Function) -> bool:
        r"""Check the quality of the transformation before the actual mesh is moved.

        Checks the quality of the transformation. The criterion is that

        .. math:: \det(I + D \texttt{transformation})

        should neither be too large nor too small in order to achieve the best
        transformations.

        Args:
            transformation: The transformation for the mesh.

        Returns:
            A boolean that indicates whether the desired transformation is feasible.

        """
        self.transformation_container.vector().vec().aypx(
            0.0, transformation.vector().vec()
        )
        self.transformation_container.vector().apply("")
        x = _utils.assemble_and_solve_linear(
            self.A_prior,
            self.l_prior,
            ksp_options=self.options_prior,
        )

        min_det: float = x.min()[1]
        max_det: float = x.max()[1]

        return bool(
            (min_det >= 1 / self.volume_change) and (max_det <= self.volume_change)
        )

    def _generate_remesh_geo(self, input_mesh_file: str) -> None:
        """Generates a .geo file used for remeshing.

        The .geo file is generated via the original .geo file for the initial geometry,
        so that mesh size fields are correctly given for the remeshing.

        Args:
            input_mesh_file: Path to the mesh file used for generating the new .geo
                file.

        """
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            with open(self.remesh_geo_file, "w", encoding="utf-8") as file:
                temp_name = os.path.split(input_mesh_file)[1]

                file.write(f"Merge '{temp_name}';\n")
                file.write("CreateGeometry;\n")
                file.write("\n")

                self.temp_dict = cast(Dict, self.temp_dict)
                geo_file = self.temp_dict["geo_file"]
                with open(geo_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line[0].islower():
                            file.write(line)
                        if line[:5] == "Field":
                            file.write(line)
                        if line[:16] == "Background Field":
                            file.write(line)
                        if line[:19] == "BoundaryLayer Field":
                            file.write(line)
                        if line[:5] == "Mesh.":
                            file.write(line)

        fenics.MPI.barrier(fenics.MPI.comm_world)

    def clean_previous_gmsh_files(self) -> None:
        """Removes the gmsh files from the previous remeshing iterations."""
        gmsh_file = f"{self.remesh_directory}/mesh_{self.remesh_counter - 1:d}.msh"
        if os.path.isfile(gmsh_file) and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            subprocess.run(["rm", gmsh_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        gmsh_pre_remesh_file = (
            f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}_pre_remesh.msh"
        )
        if (
            os.path.isfile(gmsh_pre_remesh_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", gmsh_pre_remesh_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        mesh_h5_file = f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}.h5"
        if os.path.isfile(mesh_h5_file) and fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            subprocess.run(["rm", mesh_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        mesh_xdmf_file = f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}.xdmf"
        if (
            os.path.isfile(mesh_xdmf_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", mesh_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        boundaries_h5_file = (
            f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}_boundaries.h5"
        )
        if (
            os.path.isfile(boundaries_h5_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", boundaries_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        boundaries_xdmf_file = (
            f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}_boundaries.xdmf"
        )
        if (
            os.path.isfile(boundaries_xdmf_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", boundaries_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        subdomains_h5_file = (
            f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}_subdomains.h5"
        )
        if (
            os.path.isfile(subdomains_h5_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", subdomains_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        subdomains_xdmf_file = (
            f"{self.remesh_directory}/mesh_{self.remesh_counter-1:d}_subdomains.xdmf"
        )
        if (
            os.path.isfile(subdomains_xdmf_file)
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", subdomains_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def _restart_script(self, temp_dir: str) -> None:
        """Restarts the python script with itself and replaces the process.

        Args:
              temp_dir: Path to the directory for temporary files.

        """
        if not self.config.getboolean("Debug", "restart"):
            sys.stdout.flush()
            os.execv(  # nosec 606
                sys.executable,
                [sys.executable]
                + filter_sys_argv(temp_dir)
                + ["--cashocs_remesh"]
                + ["--temp_dir"]
                + [temp_dir],
            )
        else:
            raise _exceptions.CashocsDebugException(
                "Debug flag detected. "
                "Restart of script with remeshed geometry is cancelled."
            )

    def remesh(self, solver: OptimizationAlgorithm) -> None:
        """Remeshes the current geometry with Gmsh.

        Performs a remeshing of the geometry, and then restarts the optimization problem
        with the new mesh.

        Args:
            solver: The optimization algorithm used to solve the problem.

        """
        if self.do_remesh and self.temp_dict is not None:
            self.remesh_counter += 1
            temp_file = (
                f"{self.remesh_directory}/mesh_{self.remesh_counter:d}_pre_remesh.msh"
            )
            io.write_out_mesh(self.mesh, self.gmsh_file, temp_file)
            self._generate_remesh_geo(temp_file)

            # save the output dict (without the last entries since they are "remeshed")
            self.temp_dict["output_dict"] = {}
            self.temp_dict["output_dict"][
                "state_solves"
            ] = solver.state_problem.number_of_solves
            self.temp_dict["output_dict"][
                "adjoint_solves"
            ] = solver.adjoint_problem.number_of_solves
            self.temp_dict["output_dict"]["iterations"] = solver.iteration + 1

            output_dict = solver.output_manager.result_manager.output_dict
            self.temp_dict["output_dict"]["cost_function_value"] = output_dict[
                "cost_function_value"
            ][:]
            self.temp_dict["output_dict"]["gradient_norm"] = output_dict[
                "gradient_norm"
            ][:]
            self.temp_dict["output_dict"]["stepsize"] = output_dict["stepsize"][:]
            self.temp_dict["output_dict"]["MeshQuality"] = output_dict["MeshQuality"][:]

            dim = self.mesh.geometric_dimension()

            new_gmsh_file = f"{self.remesh_directory}/mesh_{self.remesh_counter:d}.msh"

            gmsh_cmd_list = [
                "gmsh",
                self.remesh_geo_file,
                f"-{int(dim):d}",
                "-o",
                new_gmsh_file,
            ]
            if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                if not self.config.getboolean("Mesh", "show_gmsh_output"):
                    subprocess.run(  # nosec 603
                        gmsh_cmd_list,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                else:
                    subprocess.run(gmsh_cmd_list, check=True)  # nosec 603
            fenics.MPI.barrier(fenics.MPI.comm_world)

            _remove_gmsh_parametrizations(new_gmsh_file)

            self.temp_dict["remesh_counter"] = self.remesh_counter
            self.temp_dict["remesh_directory"] = self.remesh_directory
            self.temp_dict["result_dir"] = solver.output_manager.result_dir

            new_xdmf_file = f"{self.remesh_directory}/mesh_{self.remesh_counter:d}.xdmf"

            io.convert(new_gmsh_file, new_xdmf_file)

            self.clean_previous_gmsh_files()

            self.temp_dict["mesh_file"] = new_xdmf_file
            self.temp_dict["gmsh_file"] = new_gmsh_file

            self.temp_dict["OptimizationRoutine"]["iteration_counter"] = (
                solver.iteration + 1
            )
            self.temp_dict["OptimizationRoutine"][
                "gradient_norm_initial"
            ] = solver.gradient_norm_initial

            temp_dir = self.temp_dict["temp_dir"]

            if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                with open(f"{temp_dir}/temp_dict.json", "w", encoding="utf-8") as file:
                    json.dump(self.temp_dict, file)
            fenics.MPI.barrier(fenics.MPI.comm_world)

            self._restart_script(temp_dir)

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


"""Management of finite element meshes."""

from __future__ import annotations

import pathlib
import subprocess  # nosec B404
import tempfile
from typing import List, TYPE_CHECKING, Union

import fenics
import numpy as np

from cashocs import _exceptions
from cashocs import _loggers
from cashocs import _utils
from cashocs import io
from cashocs._optimization import line_search as ls
from cashocs.geometry import deformations
from cashocs.geometry import quality

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs._database import database
    from cashocs._optimization.optimization_algorithms import OptimizationAlgorithm
    from cashocs.geometry import mesh_testing


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


class _MeshHandler:
    """Handles the mesh for shape optimization problems.

    This class implements all mesh related things for the shape optimization, such as
    transformations and remeshing. Also includes mesh quality control checks.
    """

    def __init__(
        self,
        db: database.Database,
        form_handler: _forms.ShapeFormHandler,
        a_priori_tester: mesh_testing.APrioriMeshTester,
        a_posteriori_tester: mesh_testing.APosterioriMeshTester,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The corresponding shape optimization problem.
            a_priori_tester: The tester before mesh modification.
            a_posteriori_tester: The tester after mesh modification.

        """
        self.db = db
        self.form_handler = form_handler
        self.a_priori_tester = a_priori_tester
        self.a_posteriori_tester = a_posteriori_tester

        # Namespacing
        self.mesh = self.db.geometry_db.mesh
        self.deformation_handler = deformations.DeformationHandler(
            self.mesh, self.a_priori_tester, self.a_posteriori_tester
        )
        self.dx = self.db.geometry_db.dx
        self.bbtree = self.mesh.bounding_box_tree()
        self.config = self.db.config

        self._current_mesh_quality = 1.0
        self._gmsh_file = ""
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

        self.current_mesh_quality: float = quality.compute_mesh_quality(
            self.mesh, self.mesh_quality_type, self.mesh_quality_measure
        )

        self.options_frobenius: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]
        self.trial_dg0 = fenics.TrialFunction(self.db.function_db.dg_function_space)
        self.test_dg0 = fenics.TestFunction(self.db.function_db.dg_function_space)
        self.search_direction_container = fenics.Function(
            self.db.function_db.control_spaces[0]
        )
        self.a_frobenius = None
        self.l_frobenius = None

        self._setup_decrease_computation()

        self.options_prior: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]
        self.transformation_container = fenics.Function(
            self.db.function_db.control_spaces[0]
        )
        self.A_prior = None  # pylint: disable=invalid-name
        self.l_prior = None

        # Remeshing initializations
        self.do_remesh: bool = self.config.getboolean("Mesh", "remesh")
        self.save_optimized_mesh: bool = self.config.getboolean("Output", "save_mesh")

        if self.do_remesh or self.save_optimized_mesh:
            self.mesh_directory = (
                pathlib.Path(self.config.get("Mesh", "gmsh_file")).resolve().parent
            )

        self.gmsh_file: str = ""
        self.remesh_counter = 0
        if self.do_remesh and self.db.parameter_db.temp_dict:
            self.gmsh_file = self.db.parameter_db.temp_dict["gmsh_file"]
            self.remesh_counter = self.db.parameter_db.temp_dict.get(
                "remesh_counter", 0
            )

            if not self.db.parameter_db.is_remeshed:
                if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                    remesh_directory: str = tempfile.mkdtemp(
                        prefix="cashocs_remesh_", dir=self.mesh_directory
                    )
                else:
                    remesh_directory = ""
                self.db.parameter_db.remesh_directory = fenics.MPI.comm_world.bcast(
                    remesh_directory, root=0
                )
                fenics.MPI.barrier(fenics.MPI.comm_world)
            else:
                self.db.parameter_db.remesh_directory = self.db.parameter_db.temp_dict[
                    "remesh_directory"
                ]
            remesh_path = pathlib.Path(self.db.parameter_db.remesh_directory)
            if not remesh_path.is_dir():
                remesh_path.mkdir()
            self.remesh_geo_file = f"{self.db.parameter_db.remesh_directory}/remesh.geo"

        elif self.save_optimized_mesh:
            self.gmsh_file = self.config.get("Mesh", "gmsh_file")

        # create a copy of the initial mesh file
        if self.do_remesh and self.remesh_counter == 0:
            self.gmsh_file_init = (
                f"{self.db.parameter_db.remesh_directory}"
                f"/mesh_{self.remesh_counter:d}.msh"
            )
            if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                subprocess.run(  # nosec 603
                    ["cp", self.gmsh_file, self.gmsh_file_init], check=True
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)
            self.gmsh_file = self.gmsh_file_init

    @property
    def current_mesh_quality(self) -> float:
        """The current mesh quality."""
        return self._current_mesh_quality

    @current_mesh_quality.setter
    def current_mesh_quality(self, value: float) -> None:
        self.db.parameter_db.optimization_state["mesh_quality"] = value
        self._current_mesh_quality = value

    @property
    def gmsh_file(self) -> str:
        return self._gmsh_file

    @gmsh_file.setter
    def gmsh_file(self, value: str) -> None:
        self.db.parameter_db.gmsh_file_path = value
        self._gmsh_file = value

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

        if not self.a_priori_tester.test(transformation, self.volume_change):
            _loggers.debug("Mesh transformation rejected due to a priori check.")
            return False
        else:
            success_flag = self.deformation_handler.move_mesh(
                transformation, validated_a_priori=True
            )
            self.current_mesh_quality = quality.compute_mesh_quality(
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
        if self.angle_change != float("inf"):
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
            beta_armijo = self.config.getfloat("LineSearch", "beta_armijo")

            return int(
                np.maximum(
                    np.ceil(
                        np.log(self.angle_change / stepsize / frobenius_norm)
                        / np.log(1 / beta_armijo)
                    ),
                    0.0,
                )
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
                temp_name = pathlib.Path(input_mesh_file).name

                file.write(f"Merge '{temp_name}';\n")
                file.write("CreateGeometry;\n")
                file.write("\n")

                geo_file = self.db.parameter_db.temp_dict["geo_file"]
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
        gmsh_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}.msh"
        )
        if (
            pathlib.Path(gmsh_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", gmsh_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        gmsh_pre_remesh_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}_pre_remesh.msh"
        )
        if (
            pathlib.Path(gmsh_pre_remesh_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", gmsh_pre_remesh_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        mesh_h5_file = (
            f"{self.db.parameter_db.remesh_directory}/mesh_{self.remesh_counter-1:d}.h5"
        )
        if (
            pathlib.Path(mesh_h5_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", mesh_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        mesh_xdmf_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}.xdmf"
        )
        if (
            pathlib.Path(mesh_xdmf_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", mesh_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        boundaries_h5_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}_boundaries.h5"
        )
        if (
            pathlib.Path(boundaries_h5_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", boundaries_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        boundaries_xdmf_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}_boundaries.xdmf"
        )
        if (
            pathlib.Path(boundaries_xdmf_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", boundaries_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        subdomains_h5_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}_subdomains.h5"
        )
        if (
            pathlib.Path(subdomains_h5_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", subdomains_h5_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

        subdomains_xdmf_file = (
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter-1:d}_subdomains.xdmf"
        )
        if (
            pathlib.Path(subdomains_xdmf_file).is_file()
            and fenics.MPI.rank(fenics.MPI.comm_world) == 0
        ):
            subprocess.run(["rm", subdomains_xdmf_file], check=True)  # nosec 603
        fenics.MPI.barrier(fenics.MPI.comm_world)

    def _reinitialize(self, solver: OptimizationAlgorithm) -> None:
        solver.optimization_problem.__init__(  # type: ignore # pylint: disable=C2801
            solver.optimization_problem.mesh_parametrization,
            self.db.parameter_db.temp_dict["mesh_file"],
        )

        line_search_type = self.config.get("LineSearch", "method").casefold()
        if line_search_type == "armijo":
            line_search: ls.LineSearch = ls.ArmijoLineSearch(
                self.db, solver.optimization_problem
            )
        elif line_search_type == "polynomial":
            line_search = ls.PolynomialLineSearch(self.db, solver.optimization_problem)
        else:
            raise Exception("This code cannot be reached.")

        solver.__init__(  # type: ignore # pylint: disable=C2801
            solver.optimization_problem.db,
            solver.optimization_problem,
            line_search,
        )

    def remesh(self, solver: OptimizationAlgorithm) -> None:
        """Remeshes the current geometry with Gmsh.

        Performs a remeshing of the geometry, and then restarts the optimization problem
        with the new mesh.

        Args:
            solver: The optimization algorithm used to solve the problem.

        """
        if self.do_remesh and self.db.parameter_db.temp_dict:
            self.remesh_counter += 1
            temp_file = (
                f"{self.db.parameter_db.remesh_directory}"
                f"/mesh_{self.remesh_counter:d}_pre_remesh.msh"
            )
            io.write_out_mesh(self.mesh, self.gmsh_file, temp_file)
            self._generate_remesh_geo(temp_file)

            # save the output dict (without the last entries since they are "remeshed")
            self.db.parameter_db.temp_dict["output_dict"] = {}
            self.db.parameter_db.temp_dict["output_dict"][
                "state_solves"
            ] = solver.state_problem.number_of_solves
            self.db.parameter_db.temp_dict["output_dict"][
                "adjoint_solves"
            ] = solver.adjoint_problem.number_of_solves
            self.db.parameter_db.temp_dict["output_dict"]["iterations"] = (
                solver.iteration + 1
            )

            output_dict = solver.output_manager.output_dict
            self.db.parameter_db.temp_dict["output_dict"][
                "cost_function_value"
            ] = output_dict["cost_function_value"][:]
            self.db.parameter_db.temp_dict["output_dict"][
                "gradient_norm"
            ] = output_dict["gradient_norm"][:]
            self.db.parameter_db.temp_dict["output_dict"]["stepsize"] = output_dict[
                "stepsize"
            ][:]
            self.db.parameter_db.temp_dict["output_dict"]["MeshQuality"] = output_dict[
                "MeshQuality"
            ][:]

            dim = self.mesh.geometric_dimension()

            new_gmsh_file = (
                f"{self.db.parameter_db.remesh_directory}"
                f"/mesh_{self.remesh_counter:d}.msh"
            )

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

            self.db.parameter_db.temp_dict["remesh_counter"] = self.remesh_counter
            self.db.parameter_db.temp_dict[
                "remesh_directory"
            ] = self.db.parameter_db.remesh_directory
            self.db.parameter_db.temp_dict[
                "result_dir"
            ] = solver.output_manager.result_dir

            new_xdmf_file = (
                f"{self.db.parameter_db.remesh_directory}"
                f"/mesh_{self.remesh_counter:d}.xdmf"
            )

            io.convert(new_gmsh_file, new_xdmf_file)

            self.clean_previous_gmsh_files()

            self.db.parameter_db.temp_dict["mesh_file"] = new_xdmf_file
            self.db.parameter_db.temp_dict["gmsh_file"] = new_gmsh_file

            self.db.parameter_db.temp_dict["OptimizationRoutine"][
                "iteration_counter"
            ] = solver.iteration
            self.db.parameter_db.temp_dict["OptimizationRoutine"][
                "gradient_norm_initial"
            ] = solver.gradient_norm_initial

            self._reinitialize(solver)
            self._check_imported_mesh_quality(solver)

    def _check_imported_mesh_quality(self, solver: OptimizationAlgorithm) -> None:
        """Checks the quality of an imported mesh.

        This function raises exceptions when the mesh does not satisfy the desired
        quality criteria.

        Args:
            solver: The solver instance carrying the new mesh

        """
        mesh_quality_tol_lower = self.db.config.getfloat("MeshQuality", "tol_lower")
        mesh_quality_tol_upper = self.db.config.getfloat("MeshQuality", "tol_upper")

        if mesh_quality_tol_lower > 0.9 * mesh_quality_tol_upper:
            _loggers.warning(
                "You are using a lower remesh tolerance (tol_lower) close to "
                "the upper one (tol_upper). This may slow down the "
                "optimization considerably."
            )

        mesh_quality_measure = self.db.config.get("MeshQuality", "measure")
        mesh_quality_type = self.db.config.get("MeshQuality", "type")

        mesh = solver.optimization_problem.states[0].function_space().mesh()

        current_mesh_quality = quality.compute_mesh_quality(
            mesh,
            mesh_quality_type,
            mesh_quality_measure,
        )

        failed = False
        fail_msg = None
        if current_mesh_quality < mesh_quality_tol_lower:
            failed = True
            fail_msg = (
                "The quality of the mesh file you have specified is not "
                "sufficient for evaluating the cost functional.\n"
                f"It currently is {current_mesh_quality:.3e} but has to "
                f"be at least {mesh_quality_tol_lower:.3e}."
            )

        if current_mesh_quality < mesh_quality_tol_upper:
            failed = True
            fail_msg = (
                "The quality of the mesh file you have specified is not "
                "sufficient for computing the shape gradient.\n "
                + f"It currently is {current_mesh_quality:.3e} but has to "
                f"be at least {mesh_quality_tol_lower:.3e}."
            )

        if failed:
            raise _exceptions.InputError(
                "cashocs.geometry.import_mesh", "input_arg", fail_msg
            )

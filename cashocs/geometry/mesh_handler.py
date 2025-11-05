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


"""Management of finite element meshes."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
from typing import TYPE_CHECKING
import weakref

import fenics
import numpy as np

try:
    import ufl_legacy as ufl
except ImportError:
    import ufl

from cashocs import _exceptions
from cashocs import _utils
from cashocs import io
from cashocs import log
from cashocs._optimization import line_search as ls
from cashocs.geometry import deformations
from cashocs.geometry import quality
from cashocs.io.mesh import import_mesh

if TYPE_CHECKING:
    from cashocs import _forms
    from cashocs import _typing
    from cashocs._database import database
    from cashocs._optimization.optimization_algorithms import OptimizationAlgorithm
    from cashocs.geometry import mesh_testing


def check_mesh_quality_tolerance(mesh_quality: float, tolerance: float) -> None:
    """Compares the current mesh quality with the (upper) tolerance.

    This function raises an appropriate exception, when the mesh quality is not
    sufficiently high.

    Args:
        mesh_quality: The current mesh quality.
        tolerance: The upper mesh quality tolerance.

    """
    if mesh_quality < tolerance:
        fail_msg = (
            "The quality of the mesh file you have specified is not "
            "sufficient for computing the shape gradient.\n "
            + f"It currently is {mesh_quality:.3e} but has to "
            f"be at least {tolerance:.3e}."
        )
        raise _exceptions.InputError(
            "cashocs.geometry.import_mesh", "input_arg", fail_msg
        )


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
        a_posteriori_tester: mesh_testing.IntersectionTester,
    ) -> None:
        """Initializes self.

        Args:
            db: The database of the problem.
            form_handler: The corresponding shape optimization problem.
            a_priori_tester: The tester before mesh modification.
            a_posteriori_tester: The tester after mesh modification.

        """
        self.db = weakref.proxy(db)
        self.form_handler = form_handler
        self.a_priori_tester = a_priori_tester
        self.a_posteriori_tester = a_posteriori_tester

        # Namespacing
        self.mesh = self.db.geometry_db.mesh
        self.comm = self.mesh.mpi_comm()
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

        self.test_for_intersections = self.config.getboolean(
            "ShapeGradient", "test_for_intersections"
        )

        self.mesh_quality_tol_lower: float = self.config.getfloat(
            "MeshQuality", "tol_lower"
        )
        self.mesh_quality_tol_upper: float = self.config.getfloat(
            "MeshQuality", "tol_upper"
        )

        if self.mesh_quality_tol_lower > 0.9 * self.mesh_quality_tol_upper:
            log.warning(
                "You are using a lower remesh tolerance (tol_lower) close to the upper "
                "one (tol_upper). This may slow down the optimization considerably."
            )

        self.mesh_quality_measure = self.config.get("MeshQuality", "measure")

        self.mesh_quality_type = self.config.get("MeshQuality", "type")
        self.quality_quantile = self.config.getfloat("MeshQuality", "quantile")

        self.current_mesh_quality: float = quality.compute_mesh_quality(
            self.mesh,
            quality_type=self.mesh_quality_type,
            quality_measure=self.mesh_quality_measure,
            quantile=self.quality_quantile,
        )

        if not self.db.parameter_db.is_remeshed:
            check_mesh_quality_tolerance(
                self.current_mesh_quality, self.mesh_quality_tol_upper
            )

        self.options_frobenius: _typing.KspOption = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "pc_jacobi_type": "diagonal",
            "ksp_rtol": 1e-16,
            "ksp_atol": 1e-20,
            "ksp_max_it": 1000,
        }
        self.trial_dg0 = fenics.TrialFunction(self.db.function_db.dg_function_space)
        self.test_dg0 = fenics.TestFunction(self.db.function_db.dg_function_space)
        self.norm_function = fenics.Function(self.db.function_db.dg_function_space)
        self.search_direction_container = fenics.Function(
            self.db.function_db.control_spaces[0]
        )
        self.a_frobenius = None
        self.l_frobenius = None

        self._setup_decrease_computation()

        self.options_prior: _typing.KspOption = {
            "ksp_type": "preonly",
            "pc_type": "jacobi",
            "pc_jacobi_type": "diagonal",
            "ksp_rtol": 1e-16,
            "ksp_atol": 1e-20,
            "ksp_max_it": 1000,
        }
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

        self._setup_remesh()

    def _setup_remesh(self) -> None:
        self.gmsh_file: str = ""
        self.remesh_counter = 0
        if self.do_remesh and self.db.parameter_db.temp_dict:
            self.gmsh_file = self.db.parameter_db.temp_dict["gmsh_file"]
            self.remesh_counter = self.db.parameter_db.temp_dict.get(
                "remesh_counter", 0
            )

            if not self.db.parameter_db.is_remeshed:
                if self.comm.rank == 0:
                    remesh_directory: str = tempfile.mkdtemp(
                        prefix="cashocs_remesh_", dir=self.mesh_directory
                    )
                else:
                    remesh_directory = ""
                self.db.parameter_db.remesh_directory = self.comm.bcast(
                    remesh_directory, root=0
                )
                self.comm.barrier()
            else:
                self.db.parameter_db.remesh_directory = self.db.parameter_db.temp_dict[
                    "remesh_directory"
                ]
            remesh_path = pathlib.Path(self.db.parameter_db.remesh_directory)
            if not remesh_path.is_dir():
                remesh_path.mkdir(parents=True, exist_ok=True)
            self.remesh_geo_file = f"{self.db.parameter_db.remesh_directory}/remesh.geo"

        elif self.save_optimized_mesh:
            self.gmsh_file = self.config.get("Mesh", "gmsh_file")

        # create a copy of the initial mesh file
        if self.do_remesh and self.remesh_counter == 0:
            self.gmsh_file_init = (
                f"{self.db.parameter_db.remesh_directory}"
                f"/mesh_{self.remesh_counter:d}.msh"
            )
            if self.comm.rank == 0:
                subprocess.run(  # noqa: S603
                    ["cp", self.gmsh_file, self.gmsh_file_init],  # noqa: S607
                    check=True,
                )
            self.comm.barrier()
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
            return False
        else:
            success_flag = self.deformation_handler.move_mesh(
                transformation,
                validated_a_priori=True,
                test_for_intersections=self.test_for_intersections,
            )
            self.current_mesh_quality = quality.compute_mesh_quality(
                self.mesh,
                quality_type=self.mesh_quality_type,
                quality_measure=self.mesh_quality_measure,
                quantile=self.quality_quantile,
            )
            if success_flag:
                log.debug(
                    "Mesh update was successful. "
                    f"Deformed mesh has a quality of {self.current_mesh_quality:.3e}"
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
                ufl.sqrt(
                    ufl.inner(
                        ufl.grad(self.search_direction_container),
                        ufl.grad(self.search_direction_container),
                    )
                )
                * self.test_dg0
                * self.dx
            )

    def compute_decreases(
        self, search_direction: list[fenics.Function], stepsize: float
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
            _utils.assemble_and_solve_linear(
                self.a_frobenius,
                self.l_frobenius,
                self.norm_function,
                ksp_options=self.options_frobenius,
            )

            frobenius_norm = self.norm_function.vector().vec().max()[1]
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
        if self.comm.rank == 0:
            with open(self.remesh_geo_file, "w", encoding="utf-8") as file:
                temp_name = pathlib.Path(input_mesh_file).name

                file.write(f"Merge '{temp_name}';\n")
                file.write("CreateGeometry;\n")
                file.write("\n")

                geo_file = self.db.parameter_db.temp_dict["geo_file"]
                with open(geo_file, encoding="utf-8") as f:
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

        self.comm.barrier()

    def clean_previous_gmsh_files(self) -> None:
        """Removes the gmsh files from the previous remeshing iterations."""
        gmsh_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}.msh"
        )
        if gmsh_file.is_file() and self.comm.rank == 0:
            gmsh_file.unlink()
        self.comm.barrier()

        gmsh_pre_remesh_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}_pre_remesh.msh"
        )
        if gmsh_pre_remesh_file.is_file() and self.comm.rank == 0:
            gmsh_pre_remesh_file.unlink()
        self.comm.barrier()

        mesh_h5_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}.h5"
        )

        if mesh_h5_file.is_file() and self.comm.rank == 0:
            mesh_h5_file.unlink()
        self.comm.barrier()

        mesh_xdmf_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}.xdmf"
        )
        if mesh_xdmf_file.is_file() and self.comm.rank == 0:
            mesh_xdmf_file.unlink()
        self.comm.barrier()

        boundaries_h5_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}_boundaries.h5"
        )
        if boundaries_h5_file.is_file() and self.comm.rank == 0:
            boundaries_h5_file.unlink()
        self.comm.barrier()

        boundaries_xdmf_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}_boundaries.xdmf"
        )
        if boundaries_xdmf_file.is_file() and self.comm.rank == 0:
            boundaries_xdmf_file.unlink()
        self.comm.barrier()

        subdomains_h5_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}_subdomains.h5"
        )
        if subdomains_h5_file.is_file() and self.comm.rank == 0:
            subdomains_h5_file.unlink()
        self.comm.barrier()

        subdomains_xdmf_file = pathlib.Path(
            f"{self.db.parameter_db.remesh_directory}"
            f"/mesh_{self.remesh_counter - 1:d}_subdomains.xdmf"
        )
        if subdomains_xdmf_file.is_file() and self.comm.rank == 0:
            subdomains_xdmf_file.unlink()
        self.comm.barrier()

    def _reinitialize(self, solver: OptimizationAlgorithm) -> None:
        solver.optimization_problem.__init__(  # type: ignore # pylint: disable=C2801
            solver.optimization_problem.mesh_parametrization,
            self.db.parameter_db.temp_dict["mesh_file"],
        )
        solver.optimization_problem.initialize_solve_parameters()

        line_search_type = self.config.get("LineSearch", "method").casefold()
        if line_search_type == "armijo":
            line_search: ls.LineSearch = ls.ArmijoLineSearch(
                self.db, solver.optimization_problem
            )
        elif line_search_type == "polynomial":
            line_search = ls.PolynomialLineSearch(self.db, solver.optimization_problem)
        else:
            raise _exceptions.CashocsException("This code cannot be reached.")

        solver.__init__(  # type: ignore # pylint: disable=C2801
            solver.optimization_problem.db,
            solver.optimization_problem,
            line_search,
        )

    def remesh(self, solver: OptimizationAlgorithm) -> bool:
        """Remeshes the current geometry with Gmsh.

        Performs a remeshing of the geometry, and then restarts the optimization problem
        with the new mesh.

        Args:
            solver: The optimization algorithm used to solve the problem.

        Returns:
            A boolean that indicated whether a remeshing has been performed successfully
            (if it is `True`) or not (if it is `False`). A possible reason why remeshing
            was not successful is that it is not activated in the configuration.

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
            for key, value in solver.output_manager.output_dict.items():
                self.db.parameter_db.temp_dict["output_dict"][key] = value

            self.db.parameter_db.temp_dict["OptimizationRoutine"]["rtol"] = (
                self.config.getfloat("OptimizationRoutine", "rtol")
            )
            self.db.parameter_db.temp_dict["OptimizationRoutine"]["atol"] = (
                self.config.getfloat("OptimizationRoutine", "atol")
            )
            self.db.parameter_db.temp_dict["OptimizationRoutine"]["max_iter"] = (
                self.config.getint("OptimizationRoutine", "max_iter")
            )

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
            if self.comm.rank == 0:
                if not self.config.getboolean("Mesh", "show_gmsh_output"):
                    subprocess.run(  # noqa: S603
                        gmsh_cmd_list,
                        check=True,
                        stdout=subprocess.DEVNULL,
                    )
                else:
                    subprocess.run(gmsh_cmd_list, check=True)  # noqa: S603
            self.comm.barrier()

            self._remove_gmsh_parametrizations(new_gmsh_file)

            self.db.parameter_db.temp_dict["remesh_counter"] = self.remesh_counter
            self.db.parameter_db.temp_dict["remesh_directory"] = (
                self.db.parameter_db.remesh_directory
            )
            self.db.parameter_db.temp_dict["result_dir"] = (
                solver.output_manager.result_dir
            )

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

            self._update_mesh_transfer_matrix(new_xdmf_file, solver)

            self._reinitialize(solver)
            self._check_imported_mesh_quality(solver)
            return True

        else:
            return False

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
            log.warning(
                "You are using a lower remesh tolerance (tol_lower) close to "
                "the upper one (tol_upper). This may slow down the "
                "optimization considerably."
            )

        mesh_quality_measure = self.db.config.get("MeshQuality", "measure")
        mesh_quality_type = self.db.config.get("MeshQuality", "type")
        quality_quantile = self.db.config.getfloat("MeshQuality", "quantile")

        mesh = solver.optimization_problem.states[0].function_space().mesh()

        current_mesh_quality = quality.compute_mesh_quality(
            mesh,
            quality_type=mesh_quality_type,
            quality_measure=mesh_quality_measure,
            quantile=quality_quantile,
        )

        check_mesh_quality_tolerance(current_mesh_quality, mesh_quality_tol_upper)

    def _update_mesh_transfer_matrix(
        self, xdmf_filename: str, solver: OptimizationAlgorithm
    ) -> None:
        """Updates the transfer matrix for the global deformation after remeshing.

        Args:
            xdmf_filename: The filename for the new mesh (in XDMF format).
            solver: The optimization algorithm.

        """
        if self.config.getboolean("ShapeGradient", "global_deformation"):
            pre_log_level = (
                log.cashocs_logger._handler.level  # pylint: disable=protected-access
            )
            log.set_log_level(log.WARNING)
            mesh, _, _, _, _, _ = import_mesh(xdmf_filename)
            log.set_log_level(pre_log_level)

            deformation_space = fenics.VectorFunctionSpace(mesh, "CG", 1)
            interpolator = _utils.Interpolator(
                deformation_space, self.db.function_db.control_spaces[0]
            )
            new_transfer_matrix = self.db.geometry_db.transfer_matrix.matMult(
                interpolator.transfer_matrix
            )
            self.db.parameter_db.temp_dict["transfer_matrix"] = (
                new_transfer_matrix.copy()
            )
            self.db.parameter_db.temp_dict["old_transfer_matrix"] = (
                self.db.geometry_db.transfer_matrix.copy()
            )
            self.db.parameter_db.temp_dict["deformation_function"] = (
                solver.line_search.deformation_function.copy(True)
            )

    def _remove_gmsh_parametrizations(self, mesh_file: str) -> None:
        """Removes the parametrizations section from a Gmsh file.

        This is needed in case several remeshing iterations have to be executed.

        Args:
            mesh_file: Path to the Gmsh file, has to end in .msh.

        """
        temp_location = f"{mesh_file[:-4]}_temp.msh"
        if self.comm.rank == 0:
            with (
                open(mesh_file, encoding="utf-8") as in_file,
                open(temp_location, "w", encoding="utf-8") as temp_file,
            ):
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

            subprocess.run(  # noqa: S603
                ["mv", temp_location, mesh_file],  # noqa: S607
                check=True,
            )
        self.comm.barrier()

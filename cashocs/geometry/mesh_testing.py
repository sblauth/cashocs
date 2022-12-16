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

"""Testing of mesh quality."""

from __future__ import annotations

import collections
from typing import List, Union

import fenics
import numpy as np

from cashocs import _loggers
from cashocs import _utils


class APrioriMeshTester:
    """A class for testing the mesh before it is modified."""

    def __init__(self, mesh: fenics.Mesh):
        """Initializes the mesh tester.

        Args:
            mesh: The mesh that is to be tested.

        """
        self.mesh = mesh

        dg_function_space = fenics.FunctionSpace(self.mesh, "DG", 0)
        vector_cg_space = fenics.VectorFunctionSpace(self.mesh, "CG", 1)
        dx = fenics.Measure("dx", domain=mesh)

        self.transformation_container = fenics.Function(vector_cg_space)

        # pylint: disable=invalid-name
        self.A_prior = (
            fenics.TrialFunction(dg_function_space)
            * fenics.TestFunction(dg_function_space)
            * dx
        )
        self.l_prior = (
            fenics.det(
                fenics.Identity(self.mesh.geometric_dimension())
                + fenics.grad(self.transformation_container)
            )
            * fenics.TestFunction(dg_function_space)
            * dx
        )
        self.options_prior: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]

    def test(self, transformation: fenics.Function, volume_change: float) -> bool:
        r"""Check the quality of the transformation before the actual mesh is moved.

        Checks the quality of the transformation. The criterion is that

        .. math:: \det(I + D \texttt{transformation})

        should neither be too large nor too small in order to achieve the best
        transformations.

        Args:
            transformation: The transformation for the mesh.
            volume_change: The allowed factor that each element is allowed to change in
                volume.

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

        min_det = float(x.min()[1])
        max_det = float(x.max()[1])

        return bool((min_det >= 1 / volume_change) and (max_det <= volume_change))


class APosterioriMeshTester:
    """A class for testing the mesh after it has been modified."""

    def __init__(self, mesh: fenics.Mesh) -> None:
        """Initializes the posterior mesh tester.

        Args:
            mesh: The mesh that is to be tested.

        """
        self.mesh = mesh

        cells = self.mesh.cells()
        flat_cells = cells.flatten().tolist()
        self.cell_counter: collections.Counter = collections.Counter(flat_cells)
        self.occurrences = np.array(
            [self.cell_counter[i] for i in range(self.mesh.num_vertices())]
        )

    def test(self) -> bool:
        """Checks the quality of the transformation after the actual mesh is moved.

        Checks whether the mesh is a valid finite element mesh
        after it has been moved, i.e., if there are no overlapping
        or self intersecting elements.

        Returns:
            True if the test is successful, False otherwise.

        Notes:
            fenics itself does not check whether the used mesh is a valid finite
            element mesh, so this check has to be done manually.

        """
        self_intersections = False
        collisions = CollisionCounter.compute_collisions(self.mesh)
        if not (collisions == self.occurrences).all():
            self_intersections = True
        list_self_intersections = fenics.MPI.comm_world.allgather(self_intersections)

        if any(list_self_intersections):
            _loggers.debug("Mesh transformation rejected due to a posteriori check.")
            return False
        else:
            return True


class CollisionCounter:
    """Class for testing, whether a given mesh is a valid FEM mesh."""

    _cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/geometry/BoundingBoxTree.h>
#include <dolfin/geometry/Point.h>

using namespace dolfin;

Eigen::VectorXi
compute_collisions(std::shared_ptr<const Mesh> mesh)
{
  int num_vertices;
  std::vector<unsigned int> colliding_cells;

  num_vertices = mesh->num_vertices();
  Eigen::VectorXi collisions(num_vertices);

  int i = 0;
  for (VertexIterator v(*mesh); !v.end(); ++v)
  {
    colliding_cells = mesh->bounding_box_tree()->compute_entity_collisions(
      v->point()
    );
    collisions[i] = colliding_cells.size();

    ++i;
  }
  return collisions;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("compute_collisions", &compute_collisions);
}
"""
    _cpp_object = fenics.compile_cpp_code(_cpp_code)

    def __init__(self) -> None:
        """Initializes self."""
        pass

    @classmethod
    def compute_collisions(cls, mesh: fenics.Mesh) -> np.ndarray:
        """Computes the cells which (potentially) contain self intersections.

        Args:
            mesh: A FEM mesh.

        Returns:
            An array of cell indices, where ``array[i]`` contains the indices of all
            cells that vertex ``i`` collides with.

        """
        collisions: np.ndarray = cls._cpp_object.compute_collisions(mesh)
        return collisions

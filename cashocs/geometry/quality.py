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

"""Module for computing the mesh quality."""

from __future__ import annotations

from typing import List, Union

import fenics
import numpy as np
import ufl

from cashocs import _utils
from cashocs.geometry import measure


def compute_mesh_quality(
    mesh: fenics.Mesh,
    quality_type: str = "min",
    quality_measure: str = "skewness",
) -> float:
    """This computes the mesh quality of a given mesh.

    Args:
        mesh: The mesh whose quality shall be computed.
        quality_type: The type of measurement for the mesh quality, either minimum
            quality or average quality over all mesh cells, default is 'min'.
        quality_measure: The type of quality measure which is used to compute the
            quality measure, default is 'skewness'

    Returns:
        The quality of the mesh, in the interval :math:`[0,1]`, where 0 is the worst,
        and 1 the best possible quality.

    """
    min_functions = {
        "skewness": MeshQuality.min_skewness,
        "maximum_angle": MeshQuality.min_maximum_angle,
        "radius_ratios": MeshQuality.min_radius_ratios,
        "condition_number": MeshQuality.min_condition_number,
    }
    avg_functions = {
        "skewness": MeshQuality.avg_skewness,
        "maximum_angle": MeshQuality.avg_maximum_angle,
        "radius_ratios": MeshQuality.avg_radius_ratios,
        "condition_number": MeshQuality.avg_condition_number,
    }
    functions = {"min": min_functions, "avg": avg_functions}
    quality = functions[quality_type][quality_measure](mesh)

    return quality


class MeshQuality:
    r"""A class used to compute the quality of a mesh.

    This class implements either a skewness quality measure, one based on the maximum
    angle of the elements, or one based on the radius ratios. All quality measures have
    values in :math:`[0,1]`, where 1 corresponds to the reference (optimal) element, and
    0 corresponds to degenerate elements.

    Examples:
        This class can be directly used, without any instantiation, as shown here ::

            import cashocs

            mesh, _, _, _, _, _ = cashocs.regular_mesh(10)

            min_skew = cashocs.MeshQuality.min_skewness(mesh)
            avg_skew = cashocs.MeshQuality.avg_skewness(mesh)

            min_angle = cashocs.MeshQuality.min_maximum_angle(mesh)
            avg_angle = cashocs.MeshQuality.avg_maximum_angle(mesh)

            min_rad = cashocs.MeshQuality.min_radius_ratios(mesh)
            avg_rad = cashocs.MeshQuality.avg_radius_ratios(mesh)

            min_cond = cashocs.MeshQuality.min_condition_number(mesh)
            avg_cond = cashocs.MeshQuality.avg_condition_number(mesh)

        This works analogously for any mesh compatible with FEniCS.

    """

    _cpp_code_mesh_quality = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>

using namespace dolfin;


void angles_triangle(const Cell& cell, std::vector<double>& angs)
{
  const Mesh& mesh = cell.mesh();
  angs.resize(3);
  const std::size_t i0 = cell.entities(0)[0];
  const std::size_t i1 = cell.entities(0)[1];
  const std::size_t i2 = cell.entities(0)[2];

  const Point p0 = Vertex(mesh, i0).point();
  const Point p1 = Vertex(mesh, i1).point();
  const Point p2 = Vertex(mesh, i2).point();
  Point e0 = p1 - p0;
  Point e1 = p2 - p0;
  Point e2 = p2 - p1;

  e0 /= e0.norm();
  e1 /= e1.norm();
  e2 /= e2.norm();

  angs[0] = acos(e0.dot(e1));
  angs[1] = acos(-e0.dot(e2));
  angs[2] = acos(e1.dot(e2));
}



void dihedral_angles(const Cell& cell, std::vector<double>& angs)
{
  const Mesh& mesh = cell.mesh();
  angs.resize(6);

  const std::size_t i0 = cell.entities(0)[0];
  const std::size_t i1 = cell.entities(0)[1];
  const std::size_t i2 = cell.entities(0)[2];
  const std::size_t i3 = cell.entities(0)[3];

  const Point p0 = Vertex(mesh, i0).point();
  const Point p1 = Vertex(mesh, i1).point();
  const Point p2 = Vertex(mesh, i2).point();
  const Point p3 = Vertex(mesh, i3).point();

  const Point e0 = p1 - p0;
  const Point e1 = p2 - p0;
  const Point e2 = p3 - p0;
  const Point e3 = p2 - p1;
  const Point e4 = p3 - p1;

  Point n0 = e0.cross(e1);
  Point n1 = e0.cross(e2);
  Point n2 = e1.cross(e2);
  Point n3 = e3.cross(e4);

  n0 /= n0.norm();
  n1 /= n1.norm();
  n2 /= n2.norm();
  n3 /= n3.norm();

  angs[0] = acos(n0.dot(n1));
  angs[1] = acos(-n0.dot(n2));
  angs[2] = acos(n1.dot(n2));
  angs[3] = acos(n0.dot(n3));
  angs[4] = acos(n1.dot(-n3));
  angs[5] = acos(n2.dot(n3));
}



dolfin::MeshFunction<double>
skewness(std::shared_ptr<const Mesh> mesh)
{
  MeshFunction<double> cf(mesh, mesh->topology().dim(), 0.0);

  double opt_angle;
  std::vector<double> angs;
  std::vector<double> quals;

  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    if (cell->dim() == 2)
    {
      quals.resize(3);
      angles_triangle(*cell, angs);
      opt_angle = DOLFIN_PI / 3.0;
    }
    else if (cell->dim() == 3)
    {
      quals.resize(6);
      dihedral_angles(*cell, angs);
      opt_angle = acos(1.0/3.0);
    }
    else
    {
      dolfin_error(
        "cashocs_quality.cpp", "skewness", "Not a valid dimension for the mesh."
      );
    }

    for (unsigned int i = 0; i < angs.size(); ++i)
    {
      quals[i] = 1 - std::max(
        (angs[i] - opt_angle) / (DOLFIN_PI - opt_angle),
        (opt_angle - angs[i]) / opt_angle
      );
    }
    cf[*cell] = *std::min_element(quals.begin(), quals.end());
  }
  return cf;
}



dolfin::MeshFunction<double>
maximum_angle(std::shared_ptr<const Mesh> mesh)
{
  MeshFunction<double> cf(mesh, mesh->topology().dim(), 0.0);

  double opt_angle;
  std::vector<double> angs;
  std::vector<double> quals;

  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    if (cell->dim() == 2)
    {
      quals.resize(3);
      angles_triangle(*cell, angs);
      opt_angle = DOLFIN_PI / 3.0;
    }
    else if (cell->dim() == 3)
    {
      quals.resize(6);
      dihedral_angles(*cell, angs);
      opt_angle = acos(1.0/3.0);
    }
    else
    {
      dolfin_error(
        "cashocs_quality.cpp", "maximum_angle", "Not a valid dimension for the mesh."
      );
    }

    for (unsigned int i = 0; i < angs.size(); ++i)
    {
      quals[i] = 1 - std::max((angs[i] - opt_angle) / (DOLFIN_PI - opt_angle), 0.0);
    }
    cf[*cell] = *std::min_element(quals.begin(), quals.end());
  }
  return cf;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("skewness", &skewness);
  m.def("maximum_angle", &maximum_angle);
}

"""
    _quality_object = fenics.compile_cpp_code(_cpp_code_mesh_quality)

    def __init__(self) -> None:
        """Initializes self."""
        pass

    @classmethod
    def _skewness(cls, mesh: fenics.Mesh) -> np.ndarray:
        r"""Computes the skewness of the mesh.

        Args:
            mesh: The mesh whose quality shall be computed.

        Returns:
            The element wise skewness of the mesh on process 0.

        """
        comm = fenics.MPI.comm_world
        skewness_array = cls._quality_object.skewness(mesh).array()
        skewness_list: np.ndarray = comm.gather(skewness_array, root=0)
        if comm.rank == 0:
            skewness_list = np.concatenate(skewness_list, axis=None)
        else:
            skewness_list = np.zeros(1)

        return skewness_list

    @classmethod
    def min_skewness(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the minimal skewness of the mesh.

        This measure the relative distance of a triangle's angles or
        a tetrahedron's dihedral angles to the corresponding optimal
        angle. The optimal angle is defined as the angle an equilateral,
        and thus equiangular, element has. The skewness lies in
        :math:`[0,1]`, where 1 corresponds to the case of an optimal
        (equilateral) element, and 0 corresponds to a degenerate
        element. The skewness corresponding to some (dihedral) angle
        :math:`\alpha` is defined as

        .. math::

            1 - \max \left( \frac{\alpha - \alpha^*}{\pi - \alpha*} , \
            \frac{\alpha^* - \alpha}{\alpha^* - 0} \right),

        where :math:`\alpha^*` is the corresponding angle of the reference
        element. To compute the quality measure, the minimum of this expression
        over all elements and all of their (dihedral) angles is computed.

        Args:
            mesh: The mesh whose quality shall be computed.

        Returns:
            The minimum skewness of the mesh.

        """
        skewness_list = cls._skewness(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.min(skewness_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def avg_skewness(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the average skewness of the mesh.

        This measure the relative distance of a triangle's angles or
        a tetrahedron's dihedral angles to the corresponding optimal
        angle. The optimal angle is defined as the angle an equilateral,
        and thus equiangular, element has. The skewness lies in
        :math:`[0,1]`, where 1 corresponds to the case of an optimal
        (equilateral) element, and 0 corresponds to a degenerate
        element. The skewness corresponding to some (dihedral) angle
        :math:`\alpha` is defined as

        .. math::

            1 - \max \left( \frac{\alpha - \alpha^*}{\pi - \alpha*} ,
            \frac{\alpha^* - \alpha}{\alpha^* - 0} \right),

        where :math:`\alpha^*` is the corresponding angle of the reference
        element. To compute the quality measure, the average of this expression
        over all elements and all of their (dihedral) angles is computed.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The average skewness of the mesh.

        """
        skewness_list = cls._skewness(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.average(skewness_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def _maximum_angle(cls, mesh: fenics.Mesh) -> np.ndarray:
        r"""Computes the largest angle of each element.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The maximum angle quality measure for each element on process 0.

        """
        comm = fenics.MPI.comm_world
        maximum_angle_array = cls._quality_object.maximum_angle(mesh).array()
        maximum_angle_list: np.ndarray = comm.gather(maximum_angle_array, root=0)
        if comm.rank == 0:
            maximum_angle_list = np.concatenate(maximum_angle_list, axis=None)
        else:
            maximum_angle_list = np.zeros(1)

        return maximum_angle_list

    @classmethod
    def min_maximum_angle(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the minimal quality measure based on the largest angle.

        This measures the relative distance of a triangle's angles or a
        tetrahedron's dihedral angles to the corresponding optimal
        angle. The optimal angle is defined as the angle an equilateral
        (and thus equiangular) element has. This is defined as

        .. math:: 1 - \max\left( \frac{\alpha - \alpha^*}{\pi - \alpha^*} , 0 \right),

        where :math:`\alpha` is the corresponding (dihedral) angle of the element
        and :math:`\alpha^*` is the corresponding (dihedral) angle of the reference
        element. To compute the quality measure, the minimum of this expression
        over all elements and all of their (dihedral) angles is computed.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The minimum value of the maximum angle quality measure.

        """
        maximum_angle_list = cls._maximum_angle(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.min(maximum_angle_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def avg_maximum_angle(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the average quality measure based on the largest angle.

        This measures the relative distance of a triangle's angles or a
        tetrahedron's dihedral angles to the corresponding optimal
        angle. The optimal angle is defined as the angle an equilateral
        (and thus equiangular) element has. This is defined as

        .. math:: 1 - \max\left( \frac{\alpha - \alpha^*}{\pi - \alpha^*} , 0 \right),

        where :math:`\alpha` is the corresponding (dihedral) angle of the element
        and :math:`\alpha^*` is the corresponding (dihedral) angle of the reference
        element. To compute the quality measure, the average of this expression
        over all elements and all of their (dihedral) angles is computed.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The average quality, based on the maximum angle measure.

        """
        maximum_angle_list = cls._maximum_angle(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.average(maximum_angle_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def _radius_ratios(cls, mesh: fenics.Mesh) -> np.ndarray:
        r"""Computes the radius ratios of the mesh elements.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The radius ratios of the mesh elements on process 0.

        """
        comm = fenics.MPI.comm_world
        radius_ratios_array = fenics.MeshQuality.radius_ratios(mesh).array()
        radius_ratios_list: np.ndarray = comm.gather(radius_ratios_array, root=0)
        if comm.rank == 0:
            radius_ratios_list = np.concatenate(radius_ratios_list, axis=None)
        else:
            radius_ratios_list = np.zeros(1)

        return radius_ratios_list

    @classmethod
    def min_radius_ratios(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the minimal radius ratio of the mesh.

        This measures the ratio of the element's inradius to its circumradius,
        normalized by the geometric dimension. This is computed via

        .. math:: d \frac{r}{R},

        where :math:`d` is the spatial dimension, :math:`r` is the inradius, and
        :math:`R` is the circumradius. To compute the (global) quality measure, the
        minimum of this expression over all elements is returned.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The minimal radius ratio of the mesh.

        """
        radius_ratios_list = cls._radius_ratios(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.min(radius_ratios_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def avg_radius_ratios(cls, mesh: fenics.Mesh) -> float:
        r"""Computes the average radius ratio of the mesh.

        This measures the ratio of the element's inradius to its circumradius,
        normalized by the geometric dimension. This is computed via

        .. math:: d \frac{r}{R},

        where :math:`d` is the spatial dimension, :math:`r` is the inradius, and
        :math:`R` is the circumradius. To compute the (global) quality measure, the
        average of this expression over all elements is returned.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The average radius ratio of the mesh.

        """
        radius_ratios_list = cls._radius_ratios(mesh)
        comm = fenics.MPI.comm_world

        if comm.rank == 0:
            qual = float(np.average(radius_ratios_list))
        else:
            qual = None

        quality: float = comm.bcast(qual, root=0)

        return quality

    @classmethod
    def _cell_condition_number(cls, mesh: fenics.Mesh) -> fenics.Function:
        r"""Computes the condition number quality for each cell.

        This quality criterion uses the condition number (in the Frobenius norm) of the
        (linear) mapping from the elements of the mesh to the reference element.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            A fenics.Function of a piecewise constant function space which holds the
            cell's condition number quality measure.

        """
        function_space_dg0 = fenics.FunctionSpace(mesh, "DG", 0)
        jac = ufl.Jacobian(mesh)
        inv = ufl.JacobianInverse(mesh)

        options: List[List[Union[str, int, float]]] = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]

        dx = measure.NamedMeasure("dx", mesh)
        lhs = (
            fenics.TrialFunction(function_space_dg0)
            * fenics.TestFunction(function_space_dg0)
            * dx
        )
        rhs = (
            fenics.sqrt(fenics.inner(jac, jac))
            * fenics.sqrt(fenics.inner(inv, inv))
            * fenics.TestFunction(function_space_dg0)
            * dx
        )

        cond = fenics.Function(function_space_dg0)

        _utils.assemble_and_solve_linear(
            lhs, rhs, x=cond.vector().vec(), ksp_options=options
        )
        cond.vector().apply("")
        cond.vector().vec().reciprocal()
        cond.vector().apply("")
        cond.vector().vec().scale(np.sqrt(mesh.geometric_dimension()))
        cond.vector().apply("")

        return cond

    @classmethod
    def min_condition_number(cls, mesh: fenics.Mesh) -> float:
        r"""Computes quality based on the condition number of the reference mapping.

        This quality criterion uses the condition number (in the Frobenius norm) of the
        (linear) mapping from the elements of the mesh to the reference element.
        Computes the minimum of the condition number over all elements.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The minimal condition number quality measure.

        """
        cond = cls._cell_condition_number(mesh)
        quality: float = cond.vector().vec().min()[1]

        return quality

    @classmethod
    def avg_condition_number(cls, mesh: fenics.Mesh) -> float:
        """Computes quality based on the condition number of the reference mapping.

        This quality criterion uses the condition number (in the Frobenius norm) of the
        (linear) mapping from the elements of the mesh to the reference element.
        Computes the average of the condition number over all elements.

        Args:
            mesh: The mesh, whose quality shall be computed.

        Returns:
            The average mesh quality based on the condition number.

        """
        cond = cls._cell_condition_number(mesh)
        quality: float = cond.vector().vec().sum() / cond.vector().vec().getSize()

        return quality

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

from typing import TYPE_CHECKING

import fenics
import numpy as np
from petsc4py import PETSc
from typing_extensions import Literal
from ufl import Jacobian, JacobianInverse

from .measure import _NamedMeasure
from .._exceptions import InputError
from ..utils.linalg import (
    _assemble_petsc_system,
    _setup_petsc_options,
    _solve_linear_problem,
)


if TYPE_CHECKING:
    pass


def compute_mesh_quality(
    mesh: fenics.Mesh,
    type: Literal["min", "minimum", "avg", "average"] = "min",
    measure: Literal[
        "skewness", "maximum_angle", "radius_ratios", "condition_number"
    ] = "skewness",
) -> float:
    """This computes the mesh quality of a given mesh.

    Parameters
    ----------
    mesh : fenics.Mesh
        The mesh whose quality shall be computed
    type : {'min', 'minimum', 'avg', 'average'}, optional
        The type of measurement for the mesh quality, either minimum quality or average
        quality over all mesh cells, default is 'min'
    measure : {'skewness', 'maximum_angle', 'radius_ratios', 'condition_number'}, optional
        The type of quality measure which is used to compute the quality measure, default
        is 'skewness'

    Returns
    -------
    float
        The quality of the mesh, in the interval :math:`[0,1]`, where 0 is the worst, and
        1 the best possible quality.
    """

    if type in ["min", "minimum"]:
        if measure == "skewness":
            quality = MeshQuality.min_skewness(mesh)
        elif measure == "maximum_angle":
            quality = MeshQuality.min_maximum_angle(mesh)
        elif measure == "radius_ratios":
            quality = MeshQuality.min_radius_ratios(mesh)
        elif measure == "condition_number":
            quality = MeshQuality.min_condition_number(mesh)

    elif type in ["avg", "average"]:
        if measure == "skewness":
            quality = MeshQuality.avg_skewness(mesh)
        elif measure == "maximum_angle":
            quality = MeshQuality.avg_maximum_angle(mesh)
        elif measure == "radius_ratios":
            quality = MeshQuality.avg_radius_ratios(mesh)
        elif measure == "condition_number":
            quality = MeshQuality.avg_condition_number(mesh)

    return quality


class MeshQuality:
    r"""A class used to compute the quality of a mesh.

    This class implements either a skewness quality measure, one based
    on the maximum angle of the elements, or one based on the radius ratios.
    All quality measures have values in :math:`[0,1]`, where 1 corresponds
    to the reference (optimal) element, and 0 corresponds to degenerate elements.

    Examples
    --------
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
			  angs[1] = acos(e0.dot(e2));
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
				  dolfin_error("cashocs_quality.cpp", "skewness", "Not a valid dimension for the mesh.");
				}

				for (unsigned int i = 0; i < angs.size(); ++i)
				{
				  quals[i] = 1 - std::max((angs[i] - opt_angle) / (DOLFIN_PI - opt_angle), (opt_angle - angs[i]) / opt_angle);
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
				  dolfin_error("cashocs_quality.cpp", "maximum_angle", "Not a valid dimension for the mesh.");
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
        pass

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

        .. math:: 1 - \max \left( \frac{\alpha - \alpha^*}{\pi - \alpha*} , \frac{\alpha^* - \alpha}{\alpha^* - 0} \right),

        where :math:`\alpha^*` is the corresponding angle of the reference
        element. To compute the quality measure, the minimum of this expression
        over all elements and all of their (dihedral) angles is computed.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh whose quality shall be computed.

        Returns
        -------
        float
            The skewness of the mesh.
        """

        return np.min(cls._quality_object.skewness(mesh).array())

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

        .. math:: 1 - \max \left( \frac{\alpha - \alpha^*}{\pi - \alpha*} , \frac{\alpha^* - \alpha}{\alpha^* - 0} \right),

        where :math:`\alpha^*` is the corresponding angle of the reference
        element. To compute the quality measure, the average of this expression
        over all elements and all of their (dihedral) angles is computed.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The average skewness of the mesh.
        """

        return np.average(cls._quality_object.skewness(mesh).array())

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

        Parameters
        ----------
        mesh : fenics.Mesh
                The mesh, whose quality shall be computed.

        Returns
        -------
        float
                The minimum value of the maximum angle quality measure.
        """

        return np.min(cls._quality_object.maximum_angle(mesh).array())

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

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The average quality, based on the maximum angle measure.
        """

        return np.average(cls._quality_object.maximum_angle(mesh).array())

    @staticmethod
    def min_radius_ratios(mesh: fenics.Mesh) -> float:
        r"""Computes the minimal radius ratio of the mesh.

        This measures the ratio of the element's inradius to it's circumradius,
        normalized by the geometric dimension. This is computed via

        .. math:: d \frac{r}{R},

        where :math:`d` is the spatial dimension, :math:`r` is the inradius, and :math:`R` is
        the circumradius. To compute the (global) quality measure, the minimum
        of this expression over all elements is returned.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The minimal radius ratio of the mesh.
        """

        return np.min(fenics.MeshQuality.radius_ratios(mesh).array())

    @staticmethod
    def avg_radius_ratios(mesh: fenics.Mesh) -> float:
        r"""Computes the average radius ratio of the mesh.

        This measures the ratio of the element's inradius to it's circumradius,
        normalized by the geometric dimension. This is computed via

        .. math:: d \frac{r}{R},

        where :math:`d` is the spatial dimension, :math:`r` is the inradius, and :math:`R` is
        the circumradius. To compute the (global) quality measure, the average
        of this expression over all elements is returned.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The average radius ratio of the mesh.
        """

        return np.average(fenics.MeshQuality.radius_ratios(mesh).array())

    @staticmethod
    def min_condition_number(mesh: fenics.Mesh) -> float:
        r"""Computes minimal mesh quality based on the condition number of the reference mapping.

        This quality criterion uses the condition number (in the Frobenius norm) of the
        (linear) mapping from the elements of the mesh to the reference element. Computes
        the minimum of the condition number over all elements.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The minimal condition number quality measure.
        """

        DG0 = fenics.FunctionSpace(mesh, "DG", 0)
        jac = Jacobian(mesh)
        inv = JacobianInverse(mesh)

        options = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]
        ksp = PETSc.KSP().create()
        _setup_petsc_options([ksp], [options])

        dx = _NamedMeasure("dx", mesh)
        a = fenics.TrialFunction(DG0) * fenics.TestFunction(DG0) * dx
        L = (
            fenics.sqrt(fenics.inner(jac, jac))
            * fenics.sqrt(fenics.inner(inv, inv))
            * fenics.TestFunction(DG0)
            * dx
        )

        cond = fenics.Function(DG0)

        A, b = _assemble_petsc_system(a, L)
        _solve_linear_problem(ksp, A, b, cond.vector().vec(), options)
        cond.vector().apply("")
        cond.vector().vec().reciprocal()
        cond.vector().vec().scale(np.sqrt(mesh.geometric_dimension()))

        return cond.vector().vec().min()[1]

    @staticmethod
    def avg_condition_number(mesh):
        """Computes average mesh quality based on the condition number of the reference mapping.

        This quality criterion uses the condition number (in the Frobenius norm) of the
        (linear) mapping from the elements of the mesh to the reference element. Computes
        the average of the condition number over all elements.

        Parameters
        ----------
        mesh : fenics.Mesh
            The mesh, whose quality shall be computed.

        Returns
        -------
        float
            The average mesh quality based on the condition number.
        """

        DG0 = fenics.FunctionSpace(mesh, "DG", 0)
        jac = Jacobian(mesh)
        inv = JacobianInverse(mesh)

        options = [
            ["ksp_type", "preonly"],
            ["pc_type", "jacobi"],
            ["pc_jacobi_type", "diagonal"],
            ["ksp_rtol", 1e-16],
            ["ksp_atol", 1e-20],
            ["ksp_max_it", 1000],
        ]
        ksp = PETSc.KSP().create()
        _setup_petsc_options([ksp], [options])

        dx = _NamedMeasure("dx", mesh)
        a = fenics.TrialFunction(DG0) * fenics.TestFunction(DG0) * dx
        L = (
            fenics.sqrt(fenics.inner(jac, jac))
            * fenics.sqrt(fenics.inner(inv, inv))
            * fenics.TestFunction(DG0)
            * dx
        )

        cond = fenics.Function(DG0)

        A, b = _assemble_petsc_system(a, L)
        _solve_linear_problem(ksp, A, b, cond.vector().vec(), options)
        cond.vector().apply("")

        cond.vector().vec().reciprocal()
        cond.vector().vec().scale(np.sqrt(mesh.geometric_dimension()))

        return cond.vector().vec().sum() / cond.vector().vec().getSize()

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

"""Interpolations and averaging for levelset functions for topology optimization."""

import fenics
import numpy as np

from cashocs._utils import linalg

try:
    import ufl_legacy as ufl
    from ufl_legacy.core import expr as ufl_expr
except ImportError:
    import ufl
    from ufl.core import expr as ufl_expr

cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Function.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/la/Vector.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/function/FunctionSpace.h>

using namespace dolfin;


double get_volume_fraction_triangle(double psi0, double psi1, double psi2){
    std::array<double, 3> psi = {psi0, psi1, psi2};
    std::sort(psi.begin(), psi.end());
    double s;

    if (psi[2] <= 0.0){
        s = 1.0;
    }
    else if (psi[0] > 0.0){
        s = 0.0;
    }
    else if (psi[1] <= 0.0 && psi[2] > 0.0){
        s = 1.0 - pow(psi[2], 2.0) / ((psi[2] - psi[0]) * (psi[2] - psi[1]));
    }
    else if (psi[0] <= 0.0 && psi[1] > 0.0){
        s = pow(psi[0], 2.0) / ((psi[1] - psi[0]) * (psi[2] - psi[0]));
    }
    else{
        s = -1.0;
    }
    return s;
}


double get_volume_fraction_tetrahedron(
    double psi0,
    double psi1,
    double psi2,
    double psi3
){
    std::array<double, 4> psi = {psi0, psi1, psi2, psi3};
    std::sort(psi.begin(), psi.end());
    double s;

    if (psi[3] <= 0.0){
        s = 1.0;
    }
    else if (psi[0] > 0.0){
        s = 0.0;
    }
    else if (psi[2] <= 0.0 && psi[3] > 0.0){
        s = 1.0 - pow(psi[3], 3) / (
            (psi[3] - psi[0]) * (psi[3] - psi[1]) * (psi[3] - psi[2])
        );
    }
    else if (psi[1] <= 0.0 && psi[2] > 0.0){
        s = (
            psi[0] * psi[1] * (pow(psi[2], 2) + psi[2] * psi[3] + pow(psi[3], 2))
            + psi[2] * psi[3] * (
                psi[2] * psi[3] - (psi[0] + psi[1]) * (psi[2] + psi[3])
                )
        ) / (
        (psi[0] - psi[2]) * (psi[1] - psi[2]) * (psi[0] - psi[3]) * (psi[1] - psi[3])
        );
    }
    else if (psi[0] <= 0.0 && psi[1] > 0.0){
        s = -pow(psi[0], 3) / (
            (psi[1] - psi[0]) * (psi[2] - psi[0]) * (psi[3] - psi[0])
        );
    }
    else{
        s = -1.0;
    }

    return s;
}


double interpolate_levelset_to_elements(
    std::shared_ptr<Function> levelset_function,
    double val1,
    double val2,
    std::shared_ptr<Function> ratio
){
    double s = 0.0;
    std::shared_ptr<const Mesh> mesh = levelset_function->function_space()->mesh();
    std::vector<double> ratio_vector;
    std::vector<double> vertex_values;
    std::vector<double> vals;
    double psi0;
    double psi1;
    double psi2;
    double psi3;

    levelset_function->compute_vertex_values(vertex_values);
    ratio->vector()->get_local(ratio_vector);

    std::vector<unsigned int> cells = mesh->cells();
    int meshdim = mesh->geometry().dim();
    auto ghost_offset = mesh->topology().ghost_offset(meshdim);

    if (meshdim == 2){
        int index = 0;
        for (int i=0; i<ghost_offset*3; i+=3){
            psi0 = vertex_values[cells[i]];
            psi1 = vertex_values[cells[i+1]];
            psi2 = vertex_values[cells[i+2]];

            if (psi0 < 0 && psi1 < 0 && psi2 < 0){
                ratio_vector[index] = val1;
            }
            else if (psi0 > 0 && psi1 > 0 && psi2 > 0){
                ratio_vector[index] = val2;
            }
            else{
                s = get_volume_fraction_triangle(psi0, psi1, psi2);
                ratio_vector[index] = val2 + s*(val1 - val2);
            }
            index += 1;
        }
    }
    else if (meshdim == 3){
        int index = 0;
        for (int i=0; i<ghost_offset*4; i+=4){
            psi0 = vertex_values[cells[i]];
            psi1 = vertex_values[cells[i+1]];
            psi2 = vertex_values[cells[i+2]];
            psi3 = vertex_values[cells[i+3]];

            if (psi0 < 0 && psi1 < 0 && psi2 < 0 && psi3 < 0){
                ratio_vector[index] = val1;
            }
            else if (psi0 > 0 && psi1 > 0 && psi2 > 0 && psi3 > 0){
                ratio_vector[index] = val2;
            }
            else{
                s = get_volume_fraction_tetrahedron(psi0, psi1, psi2, psi3);
                ratio_vector[index] = val2 + s*(val1 - val2);
            }
            index += 1;
        }
    }

    ratio->vector()->set_local(ratio_vector);

    return 0.0;
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("get_volume_fraction_triangle", &get_volume_fraction_triangle);
  m.def("interpolate_levelset_to_elements", &interpolate_levelset_to_elements);
}

"""

interpolation_module = fenics.compile_cpp_code(cpp_code)


def interpolate_levelset_function_to_cells(
    levelset_function: fenics.Function,
    val_neg: float,
    val_pos: float,
    cell_function: fenics.Function,
) -> None:
    """Interpolates jumping values to the mesh cells based on the levelset function.

    Args:
        levelset_function: The levelset function representing the domain.
        val_neg: The value inside the domain (levelset_function < 0).
        val_pos: The value outside the domain (levelset_function > 0).
        cell_function: The piecewise constant cell function, into which the result is
            written.

    """
    interpolation_module.interpolate_levelset_to_elements(
        levelset_function.cpp_object(),
        val_neg,
        val_pos,
        cell_function.cpp_object(),
    )


def interpolate_by_volume(
    form_neg: ufl_expr.Expr,
    form_pos: ufl_expr.Expr,
    levelset_function: fenics.Function,
    node_function: fenics.Function,
) -> None:
    """Averages a piecewise constant forms and interpolates them into a CG1 space.

    The averaging is weighted by the volume of the cells.

    Args:
        form_neg: The UFL form inside the domain (levelset_function < 0).
        form_pos: The UFL form outside the domain (levelset_function > 0).
        levelset_function: The levelset function representin the domain.
        node_function: The resulting piecewise continuous function.

    """
    function_space = node_function.function_space()
    mesh = function_space.mesh()
    dg0_space = fenics.FunctionSpace(mesh, "DG", 0)
    dx = ufl.Measure("dx", mesh)

    indicator_omega = fenics.Function(dg0_space)
    interpolate_levelset_function_to_cells(levelset_function, 1.0, 0.0, indicator_omega)

    test = fenics.TestFunction(function_space)
    form_td = form_neg * indicator_omega + form_pos * (
        fenics.Constant(1.0) - indicator_omega
    )
    arr = fenics.assemble(form_td * test * dx)
    vol = fenics.assemble(test * dx)
    node_function.vector().set_local(arr[:] / vol[:])
    node_function.vector().apply("")


def interpolate_by_angle(
    form_neg: ufl_expr.Expr,
    form_pos: ufl_expr.Expr,
    levelset_function: fenics.Function,
    node_function: fenics.Function,
) -> None:
    """Averages a piecewise constant forms and interpolates them into a CG1 space.

    The averaging is weighted by the angles of the cells (relative to the node).

    Args:
        form_neg: The UFL form inside the domain (levelset_function < 0).
        form_pos: The UFL form outside the domain (levelset_function > 0).
        levelset_function: The levelset function representin the domain.
        node_function: The resulting piecewise continuous function.

    """
    cg1_space = node_function.function_space()
    mesh = cg1_space.mesh()
    dg0_space = fenics.FunctionSpace(mesh, "DG", 0)
    indicator_omega = fenics.Function(dg0_space)
    interpolate_levelset_function_to_cells(levelset_function, 1.0, 0.0, indicator_omega)

    pwc_fun = linalg.l2_projection(
        form_neg * indicator_omega
        + form_pos * (fenics.Constant(1.0) - indicator_omega),
        dg0_space,
        ksp_options={"ksp_type": "preonly", "pc_type": "jacobi"},
    )
    node_function.vector().vec().set(0.0)
    node_function.vector().apply("")

    cpp_code_angle = """
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#define _USE_MATH_DEFINES
#include <cmath>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/Vector.h>

using namespace dolfin;


void angles_triangle(const Cell& cell,
    std::vector<double>& angs,
    std::vector<double>& ents) {
  const Mesh& mesh = cell.mesh();
  angs.resize(3);
  ents.resize(3);
  ents[0] = cell.entities(0)[0];
  ents[1] = cell.entities(0)[1];
  ents[2] = cell.entities(0)[2];
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

void angles_tetrahedron(const Cell& cell,
    std::vector<double>& angs,
    std::vector<double>& ents) {
  const Mesh& mesh = cell.mesh();
  std::vector<double> dihedral_angs(6, 0.0);
  angs.resize(4);
  ents.resize(4);
  ents[0] = cell.entities(0)[0];
  ents[1] = cell.entities(0)[1];
  ents[2] = cell.entities(0)[2];
  ents[3] = cell.entities(0)[3];

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

  dihedral_angs[0] = acos(n0.dot(n1));
  dihedral_angs[1] = acos(-n0.dot(n2));
  dihedral_angs[2] = acos(n1.dot(n2));
  dihedral_angs[3] = acos(n0.dot(n3));
  dihedral_angs[4] = acos(n1.dot(-n3));
  dihedral_angs[5] = acos(n2.dot(n3));

  angs[0] = dihedral_angs[0] + dihedral_angs[1] + dihedral_angs[2] - M_PI;
  angs[1] = dihedral_angs[0] + dihedral_angs[3] + dihedral_angs[4] - M_PI;
  angs[2] = dihedral_angs[1] + dihedral_angs[3] + dihedral_angs[5] - M_PI;
  angs[3] = dihedral_angs[2] + dihedral_angs[4] + dihedral_angs[5] - M_PI;
}

std::tuple<std::vector<double>, std::vector<double>>
interpolate(std::shared_ptr<dolfin::Function> u, std::shared_ptr<dolfin::Function> v)
{
  auto mesh = u->function_space()->mesh();
  std::vector<double> angs;
  std::vector<double> ents;
  std::vector<double> dg0;
  std::vector<double> cg1;
  std::vector<double> angles;

  int i = 0;
  double val;
  u->vector()->get_local(dg0);
  v->vector()->get_local(cg1);
  v->vector()->get_local(angles);

  for (CellIterator cell(*mesh); !cell.end(); ++cell)
  {
    val = dg0[i];
    if (cell->dim() == 2)
    {
      angles_triangle(*cell, angs, ents);
    }
    else if (cell->dim() == 3)
    {
      angles_tetrahedron(*cell, angs, ents);
      cg1[ents[3]] += val*angs[3];
      angles[ents[3]] += angs[3];
    }

    cg1[ents[0]] += val * angs[0];
    cg1[ents[1]] += val * angs[1];
    cg1[ents[2]] += val * angs[2];

    angles[ents[0]] += angs[0];
    angles[ents[1]] += angs[1];
    angles[ents[2]] += angs[2];
    i += 1;
  }
  return std::make_tuple(cg1, angles);
}

PYBIND11_MODULE(SIGNATURE, m)
{
  m.def("interpolate", &interpolate);
}

"""
    module = fenics.compile_cpp_code(cpp_code_angle)
    values, weights = module.interpolate(
        pwc_fun.cpp_object(), node_function.cpp_object()
    )
    values = np.array(values)
    weights = np.array(weights)
    values /= weights
    d2v = fenics.dof_to_vertex_map(cg1_space)
    node_function.vector()[:] = values[d2v]
    node_function.vector().apply("")

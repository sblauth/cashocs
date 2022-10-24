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

"""Interpolations and averaging for levelset functions for topology optimization."""

import fenics
import numba
import numpy as np
import ufl.core.expr


@numba.vectorize
def get_volume_fraction_triangle(psi0: float, psi1: float, psi2: float) -> float:
    """Computes the volume fraction based on a levelset function in a triangle.

    The input consists of the three nodal values representing the levelset function
    in the given triangle, sorted by magnitude. The volume fraction is 1.0 if the
    triangle is inside the domain (psi <= 0) and 0 if it is not inside (psi >= 0)

    Args:
        psi0: Lowest value of the levelset function in the triangle.
        psi1: Middle value of the levelset function in the triangle.
        psi2: Highest value of the levelset function in the triangle.

    Returns:
        The volume fraction of the domain in the triangle, as represented by the
        levelset function.

    """
    if psi2 <= 0.0:
        return 1.0
    elif psi0 > 0.0:
        return 0.0
    elif psi1 <= 0.0 < psi2:
        return 1 - psi2**2 / ((psi2 - psi0) * (psi2 - psi1))
    elif psi0 <= 0.0 < psi1:
        return psi0**2 / ((psi1 - psi0) * (psi2 - psi0))
    else:
        return -1.0


@numba.vectorize
def get_volume_fraction_tetrahedron(
    psi0: float, psi1: float, psi2: float, psi3: float
) -> float:
    """Computes the volume fraction based on a levelset function in a tetrahedron.

    The input consists of the four nodal values representing the levelset function
    in the given tetrahedron, sorted by magnitude. The volume fraction is 1.0 if the
    tetrahedron is inside the domain (psi <= 0) and 0 if it is not inside (psi >= 0)

    Args:
        psi0: Lowest value of the levelset function in the tetrahedron.
        psi1: Second-lowest value of the levelset function in the tetrahedron.
        psi2: Second-highest value of the levelset function in the tetrahedron.
        psi2: Highest value of the levelset function in the tetrahedron.

    Returns:
        The volume fraction of the domain in the tetrahedron, as represented by the
        levelset function.

    """
    if psi3 <= 0.0:
        return 1.0
    elif psi0 > 0.0:
        return 0.0
    elif psi2 <= 0.0 < psi3:
        return 1.0 - psi3**3 / ((psi3 - psi0) * (psi3 - psi1) * (psi3 - psi2))
    elif psi1 <= 0.0 < psi2:
        return (
            psi0 * psi1 * (psi2**2 + psi2 * psi3 + psi3**2)
            + psi2 * psi3 * (psi2 * psi3 - (psi0 + psi1) * (psi2 + psi3))
        ) / ((psi0 - psi2) * (psi1 - psi2) * (psi0 - psi3) * (psi1 - psi3))
    elif psi0 <= 0.0 < psi1:
        return -(psi0**3) / ((psi1 - psi0) * (psi2 - psi0) * (psi3 - psi0))
    else:
        return -1.0


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
    mesh = levelset_function.function_space().mesh()
    meshdim = mesh.geometric_dimension()
    mesh_cells = mesh.cells()
    vals = np.sort(levelset_function.compute_vertex_values()[mesh_cells])

    if meshdim == 2:
        s = get_volume_fraction_triangle(vals[:, 0], vals[:, 1], vals[:, 2])
    elif meshdim == 3:
        s = get_volume_fraction_tetrahedron(
            vals[:, 0], vals[:, 1], vals[:, 2], vals[:, 3]
        )
    cell_function.vector()[:] = val_pos + s * (val_neg - val_pos)


def interpolate_by_volume(
    form_neg: ufl.core.expr.Expr,
    form_pos: ufl.core.expr.Expr,
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
    dx = fenics.Measure("dx", mesh)

    indicator_omega = fenics.Function(dg0_space)
    interpolate_levelset_function_to_cells(levelset_function, 1.0, 0.0, indicator_omega)

    test = fenics.TestFunction(function_space)
    form_td = form_neg * indicator_omega + form_pos * (
        fenics.Constant(1.0) - indicator_omega
    )
    arr = fenics.assemble(form_td * test * dx)
    vol = fenics.assemble(test * dx)
    node_function.vector()[:] = arr[:] / vol[:]


def interpolate_by_angle(
    form_neg: ufl.core.expr.Expr,
    form_pos: ufl.core.expr.Expr,
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

    pwc_fun = fenics.project(
        form_neg * indicator_omega
        + form_pos * (fenics.Constant(1.0) - indicator_omega),
        dg0_space,
    )
    node_function.vector().vec().set(0.0)
    node_function.vector().apply("")

    cpp_code = """
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
    module = fenics.compile_cpp_code(cpp_code)
    values, weights = module.interpolate(
        pwc_fun.cpp_object(), node_function.cpp_object()
    )
    values = np.array(values)
    weights = np.array(weights)
    values /= weights
    d2v = fenics.dof_to_vertex_map(cg1_space)
    node_function.vector()[:] = values[d2v]


def interpolate_by_averaging(
    form_neg: ufl.core.expr.Expr,
    form_pos: ufl.core.expr.Expr,
    levelset_function: fenics.Function,
    node_function: fenics.Function,
) -> None:
    """Averages a piecewise constant forms and interpolates them into a CG1 space.

    Args:
        form_neg: The UFL form inside the domain (levelset_function < 0).
        form_pos: The UFL form outside the domain (levelset_function > 0).
        levelset_function: The levelset function representin the domain.
        node_function: The resulting piecewise continuous function.

    """
    node_function.vector().vec().set(0.0)
    node_function.vector().apply("")
    cg1_space = node_function.function_space()
    mesh = cg1_space.mesh()
    dg0_space = fenics.FunctionSpace(mesh, "DG", 0)

    fun_neg = fenics.project(form_neg, dg0_space)
    fun_pos = fenics.project(form_pos, dg0_space)
    dx = fenics.Measure("dx", mesh)

    volumes = fenics.Function(dg0_space)
    volumes.vector()[:] = fenics.assemble(fenics.TestFunction(dg0_space) * dx)
    mesh_cells = mesh.cells()
    vals = np.sort(levelset_function.compute_vertex_values()[mesh_cells])
    volume_fraction = fenics.Function(dg0_space)
    volume_fraction.vector()[:] = get_volume_fraction_triangle(
        vals[:, 0], vals[:, 1], vals[:, 2]
    )[:]

    cpp_code = """
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

    std::tuple<std::vector<double>, std::vector<double>>
    interpolate(std::shared_ptr<dolfin::Function> td_neg,
      std::shared_ptr<dolfin::Function> td_pos,
      std::shared_ptr<dolfin::Function> volumes,
      std::shared_ptr<dolfin::Function> td_vertex,
      std::shared_ptr<dolfin::Function> volume_fraction,
      std::shared_ptr<dolfin::Function> levelset_function,
      double eps)
    {
      auto mesh = td_vertex->function_space()->mesh();
      std::vector<double> td_neg_vec;
      std::vector<double> td_pos_vec;
      std::vector<double> td_vertex_vec;
      std::vector<double> weights;
      std::vector<double> volumes_vec;
      std::vector<double> volume_fraction_vec;
      std::vector<double> psi_vec;
      std::vector<double> ents(3);

      int i = 0;
      double val;
      double vol_frac_neg;
      double vol_frac_pos;
      double elem_volume;

      td_neg->vector()->get_local(td_neg_vec);
      td_pos->vector()->get_local(td_pos_vec);
      td_vertex->vector()->get_local(td_vertex_vec);
      td_vertex->vector()->get_local(weights);
      volumes->vector()->get_local(volumes_vec);
      volume_fraction->vector()->get_local(volume_fraction_vec);
      levelset_function->compute_vertex_values(psi_vec);
      //levelset_function->vector()->get_local(psi_vec);

      for (CellIterator cell(*mesh); !cell.end(); ++cell)
      {
        ents[0] = cell->entities(0)[0];
        ents[1] = cell->entities(0)[1];
        ents[2] = cell->entities(0)[2];
        vol_frac_neg = volume_fraction_vec[i];
        vol_frac_pos = 1.0 - vol_frac_neg;
        elem_volume = volumes_vec[i];

        for (std::size_t k = 0; k < ents.size(); ++k)
        {
          val = psi_vec[ents[k]];

          if (val > eps) {
            td_vertex_vec[ents[k]] += td_pos_vec[i] * vol_frac_pos * elem_volume;
            weights[ents[k]] += vol_frac_pos * elem_volume;
          }
          else if (val < -eps) {
            td_vertex_vec[ents[k]] += td_neg_vec[i] * vol_frac_neg * elem_volume;
            weights[ents[k]] += vol_frac_neg * elem_volume;
          }
          else {
            td_vertex_vec[ents[k]] += (td_neg_vec[i] * vol_frac_neg
              + td_pos_vec[i] * vol_frac_pos) * elem_volume;
            weights[ents[k]] += elem_volume;
          }
        }
        i += 1;
      }
      return std::make_tuple(td_vertex_vec, weights);
    }

    PYBIND11_MODULE(SIGNATURE, m)
    {
      m.def("interpolate", &interpolate);
    }
    """
    module = fenics.compile_cpp_code(cpp_code)

    values, weights = module.interpolate(
        fun_neg.cpp_object(),
        fun_pos.cpp_object(),
        volumes.cpp_object(),
        node_function.cpp_object(),
        volume_fraction.cpp_object(),
        levelset_function.cpp_object(),
        1e-4,
    )
    values = np.array(values)
    weights = np.array(weights)
    values /= weights
    d2v = fenics.dof_to_vertex_map(cg1_space)
    node_function.vector()[:] = values[d2v]

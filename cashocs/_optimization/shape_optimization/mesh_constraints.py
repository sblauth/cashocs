# Copyright (C) 2020-2023 Sebastian Blauth
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

"""Constraints for the mesh / shape in shape optimization."""

import abc

import fenics
import numpy as np
from petsc4py import PETSc
from scipy import sparse

from cashocs import _utils
import cashocs.io

cpp_code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
namespace py = pybind11;

#include <tuple>

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

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

void compute_dihedral_angles(const Cell& cell, std::vector<double>& angs)
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

    Point e10 = p1 - p0;
    Point e20 = p2 - p0;
    Point e30 = p3 - p0;
    Point e21 = p2 - p1;
    Point e31 = p3 - p1;

    Point n031 = e31.cross(e10);
    Point n021 = e21.cross(e10);
    Point n231 = e21.cross(e31);
    Point n230 = e20.cross(e30);

    n031 /= n031.norm();
    n021 /= n021.norm();
    n231 /= n231.norm();
    n230 /= n230.norm();

    angs[0] = acos(n031.dot(n021));
    angs[1] = acos(-n021.dot(n231));
    angs[2] = acos(n231.dot(n031));
    angs[3] = acos(n021.dot(n230));
    angs[4] = acos(n031.dot(-n230));
    angs[5] = acos(n230.dot(n231));
}

py::array_t<double>
triangle_angles(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();
    py::array_t<double> angles(3*n);
    auto buf = angles.request();
    double *ptr = (double *) buf.ptr;

    std::vector<double> angs;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        angles_triangle(*cell, angs);
        ptr[3*idx] = angs[0];
        ptr[3*idx+1] = angs[1];
        ptr[3*idx+2] = angs[2];
        idx += 1;
    }
    return angles;
}

py::array_t<double>
tetrahedron_angles(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();
    py::array_t<double> angles(6*n);
    auto buf = angles.request();
    double *ptr = (double *) buf.ptr;

    std::vector<double> angs;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        compute_dihedral_angles(*cell, angs);
        ptr[6*idx] = angs[0];
        ptr[6*idx+1] = angs[1];
        ptr[6*idx+2] = angs[2];
        ptr[6*idx+3] = angs[3];
        ptr[6*idx+4] = angs[4];
        ptr[6*idx+5] = angs[5];
        idx += 1;
    }
    return angles;
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>>
triangle_angle_gradient(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();

    py::array_t<int> rows(18*n);
    auto buf_rows = rows.request();
    int *ptr_rows = (int *) buf_rows.ptr;

    py::array_t<int> cols(18*n);
    auto buf_cols = cols.request();
    int *ptr_cols = (int *) buf_cols.ptr;

    py::array_t<double> vals(18*n);
    auto buf_vals = vals.request();
    double *ptr_vals = (double *) buf_vals.ptr;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        auto ents = cell->entities(0);
        const std::size_t i0 = ents[0];
        const std::size_t i1 = ents[1];
        const std::size_t i2 = ents[2];

        const Point p0 = Vertex(*mesh, i0).point();
        const Point p1 = Vertex(*mesh, i1).point();
        const Point p2 = Vertex(*mesh, i2).point();
        Point e0 = p1 - p0;
        Point e1 = p2 - p0;
        Point e2 = p2 - p1;

        Point tpa1 = e0.cross(e0.cross(e1));
        Point tpa2 = e1.cross(e1.cross(e0));
        tpa1 /= tpa1.norm();
        tpa2 /= tpa2.norm();

        Point tpb0 = e0.cross(e0.cross(e2));
        Point tpb2 = e2.cross(e2.cross(-e0));
        tpb0 /= tpb0.norm();
        tpb2 /= tpb2.norm();

        Point tpc0 = e1.cross(e1.cross(-e2));
        Point tpc1 = e2.cross(e2.cross(-e1));
        tpc0 /= tpc0.norm();
        tpc1 /= tpc1.norm();

        Point dad0 = 1.0 / e0.norm() * tpa1 + 1.0 / e1.norm() * tpa2;
        Point dad1 = -1.0 / e0.norm() * tpa1;
        Point dad2 = -1.0 / e1.norm() * tpa2;

        Point dbd0 = -1.0 / e0.norm() * tpb0;
        Point dbd1 = 1.0 / e0.norm() * tpb0 + 1.0 / e2.norm() * tpb2;
        Point dbd2 = -1.0 / e2.norm() * tpb2;

        Point dcd0 = -1.0 / e1.norm() * tpc0;
        Point dcd1 = -1.0 / e2.norm() * tpc1;
        Point dcd2 = 1.0 / e1.norm() * tpc0 + 1.0 / e2.norm() * tpc1;

        ptr_rows[18*idx + 0] = 3 * idx;
        ptr_rows[18*idx + 1] = 3 * idx;
        ptr_rows[18*idx + 2] = 3 * idx;
        ptr_rows[18*idx + 3] = 3 * idx;
        ptr_rows[18*idx + 4] = 3 * idx;
        ptr_rows[18*idx + 5] = 3 * idx;

        ptr_cols[18*idx + 0] = 2 * i0 + 0;
        ptr_cols[18*idx + 1] = 2 * i0 + 1;
        ptr_cols[18*idx + 2] = 2 * i1 + 0;
        ptr_cols[18*idx + 3] = 2 * i1 + 1;
        ptr_cols[18*idx + 4] = 2 * i2 + 0;
        ptr_cols[18*idx + 5] = 2 * i2 + 1;

        ptr_vals[18*idx + 0] = dad0[0];
        ptr_vals[18*idx + 1] = dad0[1];
        ptr_vals[18*idx + 2] = dad1[0];
        ptr_vals[18*idx + 3] = dad1[1];
        ptr_vals[18*idx + 4] = dad2[0];
        ptr_vals[18*idx + 5] = dad2[1];

        ptr_rows[18*idx + 6] = 3 * idx + 1;
        ptr_rows[18*idx + 7] = 3 * idx + 1;
        ptr_rows[18*idx + 8] = 3 * idx + 1;
        ptr_rows[18*idx + 9] = 3 * idx + 1;
        ptr_rows[18*idx + 10] = 3 * idx + 1;
        ptr_rows[18*idx + 11] = 3 * idx + 1;

        ptr_cols[18*idx + 6] = 2 * i0 + 0;
        ptr_cols[18*idx + 7] = 2 * i0 + 1;
        ptr_cols[18*idx + 8] = 2 * i1 + 0;
        ptr_cols[18*idx + 9] = 2 * i1 + 1;
        ptr_cols[18*idx + 10] = 2 * i2 + 0;
        ptr_cols[18*idx + 11] = 2 * i2 + 1;

        ptr_vals[18*idx + 6] = dbd0[0];
        ptr_vals[18*idx + 7] = dbd0[1];
        ptr_vals[18*idx + 8] = dbd1[0];
        ptr_vals[18*idx + 9] = dbd1[1];
        ptr_vals[18*idx + 10] = dbd2[0];
        ptr_vals[18*idx + 11] = dbd2[1];

        ptr_rows[18*idx + 12] = 3 * idx + 2;
        ptr_rows[18*idx + 13] = 3 * idx + 2;
        ptr_rows[18*idx + 14] = 3 * idx + 2;
        ptr_rows[18*idx + 15] = 3 * idx + 2;
        ptr_rows[18*idx + 16] = 3 * idx + 2;
        ptr_rows[18*idx + 17] = 3 * idx + 2;

        ptr_cols[18*idx + 12] = 2 * i0 + 0;
        ptr_cols[18*idx + 13] = 2 * i0 + 1;
        ptr_cols[18*idx + 14] = 2 * i1 + 0;
        ptr_cols[18*idx + 15] = 2 * i1 + 1;
        ptr_cols[18*idx + 16] = 2 * i2 + 0;
        ptr_cols[18*idx + 17] = 2 * i2 + 1;

        ptr_vals[18*idx + 12] = dcd0[0];
        ptr_vals[18*idx + 13] = dcd0[1];
        ptr_vals[18*idx + 14] = dcd1[0];
        ptr_vals[18*idx + 15] = dcd1[1];
        ptr_vals[18*idx + 16] = dcd2[0];
        ptr_vals[18*idx + 17] = dcd2[1];

        idx += 1;
    }
    return std::make_tuple(rows, cols, vals);
}

std::tuple<py::array_t<int>, py::array_t<int>, py::array_t<double>>
tetrahedron_angle_gradient(std::shared_ptr<const Mesh> mesh)
{
    size_t idx = 0;
    auto n = mesh->num_cells();

    py::array_t<int> rows(72*n);
    auto buf_rows = rows.request();
    int *ptr_rows = (int *) buf_rows.ptr;

    py::array_t<int> cols(72*n);
    auto buf_cols = cols.request();
    int *ptr_cols = (int *) buf_cols.ptr;

    py::array_t<double> vals(72*n);
    auto buf_vals = vals.request();
    double *ptr_vals = (double *) buf_vals.ptr;

    Eigen::Matrix3d skew_10;
    Eigen::Matrix3d skew_20;
    Eigen::Matrix3d skew_30;
    Eigen::Matrix3d skew_21;
    Eigen::Matrix3d skew_31;
    Eigen::Matrix3d skew_32;

    Eigen::RowVector3d dadk;
    Eigen::RowVector3d dadl;
    Eigen::RowVector3d dbdk;
    Eigen::RowVector3d dbdl;
    Eigen::RowVector3d dcdk;
    Eigen::RowVector3d dcdl;
    Eigen::RowVector3d dddk;
    Eigen::RowVector3d dddl;
    Eigen::RowVector3d dedk;
    Eigen::RowVector3d dedl;
    Eigen::RowVector3d dfdk;
    Eigen::RowVector3d dfdl;

    for (CellIterator cell(*mesh); !cell.end(); ++cell)
    {
        auto ents = cell->entities(0);
        const std::size_t i0 = ents[0];
        const std::size_t i1 = ents[1];
        const std::size_t i2 = ents[2];
        const std::size_t i3 = ents[3];

        const Point p0 = Vertex(*mesh, i0).point();
        const Point p1 = Vertex(*mesh, i1).point();
        const Point p2 = Vertex(*mesh, i2).point();
        const Point p3 = Vertex(*mesh, i3).point();

        Point e10 = p1 - p0;
        Point e20 = p2 - p0;
        Point e30 = p3 - p0;
        Point e21 = p2 - p1;
        Point e31 = p3 - p1;
        Point e32 = p3 - p2;

        skew_10 <<      0.0,  e10.z(), -e10.y(),
                   -e10.z(),      0.0,  e10.x(),
                    e10.y(), -e10.x(),      0.0;

        skew_20 <<      0.0,  e20.z(), -e20.y(),
                   -e20.z(),      0.0,  e20.x(),
                    e20.y(), -e20.x(),      0.0;

        skew_30 <<      0.0,  e30.z(), -e30.y(),
                   -e30.z(),      0.0,  e30.x(),
                    e30.y(), -e30.x(),      0.0;

        skew_21 <<      0.0,  e21.z(), -e21.y(),
                   -e21.z(),      0.0,  e21.x(),
                    e21.y(), -e21.x(),      0.0;

        skew_31 <<      0.0,  e31.z(), -e31.y(),
                   -e31.z(),      0.0,  e31.x(),
                    e31.y(), -e31.x(),      0.0;

        skew_32 <<      0.0,  e32.z(), -e32.y(),
                   -e32.z(),      0.0,  e32.x(),
                    e32.y(), -e32.x(),      0.0;

        Point n031 = e31.cross(e10);
        Point n021 = e21.cross(e10);
        Point n231 = e21.cross(e31);
        Point n230 = e20.cross(e30);

        Point tpa3 = n031.cross(n031.cross(n021));
        Point tpa2 = n021.cross(n021.cross(n031));
        tpa3 /= (tpa3.norm() * n031.norm());
        tpa2 /= (tpa2.norm() * n021.norm());
        dadk << tpa3.x(), tpa3.y(), tpa3.z();
        dadl << tpa2.x(), tpa2.y(), tpa2.z();

        Point tpb0 = n021.cross(n021.cross(n231));
        Point tpb3 = n231.cross(n231.cross(-n021));
        tpb0 /= (tpb0.norm() * n021.norm());
        tpb3 /= (tpb3.norm() * n231.norm());
        dbdk << tpb0.x(), tpb0.y(), tpb0.z();
        dbdl << tpb3.x(), tpb3.y(), tpb3.z();

        Point tpc2 = n231.cross(n231.cross(-n031));
        Point tpc0 = n031.cross(n031.cross(-n231));
        tpc2 /= (tpc2.norm() * n231.norm());
        tpc0 /= (tpc0.norm() * n031.norm());
        dcdk << tpc2.x(), tpc2.y(), tpc2.z();
        dcdl << tpc0.x(), tpc0.y(), tpc0.z();

        Point tpd1 = n021.cross(n021.cross(n230));
        Point tpd3 = n230.cross(n230.cross(n021));
        tpd1 /= (tpd1.norm() * n021.norm());
        tpd3 /= (tpd3.norm() * n230.norm());
        dddk << tpd1.x(), tpd1.y(), tpd1.z();
        dddl << tpd3.x(), tpd3.y(), tpd3.z();

        Point tpe1 = n031.cross(n031.cross(-n230));
        Point tpe2 = n230.cross(n230.cross(n031));
        tpe1 /= (tpe1.norm() * n031.norm());
        tpe2 /= (tpe2.norm() * n230.norm());
        dedk << tpe1.x(), tpe1.y(), tpe1.z();
        dedl << tpe2.x(), tpe2.y(), tpe2.z();

        Point tpf0 = n230.cross(n230.cross(n231));
        Point tpf1 = n231.cross(n231.cross(n230));
        tpf0 /= (tpf0.norm() * n230.norm());
        tpf1 /= (tpf1.norm() * n231.norm());
        dfdk << tpf0.x(), tpf0.y(), tpf0.z();
        dfdl << tpf1.x(), tpf1.y(), tpf1.z();

        auto dad0 = dadk * (-skew_31) + dadl * (-skew_21);
        auto dad1 = dadk * (skew_10 + skew_31) + dadl * (skew_10 + skew_21);
        auto dad2 = dadl * (-skew_10);
        auto dad3 = dadk * (-skew_10);

        auto dbd0 = dbdk * skew_21;
        auto dbd1 = dbdk * (-skew_21 - skew_10) + dbdl * (-skew_21 + skew_31);
        auto dbd2 = dbdk * skew_10 + dbdl * (-skew_31);
        auto dbd3 = dbdl * skew_21;

        auto dcd0 = dcdl * skew_31;
        auto dcd1 = dcdk * (-skew_31 + skew_21) + dcdl * (-skew_31 - skew_10);
        auto dcd2 = dcdk * skew_31;
        auto dcd3 = dcdk * (-skew_21) + dcdl * skew_10;

        auto ddd0 = dddk * (-skew_20 + skew_10) + dddl * (-skew_20 + skew_30);
        auto ddd1 = dddk * skew_20;
        auto ddd2 = dddk * (-skew_10) + dddl * (-skew_30);
        auto ddd3 = dddl * skew_20;

        auto ded0 = dedk * (-skew_30 + skew_10) + dedl * (-skew_30 + skew_20);
        auto ded1 = dedk * skew_30;
        auto ded2 = dedl * skew_30;
        auto ded3 = dedk * (-skew_10) + dedl * (-skew_20);

        auto dfd0 = dfdk * skew_32;
        auto dfd1 = dfdl * skew_32;
        auto dfd2 = dfdk * (-skew_32 - skew_20) + dfdl * (-skew_32 - skew_21);
        auto dfd3 = dfdk * skew_20 + dfdl * skew_21;

        ptr_rows[72*idx + 0] = 6 * idx;
        ptr_rows[72*idx + 1] = 6 * idx;
        ptr_rows[72*idx + 2] = 6 * idx;
        ptr_rows[72*idx + 3] = 6 * idx;
        ptr_rows[72*idx + 4] = 6 * idx;
        ptr_rows[72*idx + 5] = 6 * idx;
        ptr_rows[72*idx + 6] = 6 * idx;
        ptr_rows[72*idx + 7] = 6 * idx;
        ptr_rows[72*idx + 8] = 6 * idx;
        ptr_rows[72*idx + 9] = 6 * idx;
        ptr_rows[72*idx + 10] = 6 * idx;
        ptr_rows[72*idx + 11] = 6 * idx;

        ptr_cols[72*idx + 0] = 3 * i0 + 0;
        ptr_cols[72*idx + 1] = 3 * i0 + 1;
        ptr_cols[72*idx + 2] = 3 * i0 + 2;
        ptr_cols[72*idx + 3] = 3 * i1 + 0;
        ptr_cols[72*idx + 4] = 3 * i1 + 1;
        ptr_cols[72*idx + 5] = 3 * i1 + 2;
        ptr_cols[72*idx + 6] = 3 * i2 + 0;
        ptr_cols[72*idx + 7] = 3 * i2 + 1;
        ptr_cols[72*idx + 8] = 3 * i2 + 2;
        ptr_cols[72*idx + 9] = 3 * i3 + 0;
        ptr_cols[72*idx + 10] = 3 * i3 + 1;
        ptr_cols[72*idx + 11] = 3 * i3 + 2;

        ptr_vals[72*idx + 0] = dad0[0];
        ptr_vals[72*idx + 1] = dad0[1];
        ptr_vals[72*idx + 2] = dad0[2];
        ptr_vals[72*idx + 3] = dad1[0];
        ptr_vals[72*idx + 4] = dad1[1];
        ptr_vals[72*idx + 5] = dad1[2];
        ptr_vals[72*idx + 6] = dad2[0];
        ptr_vals[72*idx + 7] = dad2[1];
        ptr_vals[72*idx + 8] = dad2[2];
        ptr_vals[72*idx + 9] = dad3[0];
        ptr_vals[72*idx + 10] = dad3[1];
        ptr_vals[72*idx + 11] = dad3[2];


        ptr_rows[72*idx + 12] = 6 * idx + 1;
        ptr_rows[72*idx + 13] = 6 * idx + 1;
        ptr_rows[72*idx + 14] = 6 * idx + 1;
        ptr_rows[72*idx + 15] = 6 * idx + 1;
        ptr_rows[72*idx + 16] = 6 * idx + 1;
        ptr_rows[72*idx + 17] = 6 * idx + 1;
        ptr_rows[72*idx + 18] = 6 * idx + 1;
        ptr_rows[72*idx + 19] = 6 * idx + 1;
        ptr_rows[72*idx + 20] = 6 * idx + 1;
        ptr_rows[72*idx + 21] = 6 * idx + 1;
        ptr_rows[72*idx + 22] = 6 * idx + 1;
        ptr_rows[72*idx + 23] = 6 * idx + 1;

        ptr_cols[72*idx + 12] = 3 * i0 + 0;
        ptr_cols[72*idx + 13] = 3 * i0 + 1;
        ptr_cols[72*idx + 14] = 3 * i0 + 2;
        ptr_cols[72*idx + 15] = 3 * i1 + 0;
        ptr_cols[72*idx + 16] = 3 * i1 + 1;
        ptr_cols[72*idx + 17] = 3 * i1 + 2;
        ptr_cols[72*idx + 18] = 3 * i2 + 0;
        ptr_cols[72*idx + 19] = 3 * i2 + 1;
        ptr_cols[72*idx + 20] = 3 * i2 + 2;
        ptr_cols[72*idx + 21] = 3 * i3 + 0;
        ptr_cols[72*idx + 22] = 3 * i3 + 1;
        ptr_cols[72*idx + 23] = 3 * i3 + 2;

        ptr_vals[72*idx + 12] = dbd0[0];
        ptr_vals[72*idx + 13] = dbd0[1];
        ptr_vals[72*idx + 14] = dbd0[2];
        ptr_vals[72*idx + 15] = dbd1[0];
        ptr_vals[72*idx + 16] = dbd1[1];
        ptr_vals[72*idx + 17] = dbd1[2];
        ptr_vals[72*idx + 18] = dbd2[0];
        ptr_vals[72*idx + 19] = dbd2[1];
        ptr_vals[72*idx + 20] = dbd2[2];
        ptr_vals[72*idx + 21] = dbd3[0];
        ptr_vals[72*idx + 22] = dbd3[1];
        ptr_vals[72*idx + 23] = dbd3[2];


        ptr_rows[72*idx + 24] = 6 * idx + 2;
        ptr_rows[72*idx + 25] = 6 * idx + 2;
        ptr_rows[72*idx + 26] = 6 * idx + 2;
        ptr_rows[72*idx + 27] = 6 * idx + 2;
        ptr_rows[72*idx + 28] = 6 * idx + 2;
        ptr_rows[72*idx + 29] = 6 * idx + 2;
        ptr_rows[72*idx + 30] = 6 * idx + 2;
        ptr_rows[72*idx + 31] = 6 * idx + 2;
        ptr_rows[72*idx + 32] = 6 * idx + 2;
        ptr_rows[72*idx + 33] = 6 * idx + 2;
        ptr_rows[72*idx + 34] = 6 * idx + 2;
        ptr_rows[72*idx + 35] = 6 * idx + 2;

        ptr_cols[72*idx + 24] = 3 * i0 + 0;
        ptr_cols[72*idx + 25] = 3 * i0 + 1;
        ptr_cols[72*idx + 26] = 3 * i0 + 2;
        ptr_cols[72*idx + 27] = 3 * i1 + 0;
        ptr_cols[72*idx + 28] = 3 * i1 + 1;
        ptr_cols[72*idx + 29] = 3 * i1 + 2;
        ptr_cols[72*idx + 30] = 3 * i2 + 0;
        ptr_cols[72*idx + 31] = 3 * i2 + 1;
        ptr_cols[72*idx + 32] = 3 * i2 + 2;
        ptr_cols[72*idx + 33] = 3 * i3 + 0;
        ptr_cols[72*idx + 34] = 3 * i3 + 1;
        ptr_cols[72*idx + 35] = 3 * i3 + 2;

        ptr_vals[72*idx + 24] = dcd0[0];
        ptr_vals[72*idx + 25] = dcd0[1];
        ptr_vals[72*idx + 26] = dcd0[2];
        ptr_vals[72*idx + 27] = dcd1[0];
        ptr_vals[72*idx + 28] = dcd1[1];
        ptr_vals[72*idx + 29] = dcd1[2];
        ptr_vals[72*idx + 30] = dcd2[0];
        ptr_vals[72*idx + 31] = dcd2[1];
        ptr_vals[72*idx + 32] = dcd2[2];
        ptr_vals[72*idx + 33] = dcd3[0];
        ptr_vals[72*idx + 34] = dcd3[1];
        ptr_vals[72*idx + 35] = dcd3[2];


        ptr_rows[72*idx + 36] = 6 * idx + 3;
        ptr_rows[72*idx + 37] = 6 * idx + 3;
        ptr_rows[72*idx + 38] = 6 * idx + 3;
        ptr_rows[72*idx + 39] = 6 * idx + 3;
        ptr_rows[72*idx + 40] = 6 * idx + 3;
        ptr_rows[72*idx + 41] = 6 * idx + 3;
        ptr_rows[72*idx + 42] = 6 * idx + 3;
        ptr_rows[72*idx + 43] = 6 * idx + 3;
        ptr_rows[72*idx + 44] = 6 * idx + 3;
        ptr_rows[72*idx + 45] = 6 * idx + 3;
        ptr_rows[72*idx + 46] = 6 * idx + 3;
        ptr_rows[72*idx + 47] = 6 * idx + 3;

        ptr_cols[72*idx + 36] = 3 * i0 + 0;
        ptr_cols[72*idx + 37] = 3 * i0 + 1;
        ptr_cols[72*idx + 38] = 3 * i0 + 2;
        ptr_cols[72*idx + 39] = 3 * i1 + 0;
        ptr_cols[72*idx + 40] = 3 * i1 + 1;
        ptr_cols[72*idx + 41] = 3 * i1 + 2;
        ptr_cols[72*idx + 42] = 3 * i2 + 0;
        ptr_cols[72*idx + 43] = 3 * i2 + 1;
        ptr_cols[72*idx + 44] = 3 * i2 + 2;
        ptr_cols[72*idx + 45] = 3 * i3 + 0;
        ptr_cols[72*idx + 46] = 3 * i3 + 1;
        ptr_cols[72*idx + 47] = 3 * i3 + 2;

        ptr_vals[72*idx + 36] = ddd0[0];
        ptr_vals[72*idx + 37] = ddd0[1];
        ptr_vals[72*idx + 38] = ddd0[2];
        ptr_vals[72*idx + 39] = ddd1[0];
        ptr_vals[72*idx + 40] = ddd1[1];
        ptr_vals[72*idx + 41] = ddd1[2];
        ptr_vals[72*idx + 42] = ddd2[0];
        ptr_vals[72*idx + 43] = ddd2[1];
        ptr_vals[72*idx + 44] = ddd2[2];
        ptr_vals[72*idx + 45] = ddd3[0];
        ptr_vals[72*idx + 46] = ddd3[1];
        ptr_vals[72*idx + 47] = ddd3[2];


        ptr_rows[72*idx + 48] = 6 * idx + 4;
        ptr_rows[72*idx + 49] = 6 * idx + 4;
        ptr_rows[72*idx + 50] = 6 * idx + 4;
        ptr_rows[72*idx + 51] = 6 * idx + 4;
        ptr_rows[72*idx + 52] = 6 * idx + 4;
        ptr_rows[72*idx + 53] = 6 * idx + 4;
        ptr_rows[72*idx + 54] = 6 * idx + 4;
        ptr_rows[72*idx + 55] = 6 * idx + 4;
        ptr_rows[72*idx + 56] = 6 * idx + 4;
        ptr_rows[72*idx + 57] = 6 * idx + 4;
        ptr_rows[72*idx + 58] = 6 * idx + 4;
        ptr_rows[72*idx + 59] = 6 * idx + 4;

        ptr_cols[72*idx + 48] = 3 * i0 + 0;
        ptr_cols[72*idx + 49] = 3 * i0 + 1;
        ptr_cols[72*idx + 50] = 3 * i0 + 2;
        ptr_cols[72*idx + 51] = 3 * i1 + 0;
        ptr_cols[72*idx + 52] = 3 * i1 + 1;
        ptr_cols[72*idx + 53] = 3 * i1 + 2;
        ptr_cols[72*idx + 54] = 3 * i2 + 0;
        ptr_cols[72*idx + 55] = 3 * i2 + 1;
        ptr_cols[72*idx + 56] = 3 * i2 + 2;
        ptr_cols[72*idx + 57] = 3 * i3 + 0;
        ptr_cols[72*idx + 58] = 3 * i3 + 1;
        ptr_cols[72*idx + 59] = 3 * i3 + 2;

        ptr_vals[72*idx + 48] = ded0[0];
        ptr_vals[72*idx + 49] = ded0[1];
        ptr_vals[72*idx + 50] = ded0[2];
        ptr_vals[72*idx + 51] = ded1[0];
        ptr_vals[72*idx + 52] = ded1[1];
        ptr_vals[72*idx + 53] = ded1[2];
        ptr_vals[72*idx + 54] = ded2[0];
        ptr_vals[72*idx + 55] = ded2[1];
        ptr_vals[72*idx + 56] = ded2[2];
        ptr_vals[72*idx + 57] = ded3[0];
        ptr_vals[72*idx + 58] = ded3[1];
        ptr_vals[72*idx + 59] = ded3[2];


        ptr_rows[72*idx + 60] = 6 * idx + 5;
        ptr_rows[72*idx + 61] = 6 * idx + 5;
        ptr_rows[72*idx + 62] = 6 * idx + 5;
        ptr_rows[72*idx + 63] = 6 * idx + 5;
        ptr_rows[72*idx + 64] = 6 * idx + 5;
        ptr_rows[72*idx + 65] = 6 * idx + 5;
        ptr_rows[72*idx + 66] = 6 * idx + 5;
        ptr_rows[72*idx + 67] = 6 * idx + 5;
        ptr_rows[72*idx + 68] = 6 * idx + 5;
        ptr_rows[72*idx + 69] = 6 * idx + 5;
        ptr_rows[72*idx + 70] = 6 * idx + 5;
        ptr_rows[72*idx + 71] = 6 * idx + 5;

        ptr_cols[72*idx + 60] = 3 * i0 + 0;
        ptr_cols[72*idx + 61] = 3 * i0 + 1;
        ptr_cols[72*idx + 62] = 3 * i0 + 2;
        ptr_cols[72*idx + 63] = 3 * i1 + 0;
        ptr_cols[72*idx + 64] = 3 * i1 + 1;
        ptr_cols[72*idx + 65] = 3 * i1 + 2;
        ptr_cols[72*idx + 66] = 3 * i2 + 0;
        ptr_cols[72*idx + 67] = 3 * i2 + 1;
        ptr_cols[72*idx + 68] = 3 * i2 + 2;
        ptr_cols[72*idx + 69] = 3 * i3 + 0;
        ptr_cols[72*idx + 70] = 3 * i3 + 1;
        ptr_cols[72*idx + 71] = 3 * i3 + 2;

        ptr_vals[72*idx + 60] = dfd0[0];
        ptr_vals[72*idx + 61] = dfd0[1];
        ptr_vals[72*idx + 62] = dfd0[2];
        ptr_vals[72*idx + 63] = dfd1[0];
        ptr_vals[72*idx + 64] = dfd1[1];
        ptr_vals[72*idx + 65] = dfd1[2];
        ptr_vals[72*idx + 66] = dfd2[0];
        ptr_vals[72*idx + 67] = dfd2[1];
        ptr_vals[72*idx + 68] = dfd2[2];
        ptr_vals[72*idx + 69] = dfd3[0];
        ptr_vals[72*idx + 70] = dfd3[1];
        ptr_vals[72*idx + 71] = dfd3[2];

        idx += 1;
    }
    return std::make_tuple(rows, cols, vals);
}

PYBIND11_MODULE(SIGNATURE, m)
{
    m.def("triangle_angles", &triangle_angles);
    m.def("tetrahedron_angles", &tetrahedron_angles);
    m.def("triangle_angle_gradient", &triangle_angle_gradient);
    m.def("tetrahedron_angle_gradient", &tetrahedron_angle_gradient);
}
"""
mesh_quality = fenics.compile_cpp_code(cpp_code)


class MeshConstraint(abc.ABC):
    """This class represents a generic constraint on the mesh quality."""

    type: str | None = None

    def __init__(
        self, mesh: fenics.Mesh, deformation_space: fenics.FunctionSpace
    ) -> None:
        """Abstract base class representing an interface for mesh quality constraints.

        Args:
            mesh: The underlying finite element mesh, which is subjected to the mesh
            quality constraints.
            deformation_space: A vector CG1 FEM function space used for defining
                deformations of the mesh via perturbation of identity.

        """
        self.mesh = mesh
        self.dim = self.mesh.geometry().dim()

        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)
        loc0, loc1 = deformation_space.dofmap().ownership_range()
        self.d2v_local = self.d2v[: loc1 - loc0]

        dof_ownership_range = deformation_space.dofmap().ownership_range()
        self.local_offset = dof_ownership_range[0]
        self.no_vertices = self.mesh.num_entities_global(0)
        self.l2g_dofs = deformation_space.dofmap().tabulate_local_to_global_dofs()

        self.global_vertex_indices = self.mesh.topology().global_indices(0)
        function_space = fenics.FunctionSpace(self.mesh, "CG", 1)
        loc0, loc1 = function_space.dofmap().ownership_range()
        d2v = fenics.dof_to_vertex_map(function_space)
        self.global_vertex_indices_owned = self.global_vertex_indices[
            d2v[: loc1 - loc0]
        ]

        self.ghost_offset = self.mesh.topology().ghost_offset(
            self.mesh.topology().dim()
        )

        self.no_constraints = 0
        self.is_necessary: np.ndarray = np.array([False])

    @abc.abstractmethod
    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) \leq 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        pass

    @abc.abstractmethod
    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        pass


class FixedVertexConstraint(MeshConstraint):
    """A discrete mesh constraint for fixed boundaries in shape optimization.

    This constraint is needed to incorporate other mesh constraints for the case that
    some part of the boundary is fixed during optimization. By default, this is already
    incorporated into the gradient deformation, so that this constraint alone is
    not necessary, however, it becomes so when other constraints are considered.
    The constraint is linear, so that the gradient only has to be computed once.
    """

    type = "equality"

    def __init__(
        self,
        mesh: fenics.Mesh,
        boundaries: fenics.MeshFunction,
        config: cashocs.io.Config,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the fixed boundary mesh constraint.

        Args:
            mesh: The corresponding finite element mesh.
            boundaries: The :py:class`fenics.MeshFunction` which is used to mark the
                boundaries of the mesh.
            config: The configuration of the optimization problem.
            deformation_space: The space of vector CG1 elements for mesh deformations.

        """
        super().__init__(mesh, deformation_space)
        self.boundaries = boundaries
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.mesh.init(self.mesh.topology().dim() - 1, 0)
        self.facets = np.array(
            self.mesh.topology()(self.mesh.topology().dim() - 1, 0)()
        ).reshape(-1, self.dim)

        shape_bdry_fix = self.config.getlist("ShapeGradient", "shape_bdry_fix")
        (
            self.fixed_boundary_idcs,
            self.fixed_boundary_coordinates,
        ) = self._compute_fixed_boundary_vertex_indices(shape_bdry_fix)

        shape_bdry_fix_x = self.config.getlist("ShapeGradient", "shape_bdry_fix_x")
        shape_bdry_fix_y = self.config.getlist("ShapeGradient", "shape_bdry_fix_y")
        shape_bdry_fix_z = self.config.getlist("ShapeGradient", "shape_bdry_fix_z")
        (
            self.partially_fixed_boundary_idcs,
            self.partially_fixed_boundary_coordinates,
        ) = self._compute_partially_fixed_boundary_vertex_indices(
            shape_bdry_fix_x, shape_bdry_fix_y, shape_bdry_fix_z
        )

        fixed_dimensions = self.config.getlist("ShapeGradient", "fixed_dimensions")
        (
            self.fixed_dimension_idcs,
            self.fixed_dimension_coordinates,
        ) = self._compute_fixed_dimension_vertex_indices(fixed_dimensions)

        self.fixed_idcs = np.concatenate(
            [
                self.fixed_boundary_idcs,
                self.partially_fixed_boundary_idcs,
                self.fixed_dimension_idcs,
            ]
        )
        self.fixed_coordinates = np.concatenate(
            [
                self.fixed_boundary_coordinates,
                self.partially_fixed_boundary_coordinates,
                self.fixed_dimension_coordinates,
            ]
        )

        self.no_constraints = len(self.fixed_idcs)
        self.is_necessary = np.array([False] * self.no_constraints)
        self.fixed_gradient = self._compute_fixed_gradient()

    def _compute_fixed_boundary_vertex_indices(
        self, shape_bdry_fix: list[int] | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the indices of the mesh vertices on the boundary, which are fixed.

        Args:
            shape_bdry_fix: An index or list of indices of the boundaries, which are
                fixed during shape optimization. This is specified in the configuration
                of the problem.

        Returns:
            A tuple `fixed_indices, fixed_coordinates`, where `fixed_indices` is an
            array containing the (flattened) indices of the fixed coordinates and
            `fixed_coordinates` contains the corresponding values of the fixed vertices.

        """
        bdry_fix_list = _utils.enlist(shape_bdry_fix)

        temp_fixed_idcs = []
        for i in bdry_fix_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            temp_fixed_idcs += idx_i.reshape(-1).tolist()

        fixed_idcs = np.unique(temp_fixed_idcs)

        if len(fixed_idcs) > 0:
            fixed_idcs_global = self.global_vertex_indices[fixed_idcs]
        else:
            fixed_idcs_global = []
        mask = np.isin(fixed_idcs_global, self.global_vertex_indices_owned)
        fixed_idcs_local_owned = fixed_idcs[mask]

        if self.dim == 2:
            fixed_idcs_local = np.array(
                [
                    self.dim * fixed_idcs_local_owned,
                    self.dim * fixed_idcs_local_owned + 1,
                ]
            ).T.reshape(-1)
        elif self.dim == 3:
            fixed_idcs_local = np.array(
                [
                    self.dim * fixed_idcs_local_owned,
                    self.dim * fixed_idcs_local_owned + 1,
                    self.dim * fixed_idcs_local_owned + 2,
                ]
            ).T.reshape(-1)

        if len(fixed_idcs_local) > 0:
            fixed_coordinates = (
                self.mesh.coordinates().copy().reshape(-1)[fixed_idcs_local]
            )
        else:
            fixed_coordinates = np.array([])

        return fixed_idcs_local, fixed_coordinates

    def _compute_partially_fixed_boundary_vertex_indices(
        self,
        shape_bdry_fix_x: list[int] | int,
        shape_bdry_fix_y: list[int] | int,
        shape_bdry_fix_z: list[int] | int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the indices of the partially fixed vertices on the boundary.

        Args:
            shape_bdry_fix_x: An index or list of indices of the boundaries, whose x
                coordinates are fixed during shape optimization. This is specified in
                the configuration of the problem.
            shape_bdry_fix_y: An index or list of indices of the boundaries, whose y
                coordinates are fixed during shape optimization. This is specified in
                the configuration of the problem.
            shape_bdry_fix_z: An index or list of indices of the boundaries, whose z
                coordinates are fixed during shape optimization. This is specified in
                the configuration of the problem.

        Returns:
            A tuple `partially_fixed_indices, partially_fixed_coordinates`, where
            `partially_fixed_indices` is an array containing the (flattened) indices of
            the (partially) fixed coordinates and `partially_fixed_coordinates` contains
            the corresponding values of the fixed vertices.

        """
        bdry_fix_x_list = _utils.enlist(shape_bdry_fix_x)
        bdry_fix_y_list = _utils.enlist(shape_bdry_fix_y)
        bdry_fix_z_list = _utils.enlist(shape_bdry_fix_z)

        temp_fixed_idcs_x = []
        temp_fixed_idcs_y = []
        temp_fixed_idcs_z = []
        for i in bdry_fix_x_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            temp_fixed_idcs_x += idx_i.reshape(-1).tolist()
        for i in bdry_fix_y_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            temp_fixed_idcs_y += idx_i.reshape(-1).tolist()
        for i in bdry_fix_z_list:
            idx_i = self.facets[self.boundaries.where_equal(i)]
            temp_fixed_idcs_y += idx_i.reshape(-1).tolist()

        fixed_idcs_x = np.unique(temp_fixed_idcs_x)
        fixed_idcs_y = np.unique(temp_fixed_idcs_y)
        fixed_idcs_z = np.unique(temp_fixed_idcs_z)

        if len(fixed_idcs_x) > 0:
            fixed_idcs_global_x = self.global_vertex_indices[fixed_idcs_x]
        else:
            fixed_idcs_global_x = []
        if len(fixed_idcs_y) > 0:
            fixed_idcs_global_y = self.global_vertex_indices[fixed_idcs_y]
        else:
            fixed_idcs_global_y = []
        if len(fixed_idcs_z) > 0:
            fixed_idcs_global_z = self.global_vertex_indices[fixed_idcs_z]
        else:
            fixed_idcs_global_z = []

        mask_x = np.isin(fixed_idcs_global_x, self.global_vertex_indices_owned)
        mask_y = np.isin(fixed_idcs_global_y, self.global_vertex_indices_owned)
        mask_z = np.isin(fixed_idcs_global_z, self.global_vertex_indices_owned)
        fixed_idcs_x_local_owned = fixed_idcs_x[mask_x]
        fixed_idcs_y_local_owned = fixed_idcs_y[mask_y]
        fixed_idcs_z_local_owned = fixed_idcs_z[mask_z]

        if self.dim == 2:
            fixed_idcs_local_x = np.array(
                [self.dim * fixed_idcs_x_local_owned], dtype="int64"
            ).T.reshape(-1)
            fixed_idcs_local_y = np.array(
                [self.dim * fixed_idcs_y_local_owned + 1], dtype="int64"
            ).T.reshape(-1)

            partially_fixed_idcs_local = np.concatenate(
                [fixed_idcs_local_x, fixed_idcs_local_y],
            )

        elif self.dim == 3:
            fixed_idcs_local_x = np.array(
                [self.dim * fixed_idcs_x_local_owned], dtype="int64"
            ).T.reshape(-1)
            fixed_idcs_local_y = np.array(
                [self.dim * fixed_idcs_y_local_owned + 1], dtype="int64"
            ).T.reshape(-1)
            fixed_idcs_local_z = np.array(
                [self.dim * fixed_idcs_z_local_owned + 2], dtype="int64"
            ).T.reshape(-1)

            partially_fixed_idcs_local = np.concatenate(
                [fixed_idcs_local_x, fixed_idcs_local_y, fixed_idcs_local_z],
            )

        if len(partially_fixed_idcs_local) > 0:
            partially_fixed_coordinates = (
                self.mesh.coordinates().copy().reshape(-1)[partially_fixed_idcs_local]
            )
        else:
            partially_fixed_coordinates = np.array([])

        return partially_fixed_idcs_local, partially_fixed_coordinates

    def _compute_fixed_dimension_vertex_indices(
        self, fixed_dimensions: list[int] | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the indices of the mesh vertices which are fixed.

        These are fixed through the use of the fixed_dimensions configuration parameter,
        which restricts deformations to some directions.

        Args:
            fixed_dimensions: The (list of) coordinate indices which are fixed. This
                is 0 for x, 1 for y, and 2 for z.

        Returns:
            A tuple `fixed_indices, fixed_coordinates`, where `fixed_indices` is an
            array containing the (flattened) indices of the fixed coordinates and
            `fixed_coordinates` contains the corresponding values of the fixed vertices.

        """
        fixed_dimensions_list = _utils.enlist(fixed_dimensions)

        fixed_idcs = np.arange(self.mesh.num_vertices())
        fixed_idcs_global = self.global_vertex_indices[fixed_idcs]
        mask = np.isin(fixed_idcs_global, self.global_vertex_indices_owned)
        fixed_idcs_local_owned = fixed_idcs[mask]

        fixed_idcs_local = np.array(
            [self.dim * fixed_idcs_local_owned + i for i in fixed_dimensions_list],
            dtype="int64",
        ).T.reshape(-1)

        if len(fixed_idcs_local) > 0:
            fixed_coordinates = (
                self.mesh.coordinates().copy().reshape(-1)[fixed_idcs_local]
            )
        else:
            fixed_coordinates = np.array([])

        return fixed_idcs_local, fixed_coordinates

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) = 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`. The constraint returns the mesh coordinates corresponding
        to the boundaries that shall remain fixed during shape optimization.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        if len(self.fixed_idcs) > 0:
            difference: np.ndarray = (
                coords_seq[self.fixed_idcs] - self.fixed_coordinates
            )
            return difference
        else:
            return np.array([])

    def _compute_fixed_gradient(self) -> sparse.csr_matrix:
        """Computes the gradient of the constraint function.

        As the constraint is linear, the gradient is constant and, so, it is precomputed
        with this method. Also, the gradient is independent of the actual values of the
        mesh coordinates.

        Returns:
            A sparse matrix representation of the constraint gradient.

        """
        if len(self.fixed_idcs) > 0:
            rows = np.arange(len(self.fixed_idcs))

            cols = self.l2g_dofs[self.v2d[self.fixed_idcs]]
            vals = np.ones(len(self.fixed_idcs))
            csr = rows, cols, vals
            shape = (len(self.fixed_idcs), self.no_vertices * self.dim)
            gradient = _utils.linalg.sparse2scipy(csr, shape)
        else:
            shape = (0, self.no_vertices * self.dim)
            gradient = _utils.linalg.sparse2scipy(
                (np.array([]), np.array([]), np.array([])), shape
            )
        return gradient

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        return self.fixed_gradient


class TriangleAngleConstraint(MeshConstraint):
    r"""A mesh quality constraint for the angles of triangles in the mesh.

    This ensures that the all angles :math:`\alpha` of the FEM mesh satisfy the
    constraint :math:`\alpha \geq \alpha_{min}`, which is equivalently rewritten as
    :math:`g(x) \leq 0` with :math:`g(x) = \alpha - \alpha_{min}`, where :math:`x` are
    the mesh coordinates.
    """
    type = "inequality"

    def __init__(
        self,
        mesh: fenics.Mesh,
        config: cashocs.io.Config,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the mesh quality constraint for the triangle angles.

        Args:
            mesh: The corresponding FEM mesh.
            config: The configuration of the optimization problem.
            deformation_space: A space of vector CG1 elements for mesh deformations.

        """
        super().__init__(mesh, deformation_space)
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.min_angle = self._compute_minimum_angle()

        self.cells = self.mesh.cells()

        self.no_constraints = 3 * self.ghost_offset
        self.is_necessary = np.array([True] * self.no_constraints)

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) <= 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`. The constraint returns :math:`\alpha_{min} - \alpha`,
        where :math:`\alpha` is one angle of a triangle and :math:`\alpha_{min}` is the
        minimum feasible angle in each triangle.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        values: np.ndarray = mesh_quality.triangle_angles(self.mesh)
        values = values[: 3 * self.ghost_offset]
        values = self.min_angle - values

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        return values

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        rows, cols, vals = mesh_quality.triangle_angle_gradient(self.mesh)
        rows = rows[: 18 * self.ghost_offset]
        cols = cols[: 18 * self.ghost_offset]
        vals = vals[: 18 * self.ghost_offset]

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        cols_local = self.v2d[cols]
        cols = self.l2g_dofs[cols_local]

        csr = rows, cols, vals
        shape = (int(3 * self.ghost_offset), self.no_vertices * self.dim)
        gradient = _utils.linalg.sparse2scipy(csr, shape)

        return gradient

    def _compute_minimum_angle(self) -> np.ndarray:
        constant_min_angle = self.config.getfloat("MeshQualityConstraints", "min_angle")
        constant_min_angle *= 2 * np.pi / 360.0

        feasible_angle_reduction_factor = self.config.getfloat(
            "MeshQualityConstraints", "feasible_angle_reduction_factor"
        )

        initial_angles = mesh_quality.triangle_angles(self.mesh).reshape(-1, 3)
        initial_angles = initial_angles[: self.ghost_offset]

        minimum_initial_angles = np.min(initial_angles, axis=1)
        cellwise_minimum_angle = (
            feasible_angle_reduction_factor * minimum_initial_angles
        )
        cellwise_minimum_angle = np.repeat(cellwise_minimum_angle, 3)

        if constant_min_angle > 0.0 and feasible_angle_reduction_factor > 0.0:
            minimum_angle = np.minimum(cellwise_minimum_angle, constant_min_angle)
        elif constant_min_angle > 0.0 and feasible_angle_reduction_factor == 0.0:
            minimum_angle = constant_min_angle
        elif feasible_angle_reduction_factor > 0.0 and constant_min_angle == 0.0:
            minimum_angle = cellwise_minimum_angle

        return minimum_angle


class DihedralAngleConstraint(MeshConstraint):
    r"""A mesh quality constraint for the angles of triangles in the mesh.

    This ensures that the all angles :math:`\alpha` of the FEM mesh satisfy the
    constraint :math:`\alpha \geq \alpha_{min}`, which is equivalently rewritten as
    :math:`g(x) \leq 0` with :math:`g(x) = \alpha - \alpha_{min}`, where :math:`x` are
    the mesh coordinates.
    """
    type = "inequality"

    def __init__(
        self,
        mesh: fenics.Mesh,
        config: cashocs.io.Config,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the mesh quality constraint for the triangle angles.

        Args:
            mesh: The corresponding FEM mesh.
            config: The configuration of the optimization problem.
            deformation_space: A space of vector CG1 elements for mesh deformations.

        """
        super().__init__(mesh, deformation_space)
        self.config = config

        self.dim = self.mesh.geometry().dim()
        self.min_angle = self._compute_minimum_angle()
        self.cells = self.mesh.cells()

        self.no_constraints = 6 * self.ghost_offset
        self.is_necessary = np.array([True] * self.no_constraints)

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        r"""Evaluates the constaint function at the current iterate.

        This computes :math:`g(x)` where the constraint is given by
        :math:`g(x) <= 0` and :math:`x` corresponds to mesh coordinates, i.e.,
        `coords_seq`. The constraint returns :math:`\alpha_{min} - \alpha`,
        where :math:`\alpha` is one angle of a triangle and :math:`\alpha_{min}` is the
        minimum feasible angle in each triangle.

        Args:
            coords_seq: The (flattened) list of vertex coordinates of the mesh.

        Returns:
            A numpy array containing the values of the constraint functions.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        values: np.ndarray = mesh_quality.tetrahedron_angles(self.mesh)
        values = values[: 6 * self.ghost_offset]
        values = self.min_angle - values

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        return values

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        old_coords = self.mesh.coordinates().copy()
        self.mesh.coordinates()[:, :] = coords_seq.reshape(-1, self.dim)
        self.mesh.bounding_box_tree().build(self.mesh)

        rows, cols, vals = mesh_quality.tetrahedron_angle_gradient(self.mesh)
        rows = rows[: 72 * self.ghost_offset]
        cols = cols[: 72 * self.ghost_offset]
        vals = vals[: 72 * self.ghost_offset]

        self.mesh.coordinates()[:, :] = old_coords
        self.mesh.bounding_box_tree().build(self.mesh)

        cols_local = self.v2d[cols]
        cols = self.l2g_dofs[cols_local]

        csr = rows, cols, vals
        shape = (int(6 * self.ghost_offset), self.no_vertices * self.dim)
        gradient = _utils.linalg.sparse2scipy(csr, shape)

        return gradient

    def _compute_minimum_angle(self) -> np.ndarray:
        constant_min_angle = self.config.getfloat("MeshQualityConstraints", "min_angle")
        constant_min_angle *= 2 * np.pi / 360.0

        feasible_angle_reduction_factor = self.config.getfloat(
            "MeshQualityConstraints", "feasible_angle_reduction_factor"
        )
        initial_angles = mesh_quality.tetrahedron_angles(self.mesh).reshape(-1, 6)
        initial_angles = initial_angles[: self.ghost_offset]

        minimum_initial_angles = np.min(initial_angles, axis=1)
        cellwise_minimum_angle = (
            feasible_angle_reduction_factor * minimum_initial_angles
        )
        cellwise_minimum_angle = np.repeat(cellwise_minimum_angle, 6)

        if constant_min_angle > 0.0 and feasible_angle_reduction_factor > 0.0:
            minimum_angle = np.minimum(cellwise_minimum_angle, constant_min_angle)
        elif constant_min_angle > 0.0 and feasible_angle_reduction_factor == 0.0:
            minimum_angle = constant_min_angle
        elif feasible_angle_reduction_factor > 0.0 and constant_min_angle == 0.0:
            minimum_angle = cellwise_minimum_angle

        return minimum_angle


class ConstraintManager:
    """This class manages all (selected) mesh quality constraints.

    It is used to treat arbitrary constraints (on the mesh coordinates) in a unified
    manner.
    """

    def __init__(
        self,
        config: cashocs.io.Config,
        mesh: fenics.Mesh,
        boundaries: fenics.MeshFunction,
        deformation_space: fenics.FunctionSpace,
    ) -> None:
        """Initializes the constraint manager.

        Args:
            config: The configuration of the optimization problem.
            mesh: The underlying FEM mesh.
            boundaries: The :py:class`fenics.MeshFunction` used to mark the boundaries
                of the mesh.
            deformation_space: A :py:class`fenics.FunctionSpace` of vector CG1 elements
                for mesh deformations.

        """
        self.config = config
        self.mesh = mesh
        self.boundaries = boundaries

        self.has_constraints = (
            self.config.getfloat("MeshQualityConstraints", "min_angle") > 0.0
        ) or (
            self.config.getfloat(
                "MeshQualityConstraints", "feasible_angle_reduction_factor"
            )
            > 0.0
        )
        self.v2d = fenics.vertex_to_dof_map(deformation_space)
        self.d2v = fenics.dof_to_vertex_map(deformation_space)
        loc0, loc1 = deformation_space.dofmap().ownership_range()
        self.d2v_local = self.d2v[: loc1 - loc0]

        fun = fenics.Function(deformation_space)
        self.local_petsc_size = fun.vector().vec().getSizes()[0]
        self.comm = self.mesh.mpi_comm()

        self.constraints: list[MeshConstraint] = []
        self.constraint_tolerance: float = self.config.getfloat(
            "MeshQualityConstraints", "tol"
        )
        self.no_constraints = 0

        if self.has_constraints:
            self.constraints.append(
                FixedVertexConstraint(
                    self.mesh, self.boundaries, self.config, deformation_space
                )
            )

            if self.mesh.geometry().dim() == 2:
                self.constraints.append(
                    TriangleAngleConstraint(self.mesh, self.config, deformation_space)
                )
            elif self.mesh.geometry().dim() == 3:
                self.constraints.append(
                    DihedralAngleConstraint(self.mesh, self.config, deformation_space)
                )

        if self.has_constraints:
            self.inequality_mask = self._compute_inequality_mask()
            self.no_constraints = len(self.inequality_mask)
            necessary_constraints = []
            for constraint in self.constraints:
                necessary_constraints.append(constraint.is_necessary)
            self.necessary_constraints = np.concatenate(necessary_constraints)

    def _compute_inequality_mask(self) -> np.ndarray:
        """Computes the mask which distinguishes between (in-)equality constaints.

        Returns:
            A boolean array (a mask) which is `True` when the corresponding constraint
            is an inequality constraint.

        """
        mask = []
        for constraint in self.constraints:
            if constraint.type == "equality":
                mask += [False] * constraint.no_constraints
            elif constraint.type == "inequality":
                mask += [True] * constraint.no_constraints

        return np.array(mask)

    def evaluate(self, coords_seq: np.ndarray) -> np.ndarray:
        """Evaluates all (selected) mesh quality constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A numpy array with the values of the constraint functions.

        """
        result = []
        for constraint in self.constraints:
            result.append(constraint.evaluate(coords_seq))

        if len(result) > 0:
            self.function_values: np.ndarray = np.concatenate(result)
        else:
            return np.array([])

        return self.function_values

    def evaluate_active(
        self, coords_seq: np.ndarray, active_idx: np.ndarray
    ) -> np.ndarray:
        """Evaluates only the constraints corresponding to the working set.

        Args:
            coords_seq: The flattened list of mesh coordinates.
            active_idx: The mask of active indices corresponding to the working set.

        Returns:
            A numpy array containing only the values of the constraint functions that
            are in the working set.

        """
        function_values: np.ndarray = self.evaluate(coords_seq)[active_idx]
        return function_values

    def compute_gradient(self, coords_seq: np.ndarray) -> sparse.csr_matrix:
        """Computes the gradient of the constraint functions.

        Args:
            coords_seq: The flattened list of mesh coordinates.

        Returns:
            A sparse matrix representation of the gradient.

        """
        result = []
        for constraint in self.constraints:
            result.append(constraint.compute_gradient(coords_seq))

        if len(result) > 0:
            self.gradient = sparse.vstack(result)
        else:
            return np.array([])

        return self.gradient

    def compute_active_gradient(
        self, active_idx: np.ndarray, constraint_gradient: sparse.csr_matrix
    ) -> PETSc.Mat:
        """Computes the gradient of those constraint functions that are active.

        Args:
            active_idx: The boolean array corresponding to the current working set.
            constraint_gradient: The entire (sparse) constraint gradient.

        Returns:
            A sparse matrix of the constraint gradient w.r.t. the working set.

        """
        scipy_matrix = constraint_gradient[active_idx]

        petsc_matrix = _utils.linalg.scipy2petsc(
            scipy_matrix,
            self.mesh.mpi_comm(),
            local_size=self.local_petsc_size,
        )
        return petsc_matrix

    def compute_active_set(self, coords_seq: np.ndarray) -> np.ndarray:
        """Computes the working set.

        The working set is the set of constraints which are active at a given point.

        Args:
            coords_seq: The list of flattened mesh coordinates.

        Returns:
            A boolean array which is `True` in component `i` if constraint `i` is
            active.

        """
        function_values = self.evaluate(coords_seq)
        result: np.ndarray = np.abs(function_values) <= self.constraint_tolerance
        return result

    def is_necessary(self, active_idx: np.ndarray) -> bool:
        """Checks, whether constraint treatment is necessary based on the working set.

        This can be `False`, e.g., when only fixed boundary constraints are active (as
        these are directly treated in the gradient deformation) or no constraints are
        currently active.

        Args:
            active_idx: The boolean array representing the working set.

        Returns:
            A boolean, which is `True` when active treatment of the constraints is
            necessary and `False` otherwise.

        """
        necessary_constraints_gathered = self.comm.allgather(self.necessary_constraints)
        necessary_constraints_global = np.concatenate(necessary_constraints_gathered)
        active_idx_gathered = self.comm.allgather(active_idx)
        active_idx_global = np.concatenate(active_idx_gathered)

        result = bool(
            np.any(np.logical_and(necessary_constraints_global, active_idx_global))
        )

        return result

    def is_feasible(self, coords_seq: np.ndarray) -> np.ndarray:
        """Checks, whether a given point is feasible w.r.t. all constraints.

        Args:
            coords_seq: The list of flattened mesh coordinates for the trial point.

        Returns:
            A boolean array, which is `True` in component `i` if the corresponding
            constraint is not violated, else `False`.

        """
        function_values = self.evaluate(coords_seq)
        function_values_list = self.comm.allgather(function_values)
        function_values_global = np.concatenate(function_values_list)

        result: np.ndarray = np.less_equal(
            function_values_global, self.constraint_tolerance
        )
        return result

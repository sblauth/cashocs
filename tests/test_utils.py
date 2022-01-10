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

"""Tests for the utils module.

"""

import os
import subprocess

import fenics
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs._exceptions import InputError


dir_path = os.path.dirname(os.path.realpath(__file__))

rng = np.random.RandomState(300696)
mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
V = fenics.FunctionSpace(mesh, "CG", 1)


def test_summation():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = fenics.TestFunction(V)

    for i in range(dim):
        temp = fenics.Function(V)
        temp.vector()[:] = rng.rand(V.dim())
        funcs.append(temp)

    F = cashocs.utils.summation([funcs[i] * test * dx for i in range(dim)])
    F_exact = (funcs[0] + funcs[1] + funcs[2]) * test * dx

    b = fenics.assemble(F)[:]
    b_exact = fenics.assemble(F_exact)[:]

    assert cashocs.utils.summation(a) == 10
    assert abs(cashocs.utils.summation(b) - np.sum(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert fenics.assemble(pow(cashocs.utils.summation([]), 2) * dx) == 0.0


def test_multiplication():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = fenics.TestFunction(V)

    for i in range(dim):
        temp = fenics.Function(V)
        temp.vector()[:] = rng.rand(V.dim())
        funcs.append(temp)

    F = cashocs.utils.multiplication([funcs[i] for i in range(dim)]) * test * dx
    F_exact = (funcs[0] * funcs[1] * funcs[2]) * test * dx

    b = fenics.assemble(F)[:]
    b_exact = fenics.assemble(F_exact)[:]

    assert cashocs.utils.multiplication(a) == 24
    assert abs(cashocs.utils.multiplication(b) - np.prod(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert (
        fenics.assemble(cashocs.utils.multiplication([]) * dx) / fenics.assemble(1 * dx)
        == 1.0
    )


def test_create_bcs():
    trial = fenics.TrialFunction(V)
    test = fenics.TestFunction(V)
    a = fenics.inner(fenics.grad(trial), fenics.grad(test)) * dx
    L = fenics.Constant(1) * test * dx

    bc_val = rng.rand()
    bc1 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 1)
    bc2 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 2)
    bc3 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 3)
    bc4 = fenics.DirichletBC(V, fenics.Constant(bc_val), boundaries, 4)
    bcs_ex = [bc1, bc2, bc3, bc4]
    bcs = cashocs.create_dirichlet_bcs(
        V, fenics.Constant(bc_val), boundaries, [1, 2, 3, 4]
    )

    u_ex = fenics.Function(V)
    u = fenics.Function(V)

    fenics.solve(a == L, u_ex, bcs_ex)
    fenics.solve(a == L, u, bcs)

    assert np.allclose(u_ex.vector()[:], u.vector()[:])
    assert abs(u(0, 0) - bc_val) < 1e-14


def test_interpolator():
    W = fenics.FunctionSpace(mesh, "CG", 2)
    X = fenics.FunctionSpace(mesh, "DG", 0)

    interp_W = cashocs.utils.Interpolator(V, W)
    interp_X = cashocs.utils.Interpolator(V, X)

    func_V = fenics.Function(V)
    func_V.vector()[:] = rng.rand(V.dim())

    fen_W = fenics.interpolate(func_V, W)
    fen_X = fenics.interpolate(func_V, X)

    cas_W = interp_W.interpolate(func_V)
    cas_X = interp_X.interpolate(func_V)

    assert np.allclose(fen_W.vector()[:], cas_W.vector()[:])
    assert np.allclose(fen_X.vector()[:], cas_X.vector()[:])


def test_create_named_bcs():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    cashocs._cli.convert(
        [f"{dir_path}/mesh/named_mesh.msh", f"{dir_path}/mesh/named_mesh.xdmf"]
    )

    mesh_, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        f"{dir_path}/mesh/named_mesh.xdmf"
    )
    V_ = fenics.FunctionSpace(mesh_, "CG", 1)

    bcs_str = cashocs.create_dirichlet_bcs(
        V_, fenics.Constant(0.0), boundaries, ["inlet", "wall", "outlet"]
    )
    bcs_int = cashocs.create_dirichlet_bcs(
        V_, fenics.Constant(0.0), boundaries, [1, 2, 3]
    )
    bcs_mixed = cashocs.create_dirichlet_bcs(
        V_, fenics.Constant(0.0), boundaries, ["inlet", 2, "outlet"]
    )

    fun1 = fenics.Function(V_)
    fun1.vector()[:] = rng.rand(V_.dim())

    fun2 = fenics.Function(V_)
    fun2.vector()[:] = fun1.vector()[:]

    fun3 = fenics.Function(V_)
    fun3.vector()[:] = fun1.vector()[:]

    [bc.apply(fun1.vector()) for bc in bcs_int]
    [bc.apply(fun2.vector()) for bc in bcs_str]
    [bc.apply(fun3.vector()) for bc in bcs_mixed]

    for i in range(len(bcs_str)):
        assert np.max(np.abs(fun1.vector()[:] - fun2.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun1.vector()[:] - fun3.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun2.vector()[:] - fun3.vector()[:])) <= 1e-14

    with pytest.raises(InputError) as e_info:
        cashocs.create_dirichlet_bcs(V_, fenics.Constant(0.0), boundaries, "fantasy")
        assert "The string you have supplied is not associated with a boundary" in str(
            e_info.value
        )

    assert os.path.isfile(f"{dir_path}/mesh/named_mesh.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh.h5")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh_subdomains.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh_subdomains.h5")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh_boundaries.xdmf")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh_boundaries.h5")
    assert os.path.isfile(f"{dir_path}/mesh/named_mesh_physical_groups.json")

    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_subdomains.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_subdomains.h5"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_boundaries.xdmf"], check=True)
    subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_boundaries.h5"], check=True)
    subprocess.run(
        ["rm", f"{dir_path}/mesh/named_mesh_physical_groups.json"], check=True
    )


def test_deprecated():
    cfg1 = cashocs.create_config(f"{dir_path}/config_ocp.ini")
    cfg2 = cashocs.load_config(f"{dir_path}/config_ocp.ini")

    assert cfg1 == cfg2

    zero = fenics.Constant(0.0)
    bcs1 = cashocs.create_dirichlet_bcs(V, zero, boundaries, [1])
    bcs2 = cashocs.create_bcs_list(V, zero, boundaries, [1])

    assert bcs1[0].value() == bcs2[0].value()
    assert bcs1[0].function_space() == bcs2[0].function_space()
    assert bcs1[0].domain_args == bcs2[0].domain_args

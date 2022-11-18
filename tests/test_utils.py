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

import pathlib
import subprocess

from fenics import *
import numpy as np
import pytest

import cashocs
import cashocs._cli
from cashocs._exceptions import InputError

dir_path = str(pathlib.Path(__file__).parent)
config = cashocs.load_config(dir_path + "/config_ocp.ini")
rng = np.random.RandomState(300696)
mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = cashocs.IntegralFunctional(
    Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx
)

ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)


def evaluate_function(u, x):
    comm = u.function_space().mesh().mpi_comm()
    if comm.size == 1:
        return u(*x)

    # Find whether the point lies on the partition of the mesh local
    # to this process, and evaulate u(x)
    cell, distance = mesh.bounding_box_tree().compute_closest_entity(Point(*x))
    u_eval = u(*x) if distance < DOLFIN_EPS else None

    # Gather the results on process 0
    comm = mesh.mpi_comm()
    computed_u = comm.gather(u_eval, root=0)

    # Verify the results on process 0 to ensure we see the same value
    # on a process boundary
    if comm.rank == 0:
        global_u_evals = np.array(
            [y for y in computed_u if y is not None], dtype=np.double
        )
        assert np.all(np.abs(global_u_evals[0] - global_u_evals) < 1e-9)

        computed_u = global_u_evals[0]
    else:
        computed_u = None

    # Broadcast the verified result to all processes
    computed_u = comm.bcast(computed_u, root=0)

    return computed_u


def test_summation():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = TestFunction(V)

    for i in range(dim):
        temp = Function(V)
        temp.vector().set_local(rng.rand(temp.vector().local_size()))
        temp.vector().apply("")
        funcs.append(temp)

    F = cashocs._utils.summation([funcs[i] * test * dx for i in range(dim)])
    F_exact = (funcs[0] + funcs[1] + funcs[2]) * test * dx

    b = assemble(F)[:]
    b_exact = assemble(F_exact)[:]

    assert cashocs._utils.summation(a) == 10
    assert abs(cashocs._utils.summation(b) - np.sum(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert assemble(pow(cashocs._utils.summation([]), 2) * dx) == 0.0


def test_multiplication():
    a = [1, 2, 3, 4]

    dim = 3
    funcs = []
    test = TestFunction(V)

    for i in range(dim):
        temp = Function(V)
        temp.vector().set_local(rng.rand(temp.vector().local_size()))
        temp.vector().apply("")
        funcs.append(temp)

    F = cashocs._utils.multiplication([funcs[i] for i in range(dim)]) * test * dx
    F_exact = (funcs[0] * funcs[1] * funcs[2]) * test * dx

    b = assemble(F)[:]
    b_exact = assemble(F_exact)[:]

    assert cashocs._utils.multiplication(a) == 24
    assert abs(cashocs._utils.multiplication(b) - np.prod(b)) < 1e-14
    assert np.allclose(b, b_exact)
    assert assemble(cashocs._utils.multiplication([]) * dx) / assemble(1 * dx) == 1.0


def test_create_bcs():
    rng = np.random.RandomState(300696)

    trial = TrialFunction(V)
    test = TestFunction(V)
    a = inner(grad(trial), grad(test)) * dx
    L = Constant(1) * test * dx

    bc_val = rng.rand()
    bc1 = DirichletBC(V, Constant(bc_val), boundaries, 1)
    bc2 = DirichletBC(V, Constant(bc_val), boundaries, 2)
    bc3 = DirichletBC(V, Constant(bc_val), boundaries, 3)
    bc4 = DirichletBC(V, Constant(bc_val), boundaries, 4)
    bcs_ex = [bc1, bc2, bc3, bc4]
    bcs = cashocs.create_dirichlet_bcs(V, Constant(bc_val), boundaries, [1, 2, 3, 4])

    u_ex = Function(V)
    u = Function(V)

    solve(a == L, u_ex, bcs_ex)
    solve(a == L, u, bcs)

    assert np.allclose(u_ex.vector()[:], u.vector()[:])
    assert abs(evaluate_function(u, (0, 0)) - bc_val) < 1e-14


def test_interpolator():
    W = FunctionSpace(mesh, "CG", 2)
    X = FunctionSpace(mesh, "DG", 0)

    interp_W = cashocs.Interpolator(V, W)
    interp_X = cashocs.Interpolator(V, X)

    expr = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
    func_V = interpolate(expr, V)

    fen_W = interpolate(func_V, W)
    fen_X = interpolate(func_V, X)

    cas_W = interp_W.interpolate(func_V)
    cas_X = interp_X.interpolate(func_V)

    assert np.allclose(fen_W.vector()[:], cas_W.vector()[:])
    assert np.allclose(fen_X.vector()[:], cas_X.vector()[:])


def test_create_named_bcs():
    dir_path = str(pathlib.Path(__file__).parent)
    cashocs.convert(
        f"{dir_path}/mesh/named_mesh.msh", f"{dir_path}/mesh/named_mesh.xdmf"
    )

    mesh_, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
        f"{dir_path}/mesh/named_mesh.xdmf"
    )
    V_ = FunctionSpace(mesh_, "CG", 1)

    bcs_str = cashocs.create_dirichlet_bcs(
        V_, Constant(0.0), boundaries, ["inlet", "wall", "outlet"]
    )
    bcs_int = cashocs.create_dirichlet_bcs(V_, Constant(0.0), boundaries, [1, 2, 3])
    bcs_mixed = cashocs.create_dirichlet_bcs(
        V_, Constant(0.0), boundaries, ["inlet", 2, "outlet"]
    )

    fun1 = Function(V_)
    fun1.vector().set_local(rng.rand(fun1.vector().local_size()))
    fun1.vector().apply("")

    fun2 = Function(V_)
    fun2.vector().vec().aypx(0.0, fun1.vector().vec())
    fun2.vector().apply("")

    fun3 = Function(V_)
    fun3.vector().vec().aypx(0.0, fun1.vector().vec())
    fun3.vector().apply("")

    [bc.apply(fun1.vector()) for bc in bcs_int]
    [bc.apply(fun2.vector()) for bc in bcs_str]
    [bc.apply(fun3.vector()) for bc in bcs_mixed]

    for i in range(len(bcs_str)):
        assert np.max(np.abs(fun1.vector()[:] - fun2.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun1.vector()[:] - fun3.vector()[:])) <= 1e-14
        assert np.max(np.abs(fun2.vector()[:] - fun3.vector()[:])) <= 1e-14

    with pytest.raises(InputError) as e_info:
        cashocs.create_dirichlet_bcs(V_, Constant(0.0), boundaries, "fantasy")
        assert "The string you have supplied is not associated with a boundary" in str(
            e_info.value
        )

    MPI.barrier(MPI.comm_world)
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_subdomains.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_subdomains.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_boundaries.xdmf").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_boundaries.h5").is_file()
    assert pathlib.Path(f"{dir_path}/mesh/named_mesh_physical_groups.json").is_file()
    MPI.barrier(MPI.comm_world)

    if MPI.rank(MPI.comm_world) == 0:
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.xdmf"], check=True)
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh.h5"], check=True)
        subprocess.run(
            ["rm", f"{dir_path}/mesh/named_mesh_subdomains.xdmf"], check=True
        )
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_subdomains.h5"], check=True)
        subprocess.run(
            ["rm", f"{dir_path}/mesh/named_mesh_boundaries.xdmf"], check=True
        )
        subprocess.run(["rm", f"{dir_path}/mesh/named_mesh_boundaries.h5"], check=True)
        subprocess.run(
            ["rm", f"{dir_path}/mesh/named_mesh_physical_groups.json"], check=True
        )
    MPI.barrier(MPI.comm_world)


def test_moreau_yosida_regularization():
    u.vector().vec().set(1e3)
    u.vector().apply("")
    y_bar = 1e-1
    y_low = 1e-2
    gamma = 1e3
    reg = cashocs._utils.moreau_yosida_regularization(
        y, gamma, dx, upper_threshold=y_bar, lower_threshold=y_low
    )

    max = cashocs._utils.max_
    min = cashocs._utils.min_
    reg_ana = (
        1 / (2 * gamma) * pow(max(gamma * (y - y_bar), 0.0), 2) * dx
        + 1 / (2 * gamma) * pow(min(gamma * (y - y_low), 0.0), 2) * dx
    )

    ocp.compute_state_variables()
    assert np.abs(assemble(reg) - assemble(reg_ana)) < 1e-14

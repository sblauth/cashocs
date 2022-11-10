"""
Created on 04/02/2022, 13.31

@author: blauths
"""

import pathlib
import subprocess

from fenics import *
import numpy as np

import cashocs
import cashocs._cli
import cashocs.space_mapping.shape_optimization as sosm

dir_path = str(pathlib.Path(__file__).parent)

cashocs.set_log_level(cashocs.LogLevel.ERROR)

cfg = cashocs.load_config(f"{dir_path}/config_sosm.ini")
Re_f = 50.0
Re_c = 0.0
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
    f"{dir_path}/sm_mesh/mesh.xdmf"
)
initial_coordinates = mesh.coordinates().copy()

u_in = Expression(("6.0*(0.0 - x[1])*(x[1] + 1.0)", "0.0"), degree=2)
u_des = Expression(
    (
        "0.0",
        "(x[0] >= 1.75 && x[0] <= 2.5) ? -C*6.0/pow(0.75, 2)*(x[0] - 1.75)*(2.5 - x[0]) : "
        + "(x[0] >= 3.5 && x[0] <= 4.25) ? -C*6.0/pow(0.75, 2)*(x[0] - 3.5)*(4.25 - x[0]) : "
        + "(x[0] >= 5.25 && x[0] <= 6) ? -C*6.0/pow(0.75, 2)*(x[0] - 5.25)*(6 - x[0]) : 0.0",
    ),
    degree=2,
    C=1.0 / 3.0 / 0.75,
    domain=mesh,
)


v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))


class FineModel(sosm.FineModel):
    def __init__(self, mesh, V):
        super().__init__(mesh)
        self.V_coarse = V
        self.u = Function(self.V_coarse.sub(0).collapse())

    def solve_and_evaluate(self):
        cashocs.io.write_out_mesh(
            self.mesh, f"{dir_path}/sm_mesh/mesh.msh", f"{dir_path}/sm_mesh/fine.msh"
        )
        cashocs.convert(f"{dir_path}/sm_mesh/fine.msh", f"{dir_path}/sm_mesh/fine.xdmf")

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
            f"{dir_path}/sm_mesh/fine.xdmf"
        )
        v_elem = VectorElement("CG", mesh.ufl_cell(), 2)
        p_elem = FiniteElement("CG", mesh.ufl_cell(), 1)
        V = FunctionSpace(mesh, MixedElement([v_elem, p_elem]))

        up = Function(V)
        up.set_allow_extrapolation(True)
        u, p = split(up)
        v, q = TestFunctions(V)

        F = (
            inner(grad(u), grad(v)) * dx
            + Constant(Re_f) * inner(grad(u) * u, v) * dx
            - p * div(v) * dx
            - q * div(u) * dx
        )

        bc_in = DirichletBC(V.sub(0), u_in, boundaries, 1)
        bcs_wall = cashocs.create_dirichlet_bcs(
            V.sub(0), Constant((0.0, 0.0)), boundaries, [2, 3, 4]
        )
        bc_out = DirichletBC(V.sub(0).sub(0), Constant(0.0), boundaries, 5)
        bc_pressure = DirichletBC(V.sub(1), Constant(0.0), boundaries, 5)
        bcs = [bc_in] + bcs_wall + [bc_out] + [bc_pressure]

        cashocs.newton_solve(F, up, bcs, verbose=False)

        J = cashocs.IntegralFunctional(
            Constant(0.5) * dot(u - u_des, u - u_des) * ds(5)
        )
        self.cost_functional_value = J.evaluate()

        u_temp, _ = up.split(True)
        u_temp.set_allow_extrapolation(True)
        self.u.vector().vec().aypx(
            0.0, interpolate(u_temp, self.V_coarse.sub(0).collapse()).vector().vec()
        )
        self.u.vector().apply("")

        if MPI.rank(MPI.comm_world) == 0:
            subprocess.run(["rm", f"{dir_path}/sm_mesh/fine.msh"], check=True)
            subprocess.run(["rm", f"{dir_path}/sm_mesh/fine.xdmf"], check=True)
            subprocess.run(["rm", f"{dir_path}/sm_mesh/fine.h5"], check=True)
            subprocess.run(
                ["rm", f"{dir_path}/sm_mesh/fine_boundaries.xdmf"], check=True
            )
            subprocess.run(["rm", f"{dir_path}/sm_mesh/fine_boundaries.h5"], check=True)
            subprocess.run(
                ["rm", f"{dir_path}/sm_mesh/fine_subdomains.xdmf"], check=True
            )
            subprocess.run(["rm", f"{dir_path}/sm_mesh/fine_subdomains.h5"], check=True)


up = Function(V)

u, p = split(up)
vq = Function(V)
v, q = split(vq)

F = (
    inner(grad(u), grad(v)) * dx
    + Constant(Re_c) * inner(grad(u) * u, v) * dx
    - p * div(v) * dx
    - q * div(u) * dx
)

bc_in = DirichletBC(V.sub(0), u_in, boundaries, 1)
bcs_wall = cashocs.create_dirichlet_bcs(
    V.sub(0), Constant((0.0, 0.0)), boundaries, [2, 3, 4]
)
bc_out = DirichletBC(V.sub(0).sub(0), Constant(0.0), boundaries, 5)
bc_pressure = DirichletBC(V.sub(1), Constant(0.0), boundaries, 5)
bcs = [bc_in] + bcs_wall + [bc_out] + [bc_pressure]

J = cashocs.IntegralFunctional(Constant(0.5) * dot(u - u_des, u - u_des) * ds(5))


def test_sosm_broyden_good():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=2,
        tol=1e-2,
        use_backtracking_line_search=False,
        broyden_type="good",
        memory_size=2,
    )

    space_mapping.solve()

    assert np.abs(fine_model.cost_functional_value - 7.064559500661198e-07) <= 1e-10


def test_sosm_broyden_bad():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=3,
        tol=1e-2,
        use_backtracking_line_search=False,
        broyden_type="bad",
        memory_size=2,
    )

    space_mapping.solve()

    assert np.abs(fine_model.cost_functional_value - 5.7097314142792985e-08) <= 1e-10


def test_sosm_bfgs():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="bfgs",
        max_iter=3,
        tol=1e-2,
        use_backtracking_line_search=False,
        memory_size=3,
    )

    space_mapping.solve()

    assert np.abs(fine_model.cost_functional_value - 4.054967883850159e-08) <= 1e-10


def test_sosm_steepest_descent():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="sd",
        max_iter=5,
        tol=1e-2,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 4.1434175702773817e-07) <= 1e-10


def test_sosm_ncg_FR():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="FR",
        max_iter=4,
        tol=1e-2,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 9.097862180684929e-08) <= 1e-10


def test_sosm_ncg_PR():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="PR",
        max_iter=7,
        tol=1e-2,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 1.2796532274209625e-06) <= 1e-10


def test_sosm_ncg_HS():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="HS",
        max_iter=2,
        tol=3e-1,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.0012610364722555078) <= 1e-10


def test_sosm_ncg_DY():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="DY",
        max_iter=3,
        tol=1e-2,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 2.1065112245987872e-07) <= 1e-10


def test_sosm_ncg_HZ():
    mesh.coordinates()[:, :] = initial_coordinates
    mesh.bounding_box_tree().build(mesh)

    coarse_model = sosm.CoarseModel(F, bcs, J, up, vq, boundaries, config=cfg)
    fine_model = FineModel(mesh, V)
    up_param = Function(V)
    u_param, p_param = split(up_param)
    u_des_param = fine_model.u
    J_param = cashocs.IntegralFunctional(
        Constant(0.5) * dot(u_param - u_des_param, u_param - u_des_param) * ds(5)
    )
    parameter_extraction = sosm.ParameterExtraction(
        coarse_model, J_param, up_param, config=cfg
    )

    space_mapping = sosm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="HZ",
        max_iter=4,
        tol=1e-2,
        use_backtracking_line_search=False,
    )

    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 8.460165047783625e-07) <= 1e-10

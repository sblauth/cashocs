"""
Created on 13/08/2021, 09.06

@author: blauths
"""

import pathlib

from fenics import *
import numpy as np

import cashocs
import cashocs.space_mapping.optimal_control as ocsm

dir_path = str(pathlib.Path(__file__).parent)

nonlinearity_factor = 5e2

cfg = cashocs.load_config(f"{dir_path}/config_ocsm.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)
V = FunctionSpace(mesh, "CG", 1)

y_des = Expression(
    "scale*(pow(x[0], 2)*(1 - x[0])*pow(x[1], 2)*(1 - x[1]))",
    degree=6,
    scale=729 / 16,
)
bcs = cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, [1, 2, 3, 4])


class FineModel(ocsm.FineModel):
    def __init__(self):
        super().__init__()

        self.controls = Function(V)
        self.y = Function(V)
        phi = TestFunction(V)

        self.F = (
            dot(grad(self.y), grad(phi)) * dx
            + Constant(nonlinearity_factor) * pow(self.y, 3) * phi * dx
            - self.controls * phi * dx
        )
        self.J = cashocs.IntegralFunctional(Constant(0.5) * pow(self.y - y_des, 2) * dx)

    def solve_and_evaluate(self):
        self.y.vector().vec().set(0.0)
        cashocs.newton_solve(self.F, self.y, bcs, verbose=False)
        self.cost_functional_value = self.J.evaluate()


fine_model = FineModel()

y = Function(V)
u = Function(V)
p = Function(V)

F = dot(grad(y), grad(p)) * dx - u * p * dx
J = cashocs.IntegralFunctional(Constant(0.5) * pow(y - y_des, 2) * dx)

coarse_model = ocsm.CoarseModel(F, bcs, J, y, u, p, config=cfg)

y_sm = Function(V)
u_sm = Function(V)
J_parameter = cashocs.IntegralFunctional(
    Constant(0.5) * pow(y_sm - fine_model.y, 2) * dx
)

parameter_extraction = ocsm.ParameterExtraction(
    coarse_model, J_parameter, y_sm, u_sm, config=cfg
)


def test_ocsm_parameter_extraction_single():
    config = cashocs.load_config(dir_path + "/config_ocsm.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
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

    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)
    ocp.solve()
    control_ocp = u.vector()[:]

    u.vector().vec().set(0.0)
    u.vector().apply("")
    coarse_model = ocsm.CoarseModel(F, bcs, J, y, u, p, config=config)
    coarse_model.optimize()
    control_coarse_model = u.vector()[:]

    assert np.allclose(control_ocp, control_coarse_model)

    u_pe = Function(V)
    y_pe = Function(V)
    J_pe = cashocs.IntegralFunctional(
        Constant(0.5) * (y_pe - y_d) * (y_pe - y_d) * dx
        + Constant(0.5 * alpha) * u_pe * u_pe * dx
    )

    parameter_extraction = ocsm.ParameterExtraction(
        coarse_model, J_pe, y_pe, u_pe, config=config
    )
    parameter_extraction._solve()
    control_pe = u_pe.vector()[:]
    assert np.allclose(control_ocp, control_pe)


def test_ocsm_parameter_extraction_multiple():
    config = cashocs.load_config(dir_path + "/config_ocsm.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    z = Function(V)
    p = Function(V)
    q = Function(V)
    u = Function(V)
    v = Function(V)

    states = [y, z]
    adjoints = [p, q]
    controls = [u, v]

    e_y = inner(grad(y), grad(p)) * dx - u * p * dx
    e_z = inner(grad(z), grad(q)) * dx - (y + v) * q * dx

    e = [e_y, e_z]

    bcs1 = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])
    bcs2 = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

    bcs_list = [bcs1, bcs2]

    y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
    z_d = Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)
    alpha = 1e-6
    beta = 1e-4
    J = cashocs.IntegralFunctional(
        Constant(0.5) * (y - y_d) * (y - y_d) * dx
        + Constant(0.5) * (z - z_d) * (z - z_d) * dx
        + Constant(0.5 * alpha) * u * u * dx
        + Constant(0.5 * beta) * v * v * dx
    )

    ocp = cashocs.OptimalControlProblem(
        e, bcs_list, J, states, controls, adjoints, config
    )
    ocp.solve()
    u_ocp = u.vector()[:]
    v_ocp = v.vector()[:]

    u.vector().vec().set(0.0)
    u.vector().apply("")
    v.vector().vec().set(0.0)
    v.vector().apply("")
    coarse_model = ocsm.CoarseModel(e, bcs_list, J, states, controls, adjoints, config)
    coarse_model.optimize()
    u_cm = u.vector()[:]
    v_cm = v.vector()[:]

    assert np.max(np.abs(u_ocp - u_cm)) / np.max(np.abs(u_ocp)) < 1e-10
    assert np.max(np.abs(v_ocp - v_cm)) / np.max(np.abs(v_ocp)) < 1e-10

    u_pe = Function(V)
    v_pe = Function(V)
    controls_pe = [u_pe, v_pe]
    y_pe = Function(V)
    z_pe = Function(V)
    states_pe = [y_pe, z_pe]
    J = cashocs.IntegralFunctional(
        Constant(0.5) * (y_pe - y_d) * (y_pe - y_d) * dx
        + Constant(0.5) * (z_pe - z_d) * (z_pe - z_d) * dx
        + Constant(0.5 * alpha) * u_pe * u_pe * dx
        + Constant(0.5 * beta) * v_pe * v_pe * dx
    )

    parameter_extraction = ocsm.ParameterExtraction(
        coarse_model, J, states_pe, controls_pe, config
    )
    parameter_extraction._solve()

    assert np.max(np.abs(u_ocp - u_pe.vector()[:])) / np.max(np.abs(u_ocp)) < 1e-10
    assert np.max(np.abs(v_ocp - v_pe.vector()[:])) / np.max(np.abs(v_ocp)) < 1e-10


def test_ocsm_broyden_good():
    u.vector().vec().set(0.0)
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=9,
        tol=1e-1,
        use_backtracking_line_search=False,
        broyden_type="good",
        memory_size=4,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 9.940664295004186e-05) <= 1e-8


def test_ocsm_broyden_bad():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="broyden",
        max_iter=6,
        tol=1e-1,
        use_backtracking_line_search=False,
        broyden_type="bad",
        memory_size=4,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.00034133147129726136) <= 2e-8


def test_ocsm_bfgs():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="bfgs",
        max_iter=5,
        tol=1e-1,
        use_backtracking_line_search=False,
        memory_size=5,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.00015157187957497788) <= 1e-7


def test_ocsm_steepest_descent():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="sd",
        max_iter=8,
        tol=2.5e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.008607376518100516) <= 2e-8


def test_ocsm_ncg_FR():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="FR",
        max_iter=9,
        tol=1e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.001158786222806518) <= 1e-7


def test_ocsm_ncg_PR():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="PR",
        max_iter=9,
        tol=2.5e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.008117963107823976) <= 1e-7


def test_ocsm_ncg_HS():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="HS",
        max_iter=11,
        tol=2.5e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.00864625490666504) <= 6e-5


def test_ocsm_ncg_DY():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="DY",
        max_iter=8,
        tol=1e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.00048424837926872125) <= 1e-7


def test_ocsm_ncg_HZ():
    u.vector().vec().set(0.0)
    u.vector().apply("")
    space_mapping = ocsm.SpaceMapping(
        fine_model,
        coarse_model,
        parameter_extraction,
        method="ncg",
        cg_type="HZ",
        max_iter=4,
        tol=2.5e-1,
        use_backtracking_line_search=False,
    )
    space_mapping.solve()
    assert np.abs(fine_model.cost_functional_value - 0.005701000695522027) <= 2e-7

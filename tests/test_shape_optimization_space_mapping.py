# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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

"""Test for the space mapping in the context of shpae optimization."""

from fenics import *
import numpy as np
import pytest

import cashocs
import cashocs._cli
import cashocs.space_mapping.shape_optimization as sosm

pytestmark = pytest.mark.slow


cashocs.set_log_level(cashocs.log.ERROR)


@pytest.fixture(autouse=True)
def initialize_module_state(request, config_dir, copy_test_asset):
    module = request.module
    module.sm_mesh_dir = copy_test_asset("data/meshes/sm_mesh")
    module.cfg = cashocs.load_config(str(config_dir / "config_sosm.ini"))
    module.Re_f = 50.0
    (
        module.mesh,
        module.subdomains,
        module.boundaries,
        module.dx,
        module.ds,
        module.dS,
    ) = cashocs.import_mesh(str(module.sm_mesh_dir / "mesh.xdmf"))
    module.initial_coordinates = module.mesh.coordinates().copy()
    module.u_in = Expression(("6.0*(0.0 - x[1])*(x[1] + 1.0)", "0.0"), degree=2)
    module.u_des = Expression(
        (
            "0.0",
            "(x[0] >= 1.75 && x[0] <= 2.5) ? -C*6.0/pow(0.75, 2)*(x[0] - 1.75)*(2.5 - x[0]) : "
            + "(x[0] >= 3.5 && x[0] <= 4.25) ? -C*6.0/pow(0.75, 2)*(x[0] - 3.5)*(4.25 - x[0]) : "
            + "(x[0] >= 5.25 && x[0] <= 6) ? -C*6.0/pow(0.75, 2)*(x[0] - 5.25)*(6 - x[0]) : 0.0",
        ),
        degree=2,
        C=1.0 / 3.0 / 0.75,
        domain=module.mesh,
    )
    v_elem = VectorElement("CG", module.mesh.ufl_cell(), 2)
    p_elem = FiniteElement("CG", module.mesh.ufl_cell(), 1)
    module.V = FunctionSpace(module.mesh, MixedElement([v_elem, p_elem]))
    module.up = Function(module.V)
    module.u, module.p = split(module.up)
    module.vq = Function(module.V)
    module.v, module.q = split(module.vq)
    module.F = (
        inner(grad(module.u), grad(module.v)) * module.dx
        - module.p * div(module.v) * module.dx
        - module.q * div(module.u) * module.dx
    )
    bc_in = DirichletBC(module.V.sub(0), module.u_in, module.boundaries, 1)
    bcs_wall = cashocs.create_dirichlet_bcs(
        module.V.sub(0), Constant((0.0, 0.0)), module.boundaries, [2, 3, 4]
    )
    bc_out = DirichletBC(module.V.sub(0).sub(0), Constant(0.0), module.boundaries, 5)
    bc_pressure = DirichletBC(module.V.sub(1), Constant(0.0), module.boundaries, 5)
    module.bcs = [bc_in] + bcs_wall + [bc_out] + [bc_pressure]
    module.J = cashocs.IntegralFunctional(
        Constant(0.5)
        * dot(module.u - module.u_des, module.u - module.u_des)
        * module.ds(5)
    )


class FineModel(sosm.FineModel):
    def __init__(self, mesh, V):
        super().__init__(mesh)
        self.V_coarse = V
        self.u = Function(self.V_coarse.sub(0).collapse())

    def solve_and_evaluate(self):
        fine_mesh_file = sm_mesh_dir / "fine.msh"
        fine_xdmf_file = fine_mesh_file.with_suffix(".xdmf")
        cashocs.io.write_out_mesh(
            self.mesh, str(sm_mesh_dir / "mesh.msh"), str(fine_mesh_file)
        )
        cashocs.convert(str(fine_mesh_file), str(fine_xdmf_file))

        mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh(
            str(fine_xdmf_file)
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

        cashocs.snes_solve(F, up, bcs)

        J = cashocs.IntegralFunctional(
            Constant(0.5) * dot(u - u_des, u - u_des) * ds(5)
        )
        self.cost_functional_value = J.evaluate()

        u_temp, _ = up.split(True)
        interpolator = cashocs._utils.Interpolator(
            u_temp.function_space(), self.V_coarse.sub(0).collapse()
        )
        self.u.vector().vec().aypx(0.0, interpolator.interpolate(u_temp).vector().vec())
        self.u.vector().apply("")


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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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

    space_mapping = sosm.SpaceMappingProblem(
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
